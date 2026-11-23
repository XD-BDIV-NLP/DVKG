"""真实教材接入门禁；小样本产物不得计作完整基线结果。"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
import traceback
import sqlite3
from contextlib import closing

import numpy as np

from ..manifest import digest, file_record, read_document
from ..contributions import write_immutable
from .protocol import runtime_for
from .native_provider import AuditedNativeProvider
from filelock import FileLock
from .native_corpus import native_passages
from .run_repaired_full import atomic, credentials


PARENT = (Path(__file__).resolve().parents[2] / 'outputs/target_book_dual_view_20261021_v1')


def checked(record):
    if file_record(record['path']) != record:
        raise ValueError('Frozen native source asset changed')
    return read_document(record['path'])


def select_gate_rows(rows, question, size):
    book = question['target_book_id']
    if any(r['book_id'] != book for r in rows):
        raise ValueError('Foreign source row in target-book gate')
    expected = set(question['evidence_ids'])
    essential = [r for r in rows if set(r['evidence_ids']) & expected]
    if not expected <= {e for r in essential for e in r['evidence_ids']}:
        raise ValueError('Development support absent from target-book gate')
    if len(essential) > size:
        raise ValueError('Gate sample too small for complete development evidence')
    ordered = sorted(rows,key=lambda r:digest([book,r['evidence_ids'],r['text']]))
    keys = {digest(r) for r in essential}
    selected = essential+[r for r in ordered if digest(r) not in keys][:size-len(essential)]
    return sorted(selected,key=lambda r:digest([book,r['evidence_ids'],r['text']]))


def light_callbacks(provider):
    async def chat(prompt, *, system_prompt=None, history_messages=None, purpose, **kwargs):
        # 官方内部CoT开关不能覆盖冻结的模型thinking设置。
        kwargs.pop('enable_cot',None)
        limit = kwargs.pop('max_tokens',4096)
        response_format = kwargs.pop('response_format',None)
        if kwargs:
            raise ValueError('Unsupported native LightRAG kwargs: '+','.join(sorted(kwargs)))
        messages = ([dict(role='system',content=system_prompt)] if system_prompt else [])
        messages += list(history_messages or [])+[dict(role='user',content=prompt)]
        return await provider.achat(messages,max_tokens=limit,response_format=response_format,stage=purpose)
    async def embed(texts, *, purpose):
        return await provider.aembed(texts,stage=purpose)
    return chat,embed


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value,dict):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [plain(v) for v in value]
    if isinstance(value,(np.ndarray,np.generic)):
        return value.tolist()
    if isinstance(value,Path):
        return str(value)
    return value


async def run_light(root, source, book, rows, question, provider, state):
    from .native_lightrag import NativeLightRAG, SourceDocument
    llm,embedding = light_callbacks(provider)
    adapter = await NativeLightRAG.create(checkout=source,index_root=root/'index',
        target_book_id=book,llm_callback=llm,embedding_callback=embedding,
        check_fatal=provider.raise_if_fatal)
    try:
        before = provider.statistics()
        identity = await adapter.index_book(book,[SourceDocument(digest(r['evidence_ids']),book,r['text']) for r in rows])
        provider.raise_if_fatal()
        write_immutable(root/'index_receipt.json',dict(identity=identity,before=before,after=provider.statistics()))
        state('native_retrieve')
        retrieval = await adapter.retrieve(book,question['question'])
        if not retrieval['evidence']:
            raise ValueError('Real textbook gate returned no source evidence')
        write_immutable(root/'retrieval.json',plain(retrieval))
        state('native_qa')
        result = await adapter.query_native(book,question['question'])
        answer = result['native_result']['llm_response']['content']
        if not isinstance(answer,str) or not answer.strip():
            raise ValueError('Real textbook gate produced an empty native answer')
        provider.raise_if_fatal()
        write_immutable(root/'native_qa.json',plain(result))
        return dict(native_status=result['native_result']['status'],
            selected_evidence_count=len(retrieval['evidence']),answer_utf8_bytes=len(answer.encode()),
            index_documents=identity['native_status_count'],native_commit=result['official_commit'])
    finally:
        await adapter.close()


def run_hippo(root, source, book, rows, question, provider, state):
    from .native_hipporag import NativeHippoRAG
    adapter = NativeHippoRAG(source_root=source,save_dir=root/'index',target_book_id=book,
        allowed_book_ids={book},request_executor=provider.request_executor)
    try:
        before = provider.statistics()
        identity = adapter.index(rows)
        provider.raise_if_fatal()
        write_immutable(root/'index_receipt.json',dict(identity=identity,before=before,after=provider.statistics()))
        state('native_retrieve')
        solutions = adapter.retrieve([question['question']])
        if not solutions or not solutions[0].docs:
            raise ValueError('Real textbook gate returned no source evidence')
        write_immutable(root/'retrieval.json',dict(solutions=plain(solutions),metadata=adapter.metadata()))
        state('native_qa')
        result = adapter.rag_qa(solutions)
        write_immutable(root/'native_qa.json',dict(result=plain(result),metadata=adapter.metadata()))
        provider.raise_if_fatal()
        answer = result[0][0].answer
        if not isinstance(answer,str) or not answer.strip():
            raise ValueError('Real textbook gate produced an empty native answer')
        if any(r.get('finish_reason')!='stop' for r in result[2]):
            raise ValueError('Native QA was truncated or failed; gate remains incomplete')
        return dict(selected_evidence_count=len(solutions[0].docs),answer_utf8_bytes=len(answer.encode()),
            graph_seed_count=len(solutions[0].graph_seeds or []),native_metadata=adapter.metadata())
    finally:
        adapter.close()


async def run_graphrag(root,source,book,rows,question,provider,state):
    from .native_microsoft_graphrag import NativeMicrosoftGraphRAG,GraphSourceDocument
    adapter=NativeMicrosoftGraphRAG.create(checkout=source,index_root=root/'index',target_book_id=book,
        provider=provider,max_tokens=4096,community_level=0,check_fatal=provider.raise_if_fatal)
    before=provider.statistics()
    identity=await adapter.index_book(book,[GraphSourceDocument(digest(r['evidence_ids']),book,r['text']) for r in rows])
    provider.raise_if_fatal()
    write_immutable(root/'index_receipt.json',dict(identity=identity,before=before,after=provider.statistics()))
    state('native_retrieve')
    retrieved=await adapter.retrieve(book,question['question'])
    if not retrieved['evidence']:raise ValueError('Native GraphRAG gate returned no original source evidence')
    write_immutable(root/'retrieval.json',plain(retrieved))
    answers={}
    for method in ['local','global']:
        state('native_'+method+'_qa')
        result=await adapter.query_native(book,question['question'],method=method)
        provider.raise_if_fatal()
        write_immutable(root/('native_'+method+'_qa.json'),plain(result))
        answers[method]=dict(answer_utf8_bytes=len(result['answer'].encode()),official_commit=result['official_commit'])
    return dict(index_tables=identity['tables'],selected_evidence_count=len(retrieved['evidence']),native_answers=answers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--method',choices=['hipporag','lightrag','graphrag'],required=True)
    parser.add_argument('--book',required=True)
    parser.add_argument('--scope',choices=['gate'],default='gate')
    parser.add_argument('--sample-size',type=int,default=12)
    parser.add_argument('--input-unit',choices=['evidence','source_chunk'],default='source_chunk')
    parser.add_argument('--shared-formal-budget',action='store_true')
    args = parser.parse_args()
    credentials()
    assets = read_document(PARENT/'assets/result.json')
    if args.book not in assets['unit_assets']:
        raise ValueError('Book outside frozen authorized dataset')
    book = args.book
    unit_record = assets['unit_assets'][book]
    rows = checked(unit_record)['rows']
    structure = checked(assets['structures'][book])
    rows = [dict(r,chapter_id=structure['evidence'][r['evidence_ids'][0]]['chapter_id']) for r in rows]
    original_units=len(rows)
    passage_audit=None
    if args.input_unit=='source_chunk':
        rows,passage_audit=native_passages(rows,structure,book)
    refs = read_document(PARENT/'datasets/index.json')['books'][book]
    development = checked(refs['development'])
    question = development['questions'][0]
    selected = select_gate_rows(rows,question,args.sample_size)
    cohort = next(c for c,r in assets['cohorts'].items() if book in r['active_books'])
    construction = checked(assets['cohorts'][cohort]['construction'])
    code = {name:file_record(Path(__file__).parent/name) for name in [
        'run_native_rag.py','native_provider.py','native_hipporag.py','native_lightrag.py','native_corpus.py','native_microsoft_graphrag.py']}
    from domain_ckg_workspace.ckg_incremental import runtime as original_runtime
    code['model_runtime.py'] = file_record(original_runtime.__file__)
    contract = dict(method=args.method,scope=args.scope,book=book,source_units=unit_record,
        development=refs['development'],sample_size=args.sample_size,selected_digest=digest(selected),
        question_id=question['question_id'],code=code,input_unit=args.input_unit,passage_audit=passage_audit,
        usage='integration_only_selected_development_support_included_not_final_quality_result',
        limits=dict(max_http_attempts=1200,max_seconds=86400,max_concurrency=4),
        light_default_max_tokens=4096,base_url='https://api.siliconflow.cn/v1',
        shared_formal_budget=args.shared_formal_budget)
    identity = digest(contract)
    root = args.run_root/'gates'/args.method/book/identity[:16]
    root.mkdir(parents=True,exist_ok=True)
    write_immutable(root/'contract.json',dict(identity=identity,**contract))
    if (root/'complete.json').exists():
        print(json.dumps(dict(raw_text_exported=False,summary=read_document(root/'complete.json'))))
        return
    def state(phase,status='running',**extra):
        atomic(root/'state.json',dict(status=status,phase=phase,pid=os.getpid(),updated_unix=time.time(),
            identity=identity,**extra))
    state('native_index')
    write_immutable(root/'selected_source_rows.json',selected)
    runtime=runtime_for(construction,root/'transport')
    if args.shared_formal_budget:
        provider=AuditedNativeProvider(runtime,args.run_root/'formal_provider',
            max_http_attempts=30000,max_seconds=172800,max_concurrency=4)
        carried=read_document(args.run_root/'formal_provider/gate_carry_forward.json')['http_attempts_included_in_formal_total']
        scope=args.run_root/'formal_gate_scope.sqlite'
        with closing(sqlite3.connect(scope)) as db,db:
            db.execute('CREATE TABLE IF NOT EXISTS attempts(created REAL,request_id TEXT)')
        original_record=runtime._record
        def bounded_record(key,stage,status,value):
            if status=='request_started':
                with FileLock(str(scope)+'.lock'):
                    with closing(sqlite3.connect(scope)) as db,db:
                        count=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
                        if carried+count>=1200:raise RuntimeError('Original 1200 native gate cap exhausted')
                        original_record(key,stage,status,value)
                        db.execute('INSERT INTO attempts VALUES(?,?)',(time.time(),key))
            else:original_record(key,stage,status,value)
        runtime._record=bounded_record
        write_immutable(root/'budget_allocation.json',dict(shared_global_http_cap=30000,
            original_gate_http_cap=1200,gate_prior_calls_carried=carried,
            gate_additional_scope=str(scope),old_budgets_unchanged=True))
    else:
        provider=AuditedNativeProvider(runtime,args.run_root/'gate_provider')
    source = args.run_root/'sources'/dict(hipporag='HippoRAG',lightrag='LightRAG',graphrag='MicrosoftGraphRAG')[args.method]
    try:
        if args.method=='hipporag':
            result = run_hippo(root,source,book,selected,question,provider,state)
        elif args.method=='lightrag':
            result = asyncio.run(run_light(root,source,book,selected,question,provider,state))
        else:
            result = asyncio.run(run_graphrag(root,source,book,selected,question,provider,state))
        provider.raise_if_fatal()
        summary = dict(status='completed',scope='real_textbook_integration_gate_only',method=args.method,
            book_id=book,identity=identity,root=str(root),selected_units=len(selected),
            source_evidence_units=original_units,native_passages=len(rows),question_count=1,statistics=provider.statistics(),details=result,
            completed_unix=time.time(),full_benchmark_completed=False)
        write_immutable(root/'complete.json',summary)
        state('gate_complete',status='completed')
        print(json.dumps(dict(raw_text_exported=False,summary=summary),ensure_ascii=False))
    except Exception as exc:
        error_file = root/('failure_'+str(time.time_ns())+'.log')
        error_file.write_text(traceback.format_exc(),encoding='utf-8')
        state('failed',status='failed',error_type=type(exc).__name__,error_record=file_record(error_file),
              statistics=provider.statistics())
        raise
    finally:
        provider.close()


if __name__=='__main__':
    main()
