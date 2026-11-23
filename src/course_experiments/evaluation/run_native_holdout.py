"""三个原生方法共用密封保留题；复用已完成全量索引，禁止重新构图。"""
from __future__ import annotations

import argparse
import copy
import json
import math
import importlib
from pathlib import Path
import time

from ..manifest import digest, file_record, read_document
from . import dual_view_tasks as tasks
from . import run_native_full as shared
from .native_provider import AuditedNativeProvider
from .protocol import runtime_for
from .run_repaired_full import credentials

SELECTION='dvkg_development_answers/74e07b560ec974c52fdc58b9ce425b943b74d22bbab226a780bd0b2fe73ab107/selection.json'


class _MicrosoftOnly:
    # Light/Hippo原APP不包含后建的Graph runner，只在Graph轨加载。
    def __getattr__(self,name):
        return getattr(importlib.import_module('.run_native_microsoft_full',__package__),name)


microsoft=_MicrosoftOnly()


class DependencyPending(RuntimeError):
    pass


def checked(record):
    if file_record(record['path'])!=record:raise ValueError('Frozen holdout dependency changed')
    return read_document(record['path'])


def load_holdout(sealed_path,selection_path):
    if not Path(sealed_path).is_file() or not Path(selection_path).is_file():
        raise DependencyPending('Sealed holdout and frozen development selection required')
    seal,selection=read_document(sealed_path),read_document(selection_path)
    if (selection.get('status')!='frozen_development_choice' or selection.get('confirmation_questions_used_for_selection')!=0 or
        selection.get('new_holdout_not_opened') is not True or selection.get('selected_variant') not in ['unchanged_v1','occurrence_mapping_v2'] or
        type(selection.get('created_unix')) not in [int,float] or not math.isfinite(selection['created_unix'])):
        raise ValueError('Independent quality evaluation requires the prior frozen development-only choice')
    if file_record(selection['source']['path'])!=selection['source']:
        raise ValueError('Frozen development selection source changed')
    if seal.get('status') not in ['sealed_complete','sealed_insufficient'] or seal.get('target_count')!=128:
        raise DependencyPending('Independent holdout must be sealed; insufficient is a final usable set')
    checked(seal['contract'])
    index=checked(seal['dataset_index']);books={};ids=set();counts={}
    if len(index['books'])!=8:raise ValueError('The same sealed eight-book collection is required')
    for book,record in index['books'].items():
        dataset=checked(record);questions=dataset['questions']
        if dataset['target_book_id']!=book:raise ValueError('Sealed book differs from dataset index')
        kinds={kind:0 for kind in tasks.TYPES}
        for question in questions:
            if question['target_book_id']!=book or question['type'] not in kinds or question['question_id'] in ids:
                raise ValueError('Duplicate or foreign sealed question')
            ids.add(question['question_id']);kinds[question['type']]+=1
        if any(v>4 for v in kinds.values()):raise ValueError('Sealed per-type preregistration exceeded')
        counts[book]={kind:dict(target=4,accepted=n,shortage=4-n) for kind,n in kinds.items()}
        books[book]=dataset
    n=len(ids)
    if (n!=seal['accepted_count'] or n!=index['accepted_count'] or seal['shortage']!=128-n or
        (seal['status']=='sealed_complete')!=(n==128)):
        raise ValueError('Sealed actual question denominator differs')
    return dict(books=books,book_records=index['books'],counts=counts,seal=seal,sealed_record=file_record(sealed_path),
        selection_record=file_record(selection_path),selected_variant=selection['selected_variant'])


def holdout_plan(base,data,dependency):
    book=base['book']
    if book not in data['books'] or set(data['books'])!=set(base['allowed_books']):
        raise ValueError('Target books differ from the common authorized sealed set')
    selected=data['books'][book]
    for question in selected['questions']:
        canonical=tasks.validate_task(question,base['structure']['evidence'],book,question['type'])
        if any(question.get(k)!=v for k,v in canonical.items()):raise ValueError('Sealed question no longer matches original source validation')
    plan=dict(base)
    plan['datasets']={'independent_confirmation':selected}
    contract=copy.deepcopy(base['contract'])
    contract['datasets']={'independent_confirmation':data['book_records'][book]}
    contract['dataset_index']=data['seal']['dataset_index']
    contract['global_question_counts']={b:{'independent_confirmation':len(v['questions'])} for b,v in data['books'].items()}
    contract['policy']=dict(contract['policy'],confirmation='all_actual_legal_questions_in_the_same_sealed_independent_set',
        development='frozen_choice_only_no_new_development_QA',failed_question_policy='real_sealed_questions_fixed_denominator_zero_for_incomplete')
    contract['independent_holdout']=dict(evaluation_role='independent_holdout',sealed=data['sealed_record'],
        development_selection=data['selection_record'],selected_DVKG_variant=data['selected_variant'],target_count=128,
        accepted_count=data['seal']['accepted_count'],shortage=data['seal']['shortage'],by_book_type=data['counts'],
        denominator='actual_legal_sealed_questions;target128_and_shortage_separate;no_fake_gold_for_empty_slots',
        reused_index_dependency=dependency,wrapper=file_record(__file__),index_rebuild_allowed=False,
        common_judge=shared.POLICY['judge_version'])
    plan['contract']=contract
    if plan['index_contract']!=base['index_contract'] or digest(plan['index_contract'])!=base['contract']['index_identity']:
        raise ValueError('Holdout must not change the full original index contract')
    return plan


def alive(pid):
    proc=Path('/proc')/str(pid)
    try:return (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def index_pins(run_root,method):
    books=sorted(read_document(shared.PARENT/'datasets/index.json')['books'])
    if len(books)!=8:raise ValueError('Exact eight original books required for native holdout pins')
    pins={}
    for book in books:
        base=microsoft.prepare_plan(run_root,book) if method=='graphrag' else shared.prepare_plan(run_root,method,book)
        identity=digest(base['index_contract'])
        pins[book]=dict(method=method,book=book,index_identity=identity,index_contract=base['index_contract'],
            index_root=str(Path(run_root)/'native_indices'/method/book/identity[:16]),
            original_index_code_shas={k:v['sha256'] for k,v in base['contract']['code'].items()})
    return pins


def finished_regression(value,method,counts):
    tracks=value.get('tracks',{}) if method=='graphrag' else {'native':value}
    expected={'local','global'} if method=='graphrag' else {'native'}
    if set(tracks)!=expected:return False
    for row in tracks.values():
        splits=row.get('splits',{})
        if set(splits)!={'development','confirmation'}:return False
        for split,expected_count in counts.items():
            result=splits[split]
            if result.get('question_count')!=expected_count or result.get('pending_count')!=0:return False
            if result.get('completed_count',0)+result.get('failed_count',0)!=expected_count:return False
    return True


def locate_regression(base,run_root,process_alive=alive):
    run_root=Path(run_root);method,book=base['method'],base['book'];index_id=digest(base['index_contract'])
    index_root=run_root/'native_indices'/method/book/index_id[:16]
    receipt=index_root/'index_receipt.json';contract=index_root/'contract.json'
    if not receipt.is_file() or not contract.is_file():raise DependencyPending('Full original index receipt not yet complete')
    stored=read_document(contract)
    if stored!={'index_identity':index_id,**base['index_contract']}:raise ValueError('Full original index contract differs')
    original=shared._read_checkpoint(receipt,index_id)
    candidates=[]
    for path in (run_root/'native_full'/method/book).glob('*/contract.json'):
        value=read_document(path)
        if value.get('index_identity')==index_id and not value.get('independent_holdout') and not value.get('query_mapping_revision'):
            if value.get('job_identity')!=digest({k:v for k,v in value.items() if k!='job_identity'}):
                raise ValueError('Original full-regression contract identity changed')
            counts=value.get('global_question_counts',{})
            if (value.get('method')!=method or value.get('book')!=book or len(counts)!=8 or
                sum(v.get('confirmation',0) for v in counts.values())!=313 or
                sum(v.get('development',0) for v in counts.values())!=32):
                raise ValueError('Original full-regression scope is not the frozen313+32')
            candidates.append(path.parent)
    if len(candidates)!=1:raise DependencyPending('Exactly one original full-regression job must bind this index')
    regression=candidates[0];state_path=regression/'state.json'
    if not state_path.is_file():raise DependencyPending('Original full-regression worker has not reached an observable state')
    state=read_document(state_path)
    if process_alive(state['pid']):raise DependencyPending('Original full-regression worker still running; no concurrent index reuse')
    result_path=regression/'result.json'
    if not result_path.is_file():raise DependencyPending('Original full-regression result not finalized; original QA may still be queued for recovery')
    result=read_document(result_path)
    regression_contract=read_document(regression/'contract.json')
    if result.get('job_identity')!=regression_contract['job_identity'] or not finished_regression(result,method,regression_contract['global_question_counts'][book]):
        raise DependencyPending('Original full-regression QA has unattempted questions; wait for its full worker recovery')
    return dict(index_root=str(index_root),index_identity=index_id,native_index=original['native_index'],
        index_receipt=file_record(receipt),index_contract=file_record(contract),regression_root=str(regression),
        regression_contract=file_record(regression/'contract.json'),regression_result=file_record(result_path),
        worker_exit_state=file_record(state_path),worker_pid=state['pid'])


class RequestGate:
    def __init__(self,original):self.original,self.allow=original,False
    def __call__(self,*args,**kwargs):
        if not self.allow:raise RuntimeError('Holdout index reuse attempted model access; rebuilding is forbidden')
        return self.original(*args,**kwargs)


def bind_hippo_existing(adapter,rows,native_receipt):
    from .native_hipporag import prepare_documents,_digest
    docs,identity=prepare_documents(rows,adapter.target_book_id,adapter.allowed_book_ids)
    path=adapter.save_dir/'native_corpus_identity.json'
    if read_document(path)!=identity or native_receipt['corpus_identity']!=identity:
        raise ValueError('Existing Hippo corpus identity differs from the complete target book')
    if (not adapter.native._graph_state_available or adapter.native.graph.vcount()<len(docs) or
        set(adapter.native.chunk_embedding_store.get_all_texts())!={d['text'] for d in docs}):
        raise ValueError('Persisted complete Hippo graph/passages are unavailable; rebuilding forbidden')
    adapter.corpus_identity=identity
    adapter._passage_metadata={_digest(d['text']):d['metadata'] for d in docs}
    return native_receipt


async def run_existing(plan,run_root,provider,dependency):
    expected=Path(dependency['index_root'])
    async def factory(current,index_root,owner):
        if Path(index_root)!=expected:raise ValueError('Holdout attempted a different index directory')
        original=owner.request_executor;gate=RequestGate(original);owner.request_executor=gate
        adapter=None
        try:
            adapter=await shared.open_adapter(current,index_root,owner)
            if current['method']=='hipporag':
                def reuse(rows):
                    gate.allow=False
                    try:return bind_hippo_existing(adapter,rows,dependency['native_index'])
                    finally:gate.allow=True
                reuse(current['passages']);adapter.index=reuse
            else:
                manifest=adapter._manifest();native=dependency['native_index']
                if any(native.get(k)!=v for k,v in manifest.items()):raise ValueError('Existing Light manifest differs from full index receipt')
                async def reuse(book,documents):
                    from .native_lightrag import _digest
                    import hashlib
                    records=[dict(document_id=d.document_id,text_sha256=hashlib.sha256(d.text.encode()).hexdigest(),text_utf8_bytes=len(d.text.encode())) for d in documents]
                    if book!=current['book'] or records!=manifest['records']:raise ValueError('Existing Light source corpus differs')
                    ids=['doc-'+_digest([book,d.document_id]) for d in documents]
                    statuses=await adapter.core.doc_status.get_by_ids(ids)
                    if len(statuses)!=len(ids) or any(not s or s.get('status')!='processed' for s in statuses):
                        raise ValueError('Light corpus has an incomplete source; no insertion allowed')
                    return native
                adapter.index_book=reuse
            return adapter
        except BaseException:
            if adapter is not None:
                await shared._await(adapter.close())
            raise
        finally:
            gate.allow=True;owner.request_executor=original
    async def opener(current,index_root,owner):
        if Path(index_root)!=expected/'storage':raise ValueError('Holdout attempted a different Graph index directory')
        from .native_microsoft_full import open_microsoft
        from .native_microsoft_graphrag import NativeMicrosoftGraphRAG
        original=owner.request_executor;gate=RequestGate(original);owner.request_executor=gate
        def frozen_factory(**kwargs):
            adapter=NativeMicrosoftGraphRAG.create(**kwargs)
            complete=adapter._completed()
            if complete!=dependency['native_index']['identity']:raise ValueError('Existing Graph complete index differs from original receipt')
            manifest=read_document(adapter.working_dir/'native_adapter_manifest.json')
            async def reuse(book,documents):
                import hashlib
                sources=[dict(id=d.document_id,file=digest(d.document_id)+'.txt',sha256=hashlib.sha256(d.text.encode()).hexdigest()) for d in documents]
                if book!=current['book'] or sources!=manifest['sources']:raise ValueError('Existing Graph full source corpus differs')
                return complete
            adapter.index_book=reuse
            return adapter
        try:
            session=await open_microsoft(current,index_root,owner,adapter_factory=frozen_factory)
            session.index_receipt=dependency['native_index']
            return session
        finally:
            gate.allow=True;owner.request_executor=original
    if plan['method']=='graphrag':return await microsoft.run_book(plan,run_root,provider,opener=opener,query_fn=query_with_source_mapping)
    return await shared.run_book(plan,run_root,provider,adapter_factory=factory)


async def query_with_source_mapping(session,plan,question,*,method,output_dir):
    from .native_microsoft_query_mapping import remap_native_result
    native=await microsoft.query_microsoft(session,plan,question,method=method,output_dir=output_dir)
    return remap_native_result(session,native)


def prepare(run_root,method,book,sealed,selection):
    data=load_holdout(sealed,selection)
    base=microsoft.prepare_plan(run_root,book) if method=='graphrag' else shared.prepare_plan(run_root,method,book)
    dependency=locate_regression(base,run_root)
    if method=='graphrag':
        from .native_microsoft_query_mapping import bind_query_revision
        base=bind_query_revision(base,dependency['index_contract'])
    return holdout_plan(base,data,dependency),dependency


def main():
    import asyncio
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--method',choices=['hipporag','lightrag','graphrag'],required=True);parser.add_argument('--book')
    parser.add_argument('--sealed',type=Path);parser.add_argument('--selection',type=Path)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--wait',action='store_true')
    parser.add_argument('--index-pins-only',action='store_true');parser.add_argument('--expected-index-identity')
    args=parser.parse_args();sealed=args.sealed or args.run_root/'independent_holdout_128/sealed.json'
    selection=args.selection or args.run_root/SELECTION
    limits=read_document(args.run_root/'formal_provider/limits.json')
    if limits['limits']!=shared.LIMITS or not (args.run_root/'formal_provider/audit.sqlite').is_file():
        raise ValueError('Existing formal provider must be reused; no initialization/reset')
    if args.index_pins_only:
        print(json.dumps(dict(status='index_pins_no_api',method=args.method,pins=index_pins(args.run_root,args.method),
            provider_sha256=file_record(Path(__file__).parent/'native_provider.py')['sha256'],model_calls=0,raw_text_exported=False)));return 0
    if not args.book:parser.error('--book is required unless --index-pins-only')
    while True:
        try:plan,dependency=prepare(args.run_root,args.method,args.book,sealed,selection);break
        except DependencyPending as exc:
            if args.check_only or not args.wait or time.time()>=limits['started_unix']+shared.LIMITS['max_seconds']:
                print(json.dumps(dict(status='dependencies_pending',reason=str(exc),model_calls=0,raw_text_exported=False)));return 2
            time.sleep(10)
    if args.expected_index_identity and dependency['index_identity']!=args.expected_index_identity:
        raise ValueError('Method-bound holdout index differs from its frozen per-APP pin')
    details=plan['contract']['independent_holdout']
    metadata=dict(method=args.method,book=args.book,target_count=128,accepted_count=details['accepted_count'],shortage=details['shortage'],
        book_question_count=len(plan['datasets']['independent_confirmation']['questions']),index_identity=dependency['index_identity'],
        reused_index_receipt=dependency['index_receipt'],raw_text_exported=False)
    if args.check_only:
        print(json.dumps(dict(status='ready_no_api',model_calls=0,**metadata)));return 0
    if not metadata['book_question_count']:
        print(json.dumps(dict(status='sealed_book_has_no_legal_questions',model_calls=0,**metadata)));return 0
    credentials()
    transport=args.run_root/'native_full'/args.method/args.book/digest(plan['contract'])[:16]/'transport'
    provider=AuditedNativeProvider(runtime_for(plan['construction'],transport),args.run_root/'formal_provider',**shared.LIMITS)
    result=asyncio.run(run_existing(plan,args.run_root,provider,dependency))
    print(json.dumps(dict(status=result['status'],summary=result,**metadata)))
    return 0 if result['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
