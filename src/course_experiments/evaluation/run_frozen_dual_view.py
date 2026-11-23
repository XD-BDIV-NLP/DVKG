"""冻结开发选择后的同规则回归与独立确认；旧结果只读复用。"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

from filelock import FileLock
import numpy as np

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import dual_view_run as old
from .dual_view_rag import METHODS,DualViewRetriever
from .dvkg_development import CompletedMappingsRetriever
from .native_provider import AuditedNativeProvider,NativeAccountUnavailable,NativeBudgetExceeded
from .protocol import runtime_for
from .repaired_full import request_identity
from .run_independent_holdout import HoldoutBudgetExceeded,LocalHTTPBudget
from .run_native_full import JUDGE,LIMITS,blind_judge
from .run_repaired_full import atomic,credentials

PARENT=(Path(__file__).resolve().parents[2] / 'outputs/target_book_dual_view_20261021_v1')
SELECTION='dvkg_development_answers/74e07b560ec974c52fdc58b9ce425b943b74d22bbab226a780bd0b2fe73ab107/selection.json'
POLICY=dict(version='frozen_dual_view_v2',methods=list(METHODS),answer_prompt=old.ANSWER,judge_prompt=JUDGE,
    judge_version='native_blind_original_source_v1',retrieval=old.POLICY,failed_denominator='all_frozen_questions_zero_for_failed',
    answer_reuse='exact_body_and_checked_original_receipt_only;judge_recomputed_with_common_native_rubric',
    independent_confirmation='sealed_same_set_all_methods;never_quality_selected',human_audited=False)


class FrozenBudget(LocalHTTPBudget):
    def __init__(self,provider,root,identity):
        super().__init__(provider,root,identity,cap=10000)
        self.prefix='frozen_dual/'+identity+'/'


def selected_variant(selection):
    if (selection.get('status')!='frozen_development_choice' or selection.get('confirmation_questions_used_for_selection')!=0
            or selection.get('new_holdout_not_opened') is not True
            or selection.get('selected_variant') not in {'unchanged_v1','occurrence_mapping_v2'}):
        raise ValueError('A preregistered development-only selection is required')
    return selection['selected_variant']


def jobs(assets):
    return [dict(cohort=cohort,book=book,method=method) for cohort,row in assets['cohorts'].items()
            for book in row['active_books'] for method in METHODS]


def old_answer(base,key,expected_body):
    dbpath=Path(base)/'request_cache.sqlite'
    if not dbpath.exists():return None
    with closing(sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True,timeout=30)) as db:
        found=db.execute('SELECT record FROM responses WHERE key=?',(key,)).fetchone()
    if found is None:return None
    stored=json.loads(found[0]);source=stored['source']
    if file_record(source['path'])!=source:raise ValueError('Old answer receipt changed')
    raw=read_document(source['path'])
    if (raw['request']!=expected_body or digest(raw['request'])!=stored['request_digest']
            or digest(raw['response'])!=stored['response_digest']):raise ValueError('Old answer request or response changed')
    choice=raw['response']['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Cannot reuse incomplete old answer')
    value=json.loads(choice['message']['content'])
    if not isinstance(value.get('answer'),str) or not isinstance(value.get('citations'),list):raise ValueError('Invalid old candidate')
    return value,dict(reused=True,source=source,request_digest=digest(expected_body),old_score_reused=False)


def answer(provider,question,context,folder,stage):
    folder=Path(folder);payload=dict(question=question['question'],target_book_id=question['target_book_id'],context=context)
    body=dict(provider.runtime.config['llm'].get('extra_body') or {})
    body.update(model=provider.runtime.config['llm']['model'],temperature=0,max_tokens=2048,seed=42,enable_thinking=False,
        response_format={'type':'json_object'},messages=[dict(role='system',content=old.ANSWER),dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    reused=old_answer(PARENT,request_identity(old.ANSWER,payload,2048,provider.runtime.config),body)
    if reused is not None:
        value,receipt=reused;write_immutable(folder/'answer_reuse.json',receipt)
        return value,receipt
    raw=provider.chat_response(body['messages'],max_tokens=2048,response_format=body['response_format'],stage=stage)
    write_immutable(folder/'answer_raw.json',dict(response=raw))
    choice=raw['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Frozen answer incomplete')
    value=json.loads(choice['message']['content'])
    if not isinstance(value.get('answer'),str) or not isinstance(value.get('citations'),list):raise ValueError('Invalid candidate answer')
    return value,dict(reused=False,source=file_record(folder/'answer_raw.json'))


def judge(provider,question,context,candidate,folder,identity):
    return blind_judge(provider,question,dict(provided_context=context,answer=candidate['answer']),Path(folder),identity)


def retriever(assets,cohort,variant):
    row=assets['cohorts'][cohort];index=old.asset(assets['concept_indices'][cohort])
    if file_record(index['vectors']['path'])!=index['vectors']:raise ValueError('Concept vectors changed')
    cls=CompletedMappingsRetriever if variant=='occurrence_mapping_v2' else DualViewRetriever
    return cls(old.asset(row['graph']),{b:assets['structures'][b] for b in row['active_books']},old.asset(assets['mappings'][cohort]),
        {b:assets['unit_assets'][b] for b in row['active_books']},index['concept_ids'],np.load(index['vectors']['path']))


def checkpoint(path,value):
    value=dict(value);value['checkpoint_digest']=digest(value);write_immutable(path,value);return value


def evaluate(job,assets,reference,provider,root,identity,variant,budget):
    root=Path(root);reference=old.asset(reference);questions=reference['questions'];records=[]
    engine=retriever(assets,job['cohort'],variant)
    if any(q['target_book_id']!=job['book'] for q in questions):raise ValueError('Question target changed')
    for q in questions:
        folder=root/'questions'/q['question_id'];path=folder/'record.json'
        if path.exists():
            row=read_document(path)
            if row['result_id']!=identity or row['checkpoint_digest']!=digest({k:v for k,v in row.items() if k!='checkpoint_digest'}):raise ValueError('Question checkpoint changed')
            records.append(row);continue
        folder.mkdir(parents=True,exist_ok=True)
        try:
            # 已有题向量直接读取；独立确认题走同一冻结BGE与精确请求缓存。
            vector_path=PARENT/'task_vectors'/job['book']/(q['question_id']+'.npy')
            if vector_path.exists():
                meta=read_document(vector_path.with_suffix('.npy.meta.json'))['vectors']
                if file_record(vector_path)!=meta:raise ValueError('Query vector changed')
                vector=np.load(vector_path)[0]
            else:vector=provider.embed([q['question']],stage=budget.stage('query_embedding'))[0]
            selected,trace=engine.retrieve(q['question'],job['book'],vector,job['method'])
            context=old.pack_context(selected,q['question']);old.validate_context(context,job['book'])
            candidate,receipt=answer(provider,q,context,folder,budget.stage('answer'))
            judgment=judge(provider,q,context,candidate,folder,identity)
            scores=dict(old.score_answer(candidate['answer'],q['answer']),**old.citations_score(candidate['citations'],context,engine.rows),
                **old.reference_metrics(q,context,[r['evidence_ids'][0] for r in selected],engine.candidate_ids,engine.structure))
            row=dict(result_id=identity,status='completed',question_id=q['question_id'],question_type=q['type'],target_book_id=job['book'],
                candidate=candidate,context=context,retrieval=trace,machine_judge=judgment,scores=scores,answer_receipt=receipt)
        except (NativeBudgetExceeded,NativeAccountUnavailable,HoldoutBudgetExceeded):raise
        except Exception as exc:
            error=folder/'error.log';error.write_text(traceback.format_exc(),encoding='utf-8')
            row=dict(result_id=identity,status='failed',question_id=q['question_id'],question_type=q['type'],target_book_id=job['book'],
                error_type=type(exc).__name__,error_record=file_record(error))
        records.append(checkpoint(path,row))
        atomic(root/'state.json',dict(status='running',completed=sum(r['status']=='completed' for r in records),records=len(records),total=len(questions),pid=os.getpid(),updated_unix=time.time()))
    summary=dict(old.summarize(records,[q['question_id'] for q in questions]),job=job,result_id=identity,judge_version=POLICY['judge_version'],
        failed_count=sum(r['status']=='failed' for r in records),raw_text_exported=False)
    write_immutable(root/'result.json',summary);atomic(root/'state.json',dict(status=summary['status'],records=len(records),pid=os.getpid(),updated_unix=time.time()))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--split',choices=['regression','independent'],required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();base=args.run_root;selection_path=base/SELECTION
    selection=read_document(selection_path);variant=selected_variant(selection)
    assets=read_document(PARENT/'assets/result.json');index=read_document(PARENT/'datasets/index.json')['books']
    if args.split=='independent':
        sealed=read_document(base/'independent_holdout_128/sealed.json')
        index=old.asset(sealed['dataset_index'])['books']
        if sealed['status'] not in {'sealed_complete','sealed_insufficient'}:raise ValueError('Independent set is not sealed')
        datasets={b:ref for b,ref in index.items()}
    else:datasets={b:row['confirmation'] for b,row in index.items()}
    selected=jobs(assets)
    contract=dict(selection=file_record(selection_path),assets=file_record(PARENT/'assets/result.json'),datasets=datasets,split=args.split,
        policy=POLICY,code={name:file_record(Path(__file__).parent/name) for name in ['run_frozen_dual_view.py','dvkg_development.py','dual_view_run.py','dual_view_rag.py','run_native_full.py']})
    identity=digest(contract);root=base/'frozen_dual_view'/args.split/identity
    summary=dict(status='checked_no_api',split=args.split,jobs=len(selected),unique_questions=sum(len(old.asset(r)['questions']) for r in datasets.values()),
        total_question_evaluations=sum(len(old.asset(datasets[j['book']])['questions']) for j in selected),selected_variant=variant,raw_text_exported=False)
    if args.check_only:print(json.dumps(summary));return 0
    root.mkdir(parents=True,exist_ok=True);credentials()
    provider=AuditedNativeProvider(runtime_for(old.asset(assets['cohorts']['compABCD']['construction']),root/'transport'),base/'formal_provider',**LIMITS)
    try:
        with FileLock(str(root/'execution.lock'),timeout=0),FrozenBudget(provider,root,identity) as budget:
            # 原生裁判函数使用固定stage；同一调用仍通过本阶段的持久额度守卫。
            raw_chat=provider.chat_response
            def scoped_chat(messages,**kwargs):
                if kwargs.get('stage')=='native_blind_judge':kwargs['stage']=budget.stage('judge')
                return raw_chat(messages,**kwargs)
            provider.chat_response=scoped_chat
            write_immutable(root/'contract.json',dict(result_id=identity,**contract))
            if args.split=='independent':write_immutable(root/'release.json',dict(selection=file_record(selection_path),sealed=file_record(base/'independent_holdout_128/sealed.json'),method_scores_used_for_release=False))
            results=[]
            for job in selected:
                name=job['cohort']+'__'+job['book']+'__'+job['method']
                result=evaluate(job,assets,datasets[job['book']],provider,root/name,identity,variant,budget);results.append(result)
                atomic(root/'state.json',dict(status='running',jobs_finished=len(results),jobs=len(selected),last_job=job,pid=os.getpid(),updated_unix=time.time()))
            outcome=dict(summary,status='completed' if all(r['status']=='completed' for r in results) else 'partial',jobs_finished=len(results),results=results,
                local_http_attempts=budget.count(),raw_text_exported=False)
            write_immutable(root/'result.json',outcome);atomic(root/'state.json',outcome)
            print(json.dumps({k:v for k,v in outcome.items() if k!='results'}));return 0 if outcome['status']=='completed' else 2
    except Exception as exc:
        error=root/('failure_'+str(time.time_ns())+'.log');error.write_text(traceback.format_exc(),encoding='utf-8')
        atomic(root/'state.json',dict(status='interrupted_preserved_checkpoints',error_type=type(exc).__name__,error_record=file_record(error),pid=os.getpid()))
        return 2
    finally:provider.close()


if __name__=='__main__':raise SystemExit(main())
