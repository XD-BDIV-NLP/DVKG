"""原353题双池的独立问答任务；准备零API，执行须有新的明确授权。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import threading
import time
import traceback
from types import SimpleNamespace


N=PurePosixPath(os.environ.get("DVKG_HISTORICAL_ROOT") or str(Path("outputs/complete_course_benchmark_20261022_v1").resolve()))
DEPLOYED_ROOT=N/'dvkg_evidence_upgrade_20261116/7e46e41df054027b'
CORE_SHA='bb60e953ad9d31c20848a05014ad39a1baf986e2f7c930d1c8dc4ad3e94b5bcb'
RUNTIME_REF=dict(path=str(N/'supplement_20261028/joint_handoff_v1/joint_answer_runtime.json'),sha256='aefd240245314f24d2444fddaeca85c931832cc7f71499afc83d188aa2d3e0c9')
RUNTIME_DIGEST='5b6f4bc24861a4a303c48552d514678a3e0763d4b8e74dc4dd65b716557ff67e'
HISTORICAL_ROOTS=(N/'formal_provider',DEPLOYED_ROOT/'answer_provider',N/'rag_expanded_pools_20261114_v1/formal_provider')
LIMITS=dict(max_http_attempts=1024,max_seconds=14400,max_concurrency=4)
SCOPES=('same_subject_pool','mixed_six_book_pool')
SCOPE='original353_two_pools_DVKG_rerank_answers'
PREFIX='dvkg_original353_rerank_full_20261117/answer/'
EXPECTED_COUNTS={'book_25f718562685':100,'book_49bb8d31ddb9':100,'book_0b8d87cb0667':40,
    'book_d96f472c8179':37,'book_00849acee112':38,'book_dd8183b4cb60':38}
COURSES={b:course for books,course in [(['book_25f718562685','book_49bb8d31ddb9'],'computer_organization'),
    (['book_0b8d87cb0667','book_d96f472c8179'],'operating_systems'),
    (['book_00849acee112','book_dd8183b4cb60'],'computer_networks')]for b in books}


class FullAnswerPending(RuntimeError):
    """服务或控制中断，不能作为方法零分。"""


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def record(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def checked(reference):
    if record(reference['path'])!=reference:raise ValueError('Frozen full-answer dependency changed')
    return read(reference['path'])
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Refusing to change an existing full-answer artifact')
    else:atomic(path,value)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def initialize_dependencies():
    if os.name!='posix':raise RuntimeError('Real source evidence remains on the original server')
    path=Path(str(DEPLOYED_ROOT))/'development_answer_worker.py'
    if record(path)['sha256']!=CORE_SHA:raise ValueError('Completed development answer core changed')
    core=load(path,'full_original353_isolated_answer_core')
    core.ROOT=Path(str(DEPLOYED_ROOT))
    deps=core.initialize_dependencies()
    return dict(deps,core=core,cfg=checked(deps['configuration']),historical_roots=[Path(str(p))for p in HISTORICAL_ROOTS])


def validate_rows(rows):
    if not isinstance(rows,list)or len(rows)!=706:raise ValueError('Exactly706 original question-pool records required')
    seen=set();by_book=defaultdict(set);by_question=defaultdict(list);courses=defaultdict(set)
    fields=('task_id','question_id','question','reference_answer','book_id','course_id','scope')
    for row in rows:
        if (any(not isinstance(row.get(k),str)or not row[k]for k in fields)or row.get('variant')!='rerank'
                or row.get('source')!='original353_frozen_expanded_pool_questions'or row.get('independent_test')is not False
                or row['scope']not in SCOPES or row['course_id']!=COURSES.get(row['book_id'])or not isinstance(row.get('context'),list)):
            raise ValueError('Original353 source, rerank variant and complete input fields required')
        if row['task_id']in seen:raise ValueError('Duplicate full-answer task identity')
        seen.add(row['task_id']);by_book[row['book_id']].add(row['question_id']);by_question[row['question_id']].append(row)
        courses[row['course_id']].add(row['book_id'])
    if ({b:len(v)for b,v in by_book.items()}!=EXPECTED_COUNTS or len(by_question)!=353
            or len(courses)!=3 or any(len(v)!=2 for v in courses.values())):
        raise ValueError('Original100/100/40/37/38/38 denominators and three independent courses required')
    for group in by_question.values():
        if len(group)!=2 or {r['scope']for r in group}!=set(SCOPES):raise ValueError('Every original question needs exactly both pools')
        binding={k:group[0][k]for k in ('question','reference_answer','book_id','course_id')}
        if any({k:r[k]for k in binding}!=binding for r in group):raise ValueError('Cross-pool scoring identities differ')


def source_records(root,deps):
    source=record(root/'full_rerank_retrieval_results.json');value=checked(source)
    names={'input_contract':'input_contract.json','prepared_tasks':'private_prepared_tasks.json',
        'GPU_scores':'GPU_scores.json','gpu_score_packet':'gpu_score_packet.json','finish_contract':'full_rerank_finish_contract.json'}
    for key,name in names.items():
        if value.get(key)!=record(root/name):raise ValueError('Full rerank source reference differs: '+key)
    checked(value['finish_contract'])
    input_contract=checked(value['input_contract']);prepared=checked(value['prepared_tasks'])
    identity=digest(input_contract)
    if (value.get('contract_digest')!=identity or prepared.get('contract_digest')!=identity
            or value.get('answer_generation_calls')!=0 or input_contract.get('question_count')!=353
            or input_contract.get('task_count')!=706 or input_contract.get('expected_question_counts')!=EXPECTED_COUNTS
            or set(input_contract.get('scopes',[]))!=set(SCOPES)
            or input_contract.get('comparison_policy',{}).get('variants')!=['rerank']
            or input_contract.get('comparison_policy',{}).get('added_cross_encoder')is not True
            or input_contract.get('comparison_policy',{}).get('baseline_reranker_equivalence')is not False
            or input_contract.get('comparison_policy',{}).get('graph_construction_SOTA_claim')is not False):
        raise ValueError('Full rerank preparation contract or original scope differs')
    if deps.get('cfg')is not None and input_contract['datasets']!={s['book_id']:s['dataset']for s in deps['cfg']['sources']}:
        raise ValueError('Full answer input datasets differ from frozen original353 sources')
    rows=value['records'];validate_rows(rows);expected={t['task_id']:t for t in prepared['tasks']}
    if len(prepared['tasks'])!=706 or len(expected)!=706:raise ValueError('All706 private prepared tasks required')
    for row in rows:
        prior=expected.get(row['task_id'],{})
        if any(row[k]!=prior.get(k)for k in ('question_id','question','reference_answer','book_id','course_id','scope')):
            raise ValueError('Full rerank output differs from original prepared question')
    return rows,dict(retrieval=source,**{k:value[k]for k in names}),input_contract


def ledger_snapshot(root):
    root=Path(root);database=root/'audit.sqlite'
    if not database.is_file():return dict(root=str(root),available=False,actual_http=None,max_event_id=None)
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True,timeout=30))as db:
        db.execute('BEGIN');maximum=db.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
        count=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started'AND id<=?",(maximum,)).fetchone()[0]
    return dict(root=str(root),available=True,actual_http=count,max_event_id=maximum,old_budget_reset=False)


def exact_receipt(db,key,body,runtime):
    if key!=digest(dict(endpoint='/chat/completions',body=body,config=runtime)):raise ValueError('Exact cache request key differs')
    row=db.execute('SELECT record FROM receipts WHERE key=?',(key,)).fetchone()
    if row is None:return None
    reference=json.loads(row[0]);value=checked(reference)
    if (value.get('request_id')!=key or value.get('endpoint')!='/chat/completions'or value.get('request')!=body
            or value.get('runtime_config')!=runtime):raise ValueError('Cached receipt body, endpoint or runtime differs')
    return reference


def cache_preflight(tasks,runtime,roots):
    requests={}
    for task in tasks:
        key=task['request_id'];body=task['body']
        if key!=digest(dict(endpoint='/chat/completions',body=body,config=runtime)):raise ValueError('Prepared request key differs from exact frozen body')
        if key in requests and requests[key]!=body:raise ValueError('One request key cannot identify different bodies')
        requests[key]=body
    hits={};snapshots=[]
    for root in roots:
        snapshot=ledger_snapshot(root);snapshots.append(snapshot)
        if not snapshot['available']:continue
        with closing(sqlite3.connect((Path(root)/'audit.sqlite').resolve().as_uri()+'?mode=ro',uri=True,timeout=30))as db:
            for key,body in requests.items():
                ref=exact_receipt(db,key,body,runtime)
                if ref is not None:hits.setdefault(key,[]).append(ref)
    uncached=len(requests)-len(hits);complete=all(s['available']for s in snapshots)
    return dict(task_count=len(tasks),unique_requests=len(requests),exact_cached_unique_requests=len(hits),
        exact_cached_tasks=sum(t['request_id']in hits for t in tasks),uncached_unique_requests_upper_bound=uncached,
        minimum_new_HTTP_without_retries=uncached if complete else None,lookup_scope_complete=complete,
        retries_additional_and_unknown=True,cache_does_not_extend_or_reset_old_budget=True,
        ledger_snapshots=snapshots,eligible_receipts=hits,
        per_task=[dict(task_key=t['task_key'],request_id=t['request_id'],exact_receipts=hits.get(t['request_id'],[]))for t in tasks])


def prepare(root,dependencies=None):
    root=Path(root);deps=dependencies or initialize_dependencies();core=deps['core']
    rows,refs,input_contract=source_records(root,deps)
    tasks=[]
    for row in rows:
        task=core.task_contract(row,deps);task.update(source=row['source'],method='DVKG_rerank')
        tasks.append(task)
    path=root/'full_answer_contract.json'
    if path.exists():
        saved=read(path);eligibility=checked(saved['cache_eligibility'])
    else:eligibility=cache_preflight(tasks,deps['runtime'],deps['historical_roots'])
    immutable(root/'full_answer_cache_eligibility.json',eligibility)
    contract=dict(version='original353_DVKG_rerank_answers_v1',scope=SCOPE,source_refs=refs,
        original_input_contract_digest=digest(input_contract),runtime=RUNTIME_REF,runtime_digest=RUNTIME_DIGEST,
        dependencies=deps['dependencies'],development_answer_core=core.record(core.__file__),worker=record(__file__),
        question_count=353,task_count=706,expected_question_counts=EXPECTED_COUNTS,scopes=list(SCOPES),variants=['rerank'],
        comparison_policy=input_contract['comparison_policy'],new_bounded_scope_distinct_from_development192=True,
        historical_budget_reset=False,historical_provider_roots=[str(p)for p in deps['historical_roots']],
        cache_eligibility=record(root/'full_answer_cache_eligibility.json'),limits=LIMITS,
        shared_global_slots=str(N/'formal_provider/locks'),consent_required_before_run=True,
        judge_calls=0,independent_test=False,pending_scored_zero=False,failed_model_responses_scored_zero=True,
        aggregate='within_book_mean_then_six_equal_books;course_two_equal_books',tasks=tasks)
    contract['result_identity']=digest(contract)
    for task in tasks:core.immutable(core.task_folder(root,task)/'contract.json',task)
    immutable(path,contract);update_status(root,contract,deps)
    return contract


def required_consent(contract_identity):
    return dict(version='explicit_original353_full_answer_consent_v1',scope=SCOPE,allow=False,user_authorization_text='',
        full_answer_contract_identity=contract_identity,question_count=353,answer_task_count=706,
        max_actual_HTTP=1024,max_seconds=14400,max_concurrency=4,runtime_digest=RUNTIME_DIGEST,
        destination='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',old_budget_reset=False)


def validate_consent(value,identity):
    expected=required_consent(identity)
    if (not isinstance(value,dict)or value.get('allow')is not True or not isinstance(value.get('user_authorization_text'),str)
            or not value['user_authorization_text'].strip()
            or any(value.get(k)!=v or type(value.get(k))is not type(v)for k,v in expected.items()if k not in {'allow','user_authorization_text'})):
        raise ValueError('New explicit original353/706-answer/1024-HTTP consent required; development192 approval is insufficient')


def summarize(root,contract,deps):
    core=deps['core'];outcomes={}
    for task in contract['tasks']:
        path=core.task_folder(root,task)/'record.json'
        if path.exists():outcomes[task['task_key']]=core.validate_terminal(core.read_checkpoint(path),task,contract)
    groups={scope:core.aggregate([t for t in contract['tasks']if t['scope']==scope],outcomes)for scope in SCOPES}
    completed=sum(r['status']=='completed'for r in outcomes.values());failed=len(outcomes)-completed
    current=ledger_snapshot(Path(root)/'full_answer_provider')
    return dict(status='all706_terminal'if len(outcomes)==706 else'partial_pending_not_scored',result_identity=contract['result_identity'],
        question_count=353,target_tasks=706,completed=completed,failed=failed,pending=706-len(outcomes),groups=groups,
        actual_new_HTTP=current['actual_http'],new_provider_initialized=current['available'],limits=LIMITS,
        pending_scored_zero=False,judge_calls=0,independent_test=False,comparison_policy=contract['comparison_policy'])


def update_status(root,contract,deps,**extra):
    result=summarize(root,contract,deps)
    atomic(Path(root)/'full_answer_summary.json',result)
    atomic(Path(root)/'full_answer_state.json',dict(result,pid=os.getpid(),observed_unix=time.time(),**extra))
    return result


def provider_type(module,core,slots,roots,stop,consent_ref):
    Base=core.provider_type(module,slots)
    class FullProvider(Base):
        def raise_if_fatal(self):
            super().raise_if_fatal()
            if stop.is_set():raise FullAnswerPending('Full answer service stopped; unfinished tasks remain pending')
        def _record(self,key,stage,status,value):
            if not stage.startswith(PREFIX):raise ValueError('Full-answer request is outside the new authorized namespace')
            blocking=status=='http_error'and isinstance(value,dict)and(value.get('status_code')in{401,402,403,429}
                or isinstance(value.get('status_code'),int)and 500<=value['status_code']<=599)
            try:result=super()._record(key,stage,status,value)
            finally:
                if blocking:stop.set()
            if blocking:raise FullAnswerPending('Official service unavailable; do not score pending tasks zero')
            return result
        def _cached(self,key,endpoint,body):
            value=super()._cached(key,endpoint,body)
            if value is not None:
                receipt=read(self.root/'receipts'/(key+'.json'))
                if receipt.get('runtime_config')!=self.runtime.config:raise ValueError('Full cached runtime differs')
                if receipt.get('inherited_receipt'):checked(receipt['inherited_receipt'])
                return value
            if endpoint!='/chat/completions':raise ValueError('Full answers cannot call other endpoints')
            for root in roots:
                path=Path(root)/'audit.sqlite'
                if not path.is_file():continue
                with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=30))as db:
                    reference=exact_receipt(db,key,body,self.runtime.config)
                if reference is None:continue
                old=checked(reference);destination=self.root/'receipts'/(key+'.json')
                immutable(destination,dict(old,response_source='exact_historical_receipt_reuse',inherited_receipt=reference,full_answer_authorization=consent_ref))
                published=record(destination)
                self._write_transaction(lambda db:db.execute('INSERT INTO receipts(key,record)VALUES(?,?)',(key,json.dumps(published))),operation='receipt_publish',request_id=key)
                return old['response']
            return None
    return FullProvider


def scoped_dependencies(deps):
    core=deps['core'];original=deps['joint']
    def generate(provider,retrieval,folder,stage):
        if not stage.startswith(core.STAGE_PREFIX):raise ValueError('Unexpected old-core stage')
        return original.generate_candidate(provider,retrieval,folder,PREFIX+stage[len(core.STAGE_PREFIX):])
    return dict(deps,joint=SimpleNamespace(generate_candidate=generate,JointIntegrityError=original.JointIntegrityError))


def execute(root):
    root=Path(root)
    # 新授权必须先存在，不能借旧开发任务的授权初始化付费执行。
    consent_ref=record(root/'full_answer_consent.json');consent=checked(consent_ref)
    saved=read(root/'full_answer_contract.json');validate_consent(consent,saved['result_identity'])
    deps=initialize_dependencies();core=deps['core']
    from filelock import FileLock
    with FileLock(str(root/'full_answer_run.lock'),timeout=0):
        contract=prepare(root,deps);validate_consent(consent,contract['result_identity'])
        immutable(root/'full_answer_execution_authorization.json',dict(consent=consent_ref,contract=record(root/'full_answer_contract.json'),limits=LIMITS))
        result=summarize(root,contract,deps)
        if not result['pending']:return update_status(root,contract,deps)
        deps['m'].credentials()
        runtime=deps['m'].runtime_for({'files':{'assets':{'runtime_config':RUNTIME_REF}}},root/'full_answer_transport')
        if digest(runtime.config)!=RUNTIME_DIGEST:raise ValueError('Full answer runtime differs from frozen official generator')
        stop=threading.Event();lock=threading.Lock();errors=[]
        Provider=provider_type(deps['native_provider'],core,contract['shared_global_slots'],deps['historical_roots'],stop,consent_ref)
        provider=Provider(runtime,root/'full_answer_provider',**LIMITS)
        run_deps=scoped_dependencies(deps)
        def worker(task):
            if stop.is_set():return
            try:
                with FileLock(str(core.task_folder(root,task)/'execution.lock'),timeout=0):core.answer_task(provider,task,root,contract,run_deps)
            except core.PendingTask as exc:
                stop.set()
                with lock:errors.append(dict(task_key=task['task_key'],error_type=type(exc.__cause__).__name__))
            finally:
                with lock:update_status(root,contract,deps,worker_errors=errors)
        try:
            provider.raise_if_fatal()
            pending=[t for t in contract['tasks']if not(core.task_folder(root,t)/'record.json').exists()]
            with ThreadPoolExecutor(max_workers=4)as pool:
                for future in as_completed([pool.submit(worker,t)for t in pending]):future.result()
            return update_status(root,contract,deps,worker_errors=errors)
        finally:provider.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    phase=parser.add_mutually_exclusive_group(required=True);phase.add_argument('--prepare',action='store_true');phase.add_argument('--run',action='store_true');phase.add_argument('--summary',action='store_true')
    args=parser.parse_args()
    try:
        if args.run:result=execute(args.root)
        elif args.prepare:
            contract=prepare(args.root);eligibility=checked(contract['cache_eligibility'])
            result=dict(status='full706_prepared_zero_API',result_identity=contract['result_identity'],task_count=len(contract['tasks']),
                cache_preflight={k:v for k,v in eligibility.items()if k not in {'eligible_receipts','per_task'}},
                exact_cache_qualification_ref=contract['cache_eligibility'],consent_required_before_run=True)
        else:
            deps=initialize_dependencies();contract=read(args.root/'full_answer_contract.json')
            if contract.get('result_identity')!=digest({k:v for k,v in contract.items()if k!='result_identity'}):raise ValueError('Full-answer contract changed')
            result=update_status(args.root,contract,deps)
    except Exception as exc:
        args.root.mkdir(parents=True,exist_ok=True);(args.root/'full_answer_error.log').write_text(traceback.format_exc(),encoding='utf-8')
        atomic(args.root/'full_answer_error_state.json',dict(status='pending_error_preserved',error_type=type(exc).__name__));raise
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
