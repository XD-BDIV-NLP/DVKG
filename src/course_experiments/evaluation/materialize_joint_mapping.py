"""固定联合检索预检所需的原始核验批次；默认零API准备，显式--run才执行。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import threading
import traceback


VERSION='supplement_joint_exact_mapping_v1'
STAGE_PREFIX='echa_v1/supplement_joint_mapping_20261028/'
STAGE_CAP=3000
MAX_STAGE_CAP=12000
BATCH_SUFFIX='\n本批次逐对返回JSON {"verdicts":[{"pair_id":"输入ID",...上述判定字段}]}，每对必须且仅有一个判定。'


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def file_record(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    if Path(path).suffix.lower() in {'.yaml','.yml'}:
        from course_experiments.manifest import read_document
        return read_document(path)
    return json.loads(Path(path).read_text('utf-8-sig'))


def checked(reference):
    if file_record(reference['path'])!=reference:raise ValueError('Pinned mapping-materialization dependency changed')
    return read(reference['path'])


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def worker_count(configuration):
    count=configuration.get('worker_count',4)
    if type(count) is not int or not 1<=count<=4:
        raise ValueError('Joint mapping book-worker count must be an integer within1..4')
    return count


def prepare_provider(configuration):
    """预检真实相对导入及授权工厂；不构造provider，不初始化账本。"""
    for key in ('native_provider_module','provider_factory','authorization'):
        if file_record(configuration[key]['path'])!=configuration[key]:
            raise ValueError('Supplementary provider dependency changed')
    module=load(configuration['native_provider_module']['path'],
        'course_experiments.evaluation.supplement_joint_mapping_native_provider')
    factory=load(configuration['provider_factory']['path'],
        'course_experiments.evaluation.supplement_joint_mapping_provider_factory')
    Provider=factory.provider_type(module,configuration['authorization'])
    if hasattr(factory,'retry_writer'):
        oldlock=Provider._writer_lock;oldwrite=Provider._write_transaction
        Provider._writer_lock=lambda provider,timeout:oldlock(provider,max(timeout,600))
        def write(provider,fn,*,operation,request_id=None):
            return factory.retry_writer(provider,lambda f,**kw:oldwrite(provider,f,**kw),fn,
                operation=operation,request_id=request_id)
        Provider._write_transaction=write
    return Provider


def accept_exact_batch(dispatcher,batch,reference):
    if batch.get('full_original_plan')!=dispatcher.plan_record:return False
    ordinal=batch['original_batch_ordinal']
    if (type(ordinal) is not int or not 0<=ordinal<len(dispatcher.batch_ids)
            or batch['original_batch_id']!=dispatcher.batch_ids[ordinal]):
        raise ValueError('Matching-plan batch has a different original ordinal or batch identity')
    original=dispatcher.flat[ordinal*8:ordinal*8+8]
    expected={(cid,c['evidence_id']) for cid,c in original}
    actual={(r['concept_id'],r['evidence_id']) for r in batch['records']}
    if actual!=expected or len(batch['records'])!=len(expected):
        raise ValueError('Matching-plan batch does not preserve exact original pair identities')
    if 'pair_keys' in batch and batch['pair_keys']!=[digest([cid,c]) for cid,c in original]:
        raise ValueError('Cached batch pair order differs from the original eight-pair plan')
    if ordinal in dispatcher.loaded:
        if dispatcher.loaded[ordinal]!=batch['records']:raise ValueError('Conflicting original batch decisions')
        return True
    dispatcher.loaded[ordinal]=batch['records'];dispatcher.proofs[ordinal]=reference
    return True


def original_plans(dispatcher,ordinal):
    plans=[]
    for cid,candidate in dispatcher.flat[ordinal*8:ordinal*8+8]:
        if not plans or plans[-1]['concept_id']!=cid:plans.append(dict(concept_id=cid,candidates=[]))
        plans[-1]['candidates'].append(candidate)
    return plans


def http_bounds(*,missing_batches,cached_first_attempts,transport_attempts,stage_used,stage_cap=STAGE_CAP):
    uncached=missing_batches-cached_first_attempts
    upper=(uncached+missing_batches)*transport_attempts
    remaining=max(0,stage_cap-stage_used)
    return dict(missing_original_batches=missing_batches,cached_first_attempts=cached_first_attempts,
        base_uncached_logical_requests_estimate=uncached,format_repair_logical_upper=missing_batches,
        transport_attempts_per_uncached_logical_upper=transport_attempts,
        uncached_actual_http_upper=upper,stage_used=stage_used,stage_cap=stage_cap,
        stage_remaining=remaining,sufficient_under_worst_case=upper<=remaining,
        completion_guaranteed=False,actual_cache_and_repair_outcomes_not_assumed=True,
        insufficient_policy='bounded_execution_may_stop_incomplete;never_publish_missing_batches_as_complete')


class MappingBudgetExhausted(RuntimeError):
    pass


class MappingHTTPBudget:
    def __init__(self,provider,root,cap=STAGE_CAP):
        from filelock import FileLock
        if type(cap) is not int or not 1<=cap<=MAX_STAGE_CAP:
            raise ValueError('Mapping stage cap must stay within the authorized12000 upper bound')
        self.provider=provider;self.root=Path(root);self.cap=cap
        self.lock=FileLock(str(provider.root/'supplement_joint_mapping_20261028_http.lock'),timeout=600)

    def stage(self,name):return STAGE_PREFIX+name

    def count(self):
        with closing(sqlite3.connect(self.provider.database,timeout=30)) as db:
            return db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,?)=?",
                (len(STAGE_PREFIX),STAGE_PREFIX)).fetchone()[0]

    def __enter__(self):
        from course_experiments.contributions import write_immutable
        write_immutable(self.root/('http_scope_cap_'+str(self.cap)+'.json'),dict(prefix=STAGE_PREFIX,cap=self.cap,
            accounting='all_actual_HTTP_starts_including_every_transport_retry;exact_cache_hits_free',
            inherited_global_allocation=200000,inherited_echa_allocation=60000,budget_reset=False))
        self.original=self.provider.runtime._record
        def bounded(key,stage,status,value):
            if not stage.startswith(STAGE_PREFIX):raise ValueError('Mapping request exceeds the fixed supplementary scope')
            if status!='request_started':return self.original(key,stage,status,value)
            with self.lock:
                if self.count()>=self.cap:raise MappingBudgetExhausted('Joint-mapping cumulative actual HTTP cap exhausted')
                return self.original(key,stage,status,value)
        self.provider.runtime._record=bounded;return self

    def __exit__(self,*args):self.provider.runtime._record=self.original


def first_attempt_body(m,ctx,source,ordinal):
    plans=original_plans(source['dispatcher'],ordinal)
    pairs=list(m.pipeline._pairs(ctx['entries'][source['state_key']]['graph_data'],ctx['structures'][source['book_id']],plans))
    payload=dict(target_book_id=source['book_id'],pairs=[p['payload'] for p in pairs])
    return dict(model=source['runtime']['llm']['model'],temperature=0,seed=42,enable_thinking=False,
        max_tokens=4096,response_format=dict(type='json_object'),messages=[
            dict(role='system',content=m.pipeline.MODEL_PROMPT+BATCH_SUFFIX),
            dict(role='user',content=json.dumps(payload,ensure_ascii=False))])


def cached_first_attempts(m,ctx,sources,database):
    result={};proofs=[]
    if not Path(database).exists():return {s['book_id']:0 for s in sources},proofs
    uri=Path(database).resolve().as_uri()+'?mode=ro'
    with closing(sqlite3.connect(uri,uri=True,timeout=30)) as db:
        for source in sources:
            count=0
            for ordinal in source['missing']:
                body=first_attempt_body(m,ctx,source,ordinal)
                request_id=digest(dict(endpoint='/chat/completions',body=body,config=source['runtime']))
                found=db.execute('SELECT record FROM receipts WHERE key=?',(request_id,)).fetchone()
                if found is None:continue
                reference=json.loads(found[0]);value=checked(reference)
                if (value['request_id']!=request_id or value['endpoint']!='/chat/completions'
                        or value['request']!=body or value['runtime_config']!=source['runtime']):
                    raise ValueError('Cached original-batch provider receipt has different exact inputs')
                count+=1;proofs.append(reference)
            result[source['book_id']]=count
    return result,proofs


def _references(m,configuration,retrieval,preflight,root):
    refs={};unreadable=[]
    def manifest(reference):
        for proof in checked(reference).get('proofs',[]):refs.setdefault(proof['path'],proof)
    for source in retrieval['sources']:
        own=root/'books'/source['book_id']/'mapping_manifest.json'
        if own.exists():manifest(file_record(own))
        for reference in source.get('mapping_manifests',[]):manifest(reference)
    for values in preflight.get('mapping_proofs',{}).values():
        for reference in values:refs.setdefault(reference['path'],reference)
    for reference in configuration.get('additional_manifest_refs',[]):manifest(reference)
    roots=list(configuration.get('proof_search_roots',[]))
    roots.extend(p for s in retrieval['sources'] for p in s.get('alignment_roots',[]))
    roots.append(str(root/'books'))
    for folder in dict.fromkeys(roots):
        for path in sorted(Path(folder).rglob('batch_result.json')):
            if str(path) in refs:continue
            try:
                value=read(path)
                if not isinstance(value,dict) or 'full_original_plan' not in value:continue
                refs[str(path)]=file_record(path)
            except (ValueError,OSError) as exc:unreadable.append(dict(path=str(path),error_type=type(exc).__name__))
    return list(refs.values()),unreadable


def prepare(configuration,reference):
    if configuration.get('version')!=VERSION:raise ValueError('Explicit exact supplementary mapping configuration required')
    workers=worker_count(configuration)
    cap=configuration.get('actual_http_cap',STAGE_CAP)
    if type(cap) is not int or not 1<=cap<=MAX_STAGE_CAP:raise ValueError('Authorized mapping stage cap must be at most12000')
    original=checked(configuration['original_contract'])
    sys.path.insert(0,str(Path(original['code']['run_echa.py']['path']).parents[2]))
    from course_experiments.evaluation import run_echa as m
    from course_experiments.contributions import write_immutable
    for record in original['code'].values():
        if file_record(record['path'])!=record:raise ValueError('Original frozen mapping implementation changed')
    for name in ['authorization','native_provider_module','provider_factory']:
        if file_record(configuration[name]['path'])!=configuration[name]:raise ValueError('Supplementary provider dependency changed')
    auth=checked(configuration['authorization'])
    if auth.get('scope')!='supplement_8_9_10' or not any(STAGE_PREFIX.startswith(p) for p in auth.get('allowed_stage_prefixes',[])):
        raise ValueError('Mapping materialization requires the actual supplementary8/9/10 authorization')
    Provider=prepare_provider(configuration)
    if 'batch_runtime' in configuration:
        if file_record(configuration['batch_runtime']['path'])!=configuration['batch_runtime']:
            raise ValueError('Pinned batch cache adapter changed')
        perf=load(configuration['batch_runtime']['path'],
            'course_experiments.evaluation.supplement_joint_mapping_original_batch_cache')
        if not callable(getattr(perf,'cached_batch_verifier',None)):
            raise ValueError('Original batch cache adapter interface is unavailable')
    retrieval=checked(configuration['retrieval_configuration']);preflight=checked(configuration['mapping_preflight'])
    planned=m.asset(retrieval['deferred_plan'])
    if planned['plan_identity']!=m.digest({k:v for k,v in planned.items() if k!='plan_identity'}):
        raise ValueError('Original deferred-plan identity changed')
    ctx=m.load_context(original['inventory']['path'],original['inventory']['sha256'],original['tokenizer']['path'],original['tokenizer']['sha256'])
    if ctx['identity']!=planned['context_identity']:raise ValueError('Original frozen context differs from deferred plan')
    runtime=m.asset(ctx['inventory']['runtime_config'])
    if checked(configuration['runtime_config'])!=runtime:raise ValueError('Mapping runtime must preserve the exact original receipt-cache configuration')
    if m.pipeline.POLICY['batch_pairs']!=8 or m.pipeline.POLICY['format_repairs']!=1:
        raise ValueError('Exact original eight-pair batches and bounded single repair required')
    m._recheck(ctx,deep=True);root=Path(configuration['output_root']);root.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(configuration=reference,worker=file_record(__file__)))
    write_immutable(root/'execution_contract.json',dict(version=VERSION,execution_identity=identity,configuration=reference,
        worker=file_record(__file__),preflight=configuration['mapping_preflight'],context_identity=ctx['identity'],
        selection='only_concept_ids_in_the_hashed_all-question_joint_preflight;QA_answers_not_opened',
        original_eight_pair_batches=True,original_order_preserved=True,format_repair_maximum=1,
        prefix=STAGE_PREFIX,actual_http_cap=cap,worker_count=workers,
        scheduling='one_book_per_worker;original_batches_and_model_requests_unchanged',
        provider_preflight='qualified_module_imports_and_authorization_factory_verified_without_instantiation',
        inherited_global_cap=200000,inherited_echa_cap=60000,
        budget_reset=False,full_alignment_claim=False))
    reports={r['book_id']:r for r in preflight['books']};sources=[]
    if set(reports)!={s['book_id'] for s in retrieval['sources']} or len(reports)!=6:
        raise ValueError('Exact same six-book joint-preflight source set required')
    for spec in retrieval['sources']:
        book=spec['book_id'];key=spec['state_key'];report=reports[book]
        d=m.ExactDeferredAlignment(ctx,key,book,planned['full_plans'][key][book],root/'books'/book/'alignment')
        concepts=sorted(set(report['required_concept_ids']));required=d.required_batches(concepts)
        if (report['full_original_plan']!=d.plan_record or report['graph_id']!=digest(ctx['entries'][key]['graph_data'])
                or report['required_original_batch_count']!=len(required)
                or not set(report['missing_original_batch_ordinals'])<=set(required)
                or report['missing_original_batch_ids']!=[d.batch_ids[i] for i in report['missing_original_batch_ordinals']]):
            raise ValueError('Actual joint preflight has a different graph, full plan or required batch identity')
        sources.append(dict(spec,dispatcher=d,concepts=concepts,required=required,runtime=runtime))
    references,unreadable=_references(m,configuration,retrieval,preflight,root)
    plan_lookup={digest(s['dispatcher'].plan_record):s for s in sources};reused=0;foreign=0
    for record in references:
        batch=checked(record);key=digest(batch.get('full_original_plan'))
        if key not in plan_lookup:foreign+=1;continue
        source=plan_lookup[key]
        accept_exact_batch(source['dispatcher'],batch,record);reused+=1
    for source in sources:
        d=source['dispatcher'];qualified=[r for i in d.loaded for r in d.loaded[i]]
        m.core.lift_coverage(ctx['entries'][source['state_key']]['graph_data'],ctx['structures'][source['book_id']],qualified)
        source['missing']=[i for i in source['required'] if i not in d.loaded]
    database=Path(configuration['run_root'])/'formal_provider/audit.sqlite'
    cache_counts,cache_proofs=cached_first_attempts(m,ctx,sources,database)
    if database.exists():
        uri=database.resolve().as_uri()+'?mode=ro'
        with closing(sqlite3.connect(uri,uri=True,timeout=30)) as db:
            stage_used=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,?)=?",
                (len(STAGE_PREFIX),STAGE_PREFIX)).fetchone()[0]
    else:raise ValueError('Existing shared provider ledger required; never initialize a new budget')
    total_missing=sum(len(s['missing']) for s in sources)
    bounds=http_bounds(missing_batches=total_missing,cached_first_attempts=sum(cache_counts.values()),
        transport_attempts=ctx['transport_attempts'],stage_used=stage_used,stage_cap=cap)
    preparation=dict(status='prepared_NO_API;mapping_incomplete' if total_missing else 'all_required_original_batches_already_available_NO_API',
        execution_identity=identity,books={s['book_id']:dict(required_batches=len(s['required']),missing_batches=len(s['missing']),
            missing_original_ordinals=s['missing'],cached_first_attempts=cache_counts[s['book_id']]) for s in sources},
        http_bounds=bounds,worker_count=workers,additional_compatible_batch_files=reused,foreign_plan_files_ignored=foreign,
        unreadable_search_files=unreadable,cache_receipt_refs=cache_proofs,model_calls_this_process=0,
        budget_snapshot=m._budget_snapshot(configuration['run_root']))
    if not (root/'initial_preparation.json').exists():write_immutable(root/'initial_preparation.json',preparation)
    m.atomic(root/'current_preparation.json',preparation)
    return m,ctx,root,identity,sources,preparation,Provider


def mapping_manifest(m,ctx,source,identity):
    d=source['dispatcher'];required=source['required'];missing=[i for i in required if i not in d.loaded]
    qualified=[r for i in required if i in d.loaded for r in d.loaded[i]]
    m.core.lift_coverage(ctx['entries'][source['state_key']]['graph_data'],ctx['structures'][source['book_id']],qualified)
    return dict(status='complete_required_joint_query_batches' if not missing else 'incomplete_required_joint_query_batches',
        execution_identity=identity,book_id=source['book_id'],state_key=source['state_key'],full_original_plan=d.plan_record,
        required_concept_ids=source['concepts'],required_original_batches=len(required),missing_original_ordinals=missing,
        proofs=[d.proofs[i] for i in required if i in d.loaded],mapping_record_status_counts=dict(Counter(r['status'] for r in qualified)),
        entire_concept_plan_claimed_complete=False,full_alignment_complete=False,
        QA_answers_used_for_selection=False,accepted_rejected_unresolved_all_retained=True)


def execute(configuration,reference,*,run=False):
    m,ctx,root,identity,sources,preparation,Provider=prepare(configuration,reference)
    from course_experiments.contributions import write_immutable
    def publish(source):
        value=mapping_manifest(m,ctx,source,identity);folder=root/'books'/source['book_id']
        if not value['missing_original_ordinals']:write_immutable(folder/'mapping_manifest.json',value)
        else:m.atomic(folder/'mapping_manifest.partial.json',value)
        return value
    if not run:
        manifests=[publish(s) for s in sources]
        return dict(status=preparation['status'],books=len(sources),missing_batches=sum(len(v['missing_original_ordinals']) for v in manifests),
            http_bounds=preparation['http_bounds'],model_calls_this_process=0)
    if 'batch_runtime' in configuration:
        if file_record(configuration['batch_runtime']['path'])!=configuration['batch_runtime']:raise ValueError('Pinned batch cache adapter changed')
        perf=load(configuration['batch_runtime']['path'],
            'course_experiments.evaluation.supplement_joint_mapping_original_batch_cache')
        m.pipeline.verify_candidates=perf.cached_batch_verifier(m.pipeline.verify_candidates)
    m.credentials();stop=threading.Event();errors={}
    def worker(source):
        from filelock import FileLock
        book=source['book_id'];folder=root/'books'/book;provider=tracker=None
        try:
            with FileLock(str(folder/'execution.lock'),timeout=0):
                if not stop.is_set():
                    runtime=m.runtime_for({'files':{'assets':{'runtime_config':configuration['runtime_config']}}},folder/'transport')
                    provider=Provider(runtime,Path(configuration['run_root'])/'formal_provider',**m.LIMITS)
                    m._validate_provider(provider)
                    if provider.effective_http_cap!=200000 or provider.allocation['caps']['echa']!=60000:
                        raise ValueError('Original200k global/60k ECHA ledger required')
                    tracker=m.ReceiptTracker(provider)
                    with m.ECHABudget(provider,configuration['run_root']),MappingHTTPBudget(provider,
                            Path(configuration['run_root'])/'supplement_joint_mapping_budget_20261028',
                            cap=configuration.get('actual_http_cap',STAGE_CAP)) as budget,tracker.scope(folder/'requests.jsonl'):
                        source['dispatcher'].ensure(source['concepts'],provider,budget)
                return publish(source)
        except Exception as exc:
            stop.set();errors[book]=type(exc).__name__
            (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
            return publish(source)
        finally:
            if tracker is not None:tracker.close()
            if provider is not None:provider.close()
    with ThreadPoolExecutor(max_workers=worker_count(configuration)) as pool:
        futures=[pool.submit(worker,s) for s in sources]
        while any(not f.done() for f in futures):
            wait(futures,timeout=30)
            m.atomic(root/'current_progress.json',dict(completed_books=sum(f.done() for f in futures),target_books=len(sources),
                errors=errors,full_alignment_claim=False))
        manifests=[f.result() for f in futures]
    m._recheck(ctx,deep=True)
    missing=sum(len(v['missing_original_ordinals']) for v in manifests)
    summary=dict(status='all_required_joint_mapping_batches_materialized' if not missing else 'incomplete_joint_mapping_batches_preserved',
        execution_identity=identity,missing_original_batches=missing,books={v['book_id']:dict(status=v['status'],
            required_batches=v['required_original_batches'],missing_batches=len(v['missing_original_ordinals']),
            manifest=file_record(root/'books'/v['book_id']/'mapping_manifest.json') if not v['missing_original_ordinals'] else None) for v in manifests},
        errors=errors,full_alignment_claim=False,budget_reset=False,budget_snapshot=m._budget_snapshot(configuration['run_root']))
    m.atomic(root/'current_summary.json',summary)
    if not missing:write_immutable(root/'complete_manifest_index.json',dict(execution_identity=identity,
        books={b:r['manifest'] for b,r in summary['books'].items()},full_alignment_claim=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--configuration',required=True)
    parser.add_argument('--configuration-sha',required=True);parser.add_argument('--run',action='store_true')
    args=parser.parse_args();reference=dict(path=str(Path(args.configuration).resolve()),sha256=args.configuration_sha)
    result=execute(checked(reference),reference,run=args.run)
    print(json.dumps(result,ensure_ascii=False))
    return 2 if result.get('missing_original_batches',result.get('missing_batches',0)) else 0


if __name__=='__main__':raise SystemExit(main())
