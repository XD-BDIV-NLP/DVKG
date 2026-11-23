"""仅追加G0+S+M/T+S+M二十条件；不更改运行中的F实验。"""
from __future__ import annotations
import argparse,json,os,time,traceback
from pathlib import Path
import numpy as np
from filelock import FileLock
from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable
from . import state_dual_view as state
from . import run_frozen_dual_view as frozen
from .run_native_full import LIMITS,ReceiptTracker
from .native_provider import AuditedNativeProvider,NativeAccountUnavailable,NativeBudgetExceeded
from .run_independent_holdout import LocalHTTPBudget,HoldoutBudgetExceeded
from .protocol import runtime_for
from .run_repaired_full import atomic,credentials


LOCAL_CAPS={'regression':2200,'independent':1200}


class StateBudget(LocalHTTPBudget):
    def __init__(self,provider,root,identity,cap):
        super().__init__(provider,root,identity,cap=cap)
        self.prefix='semantic_states/'+identity+'/'


def checked_inventory(path,sha):
    record=file_record(path)
    if record['sha256']!=sha:raise ValueError('Frozen semantic-state inventory changed')
    value=read_document(path)
    if value['identity']!=digest({k:v for k,v in value.items() if k!='identity'}):raise ValueError('State inventory identity changed')
    if value['code']['sha256']!=file_record(state.__file__)['sha256']:raise ValueError('Inventory state adapter changed')
    if file_record(value['asset_manifest']['path'])!=value['asset_manifest'] or file_record(value['parent_matrix']['path'])!=value['parent_matrix']:
        raise ValueError('Original frozen state assets changed')
    for entry in value['states'].values():
        for name in ['graph','summary','sources']:
            proof=entry['provenance'][name]
            if file_record(proof['path'])!=proof:raise ValueError('Original independently constructed state provenance changed')
    return value,record


class DependencyPending(RuntimeError):
    pass


def independent_datasets(run_root,assets):
    path=Path(run_root)/'independent_holdout_128/sealed.json'
    if not path.exists():raise DependencyPending('Same independent sealed set not ready')
    seal=read_document(path)
    if seal.get('status') not in ['sealed_complete','sealed_insufficient'] or seal.get('target_count')!=128:
        raise DependencyPending('Same independent set is not sealed')
    state.asset(seal['contract']);index=state.asset(seal['dataset_index']);books=index['books']
    if set(books)!=set(assets['unit_assets']):raise ValueError('Sealed independent eight-book scope differs')
    n=0
    for book,reference in books.items():
        dataset=state.asset(reference);questions=dataset['questions'];counts={kind:0 for kind in ['definition','paraphrase','conditions','cross_chapter']}
        if dataset['target_book_id']!=book:raise ValueError('Independent dataset book differs')
        for q in questions:
            if q['type'] not in counts:raise ValueError('Independent task type differs')
            counts[q['type']]+=1
        if any(v>4 for v in counts.values()):raise ValueError('Independent preregistration exceeded')
        n+=len(questions)
    if n!=seal['accepted_count'] or n!=index['accepted_count'] or seal['shortage']!=128-n or (seal['status']=='sealed_complete')!=(n==128):
        raise ValueError('Independent actual legal denominator differs')
    return books,file_record(path),dict(target_count=128,accepted_count=n,shortage=128-n)


def prepare(inventory,assets,datasets):
    jobs=[dict(cohort=row['cohort'],state=row['state'],method=row['method'],book=b,state_key=key)
        for key,row in inventory['states'].items() for b in row['active_books']]
    if len(jobs)!=20 or {j['state'] for j in jobs}!={'G0','T'}:raise ValueError('Exact paired20 semantic state conditions required')
    if set(datasets)!=set(assets['unit_assets']):raise ValueError('State datasets must cover same eight books')
    counts={b:len(state.asset(ref)['questions']) for b,ref in datasets.items()}
    for book,reference in datasets.items():
        if any(q['target_book_id']!=book for q in state.asset(reference)['questions']):raise ValueError('State question target differs')
    qids={b:sorted(q['question_id'] for q in state.asset(ref)['questions']) for b,ref in datasets.items()}
    if any(len(ids)!=len(set(ids)) for ids in qids.values()):raise ValueError('Duplicate question identifiers in state evaluation')
    return jobs,dict(unique_questions=sum(counts.values()),total_question_evaluations=sum(counts[j['book']] for j in jobs),by_book=counts,
        question_ids_digest=digest(qids),book_question_ids_digests={b:digest(ids) for b,ids in qids.items()},same_frozen_question_pool_as_F=True)


def vectors(inventory,root,provider,budget):
    root=Path(root);new={};groups={}
    for key,row in inventory['unique_missing_embeddings'].items():
        groups.setdefault(digest(row['embedding_configuration']),[]).append(key)
    for group,keys in sorted(groups.items()):
        keys=sorted(keys)
        for first in range(0,len(keys),16):
            chosen=keys[first:first+16];path=root/'new_vectors'/(digest(chosen)+'.npy');meta=path.with_suffix('.meta.json')
            if meta.exists():
                record=read_document(meta)
                if record['keys']!=chosen or file_record(path)!=record['vectors']:raise ValueError('State vector checkpoint differs')
            else:
                config=inventory['unique_missing_embeddings'][chosen[0]]['embedding_configuration']
                if provider.runtime.config['embedding']!=config:raise ValueError('State vector model configuration differs')
                matrix=provider.embed([inventory['unique_missing_embeddings'][k]['text'] for k in chosen],stage=budget.stage('state_concept_embeddings'))
                if matrix.shape!=(len(chosen),1024) or not np.isfinite(matrix).all():raise ValueError('State embeddings invalid')
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():path.rename(path.with_suffix('.uncommitted_'+str(time.time_ns())))
                with path.open('xb') as f:np.save(f,matrix)
                record=dict(keys=chosen,vectors=file_record(path),embedding_configuration=config,formal_receipts_scope=str(root/'requests.jsonl'))
                write_immutable(meta,record)
            for i,key in enumerate(chosen):new[key]=dict(vectors=record['vectors'],row=i)
    result={}
    for key,row in inventory['states'].items():
        plan=dict(row['vector_plan']);plan['entries']=[dict(r,reuse=r['reuse'] or new[r['text_key']]) for r in plan['entries']];plan['missing_count']=0
        result[key]=state.materialize_reused(plan)
    return result


def evaluate(job,entry,assets,matrix,variant,dataset,provider,root,identity,budget):
    root=Path(root);questions=state.asset(dataset)['questions'];records=[]
    engine=state.StateRetriever(job['state'],state.asset(entry['graph']),{b:assets['structures'][b] for b in entry['active_books']},
        entry['mappings'],{b:assets['unit_assets'][b] for b in entry['active_books']},entry['vector_plan']['concept_ids'],matrix,variant=variant)
    for q in questions:
        folder=root/'questions'/q['question_id'];path=folder/'record.json'
        if path.exists():
            row=read_document(path)
            if row['result_id']!=identity or row['checkpoint_digest']!=digest({k:v for k,v in row.items() if k!='checkpoint_digest'}):raise ValueError('State question checkpoint changed')
            records.append(row);continue
        folder.mkdir(parents=True,exist_ok=True)
        try:
            original=frozen.PARENT/'task_vectors'/job['book']/(q['question_id']+'.npy')
            if original.exists():
                meta=read_document(original.with_suffix('.npy.meta.json'))['vectors']
                if file_record(original)!=meta:raise ValueError('Original frozen query vector changed')
                query=np.load(original)[0]
            else:query=provider.embed([q['question']],stage=budget.stage('query_embedding'))[0]
            row=state.evaluate_state_question(engine,q,query,provider,folder,identity,budget.stage('state_qa'))
            row.update(result_id=identity)
        except (NativeAccountUnavailable,NativeBudgetExceeded,HoldoutBudgetExceeded):raise
        except Exception as exc:
            error=folder/'error.log';error.write_text(traceback.format_exc())
            row=dict(result_id=identity,status='failed',question_id=q['question_id'],question_type=q['type'],target_book_id=job['book'],error_type=type(exc).__name__,error_record=file_record(error))
        records.append(frozen.checkpoint(path,row))
        atomic(root/'state.json',dict(status='running',records=len(records),total=len(questions),pid=os.getpid(),updated_unix=time.time()))
    result=dict(frozen.old.summarize(records,[q['question_id'] for q in questions]),job=job,result_id=identity,
        judge_version=state.POLICY['judge_version'],failed_count=sum(r['status']=='failed' for r in records),raw_text_exported=False)
    if not questions:result.update(status='completed',empty_sealed_book=True,denominator='zero_legal_sealed_questions;metrics_NA;no_missing_gold_scored')
    write_immutable(root/'result.json',result);atomic(root/'state.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--inventory-sha',required=True)
    p.add_argument('--split',choices=['regression','independent'],required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--max-http-attempts',type=int)
    args=p.parse_args();inventory,inventory_record=checked_inventory(args.inventory,args.inventory_sha)
    assets=state.asset(inventory['asset_manifest']);selection_path=args.run_root/frozen.SELECTION;selection=read_document(selection_path);variant=frozen.selected_variant(selection)
    if file_record(selection['source']['path'])!=selection['source']:raise ValueError('Frozen development selection source changed')
    if args.split=='independent':
        try:datasets,seal,counts_extra=independent_datasets(args.run_root,assets)
        except DependencyPending:
            if not args.check_only:raise
            print(json.dumps(dict(status='dependencies_pending',split='independent',jobs=20,target_count=128,actual_count=None,
                local_http_cap=LOCAL_CAPS['independent'],same_frozen_question_pool_as_F='will_verify_exact_sealed_common_pool',model_api_calls=0,raw_text_exported=False)));return 2
    else:
        datasets={b:r['confirmation'] for b,r in read_document(frozen.PARENT/'datasets/index.json')['books'].items()};seal=None;counts_extra={}
    jobs,counts=prepare(inventory,assets,datasets)
    summary=dict(status='checked_no_api',split=args.split,jobs=len(jobs),**counts,**counts_extra,new_vectors=inventory['new_vector_count'],embedding_http_base_upper=inventory['embedding_http_base_upper'],
        answer_judge_uncached_base_upper=counts['total_question_evaluations']*2,model_api_calls=0,raw_text_exported=False)
    cap=LOCAL_CAPS[args.split]
    if args.max_http_attempts not in [None,cap]:raise ValueError('Local cap differs from root frozen split-specific policy')
    limits=read_document(args.run_root/'formal_provider/limits.json')
    if limits['limits']!=LIMITS or not (args.run_root/'formal_provider/audit.sqlite').exists():raise ValueError('Existing formal budget required; never reset')
    summary.update(local_http_cap=cap,original_formal_started_unix=limits['started_unix'],global_limits=LIMITS,
        F_pool_reference=str(args.run_root/'independent_holdout_128/sealed.json') if args.split=='independent' else str(frozen.PARENT/'datasets/index.json'))
    if args.check_only:print(json.dumps(summary));return 0
    contract=dict(inventory=inventory_record,selection=file_record(selection_path),sealed=seal,datasets=datasets,split=args.split,counts=counts,shortage=counts_extra,
        policy=state.POLICY,local_http_cap=cap,code={n:file_record(Path(__file__).parent/n) for n in ['state_dual_view.py','run_state_dual_view.py','run_frozen_dual_view.py','dual_view_rag.py','dvkg_development.py','run_native_full.py']})
    identity=digest(contract);root=args.run_root/'state_dual_view'/args.split/identity;root.mkdir(parents=True,exist_ok=True);credentials()
    provider=AuditedNativeProvider(runtime_for(state.asset(assets['cohorts']['compABCD']['construction']),root/'transport'),args.run_root/'formal_provider',**LIMITS);tracker=ReceiptTracker(provider)
    try:
        with FileLock(str(root/'execution.lock'),timeout=0),StateBudget(provider,root,identity,cap) as budget,tracker.scope(root/'requests.jsonl'):
            write_immutable(root/'contract.json',dict(result_id=identity,**contract))
            raw=provider.chat_response
            def scoped(messages,**kwargs):
                if kwargs.get('stage')=='native_blind_judge':kwargs['stage']=budget.stage('judge')
                return raw(messages,**kwargs)
            provider.chat_response=scoped
            matrices=vectors(inventory,root,provider,budget);results=[]
            for job in jobs:
                entry=inventory['states'][job['state_key']];folder=root/(job['cohort']+'__'+job['book']+'__'+job['state'])
                results.append(evaluate(job,entry,assets,matrices[job['state_key']],variant,datasets[job['book']],provider,folder,identity,budget))
                atomic(root/'state.json',dict(status='running',jobs_finished=len(results),jobs=20,pid=os.getpid(),updated_unix=time.time()))
            result=dict(summary,status='completed' if all(r['status']=='completed' for r in results) else 'partial',results=results,result_id=identity,local_http_attempts=budget.count(),model_api_calls=budget.count())
            write_immutable(root/'result.json',result);atomic(root/'state.json',result);print(json.dumps({k:v for k,v in result.items() if k!='results'}));return 0 if result['status']=='completed' else 2
    except Exception as exc:
        path=root/('failure_'+str(time.time_ns())+'.log');path.write_text(traceback.format_exc());atomic(root/'state.json',dict(status='interrupted_preserved_checkpoints',error_type=type(exc).__name__,error_record=file_record(path),pid=os.getpid()));return 2
    finally:tracker.close();provider.close()


if __name__=='__main__':raise SystemExit(main())
