"""仅追加既有原文复核修图的十个dual条件；共享原formal预算，不改运行快照。"""
from __future__ import annotations
import argparse,json,os,time,traceback
from pathlib import Path
from types import FunctionType,SimpleNamespace
from filelock import FileLock
from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable
from . import corrected_f_dual_view as corrected
from . import state_dual_view as state
from . import run_state_dual_view as common
from . import run_frozen_dual_view as frozen
from .run_independent_holdout import LocalHTTPBudget
from .native_provider import AuditedNativeProvider
from .run_native_full import LIMITS,ReceiptTracker
from .protocol import runtime_for
from .run_repaired_full import atomic,credentials

LOCAL_CAPS={'regression':1100,'independent':400}
COHORTS={'compAB','compABCD','osAB','netAB'}


def engine_factory(semantic_state,*args,**kwargs):
    if semantic_state!='F_corrected':raise ValueError('Only independently source-reviewed corrected F required')
    return corrected.CorrectedFRetriever(*args,**kwargs)


# 只在独立函数命名空间注入薄接口；原评测字节码及原模块全局变量保持原样。
_evaluation_state=SimpleNamespace(asset=state.asset,StateRetriever=engine_factory,
    evaluate_state_question=corrected.evaluate_question,POLICY=corrected.POLICY)
evaluate=FunctionType(common.evaluate.__code__,dict(common.evaluate.__globals__,state=_evaluation_state),'evaluate')


class CorrectedBudget(LocalHTTPBudget):
    def __init__(self,provider,run_root,split):
        super().__init__(provider,Path(run_root)/'corrected_F_allowances'/split,digest(['corrected_F',split]),cap=LOCAL_CAPS[split])
        self.prefix='corrected_F/'+split+'/'


def checked_inventory(path,sha):
    record=file_record(path)
    if record['sha256']!=sha:raise ValueError('Frozen corrected F inventory SHA differs')
    value=read_document(path)
    if value['identity']!=digest({k:v for k,v in value.items() if k!='identity'}) or set(value['states'])!=COHORTS:
        raise ValueError('Corrected F inventory identity/cohort scope differs')
    state.asset(value['asset_manifest']);proof_path=Path(path).parent/'receipt.json';proof=read_document(proof_path)
    if proof['inventory']!=record or proof['status']!='completed_no_api':raise ValueError('Original zero-API correction provenance receipt differs')
    for co,entry in value['states'].items():
        row=proof['cohorts'][co]
        if row['graph']!=entry['graph'] or not row['same_native_source_input_hashes'] or not row['same_S_reconstructed_from_exact_B_corpus']:
            raise ValueError('Corrected graph or same-S original-source proof differs')
        state.asset(entry['graph']);review=state.asset(row['review'])
        for ref in review['reviews']:state.asset(ref)
    return value,record,file_record(proof_path)


def prepare(inventory,assets,datasets):
    if set(inventory['states'])!=COHORTS or set(datasets)!=set(assets['unit_assets']):raise ValueError('Four corrected graphs and common eight-book questions required')
    jobs=[dict(cohort=co,state='F_corrected',method='F_corrected+S+M',book=b,state_key=co)
        for co,row in sorted(inventory['states'].items()) for b in row['active_books']]
    if len(jobs)!=10 or any(row['cohort']!=co or row['state']!='F_corrected' for co,row in inventory['states'].items()):raise ValueError('Exactly ten corrected-F dual conditions required')
    qids={}
    for book,ref in datasets.items():
        questions=state.asset(ref)['questions'];qids[book]=sorted(q['question_id'] for q in questions)
        if len(qids[book])!=len(set(qids[book])) or any(q['target_book_id']!=book for q in questions):raise ValueError('Same target-book question identity differs')
    return jobs,dict(unique_questions=sum(map(len,qids.values())),total_question_evaluations=sum(len(qids[j['book']]) for j in jobs),
        by_book={b:len(v) for b,v in qids.items()},question_ids_digest=digest(qids),book_question_ids_digests={b:digest(v) for b,v in qids.items()},same_frozen_question_pool_as_F=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--inventory-sha',required=True)
    p.add_argument('--split',choices=list(LOCAL_CAPS),required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--max-http-attempts',type=int);args=p.parse_args()
    inv,inventory_record,review_proof=checked_inventory(args.inventory,args.inventory_sha);assets=state.asset(inv['asset_manifest'])
    selection_path=args.run_root/frozen.SELECTION;selection=read_document(selection_path);variant=frozen.selected_variant(selection);state.asset(selection['source'])
    cap=LOCAL_CAPS[args.split]
    if args.max_http_attempts not in [None,cap]:raise ValueError('Root-frozen corrected F split allowance differs')
    limits=read_document(args.run_root/'formal_provider/limits.json')
    if limits['limits']!=LIMITS or not (args.run_root/'formal_provider/audit.sqlite').exists():raise ValueError('Existing original formal budget required; never reset')
    if args.split=='independent':
        try:datasets,seal,extra=common.independent_datasets(args.run_root,assets)
        except common.DependencyPending:
            if not args.check_only:raise
            print(json.dumps(dict(status='dependencies_pending',split=args.split,jobs=10,target_count=128,actual_count=None,local_http_cap=cap,model_api_calls=0,raw_text_exported=False)));return 2
    else:datasets={b:r['confirmation'] for b,r in read_document(frozen.PARENT/'datasets/index.json')['books'].items()};seal=None;extra={}
    jobs,counts=prepare(inv,assets,datasets);new=len(inv['unique_missing_embeddings'])
    if args.split=='regression' and counts['total_question_evaluations']!=393:raise ValueError('Same fixed313/393 paired regression scope required')
    summary=dict(status='checked_no_api',split=args.split,jobs=10,**counts,**extra,new_vectors=new,embedding_http_base_upper=(new+15)//16,
        embedding_reuse_independent='exact same regression concept embedding batch/config;provider exact cache after regression',answer_judge_uncached_base_upper=counts['total_question_evaluations']*2,
        local_http_cap=cap,original_formal_started_unix=limits['started_unix'],global_limits=LIMITS,source_review_proof=review_proof,model_api_calls=0,raw_text_exported=False)
    if args.check_only:print(json.dumps(summary));return 0
    contract=dict(inventory=inventory_record,source_review_proof=review_proof,selection=file_record(selection_path),sealed=seal,datasets=datasets,split=args.split,counts=counts,shortage=extra,
        policy=corrected.POLICY,local_http_cap=cap,code={n:file_record(Path(__file__).parent/n) for n in ['corrected_f_dual_view.py','run_corrected_f_dual_view.py','state_dual_view.py','run_state_dual_view.py','run_frozen_dual_view.py','dual_view_rag.py','dvkg_development.py','native_provider.py','run_native_full.py']})
    identity=digest(contract);root=args.run_root/'corrected_F_dual_view'/args.split/identity;root.mkdir(parents=True,exist_ok=True);credentials()
    provider=AuditedNativeProvider(runtime_for(state.asset(assets['cohorts']['compABCD']['construction']),root/'transport'),args.run_root/'formal_provider',**LIMITS);tracker=ReceiptTracker(provider)
    try:
        with FileLock(str(root/'execution.lock'),timeout=0),CorrectedBudget(provider,args.run_root,args.split) as budget,tracker.scope(root/'requests.jsonl'):
            write_immutable(root/'contract.json',dict(result_id=identity,**contract));raw=provider.chat_response
            def scoped(messages,**kwargs):
                if kwargs.get('stage')=='native_blind_judge':kwargs['stage']=budget.stage('judge')
                return raw(messages,**kwargs)
            provider.chat_response=scoped
            matrices=common.vectors(inv,root,provider,budget);results=[]
            for job in jobs:
                results.append(evaluate(job,inv['states'][job['state_key']],assets,matrices[job['state_key']],variant,datasets[job['book']],provider,
                    root/(job['cohort']+'__'+job['book']+'__F_corrected'),identity,budget))
                atomic(root/'state.json',dict(status='running',jobs_finished=len(results),jobs=10,pid=os.getpid(),updated_unix=time.time()))
            result=dict(summary,status='completed' if all(r['status']=='completed' for r in results) else 'partial',results=results,result_id=identity,local_http_attempts=budget.count(),model_api_calls=budget.count())
            write_immutable(root/'result.json',result);atomic(root/'state.json',result);print(json.dumps({k:v for k,v in result.items() if k!='results'}));return 0 if result['status']=='completed' else 2
    except Exception as exc:
        path=root/('failure_'+str(time.time_ns())+'.log');path.write_text(traceback.format_exc());atomic(root/'state.json',dict(status='interrupted_preserved_checkpoints',error_type=type(exc).__name__,error_record=file_record(path),pid=os.getpid()));return 2
    finally:tracker.close();provider.close()

if __name__=='__main__':raise SystemExit(main())
