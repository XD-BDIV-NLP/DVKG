"""固定开发选择的DVKG三状态同64题入口；原检索和评分实现保持不变。"""
from __future__ import annotations
import argparse,json,os,time,traceback
from pathlib import Path
from filelock import FileLock
from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable
from . import run_controlled_blind64 as controlled
from . import run_state_dual_view as common
from . import run_corrected_f_dual_view as corrected_run
from . import corrected_f_dual_view,dual_view_run
from . import state_dual_view as state
from . import run_frozen_dual_view as frozen
from .blind64_pipeline import require_freeze,FreezeError,validate_provider
from .run_independent_holdout import LocalHTTPBudget
from .native_provider import AuditedNativeProvider
from .protocol import runtime_for
from .run_native_full import LIMITS,ReceiptTracker
from .run_repaired_full import atomic,credentials

METHOD_ID='dvkg_selected_control'
LOCAL_CAP=800


def implementation_records():
    records=controlled.implementation_records()+[file_record(m.__file__) for m in [corrected_run,corrected_f_dual_view,dual_view_run]]+[file_record(__file__)]
    return list({r['path']:r for r in records}.values())


def selected_variant(selection):
    return frozen.selected_variant(selection)


def evaluator(semantic_state):
    if semantic_state in ['G0','T']:return common.evaluate
    if semantic_state=='F_corrected':return corrected_run.evaluate
    raise ValueError('Only original G0/T and independently corrected F are registered')


def load_datasets(run_root,freeze_path,freeze_sha):
    algorithm,_=require_freeze(freeze_path,freeze_sha);method=controlled._method(algorithm,METHOD_ID)
    covered={(r['path'],r['sha256']) for r in method['code']}
    if any((r['path'],r['sha256']) not in covered for r in implementation_records()):raise FreezeError('Current selected DVKG implementation is not in final freeze')
    return controlled.load_datasets(run_root,freeze_path,freeze_sha,METHOD_ID)


def prepare(states,corrected,assets,datasets):
    if len(states['states'])!=8 or len(corrected['states'])!=4:raise ValueError('Exactly original eight G0/T and four corrected F states required')
    for inventory in [states,corrected]:
        for entry in inventory['states'].values():
            if entry['active_books']!=assets['cohorts'][entry['cohort']]['active_books']:raise ValueError('Original AB/ABCD/OS/NET paired graph book scope differs')
    jobs1,counts1=common.prepare(states,assets,datasets);jobs2,counts2=corrected_run.prepare(corrected,assets,datasets)
    if any(n!=8 for n in counts1['by_book'].values()) or counts1['unique_questions']!=64 or counts1['question_ids_digest']!=counts2['question_ids_digest']:
        raise ValueError('All selected DVKG states must use exactly the same sealed64 questions')
    jobs=[dict(j,inventory='states') for j in jobs1]+[dict(j,inventory='corrected') for j in jobs2]
    if len(jobs)!=30 or counts1['total_question_evaluations']+counts2['total_question_evaluations']!=240:raise ValueError('Exactly thirty book-state conditions and240 evaluations required')
    return jobs,dict(unique_questions=64,total_question_evaluations=240,book_state_conditions=30,graph_states=12,by_book=counts1['by_book'],question_ids_digest=counts1['question_ids_digest'],AB_vs_ABCD_paired=True)


class Allowance(LocalHTTPBudget):
    def __init__(self,provider,run_root):
        super().__init__(provider,Path(run_root)/'dvkg_selected_allowances'/'blind64',digest([METHOD_ID,'blind64']),cap=LOCAL_CAP)
        self.prefix=METHOD_ID+'/blind64/'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    for key in ['state','corrected']:
        p.add_argument('--'+key+'-inventory',type=Path,required=True);p.add_argument('--'+key+'-inventory-sha',required=True)
    p.add_argument('--freeze-manifest',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    datasets,seal=load_datasets(a.run_root,a.freeze_manifest,a.freeze_sha256)
    inv1,r1=common.checked_inventory(a.state_inventory,a.state_inventory_sha);inv2,r2,proof=corrected_run.checked_inventory(a.corrected_inventory,a.corrected_inventory_sha)
    if inv1['asset_manifest']!=inv2['asset_manifest']:raise ValueError('Original selected DVKG source evidence assets differ')
    assets=state.asset(inv1['asset_manifest']);jobs,counts=prepare(inv1,inv2,assets,datasets)
    selection_path=a.run_root/frozen.SELECTION;selection=read_document(selection_path);variant=selected_variant(selection);state.asset(selection['source']);selection_record=file_record(selection_path)
    algorithm,_=require_freeze(a.freeze_manifest,a.freeze_sha256);method=controlled._method(algorithm,METHOD_ID)
    for record in [r1,r2,proof,selection_record]:controlled._require_record(method,'parameters',record)
    for inventory in [inv1,inv2]:
        for entry in inventory['states'].values():controlled._require_record(method,'graphs',entry['graph'])
    summary=dict(status='checked_no_api',method=METHOD_ID,split='blind64',**counts,local_http_cap=LOCAL_CAP,native_group_cap=45000,model_api_calls=0,raw_text_exported=False,selected_variant=variant,old_results_unmodified=True)
    if a.check_only:
        print(json.dumps(dict(summary,answer_judge_HTTP_base_upper=480,query_embedding_nominal_upper=64,concept_embedding_HTTP_base_upper=sum((len(inv['unique_missing_embeddings'])+15)//16 for inv in [inv1,inv2]),stage_group='native')));return 0
    contract=dict(version='selected_DVKG_same_sealed64_v1',method=METHOD_ID,split='blind64',inventories=dict(states=r1,corrected=r2),corrected_source_proof=proof,selection=selection_record,
        selected_variant=variant,seal=seal,datasets=datasets,counts=counts,local_cap=LOCAL_CAP,policy=dict(states=state.POLICY,corrected=corrected_f_dual_view.POLICY),code=implementation_records())
    identity=digest(contract);root=a.run_root/METHOD_ID/'blind64'/identity;root.mkdir(parents=True,exist_ok=True);credentials()
    provider=AuditedNativeProvider(runtime_for(state.asset(assets['cohorts']['compABCD']['construction']),root/'transport'),a.run_root/'formal_provider',**LIMITS);tracker=None
    try:
        tracker=ReceiptTracker(provider);authorization=validate_provider(provider);allowance=Allowance(provider,a.run_root)
        with FileLock(str(allowance.root/'execution.lock'),timeout=0),allowance as budget,tracker.scope(root/'requests.jsonl'):
            write_immutable(root/'contract.json',dict(result_id=identity,**contract,effective_execution_authorization=authorization));raw=provider.chat_response
            def scoped(messages,**kwargs):
                if kwargs.get('stage')=='native_blind_judge':kwargs['stage']=budget.stage('judge')
                return raw(messages,**kwargs)
            provider.chat_response=scoped
            inventories=dict(states=inv1,corrected=inv2);matrices={k:common.vectors(v,root/k,provider,budget) for k,v in inventories.items()};results=[]
            for job in jobs:
                key=job['inventory'];entry=inventories[key]['states'][job['state_key']]
                results.append(evaluator(job['state'])(job,entry,assets,matrices[key][job['state_key']],variant,datasets[job['book']],provider,root/(job['cohort']+'__'+job['book']+'__'+job['state']),identity,budget))
                atomic(root/'state.json',dict(status='running',jobs_finished=len(results),jobs=30,pid=os.getpid(),updated_unix=time.time()))
            terminal=controlled.fixed_scope_complete(results)
            result=dict(summary,status='completed' if terminal else 'partial',result_id=identity,results=results,local_http_attempts=budget.count(),failed_count=sum(r['failed_count'] for r in results),all_questions_successful=all(r['failed_count']==0 for r in results),denominator='same64;240paired_evaluations;failures_retained_zero;no_score_selection')
            write_immutable(root/'result.json',result);atomic(root/'state.json',result);print(json.dumps({k:v for k,v in result.items() if k!='results'}));return 0 if terminal else 2
    except Exception as exc:
        path=root/('failure_'+str(time.time_ns())+'.log');path.write_text(traceback.format_exc());atomic(root/'state.json',dict(status='interrupted_preserved_checkpoints',error_type=type(exc).__name__,error_record=file_record(path),raw_text_exported=False));return 2
    finally:
        if tracker is not None:tracker.close()
        provider.close()


if __name__=='__main__':raise SystemExit(main())
