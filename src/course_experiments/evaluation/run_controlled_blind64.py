"""新64同题集的共同构图薄入口；复用原检索、生成、裁判和逐题检查点。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import traceback
from filelock import FileLock
from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import run_baseline_rag_bridge as shared
from . import baseline_rag_bridge as bridge
from . import state_dual_view as state
from . import run_state_dual_view as common
from . import run_frozen_dual_view as frozen
from .blind64_pipeline import QUOTAS,FreezeError,require_freeze,validate_provider
from .native_provider import AuditedNativeProvider
from .protocol import runtime_for
from .run_independent_holdout import LocalHTTPBudget,HoldoutBudgetExceeded
from .run_native_full import LIMITS,ReceiptTracker
from .run_repaired_full import atomic,credentials


def implementation_records():
    from . import blind64_pipeline,native_provider,protocol,run_independent_holdout,run_native_full,run_repaired_full,dvkg_development,dual_view_rag,dual_view_pipeline,rag_repair_v3,repaired_full,rag,graph_rag_v2
    from .. import manifest,contributions,dual_view,textbook
    from domain_ckg_workspace.ckg_incremental import runtime
    paths=[Path(__file__)]+[Path(m.__file__) for m in [shared,bridge,state,common,frozen,blind64_pipeline,native_provider,protocol,run_independent_holdout,run_native_full,run_repaired_full,dvkg_development,dual_view_rag,dual_view_pipeline,rag_repair_v3,repaired_full,rag,graph_rag_v2,manifest,contributions,dual_view,textbook,runtime]]
    return [file_record(p) for p in paths]


def _method(algorithm,method_id):
    rows=[r for r in algorithm['methods'] if r['method_id']==method_id]
    if len(rows)!=1:raise FreezeError('Controlled method must be preregistered in the final blind64 freeze')
    return rows[0]


def load_datasets(run_root,freeze_path,freeze_sha,method_id):
    algorithm,freeze=require_freeze(freeze_path,freeze_sha)
    method=_method(algorithm,method_id);covered={(r['path'],r['sha256']) for r in method['code']}
    if any((r['path'],r['sha256']) not in covered for r in implementation_records()):raise FreezeError('Current controlled implementation is not in final freeze')
    path=Path(run_root)/'blind64_20261022_v1/sealed.json';sealed=read_document(path)
    if (sealed.get('status')!='sealed_complete' or sealed.get('accepted_count')!=64 or sealed.get('target_count')!=64 or sealed.get('released_for_quality_evaluation') is not True):raise ValueError('Only the full released sealed64 may be evaluated')
    contract=state.asset(sealed['contract']);registered=sorted(r['method_id'] for r in algorithm['methods'])
    if contract.get('algorithm_freeze')!=freeze or contract.get('registered_methods')!=registered or sealed.get('registered_methods')!=registered:raise FreezeError('Sealed64 belongs to a different final method freeze')
    index=state.asset(sealed['dataset_index']);refs=index['books'];books={b for rows in algorithm['scope']['courses'].values() for b in rows}
    if set(refs)!=books or index.get('registered_methods')!=registered:raise ValueError('Same sealed64 index must cover all eight books and methods')
    ids=[]
    for book,record in refs.items():
        dataset=state.asset(record);questions=dataset['questions'];types=Counter(q['type'] for q in questions)
        if dataset.get('target_book_id')!=book or len(questions)!=8 or types!=QUOTAS or any(q['target_book_id']!=book for q in questions):raise ValueError('Sealed64 per-book count/type/target differs')
        ids.extend(q['question_id'] for q in questions)
    if len(ids)!=len(set(ids)) or len(ids)!=64:raise ValueError('Sealed64 fixed question identity differs')
    return refs,dict(sealed=file_record(path),dataset_index=sealed['dataset_index'],algorithm_freeze=freeze,accepted_count=64,registered_methods=registered)


class Allowance(LocalHTTPBudget):
    def __init__(self,provider,run_root,method,cohort,cap):
        if method not in bridge.METHODS or cohort not in bridge.POLICY['cohorts'] or type(cap) is not int or not 1<=cap<=45000:raise ValueError('Explicit controlled blind64 native-group allowance required')
        super().__init__(provider,Path(run_root)/'controlled_baseline_allowances'/method/cohort/'blind64',digest([method,cohort,'blind64']),cap=cap)
        self.prefix='controlled_baseline/'+method+'/'+cohort+'/blind64/'


def _require_record(method,role,record):
    if record not in method[role]:raise FreezeError('Current controlled '+role+' asset is not in final freeze')


def fixed_scope_complete(results):
    return bool(results) and all(r['question_count']==8 and r['completed_count']+r['failed_count']==8 for r in results)


def validate_entry_scope(entry,assets,method,cohort,refs):
    if (entry.get('method')!=method or entry.get('cohort')!=cohort or entry.get('source_scope_verified') is not True
            or entry['active_books']!=assets['cohorts'][cohort]['active_books'] or set(refs)!=set(assets['unit_assets'])):
        raise ValueError('Complete original joint cohort and exact same eight-book question scope required')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--inventory-sha',required=True)
    p.add_argument('--method',choices=bridge.METHODS,required=True);p.add_argument('--cohort',choices=bridge.POLICY['cohorts'],required=True)
    p.add_argument('--freeze-manifest',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);p.add_argument('--freeze-method-id')
    p.add_argument('--max-http-attempts',type=int,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args();method_id=a.freeze_method_id or a.method
    refs,seal=load_datasets(a.run_root,a.freeze_manifest,a.freeze_sha256,method_id)
    inventory,inventory_record=shared.checked_inventory(a.inventory,a.inventory_sha);key=a.method+':'+a.cohort
    if key not in inventory['ready']:raise common.DependencyPending('Complete native joint-cohort artifact required')
    entry=inventory['ready'][key];assets=state.asset(inventory['asset_manifest'])
    validate_entry_scope(entry,assets,a.method,a.cohort,refs)
    if entry.get('source_scope_verified') is not True or file_record(entry['artifact']['path'])!=entry['artifact']:raise ValueError('Complete original source coverage or artifact changed')
    algorithm,_=require_freeze(a.freeze_manifest,a.freeze_sha256);method=_method(algorithm,method_id)
    _require_record(method,'parameters',inventory_record);_require_record(method,'graphs',entry['artifact'])
    selection_path=a.run_root/frozen.SELECTION;selection=read_document(selection_path);variant=frozen.selected_variant(selection);selection_record=file_record(selection_path)
    _require_record(method,'parameters',selection_record)
    if file_record(selection['source']['path'])!=selection['source']:raise ValueError('Original development-only selection changed')
    cap=a.max_http_attempts
    if not 1<=cap<=45000:raise ValueError('Explicit native-group local cap must be within45000')
    summary=shared.scope_summary(entry,refs);summary.update(split='blind64',unique_frozen_questions=64,target_count=64,method_scores_used_for_dataset_selection=False,model_api_calls=0,raw_text_exported=False)
    contract=dict(version='controlled_same_sealed64_v1',method=a.method,cohort=a.cohort,split='blind64',inventory=inventory_record,artifact=entry['artifact'],selection=selection_record,
        seal=seal,datasets={b:refs[b] for b in entry['active_books']},policy=bridge.POLICY,local_cap=cap,code=implementation_records())
    if a.check_only:
        print(json.dumps(dict(status='checked_no_api',scope=summary,local_http_cap=cap,embedding_HTTP_base_upper=(entry['vector_plan']['unique_missing_count']+15)//16,answer_judge_HTTP_base_upper=2*summary['question_evaluations'],stage_group='native',generation_scope_1200_unmodified=True)));return 0
    identity=digest(contract);root=a.run_root/'controlled_baseline_rag'/a.method/a.cohort/'blind64'/identity;root.mkdir(parents=True,exist_ok=True);credentials()
    provider=AuditedNativeProvider(runtime_for(state.asset(entry['construction']),root/'transport'),a.run_root/'formal_provider',**LIMITS);tracker=None
    try:
        tracker=ReceiptTracker(provider)
        authorization=validate_provider(provider);allowance=Allowance(provider,a.run_root,a.method,a.cohort,cap)
        with FileLock(str(allowance.root/'execution.lock'),timeout=0),allowance as budget,tracker.scope(root/'requests.jsonl'):
            write_immutable(root/'contract.json',dict(result_id=identity,**contract,effective_execution_authorization=authorization));raw=provider.chat_response
            def scoped(messages,**kwargs):
                if kwargs.get('stage')=='native_blind_judge':kwargs['stage']=budget.stage('judge')
                return raw(messages,**kwargs)
            provider.chat_response=scoped
            matrix=common.vectors(shared.vector_inventory(entry),root,provider,budget)['selected']
            engine=bridge.BaselineRetriever(a.method,entry['graph'],{b:assets['structures'][b] for b in entry['active_books']},entry['mappings'],
                {b:assets['unit_assets'][b] for b in entry['active_books']},entry['vector_plan']['concept_ids'],matrix,variant=variant)
            results=[shared.evaluate_book(entry,b,refs[b],engine,provider,root/b,identity,budget) for b in entry['active_books']]
            terminal=fixed_scope_complete(results)
            result=dict(summary,status='completed' if terminal else 'partial',result_id=identity,results=results,failed_count=sum(r['failed_count'] for r in results),all_questions_successful=all(r['failed_count']==0 for r in results),
                denominator='all_same_sealed_questions;failures_retained_zero;no_score_selection',local_http_attempts=budget.count(),native_group_cap=45000,generation_scope_1200_unmodified=True)
            write_immutable(root/'result.json',result);atomic(root/'state.json',result);print(json.dumps({k:v for k,v in result.items() if k!='results'}));return 0 if terminal else 2
    except Exception as exc:
        path=root/('failure_'+str(time.time_ns())+'.log');path.write_text(traceback.format_exc());atomic(root/'state.json',dict(status='interrupted_preserved_checkpoints',error_type=type(exc).__name__,error_record=file_record(path),raw_text_exported=False));return 2
    finally:
        if tracker is not None:tracker.close()
        provider.close()


if __name__=='__main__':raise SystemExit(main())
