"""逐完整cohort的原生构图受控RAG；缺完整图明确pending，预算跨APP/目录恢复累计。"""
from __future__ import annotations
import argparse,collections,json,os,time,traceback
from pathlib import Path
import numpy as np
from filelock import FileLock
from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable
from . import baseline_rag_bridge as bridge
from . import state_dual_view as state
from . import run_state_dual_view as common
from . import run_frozen_dual_view as frozen
from .run_independent_holdout import LocalHTTPBudget,HoldoutBudgetExceeded
from .native_provider import AuditedNativeProvider,NativeAccountUnavailable,NativeBudgetExceeded
from .protocol import runtime_for
from .run_repaired_full import atomic,credentials
from .run_native_full import LIMITS,ReceiptTracker

LOCAL_CAPS={('k12_kgraph','compAB','regression'):550,('k12_kgraph','osAB','regression'):750,
            ('k12_kgraph','compAB','independent'):200,('k12_kgraph','osAB','independent'):200}


class Allowance(LocalHTTPBudget):
    def __init__(self,provider,run_root,method,cohort,split,cap):
        if method not in bridge.METHODS or cohort not in bridge.POLICY['cohorts'] or split not in ['regression','independent']:
            raise ValueError('Allowance requires the declared method/cohort/split')
        root=Path(run_root)/'controlled_baseline_allowances'/method/cohort/split
        super().__init__(provider,root,digest([method,cohort,split]),cap=cap)
        self.prefix='controlled_baseline/'+method+'/'+cohort+'/'+split+'/'


def checked_inventory(path,sha):
    proof=file_record(path)
    if proof['sha256']!=sha:raise ValueError('Frozen complete baseline inventory changed')
    value=read_document(path)
    if value['identity']!=digest({k:v for k,v in value.items() if k!='identity'}):raise ValueError('Baseline inventory identity differs')
    if value['code']['sha256']!=file_record(bridge.__file__)['sha256'] or value['state_code']['sha256']!=file_record(state.__file__)['sha256']:
        raise ValueError('Frozen thin bridge or shared state retrieval code changed')
    if file_record(value['asset_manifest']['path'])!=value['asset_manifest']:raise ValueError('Original same-S assets changed')
    if value['pending']!=bridge.pending_matrix(value['ready'],value['pending']):raise ValueError('Baseline inventory must explicitly cover all five methods/four cohorts, including GraphExtract pending')
    return value,proof


def scope_summary(entry,datasets):
    graph=entry['graph'];entities=graph['entities'];indexed=[e for e in entities if e.get('type') not in state.STRUCTURAL]
    qids={b:sorted(q['question_id'] for q in state.asset(datasets[b])['questions']) for b in entry['active_books']}
    for b in entry['active_books']:
        qs=state.asset(datasets[b])['questions']
        if len(qids[b])!=len(set(qids[b])) or any(q['target_book_id']!=b for q in qs):raise ValueError('Same-book frozen question scope differs')
    if len(indexed)!=len(entry['vector_plan']['concept_ids']):raise ValueError('Common structural filter differs from concept vector plan')
    return dict(method=entry['method'],cohort=entry['cohort'],target_books=entry['active_books'],target_book_conditions=len(entry['active_books']),
        question_evaluations=sum(len(v) for v in qids.values()),by_book={b:len(v) for b,v in qids.items()},question_ids_digest=digest(qids),
        book_question_ids_digests={b:digest(v) for b,v in qids.items()},same_frozen_question_pool_as_F=True,
        native_retained_nodes=len(entities),native_retained_edges=len(graph['relations']),dense_indexed_concepts=len(indexed),
        common_structural_filtered_nodes=len(entities)-len(indexed),actual_native_type_counts=dict(collections.Counter(str(e.get('type','<missing>')) for e in entities)),
        actual_dense_native_type_counts=dict(collections.Counter(str(e.get('type','<missing>')) for e in indexed)),
        vector_reusable_count=entry['vector_plan']['reusable_count'],vector_missing_count=entry['vector_plan']['missing_count'],
        embedding_http_base_upper=(entry['vector_plan']['unique_missing_count']+15)//16,
        untyped_nodes_are_not_new_Concept_claims=True,source_unit_correlated_is_fact_support=False)


def vector_inventory(entry):
    plan=entry['vector_plan']
    return dict(states={'selected':entry},unique_missing_embeddings={k:dict(text=t,embedding_configuration=plan['embedding_configuration']) for k,t in plan['missing_texts'].items()})


def evaluate_book(entry,book,dataset,engine,provider,root,identity,budget):
    root=Path(root);records=[];questions=state.asset(dataset)['questions']
    for q in questions:
        folder=root/'questions'/q['question_id'];path=folder/'record.json'
        if path.exists():
            row=read_document(path)
            if row['result_id']!=identity or row['checkpoint_digest']!=digest({k:v for k,v in row.items() if k!='checkpoint_digest'}):raise ValueError('Controlled baseline question checkpoint differs')
            records.append(row);continue
        folder.mkdir(parents=True,exist_ok=True)
        try:
            original=frozen.PARENT/'task_vectors'/book/(q['question_id']+'.npy')
            if original.exists():
                meta=read_document(original.with_suffix('.npy.meta.json'))['vectors']
                if file_record(original)!=meta:raise ValueError('Frozen shared query vector changed')
                vector=np.load(original)[0]
            else:vector=provider.embed([q['question']],stage=budget.stage('query_embedding'))[0]
            row=state.evaluate_state_question(engine,q,vector,provider,folder,identity,budget.stage('controlled_qa'))
            row.update(result_id=identity,controlled_bridge_policy=bridge.POLICY['version'],native_method=entry['method'])
        except (NativeAccountUnavailable,NativeBudgetExceeded,HoldoutBudgetExceeded):raise
        except Exception as exc:
            error=folder/'error.log';error.write_text(traceback.format_exc());row=dict(result_id=identity,status='failed',question_id=q['question_id'],question_type=q['type'],target_book_id=book,error_type=type(exc).__name__,error_record=file_record(error))
        records.append(frozen.checkpoint(path,row));atomic(root/'state.json',dict(status='running',records=len(records),total=len(questions),pid=os.getpid(),updated_unix=time.time()))
    result=dict(frozen.old.summarize(records,[q['question_id'] for q in questions]),method=entry['method'],cohort=entry['cohort'],book=book,result_id=identity,
        judge_version=state.POLICY['judge_version'],failed_count=sum(r['status']=='failed' for r in records),source_unit_correlated_is_fact_support=False,raw_text_exported=False)
    if not questions:result.update(status='completed',empty_sealed_book=True,denominator='zero_legal_sealed_questions;metrics_NA;no_fake_gold_scored')
    write_immutable(root/'result.json',result);atomic(root/'state.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--inventory-sha',required=True)
    p.add_argument('--method',choices=bridge.METHODS,required=True);p.add_argument('--cohort',choices=bridge.POLICY['cohorts'],required=True)
    p.add_argument('--split',choices=['regression','independent'],required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--max-http-attempts',type=int)
    args=p.parse_args();inventory,inventory_record=checked_inventory(args.inventory,args.inventory_sha);key=args.method+':'+args.cohort
    if key not in inventory['ready']:
        print(json.dumps(dict(status='pending_complete_cohort_artifact',method=args.method,cohort=args.cohort,
            reason=inventory['pending'].get(key,'complete_joint_artifact_not_declared_in_frozen_inventory'),question_evaluations=None,model_api_calls=0,raw_text_exported=False)));return 2
    entry=inventory['ready'][key];assets=state.asset(inventory['asset_manifest'])
    if file_record(entry['artifact']['path'])!=entry['artifact']:raise ValueError('Completed native artifact changed')
    selection_path=args.run_root/frozen.SELECTION;selection=read_document(selection_path);variant=frozen.selected_variant(selection)
    if file_record(selection['source']['path'])!=selection['source']:raise ValueError('Prior development-only selection source changed')
    cap=LOCAL_CAPS.get((args.method,args.cohort,args.split),args.max_http_attempts)
    if cap is None or not 1<=cap<=30000 or args.max_http_attempts not in [None,cap]:raise ValueError('Explicit root frozen cohort/split HTTP allowance required')
    limits=read_document(args.run_root/'formal_provider/limits.json')
    if limits['limits']!=LIMITS or not (args.run_root/'formal_provider/audit.sqlite').is_file():raise ValueError('Original formal provider budget required; no reset')
    if args.split=='independent':
        try:datasets,seal,shortage=common.independent_datasets(args.run_root,assets)
        except common.DependencyPending:
            if not args.check_only:raise
            print(json.dumps(dict(status='dependencies_pending',method=args.method,cohort=args.cohort,split=args.split,target_count=128,actual_count=None,
                target_book_conditions=len(entry['active_books']),local_http_cap=cap,model_api_calls=0,raw_text_exported=False)));return 2
    else:
        datasets={b:r['confirmation'] for b,r in read_document(frozen.PARENT/'datasets/index.json')['books'].items()};seal=None;shortage={}
    summary=scope_summary(entry,datasets);summary.update(status='checked_no_api',split=args.split,local_http_cap=cap,
        all_unique_confirmation_pool=313 if args.split=='regression' else shortage['accepted_count'],**shortage,
        original_formal_started_unix=limits['started_unix'],global_limits=LIMITS,model_api_calls=0,raw_text_exported=False,
        complete_artifact=entry['artifact'],pending_other_method_cohorts=inventory['pending'],budget_directory_independent=True,
        uncached_answer_judge_base_upper=summary['question_evaluations']*2)
    if args.check_only:print(json.dumps(summary));return 0
    selected_refs={b:datasets[b] for b in entry['active_books']}
    contract=dict(method=args.method,cohort=args.cohort,split=args.split,inventory=inventory_record,artifact=entry['artifact'],datasets=selected_refs,sealed=seal,
        shortage=shortage,selection=file_record(selection_path),policy=bridge.POLICY,local_cap=cap,scope_summary=summary,
        code={n:file_record(Path(__file__).parent/n) for n in ['baseline_rag_bridge.py','run_baseline_rag_bridge.py','state_dual_view.py','run_state_dual_view.py','run_frozen_dual_view.py','dual_view_rag.py','dvkg_development.py','run_native_full.py']})
    identity=digest(contract);root=args.run_root/'controlled_baseline_rag'/args.method/args.cohort/args.split/identity;root.mkdir(parents=True,exist_ok=True);credentials()
    provider=AuditedNativeProvider(runtime_for(state.asset(entry['construction']),root/'transport'),args.run_root/'formal_provider',**LIMITS);tracker=ReceiptTracker(provider)
    allowance=Allowance(provider,args.run_root,args.method,args.cohort,args.split,cap)
    try:
        with FileLock(str(allowance.root/'execution.lock'),timeout=0),allowance as budget,tracker.scope(root/'requests.jsonl'):
            write_immutable(root/'contract.json',dict(result_id=identity,**contract));raw=provider.chat_response
            def scoped(messages,**kwargs):
                if kwargs.get('stage')=='native_blind_judge':kwargs['stage']=budget.stage('judge')
                return raw(messages,**kwargs)
            provider.chat_response=scoped
            matrix=common.vectors(vector_inventory(entry),root,provider,budget)['selected']
            engine=bridge.BaselineRetriever(args.method,entry['graph'],{b:assets['structures'][b] for b in entry['active_books']},entry['mappings'],
                {b:assets['unit_assets'][b] for b in entry['active_books']},entry['vector_plan']['concept_ids'],matrix,variant=variant)
            results=[]
            for book in entry['active_books']:
                results.append(evaluate_book(entry,book,datasets[book],engine,provider,root/book,identity,budget));atomic(root/'state.json',dict(status='running',books_finished=len(results),books=len(entry['active_books']),pid=os.getpid(),updated_unix=time.time()))
            result=dict(summary,status='completed' if all(r['status']=='completed' for r in results) else 'partial',result_id=identity,results=results,
                model_api_calls=budget.count(),local_http_attempts=budget.count(),allowance_scope=str(allowance.root))
            write_immutable(root/'result.json',result);atomic(root/'state.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['results','pending_other_method_cohorts']}));return 0 if result['status']=='completed' else 2
    except Exception as exc:
        path=root/('failure_'+str(time.time_ns())+'.log');path.write_text(traceback.format_exc());atomic(root/'state.json',dict(status='interrupted_preserved_checkpoints',error_type=type(exc).__name__,error_record=file_record(path),pid=os.getpid()));return 2
    finally:tracker.close();provider.close()


if __name__=='__main__':raise SystemExit(main())
