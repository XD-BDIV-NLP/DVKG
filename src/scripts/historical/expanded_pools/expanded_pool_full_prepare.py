"""仅零API准备原353题双池输入；新增精排作为探索行，不宣称构图SOTA。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import traceback


ROOT=Path(__file__).resolve().parent/'full_comparison'
N=PurePosixPath(os.environ.get("DVKG_HISTORICAL_ROOT") or str(Path("outputs/complete_course_benchmark_20261022_v1").resolve()))
DEPLOYED_ROOT=N/'dvkg_evidence_upgrade_20261116/7e46e41df054027b'
CANONICAL_REF=dict(path=str(N/'rag_expanded_pools_20261114_v1/common_recovery_v2/final_audits/two_stage_final_1794797874424841242/final_audited_five_method_results.json'),
    sha256='43cb0cb57a2288f1b39ebde8d77575c76b4b847764d279392f0e4b02922e1fcb')
DEPLOYED_FILES={
    'development_worker.py':'cd7d2d9d05a498d0605ce6913d74973d4f8f17b576a65d8f7cbedf0027f88028',
    'modules/query_evidence_rerank.py':'921d73c0d119ae27aa33e5e157802352ce517220afa9cfe77b9de8d18cab83ac',
    'modules/evidence_window_packing.py':'447316cbb69db14fbeb557a457e2c01e5c7c5a9a055db0150ad6c29c9c926756',
    'modules/evidence_stage_metrics.py':'fa9b6cb1a3fa9676c97f2ec9944e3bed2c1f49175d4d8d9741947716dd2ed571',
    'modules/evidence_development_run.py':'bdbbe7eaabd28bb09ef757d8c272d1168213ed6829937c66d31098cc168a571e',
    'modules/evidence_selection_pool.py':'6c4fe53a368b5823e3de354b7a8d59183eaec8e5ac406fc2f6d8043150f66d80',
    'modules/run_joint_pool_retrieval.py':'50dc877bb749edd30eb6a16eec1080441e45522c30d93807ec4675e4a2a6969f'}
EXPECTED_COUNTS={'book_25f718562685':100,'book_49bb8d31ddb9':100,
    'book_0b8d87cb0667':40,'book_d96f472c8179':37,'book_00849acee112':38,'book_dd8183b4cb60':38}
SCOPES=('same_subject_pool','mixed_six_book_pool')
BASELINES=('K12-KGraph','GraphRAG','LightRAG','HippoRAG')


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))


def record(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(reference):
    if record(reference['path'])!=reference:raise ValueError('Frozen full-comparison source changed')
    return read(reference['path'])


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Refusing to modify an existing full-comparison artifact')
    else:atomic(path,value)


def state(root,status,**fields):
    atomic(Path(root)/'state.json',dict(status=status,pid=os.getpid(),observed_unix=time.time(),model_calls=0,**fields))


def initialize_dependencies():
    if os.name!='posix':raise RuntimeError('Real textbook preparation runs only on the original server; no local raw-text export')
    frozen=Path(str(DEPLOYED_ROOT));refs={}
    for relative,expected in DEPLOYED_FILES.items():
        reference=record(frozen/relative)
        if reference['sha256']!=expected:raise ValueError('Completed development bundle changed: '+relative)
        refs[relative]=reference
    spec=importlib.util.spec_from_file_location('expanded_full_frozen_development',frozen/'development_worker.py')
    development=importlib.util.module_from_spec(spec);sys.modules[spec.name]=development;spec.loader.exec_module(development)
    cfg,original,m,joint=development.initialize()
    from course_experiments.evaluation.echa_structured_focus import StructuredFocusRetriever
    from course_experiments.evaluation import run_joint_pool_retrieval as old
    from course_experiments.evaluation.evidence_selection_pool import EvidenceSelectionPool, POLICY
    from course_experiments.evaluation.evidence_development_run import build_score_packet
    return dict(cfg=cfg,original=original,m=m,joint=joint,old=old,Engine=StructuredFocusRetriever,
        SelectionPool=EvidenceSelectionPool,selection_policy=POLICY,build_score_packet=build_score_packet,
        configuration=dict(path=str(development.CONFIG),sha256=development.CONFIG_SHA),
        module_refs=refs,canonical_ref=CANONICAL_REF)


def dataset_questions(cfg,m):
    specs=cfg['sources'];ids=[s['book_id']for s in specs];courses={s['course_id']for s in specs}
    if (len(ids)!=6 or set(ids)!=set(EXPECTED_COUNTS)or len(courses)!=3
            or any(sum(s['course_id']==c for s in specs)!=2 for c in courses)):
        raise ValueError('Original six textbook and three course scope required')
    questions=[];datasets={};seen=set()
    for spec in specs:
        book=spec['book_id']
        if spec['expected_question_count']!=EXPECTED_COUNTS[book]:raise ValueError('Original353 fixed per-book denominator changed')
        values=m._dataset_questions(spec['dataset'],book)
        if len(values)!=EXPECTED_COUNTS[book]:raise ValueError('Original353 source dataset count differs')
        datasets[book]=spec['dataset']
        for q in values:
            if q['target_book_id']!=book or q['question_id']in seen:raise ValueError('Original question source or identity differs')
            seen.add(q['question_id']);questions.append(dict(q,evaluation_course_id=spec['course_id']))
    if len(questions)!=353:raise ValueError('All353 original questions are required')
    return questions,datasets


def comparison_policy():
    return dict(variants=['rerank'],added_cross_encoder=True,added_global_candidate_union=True,
        baseline_reranker_equivalence=False,shared_reranker_control_pending=True,
        graph_construction_SOTA_claim=False,uniform_superiority_claim=False,
        comparison='exploratory_DVKG_plus_cross_encoder_against_unchanged_canonical_adapter_rows',
        additional_compute_must_be_reported=True,final_packer='frozen_original_JointPoolRetriever.pack_context',
        scoring_windows_are_not_final_answer_context=True,candidate_limit=64,retrieval_top_k=5,context_bytes=3000,
        target_book_or_course_oracle=False,independent_unseen_test=False)


def comparison_anchors(reference):
    canonical=checked(reference)
    if (canonical.get('question_count')!=353 or canonical.get('observations')!=3530
            or canonical.get('results_publishable')is not True or len(canonical.get('rows',[]))!=3530):
        raise ValueError('Complete audited five-method canonical source required')
    index={};anchors={}
    for method in (*BASELINES,'DVKG'):
        for scope in SCOPES:
            rows=[r for r in canonical['rows']if r['method']==method and r['scope']==scope]
            if len(rows)!=353 or any(r['status']!='completed'for r in rows):raise ValueError('Canonical comparison rows are incomplete')
            if method=='DVKG':
                for row in rows:
                    key=scope,row['question_id']
                    if key in index:raise ValueError('Repeated canonical DVKG question')
                    index[key]=row
        if method in BASELINES:anchors[method]=canonical['metrics'][method]['by_scope']
    return anchors,index


def match_baseline(cfg,q,scope,selected,context,canonical_index):
    path=Path(cfg['output_root'])/'retrieval'/scope/'dvkg'/digest(q['question_id'])/'retrieval.json'
    anchor=canonical_index.get((scope,q['question_id']))
    if not path.is_file()or anchor is None:
        return dict(status='matched_baseline_refs_pending',expected_path=str(path),canonical_row_present=anchor is not None)
    reference=record(path)
    if reference!=anchor['source_retrieval']:raise ValueError('Original baseline retrieval differs from audited canonical reference')
    if (anchor['book_id']!=q['target_book_id']or anchor['question_text_sha256']!=hashlib.sha256(q['question'].encode('utf-8')).hexdigest()):
        raise ValueError('Baseline source question differs from original dataset')
    saved=read(path)
    if saved.get('checkpoint_digest')!=digest({k:v for k,v in saved.items()if k!='checkpoint_digest'}):
        raise ValueError('Original baseline retrieval checkpoint changed')
    native=[[r['book_id'],r['source_evidence_id']]for r in selected]
    if (saved.get('question_id')!=q['question_id']or saved.get('scope')!=scope or saved.get('method')!='dvkg'
            or saved.get('selected_native_sources')!=native
            or json.dumps(saved.get('context'),ensure_ascii=False)!=json.dumps(context,ensure_ascii=False)):
        raise ValueError('New wrapper baseline does not replay original selected IDs and serialized context')
    return dict(status='matched',original_retrieval=reference,context_digest=digest(context),
        selected_native_sources=native,canonical_question_key=anchor['canonical_question_key'])


def score_packet(tasks,builder):
    packet=builder(tasks)
    if set(packet)!={'version','answers_included','model','revision','max_length','packets'}or packet['answers_included']is not False:
        raise ValueError('GPU score packet must use the existing whitelist')
    for item in packet['packets']:
        if set(item)!={'task_id','question','candidates'}:raise ValueError('Private scoring labels cannot enter GPU packet')
        for candidate in item['candidates']:
            if set(candidate)!={'key','text','char_start','char_end','input_sha256','source_sha256'}:
                raise ValueError('Only question and exact source scoring windows may enter GPU packet')
    return packet


def cache_frozen_pool_indexes(pools):
    # 本轮映射已冻结，只缓存实例索引；原检索器及其参数保持不变。
    guards={}
    for name,pool in pools.items():
        original=pool._pool_index;cached=original();snapshot=deepcopy(cached)
        pool._pool_index=lambda index=cached:index
        guards[name]=(original,cached,snapshot)
    return guards


def verify_frozen_pool_indexes(guards):
    for original,cached,snapshot in guards.values():
        if cached!=snapshot:raise ValueError('Static pool index changed during original353 retrieval')
        if original()!=snapshot:raise ValueError('Recomputed original pool index differs from frozen snapshot')
    return dict(version='run_local_frozen_pool_index_cache_v1',performance_only=True,
        frozen_source_code_modified=False,retrieval_parameters_changed=False,
        pool_count=len(guards),initial_original_builds_per_pool=1,final_original_rebuilds_per_pool=1,
        cached_index_unchanged=True,end_original_index_equal=True,
        comparison='exact_python_deep_equality_including_native_sets_and_tuple_keys',
        source_validation='original_load_and_final_deep_SHA_recheck_not_mtime_substitution')


def prepare(root=ROOT,dependencies=None):
    deps=dependencies or initialize_dependencies();root=Path(root);cfg=deps['cfg'];m=deps['m'];original=deps['original'];old=deps['old'];joint=deps['joint']
    if root.resolve()==Path(cfg['output_root']).resolve():raise ValueError('Original retrieval outputs cannot be overwritten')
    root.mkdir(parents=True,exist_ok=True);state(root,'loading_original353_frozen_context_NO_API')
    anchors,canonical_index=comparison_anchors(deps['canonical_ref'])
    questions,datasets=dataset_questions(cfg,m)
    ctx=m.load_context(original['inventory']['path'],original['inventory']['sha256'],original['tokenizer']['path'],original['tokenizer']['sha256'])
    m._recheck(ctx,deep=True);plan=checked(cfg['deferred_plan'])
    if ctx['identity']!=plan['context_identity']:raise ValueError('Original frozen context differs from mapping plan')
    sources=[];queries={};query_refs={};missing_queries=[]
    for spec in cfg['sources']:
        book=spec['book_id'];key=spec['state_key'];entry=ctx['entries'][key]
        if entry['state']!='F_corrected'or set(entry['active_books'])!={s['book_id']for s in cfg['sources']if s['course_id']==spec['course_id']}:
            raise ValueError('Course source graph differs from original two-book release')
        dispatcher=m.ExactDeferredAlignment(ctx,key,book,plan['full_plans'][key][book],root/'alignment'/book)
        old._load_mapping_proofs(m,spec,dispatcher)
        qualified=[r for i in sorted(dispatcher.loaded)for r in dispatcher.loaded[i]]
        engine=deps['Engine'](entry['graph_data'],{book:ctx['structures'][book]},entry['mappings'],{book:ctx['units'][book]},
            entry['vector_plan']['concept_ids'],entry['vectors'],{book:qualified},ctx['paragraph_tokens'])
        sources.append(dict(spec,engine=engine,dispatcher=dispatcher,graph_id=engine.graph_id))
        for q in (v for v in questions if v['target_book_id']==book):
            cached=m._existing_query(ctx,q,cfg['model_cache_root'],root/'query_cache')
            if cached is None:missing_queries.append(dict(book_id=book,question_id=q['question_id']))
            else:queries[q['question_id']]=cached[0];query_refs[q['question_id']]=cached[1]
    if missing_queries:
        result=dict(status='cached_query_dependencies_pending',missing=missing_queries,question_count=353,model_calls=0)
        atomic(root/'pending_dependencies.json',result);state(root,result['status'],pending=len(missing_queries));return result
    pools={course:joint.JointPoolRetriever([s for s in sources if s['course_id']==course],paragraph_dedup=True)
        for course in sorted({s['course_id']for s in sources})}
    pools['mixed_six_books']=joint.JointPoolRetriever(sources,paragraph_dedup=True);required=defaultdict(set)
    for q in questions:
        for label in (q['evaluation_course_id'],'mixed_six_books'):
            for graph,concepts in pools[label].query_concepts(q['question'],queries[q['question_id']]).items():required[graph].update(concepts)
    preflight=old.mapping_preflight(sources,required);immutable(root/'mapping_preflight.json',preflight)
    if preflight['missing_batch_count']:
        result=dict(status='mapping_dependencies_pending',missing_batch_count=preflight['missing_batch_count'],question_count=353,model_calls=0)
        atomic(root/'pending_dependencies.json',result);state(root,result['status'],pending=preflight['missing_batch_count']);return result
    index_guards=cache_frozen_pool_indexes(pools)
    wrappers={key:deps['SelectionPool'](pool)for key,pool in pools.items()};tasks=[];baseline_checks=[]
    for q in questions:
        for scope,label in (('same_subject_pool',q['evaluation_course_id']),('mixed_six_book_pool','mixed_six_books')):
            pool=pools[label];candidates,routes,trace=wrappers[label].candidates(q['question'],queries[q['question_id']])
            if not 1<=len(candidates)<=64:raise ValueError('Frozen full comparison requires one to64 unique candidate paragraphs per task')
            selected=trace.pop('baseline_selected');context=pool.pack_context(selected,q['question'])
            matched=match_baseline(cfg,q,scope,selected,context,canonical_index);baseline_checks.append(matched)
            tasks.append(dict(task_id=digest(dict(question=q['question_id'],scope=scope)),question_id=q['question_id'],question=q['question'],
                book_id=q['target_book_id'],course_id=q['evaluation_course_id'],scope=scope,variant='rerank',reference_answer=q['answer'],
                gold_paragraphs=sorted(old.paragraph_keys(ctx['structures'][q['target_book_id']],q['target_book_id'],q['evidence_ids'])),
                candidates=candidates,route_scores=[dict(key=list(k),score=v)for k,v in routes.items()],baseline_selected=selected,
                baseline_context=context,baseline_replay=matched,candidate_trace=trace,query_vector_ref=query_refs[q['question_id']],
                source='original353_frozen_expanded_pool_questions',independent_test=False))
            state(root,'preparing_original353_candidates_NO_API',completed=len(tasks),total=706)
    if len(tasks)!=706 or len({t['task_id']for t in tasks})!=706:raise ValueError('All706 original question-pool tasks required')
    packet=score_packet(tasks,deps['build_score_packet']);pending=sum(r['status']!='matched'for r in baseline_checks)
    index_cache_proof=verify_frozen_pool_indexes(index_guards)
    m._recheck(ctx,deep=True)
    contract=dict(version='original353_expanded_DVKG_rerank_exploratory_v1',deployed_root=str(DEPLOYED_ROOT),worker=record(__file__),
        module_refs=deps['module_refs'],original_configuration=deps['configuration'],original_contract=cfg['original_contract'],
        context_identity=ctx['identity'],deferred_plan=cfg['deferred_plan'],datasets=datasets,expected_question_counts=EXPECTED_COUNTS,
        question_count=353,task_count=706,scopes=list(SCOPES),comparison_policy=comparison_policy(),selection_policy=deps['selection_policy'],
        canonical_comparison_anchor=deps['canonical_ref'],baseline_methods=list(BASELINES),baseline_metrics=anchors,
        original_baseline_replay=dict(matched=706-pending,pending=pending,status='matched_baseline_refs_pending'if pending else'all706_baseline_rows_exactly_replayed'),
        mapping_proofs={s['book_id']:list(s['dispatcher'].proofs.values())for s in sources},query_vector_records=query_refs,
        source_graphs_modified=False,paid_generation_enabled=False,model_calls=0,raw_text_destination='original_server_private_output_only',
        performance_cache=index_cache_proof,
        candidate_scoring_model=dict(model=packet['model'],revision=packet['revision'],max_length=packet['max_length']),
        original_baselines_reexecuted=False,full_comparison_answers_generated=False,independent_test=False)
    immutable(root/'input_contract.json',contract)
    immutable(root/'private_prepared_tasks.json',dict(contract_digest=digest(contract),tasks=tasks))
    immutable(root/'gpu_score_packet.json',packet)
    result=dict(status='matched_baseline_refs_pending'if pending else'full_original353_inputs_prepared_NO_answers',question_count=353,task_count=706,
        variant='rerank',baseline_replayed=706-pending,baseline_pending=pending,contract_digest=digest(contract),packet_digest=digest(packet),
        candidate_count_min=min(len(t['candidates'])for t in tasks),candidate_count_max=max(len(t['candidates'])for t in tasks),
        added_cross_encoder=True,baseline_reranker_equivalence=False,graph_construction_SOTA_claim=False,model_calls=0)
    state(root,result['status'],**{k:v for k,v in result.items()if k not in {'status','model_calls'}});return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true',required=True)
    parser.parse_args()
    try:result=prepare()
    except Exception as exc:
        ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        state(ROOT,'full_prepare_error_preserved',error_type=type(exc).__name__);raise
    print(json.dumps(result,ensure_ascii=False))
    if result['status']!='full_original353_inputs_prepared_NO_answers':raise SystemExit(2)


if __name__=='__main__':main()
