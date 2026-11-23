"""零API封存原353题双池精排上下文；额外交叉编码器仅作探索比较。"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


EXPECTED_COUNTS={'book_25f718562685':100,'book_49bb8d31ddb9':100,'book_0b8d87cb0667':40,
    'book_d96f472c8179':37,'book_00849acee112':38,'book_dd8183b4cb60':38}
COURSES={b:course for books,course in [(['book_25f718562685','book_49bb8d31ddb9'],'computer_organization'),
    (['book_0b8d87cb0667','book_d96f472c8179'],'operating_systems'),
    (['book_00849acee112','book_dd8183b4cb60'],'computer_networks')]for b in books}
SCOPES=('same_subject_pool','mixed_six_book_pool')
BASELINES=('K12-KGraph','GraphRAG','LightRAG','HippoRAG')
SOURCE='original353_frozen_expanded_pool_questions'
CANONICAL_SHA='43cb0cb57a2288f1b39ebde8d77575c76b4b847764d279392f0e4b02922e1fcb'
FROZEN_MODULES={
    'development_worker.py':'cd7d2d9d05a498d0605ce6913d74973d4f8f17b576a65d8f7cbedf0027f88028',
    'modules/query_evidence_rerank.py':'921d73c0d119ae27aa33e5e157802352ce517220afa9cfe77b9de8d18cab83ac',
    'modules/evidence_window_packing.py':'447316cbb69db14fbeb557a457e2c01e5c7c5a9a055db0150ad6c29c9c926756',
    'modules/evidence_stage_metrics.py':'fa9b6cb1a3fa9676c97f2ec9944e3bed2c1f49175d4d8d9741947716dd2ed571',
    'modules/evidence_development_run.py':'bdbbe7eaabd28bb09ef757d8c272d1168213ed6829937c66d31098cc168a571e',
    'modules/evidence_selection_pool.py':'6c4fe53a368b5823e3de354b7a8d59183eaec8e5ac406fc2f6d8043150f66d80',
    'modules/run_joint_pool_retrieval.py':'50dc877bb749edd30eb6a16eec1080441e45522c30d93807ec4675e4a2a6969f'}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))


def record(path):
    path=Path(path).resolve(strict=True);hasher=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):hasher.update(chunk)
    return dict(path=str(path),sha256=hasher.hexdigest())


def checked(reference):
    if record(reference['path'])!=reference:raise ValueError('Frozen full-rerank dependency changed')
    return read(reference['path'])


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Full-rerank frozen output already exists with different content')
    else:atomic(path,value)


def state(root,status,**fields):
    atomic(Path(root)/'state.json',dict(status=status,pid=os.getpid(),observed_unix=time.time(),model_calls=0,**fields))


def initialize_dependencies(contract):
    if os.name!='posix':raise RuntimeError('Real source contexts remain on their original server')
    refs=contract['module_refs']
    if set(refs)!=set(FROZEN_MODULES):raise ValueError('Full-rerank module freeze is incomplete')
    for name,expected in FROZEN_MODULES.items():
        if refs[name]['sha256']!=expected or record(refs[name]['path'])!=refs[name]:
            raise ValueError('Completed development module differs: '+name)
    spec=importlib.util.spec_from_file_location('full_rerank_frozen_development',refs['development_worker.py']['path'])
    development=importlib.util.module_from_spec(spec);sys.modules[spec.name]=development;spec.loader.exec_module(development)
    cfg,original,m,_joint=development.initialize()
    if (contract['original_configuration']!=dict(path=str(development.CONFIG),sha256=development.CONFIG_SHA)
            or contract['original_contract']!=cfg['original_contract']):
        raise ValueError('Full preparation does not use the exact original configuration')
    if record(contract['worker']['path'])!=contract['worker']:raise ValueError('Original full-preparation worker changed')
    if contract['canonical_comparison_anchor']['sha256']!=CANONICAL_SHA:
        raise ValueError('Audited original comparison anchor differs')
    from course_experiments.evaluation.evidence_development_run import build_score_packet,verify_score_receipt
    from course_experiments.evaluation.query_evidence_rerank import rerank_evidence,paragraph_key
    from course_experiments.evaluation.graph_rag_v2 import pack_context
    from course_experiments.evaluation.evidence_stage_metrics import score_stages
    inventory=m.asset(original['inventory']);assets=m.asset(inventory['assets'])
    structures={};units={};datasets={};sources={}
    for source in cfg['sources']:
        book=source['book_id']
        if contract['datasets'].get(book)!=source['dataset']:raise ValueError('Full source dataset reference changed')
        structures[book]=m.asset(assets['structures'][book]);unit=m.asset(assets['unit_assets'][book])
        units[book]={r['evidence_ids'][0]:r for r in unit['rows']}
        if len(units[book])!=len(unit['rows']):raise ValueError('Original unit identities are duplicated')
        datasets[book]=m._dataset_questions(source['dataset'],book)
        sources[book]=dict(structure=assets['structures'][book],units=assets['unit_assets'][book],dataset=source['dataset'])
    return dict(cfg=cfg,datasets=datasets,structures=structures,unit_rows=units,source_refs=sources,
        canonical=checked(contract['canonical_comparison_anchor']),canonical_ref=contract['canonical_comparison_anchor'],
        build_score_packet=build_score_packet,verify_score_receipt=verify_score_receipt,
        rerank_evidence=rerank_evidence,paragraph_key=paragraph_key,pack_context=pack_context,score_stages=score_stages)


def validate_candidate(row,allowed,deps):
    key=deps['paragraph_key'](row);book=key[0];eid=row.get('source_evidence_id')
    if book not in allowed or not isinstance(eid,str):raise ValueError('Candidate belongs to a foreign source pool')
    source=deps['structures'][book]['evidence'].get(eid);unit=deps['unit_rows'][book].get(eid)
    if (source is None or unit is None or row['text']!=source['text'] or row['text']!=unit['text']
            or row['chunk_id']!=unit['chunk_id'] or unit['book_id']!=book or source['book_id']!=book
            or key[1]!=source.get('source',{}).get('source_id')
            or row['evidence_ids']!=[json.dumps([book,eid],ensure_ascii=False,separators=(',',':'))]):
        raise ValueError('Candidate differs from frozen original source/chunk/paragraph namespace')
    return key


def validate_inputs(contract,prepared,packet,deps):
    policy=contract.get('comparison_policy',{})
    if (contract.get('version')!='original353_expanded_DVKG_rerank_exploratory_v1'
            or contract.get('question_count')!=353 or contract.get('task_count')!=706
            or contract.get('expected_question_counts')!=EXPECTED_COUNTS or contract.get('scopes')!=list(SCOPES)
            or contract.get('source_graphs_modified')is not False or prepared.get('contract_digest')!=digest(contract)
            or policy.get('variants')!=['rerank']or policy.get('added_cross_encoder')is not True
            or policy.get('baseline_reranker_equivalence')is not False
            or policy.get('graph_construction_SOTA_claim')is not False
            or policy.get('context_bytes')!=3000 or policy.get('retrieval_top_k')!=5
            or contract.get('canonical_comparison_anchor')!=deps['canonical_ref']):
        raise ValueError('Original353 full-rerank frozen input contract differs')
    replay=contract.get('original_baseline_replay',{})
    if replay.get('matched')!=706 or replay.get('pending')!=0:raise ValueError('All original baseline replays must be verified')
    sources=deps['cfg']['sources'];book_courses={s['book_id']:s['course_id']for s in sources}
    if len(sources)!=6 or book_courses!=COURSES:raise ValueError('Fixed six textbooks and three courses are required')
    questions={}
    for book,count in EXPECTED_COUNTS.items():
        values=deps['datasets'].get(book,[])
        if len(values)!=count:raise ValueError('Original per-book question count changed')
        for q in values:
            if q['question_id']in questions or q['target_book_id']!=book:raise ValueError('Original question identity collision')
            questions[q['question_id']]=q
    tasks=prepared.get('tasks',[]);seen=set();scopes=defaultdict(set)
    if len(tasks)!=706:raise ValueError('Exactly706 original question/pool tasks required')
    for task in tasks:
        qid=task.get('question_id');scope=task.get('scope');key=(qid,scope);q=questions.get(qid)
        if (q is None or scope not in SCOPES or key in seen
                or task.get('task_id')!=digest(dict(question=qid,scope=scope))
                or task.get('book_id')!=q['target_book_id']or task.get('course_id')!=COURSES[q['target_book_id']]
                or task.get('question')!=q['question']or task.get('reference_answer')!=q['answer']
                or task.get('variant')!='rerank'or task.get('source')!=SOURCE or task.get('independent_test')is not False
                or task.get('baseline_replay',{}).get('status')!='matched'):
            raise ValueError('Full task differs from its frozen original question/scoring identity')
        seen.add(key);scopes[qid].add(scope);book=q['target_book_id']
        gold={(book,deps['structures'][book]['evidence'][eid]['source']['source_id'])for eid in q['evidence_ids']}
        if not gold or {tuple(p)for p in task['gold_paragraphs']}!=gold:raise ValueError('Original gold paragraph identity changed')
        allowed=set(EXPECTED_COUNTS)if scope=='mixed_six_book_pool'else {b for b,c in COURSES.items()if c==COURSES[book]}
        candidates=task.get('candidates',[])
        if not 1<=len(candidates)<=64:raise ValueError('One to64 frozen paragraph candidates required')
        keys=[validate_candidate(r,allowed,deps)for r in candidates]
        if len(set(keys))!=len(keys):raise ValueError('Repeated original candidate paragraph')
        routes=task.get('route_scores',[])
        if len({tuple(r['key'])for r in routes})!=len(routes):raise ValueError('Repeated route prior identity')
    if len(scopes)!=353 or any(value!=set(SCOPES)for value in scopes.values()):raise ValueError('Every353 question requires both pools')
    rebuilt=deps['build_score_packet'](tasks)
    if digest(rebuilt)!=digest(packet):raise ValueError('Actual GPU input does not exactly match frozen prepared original candidates')
    if contract['candidate_scoring_model']!={k:packet[k]for k in ('model','revision','max_length')}:
        raise ValueError('Frozen scoring model differs from GPU input')
    return questions


def canonical_rows(deps,questions):
    value=deps['canonical'];rows=value.get('rows',[])
    if value.get('question_count')!=353 or value.get('observations')!=3530 or len(rows)!=3530 or value.get('results_publishable')is not True:
        raise ValueError('Complete audited original five-method results required')
    index={}
    for row in rows:
        key=(row.get('method'),row.get('scope'),row.get('question_id'));q=questions.get(row.get('question_id'))
        if (key in index or key[0]not in (*BASELINES,'DVKG')or key[1]not in SCOPES or q is None
                or row.get('status')!='completed'or row.get('book_id')!=q['target_book_id']
                or row.get('course_id')!=COURSES[q['target_book_id']]
                or row.get('question_text_sha256')!=sha(q['question'])):
            raise ValueError('Canonical comparison question/source scope differs')
        for metric in ('f1','paragraph_recall_at5'):
            score=row.get(metric)
            if type(score)not in (int,float)or not math.isfinite(score)or not 0<=score<=1:raise ValueError('Canonical metric is invalid')
        index[key]=row
    return index


def summarize(records):
    metrics=('candidate_paragraph_recall','paragraph_recall_at5','packed_paragraph_recall','context_bytes')
    summary={}
    for scope in SCOPES:
        subset=[r for r in records if r['scope']==scope]
        books={b:dict(question_count=EXPECTED_COUNTS[b],**{k:statistics.mean(r['metrics'][k]for r in subset if r['book_id']==b)for k in metrics})for b in EXPECTED_COUNTS}
        courses={course:dict(book_count=2,question_count=sum(EXPECTED_COUNTS[b]for b in books if COURSES[b]==course),
            equal_book_macro={k:statistics.mean(v[k]for b,v in books.items()if COURSES[b]==course)for k in metrics})
            for course in sorted(set(COURSES.values()))}
        summary[scope]=dict(tasks=len(subset),by_book=books,
            by_course=courses,
            question_micro={k:statistics.mean(r['metrics'][k]for r in subset)for k in metrics},
            equal_book_macro={k:statistics.mean(v[k]for v in books.values())for k in metrics})
    return summary


def exploration(summary,index,records,reference):
    table=[];deltas={}
    for scope in SCOPES:
        for method in BASELINES:
            by_book={b:[r for(key,s,q),r in index.items()if key==method and s==scope and r['book_id']==b]for b in EXPECTED_COUNTS}
            book_metrics={b:dict(question_count=len(group),**{k:statistics.mean(r[k]for r in group)
                for k in ('f1','paragraph_recall_at5')})for b,group in by_book.items()}
            course_metrics={course:dict(book_count=2,question_count=sum(v['question_count']for b,v in book_metrics.items()if COURSES[b]==course),
                **{k:statistics.mean(v[k]for b,v in book_metrics.items()if COURSES[b]==course)for k in ('f1','paragraph_recall_at5')})
                for course in sorted(set(COURSES.values()))}
            table.append(dict(method=method,scope=scope,question_count=353,by_book=book_metrics,by_course=course_metrics,
                f1=statistics.mean(v['f1']for v in book_metrics.values()),
                paragraph_recall_at5=statistics.mean(v['paragraph_recall_at5']for v in book_metrics.values()),
                score_source='unchanged_audited_canonical',source_reference=reference))
        table.append(dict(method='DVKG+cross-encoder',scope=scope,question_count=353,f1=None,
            by_book=summary[scope]['by_book'],by_course=summary[scope]['by_course'],
            paragraph_recall_at5=summary[scope]['equal_book_macro']['paragraph_recall_at5'],
            score_source='new_verified_GPU_rerank_retrieval_only',answer_status='not_generated'))
        scoped=[r for r in records if r['scope']==scope]
        deltas[scope]={method:statistics.mean(statistics.mean(r['metrics']['paragraph_recall_at5']-
            index[method,scope,r['question_id']]['paragraph_recall_at5']for r in scoped if r['book_id']==b)
            for b in EXPECTED_COUNTS)for method in (*BASELINES,'DVKG')}
    return dict(rows=table,paired_equal_book_recall_differences=deltas,added_cross_encoder=True,
        baseline_shared_reranker=False,shared_ranker_fairness_control_complete=False,SOTA_claim=False,
        comparison='exploratory_extra_compute_system_row_against_unchanged_canonical_methods',
        f1_for_new_method_pending=True,independent_unseen_test=False)


def prepare_contexts(root,dependencies=None):
    root=Path(root);state(root,'full_rerank_validating_frozen_inputs_NO_API')
    contract=read(root/'input_contract.json');prepared=read(root/'private_prepared_tasks.json')
    packet=read(root/'gpu_score_packet.json');receipt=read(root/'GPU_scores.json')
    deps=dependencies or initialize_dependencies(contract)
    if 'output_root'in deps['cfg']and root.resolve()==Path(deps['cfg']['output_root']).resolve():
        raise ValueError('Original retrieval outputs cannot be overwritten')
    questions=validate_inputs(contract,prepared,packet,deps)
    scores=deps['verify_score_receipt'](packet,receipt);canonical=canonical_rows(deps,questions)
    references={name:record(root/filename)for name,filename in [('input_contract','input_contract.json'),
        ('prepared_tasks','private_prepared_tasks.json'),('gpu_score_packet','gpu_score_packet.json'),('GPU_scores','GPU_scores.json')]}
    frozen=dict(version='original353_rerank_context_finish_v1',worker=record(__file__),inputs=references,
        contract_digest=digest(contract),source_refs=deps['source_refs'],comparison_policy=contract['comparison_policy'],
        question_count=353,task_count=706,variant='rerank',context_bytes=3000,top_k=5,
        answer_generation_calls=0,old_outputs_modified=False)
    records=[]
    for task in prepared['tasks']:
        ranked,ranking=deps['rerank_evidence'](task['question'],task['candidates'],
            route_scores={tuple(r['key']):r['score']for r in task['route_scores']},
            pair_scores=scores[task['task_id']],limit=5)
        context=deps['pack_context'](ranked,task['question'])
        metrics=deps['score_stages'](task['gold_paragraphs'],task['candidates'],ranked,context)
        anchor=canonical['DVKG',task['scope'],task['question_id']]
        records.append(dict(**{k:task[k]for k in ('task_id','question_id','question','reference_answer','book_id','course_id','scope')},
            variant='rerank',source=SOURCE,independent_test=False,context=context,metrics=metrics,ranking=ranking,
            packing=dict(policy='frozen_original_graph_rag_v2.pack_context',top_k=5,context_bytes=metrics['context_bytes']),
            selected_paragraphs=[list(deps['paragraph_key'](r))for r in ranked],
            scoring_ids=dict(question_id=task['question_id'],target_book_id=task['book_id'],
                gold_paragraphs=task['gold_paragraphs'],native_evidence_ids=questions[task['question_id']]['evidence_ids'],
                canonical_question_key=anchor['canonical_question_key']),
            source_refs=dict(original_dataset=contract['datasets'][task['book_id']],
                selected_books={b:deps['source_refs'][b]for b in sorted({r['book_id']for r in ranked})},**references)))
    summary=summarize(records);comparison=exploration(summary,canonical,records,deps['canonical_ref'])
    immutable(root/'full_rerank_finish_contract.json',frozen)
    result=dict(version='original353_full_rerank_retrieval_results_v1',records=records,summary=summary,
        contract_digest=digest(contract),comparison_policy=contract['comparison_policy'],exploratory_comparison=comparison,
        finish_contract=record(root/'full_rerank_finish_contract.json'),answer_generation_calls=0,
        independent_test=False,question_count=353,task_count=706,**references)
    immutable(root/'full_rerank_retrieval_results.json',result)
    state(root,'full_rerank_contexts_complete_NO_answers',question_count=353,task_count=706,
        result_reference=record(root/'full_rerank_retrieval_results.json'))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--prepare-contexts',required=True,action='store_true');args=parser.parse_args(argv)
    try:
        result=prepare_contexts(args.root)
        print(json.dumps(dict(status='full_rerank_contexts_complete_NO_answers',tasks=result['task_count'],model_calls=0)))
    except Exception as exc:
        (args.root/'full_rerank_finish_error.log').write_text(traceback.format_exc(),encoding='utf-8')
        state(args.root,'full_rerank_finish_error_preserved',error_type=type(exc).__name__);raise


if __name__=='__main__':main()
