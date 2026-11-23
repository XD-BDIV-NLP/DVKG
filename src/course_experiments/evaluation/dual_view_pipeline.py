from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable
from . import dual_view_tasks as task
from . import dual_view_run as run
from .dual_view_rag import asset,POLICY,METHODS
from .protocol import runtime_for,check_runtime
from .repaired_full import limited_runtime
from .chat import call_json,account_status
from .run_repaired_full import atomic,credentials,code_records

PARENT=(Path(__file__).resolve().parents[2] / 'outputs/course_followup_20261019_v3')


def store_vectors(path,matrix):
    path=Path(path);meta=path.with_suffix('.npy.meta.json')
    if meta.exists():
        record=read_document(meta)['vectors']
        if file_record(path)!=record:raise ValueError('Frozen vectors changed')
        return record
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.rename(path.with_suffix('.uncommitted_'+str(time.time_ns())))
    with path.open('xb') as stream:np.save(stream,np.asarray(matrix,dtype=np.float32))
    record=file_record(path);write_immutable(meta,dict(vectors=record));return record


def embeddings(runtime,texts,path):
    path=Path(path);record_path=path.with_suffix('.input.json')
    identity=dict(texts_digest=digest(texts),count=len(texts),configuration=runtime.config['embedding'])
    write_immutable(record_path,identity);parts=[]
    for first in range(0,len(texts),32):
        batch=texts[first:first+32];part=path.parent/(path.stem+'_batches')/(digest(batch)+'.npy')
        if part.with_suffix('.npy.meta.json').exists():
            vec=read_document(part.with_suffix('.npy.meta.json'))['vectors']
            if file_record(part)!=vec:raise ValueError('Embedding batch changed')
            values=np.load(part)
        else:
            values=np.asarray(runtime.embed(batch),dtype=np.float32);norms=np.linalg.norm(values,axis=1,keepdims=True)
            if values.shape!=(len(batch),1024) or not np.isfinite(values).all() or (norms<=0).any():raise ValueError('Invalid task embeddings')
            values=values/norms;store_vectors(part,values)
        parts.append(values)
    result=np.concatenate(parts) if parts else np.empty((0,1024),dtype=np.float32)
    return store_vectors(path,result)


def observed_sources(assets):
    structures={book:asset(rec) for book,rec in assets['structures'].items()};owners={};chunks={}
    for book,s in structures.items():
        for eid,row in s['evidence'].items():
            owners.setdefault(eid,set()).add(book);chunks.setdefault(row['chunk_id'],set()).add((book,eid))
    excluded={book:set() for book in structures};questions=set();ids=set();chunk_ids=set()
    def visit(value,key=''):
        if isinstance(value,dict):
            if isinstance(value.get('question'),str):questions.add(value['question'])
            for k,v in value.items():visit(v,k)
        elif isinstance(value,list):
            for v in value:visit(v,key)
        elif isinstance(value,str):
            if key in {'evidence_id','evidence_ids','source_node_ids'}:ids.add(value.rsplit('/evidence/',1)[-1])
            if key in {'chunk_id','source_chunk_id','source_chunk_ids'}:chunk_ids.add(value)
    records={rec['path']:rec for rec in [v['reference'] for v in assets['cohorts'].values()]}
    for p in (PARENT/'followup/supplement').glob('*/reference.json'):records[str(p)]=file_record(p)
    for p in (PARENT/'followup/corrections').glob('*/review/*.json'):records[str(p)]=file_record(p)
    for rec in records.values():visit(asset(rec))
    for eid in ids:
        for book in owners.get(eid,set()):excluded[book].add(eid)
    for chunk in chunk_ids:
        for book,eid in chunks.get(chunk,set()):excluded[book].add(eid)
    return dict(excluded={b:sorted(v) for b,v in excluded.items()},questions=sorted(questions),source_records=list(records.values()),
        policy='exclude_observed_question_quotes_source_chunks_and_diagnostic_quote_units; chapter_disjoint_new_development_confirmation')


def candidate_bundles(structure,units,indices,kind):
    rows=units['rows'];positions={r['evidence_ids'][0]:i for i,r in enumerate(rows)}
    if file_record(units['vectors']['path'])!=units['vectors']:raise ValueError('Candidate selection source vectors changed')
    matrix=np.load(units['vectors']['path'],mmap_mode='r');evidence=structure['evidence'];eligible=set(indices)
    order=list(indices)
    if kind=='conditions':
        order=[e for e in order if re.search(r'只有|仅当|当.{0,30}时|条件|前提|如果|必须|除非',evidence[e]['text'])]
    used=set()
    for eid in order:
        first=evidence[eid];bundle=[eid]
        adjacent=[e for e in indices if e!=eid and evidence[e]['chunk_id']==first['chunk_id'] and evidence[e]['section_id']==first['section_id']]
        adjacent.sort(key=lambda e:abs(evidence[e]['source'].get('paragraph_order',0)-first['source'].get('paragraph_order',0)))
        bundle+=adjacent[:1]
        if kind=='cross_chapter':
            possible=[e for e in indices if evidence[e]['chapter_id']!=first['chapter_id']]
            if not possible:continue
            positions2=[positions[e] for e in possible];scores=matrix[positions2]@matrix[positions[eid]]
            second=possible[min(range(len(scores)),key=lambda i:(-float(scores[i]),possible[i]))]
            bundle.append(second)
        bundle=sorted(set(bundle))
        key=digest(bundle)
        if key in used:continue
        used.add(key)
        if not set(bundle)<=eligible:raise AssertionError('Candidate crosses source split')
        yield bundle


def audited_call(runtime,prompt,payload,root):
    errors=[]
    for attempt in range(2):
        try:return call_json(runtime,prompt,payload,1536,Path(root),attempt)[0]
        except (ValueError,KeyError,TypeError) as exc:
            errors.append(dict(attempt=attempt,error_type=type(exc).__name__,message=str(exc)))
            write_immutable(Path(root)/('format_error_'+str(attempt)+'.json'),errors[-1])
    raise ValueError('Task model format failed after bounded retries')


def prepare_book(base,assets,book,observed,old_vectors):
    base=Path(base);root=base/'task_generation'/book;root.mkdir(parents=True,exist_ok=True)
    structure=asset(assets['structures'][book]);units=asset(assets['unit_assets'][book]);splits=task.split_sources(structure,set(observed['excluded'][book]))
    write_immutable(root/'source_split.json',dict(split=splits,excluded_count=len(observed['excluded'][book]),policy=task.TASK_POLICY))
    cohort=next(c for c,row in assets['cohorts'].items() if book in row['active_books'])
    runtime=limited_runtime(runtime_for(asset(assets['cohorts'][cohort]['construction']),base/'runtime'/'tasks'/book),base)
    known=list(observed['questions']);vectors=[v for v in old_vectors];result={}
    try:
        for split in ['development','confirmation']:
            output=base/'datasets'/book/(split+'.json')
            if output.exists():
                frozen=read_document(output);result[split]=file_record(output)
                for q in frozen['questions']:
                    known.append(q['question']);vp=base/'task_vectors'/book/(q['question_id']+'.npy')
                    vec=read_document(vp.with_suffix('.npy.meta.json'))['vectors']
                    if file_record(vp)!=vec:raise ValueError('Accepted task vector changed')
                    vectors.append(np.load(vp)[0])
                continue
            target=task.TASK_POLICY[split+'_per_type'];questions=[];counts={}
            for kind in task.TYPES:
                good=[];attempts=0
                for bundle in candidate_bundles(structure,units,splits[split],kind):
                    if len(good)>=target or attempts>=task.TASK_POLICY['max_candidates_per_type']:break
                    attempts+=1;folder=root/split/kind/(str(attempts).zfill(3)+'_'+digest(bundle)[:16]);checkpoint=folder/'decision.json'
                    if checkpoint.exists():
                        decision=read_document(checkpoint)
                        if decision['accepted']:
                            q=decision['question'];good.append(q);known.append(q['question']);vp=base/'task_vectors'/book/(q['question_id']+'.npy')
                            vec=read_document(vp.with_suffix('.npy.meta.json'))['vectors']
                            if file_record(vp)!=vec:raise ValueError('Accepted task vector changed')
                            vectors.append(np.load(vp)[0])
                        continue
                    evidence={e:structure['evidence'][e] for e in bundle}
                    sources=[dict(evidence_id=e,chapter_id=r['chapter_id'],text=r['text']) for e,r in evidence.items()]
                    decision=None
                    try:
                        generated=audited_call(runtime,task.GENERATE,dict(target_book_id=book,type=kind,sources=sources),folder/'generate')
                        if generated.get('skip') is True:raise ValueError('Generator declined unsupported candidate')
                        q=task.validate_task(generated,evidence,book,kind)
                        if task.near_duplicate(q['question'],known):raise ValueError('Character duplicate of observed or accepted task')
                        vp=base/'task_vectors'/book/(q['question_id']+'.npy');vec=embeddings(runtime,[q['question']],vp);vector=np.load(vec['path'])[0]
                        if vectors and float(np.max(np.asarray(vectors)@vector))>=task.TASK_POLICY['duplicate_embedding_cosine']:raise ValueError('Semantic near duplicate of observed or accepted task')
                        audit=audited_call(runtime,task.SUPPORT,dict(target_book_id=book,type=kind,task=q,sources=sources),folder/'support')
                        if audit.get('supported') is not True or audit.get('coherent') is not True or kind=='conditions' and audit.get('explicit_conditions') is not True:raise ValueError('Independent source support audit rejected')
                        if kind=='cross_chapter':
                            for chapter in q['chapter_ids']:
                                partial=[s for s in sources if s['chapter_id']!=chapter]
                                necessary=audited_call(runtime,task.NECESSITY,dict(question=q['question'],sources=partial),folder/'leave_out'/digest(chapter))
                                if necessary.get('answerable') is not False:raise ValueError('Cross-chapter evidence not independently necessary')
                        decision=dict(accepted=True,question=q,source_bundle=bundle,source_audit=audit)
                        good.append(q);known.append(q['question']);vectors.append(vector)
                    except (ValueError,KeyError,TypeError) as exc:
                        decision=dict(accepted=False,source_bundle=bundle,error_type=type(exc).__name__,reason=str(exc))
                    write_immutable(checkpoint,decision)
                    atomic(root/'state.json',dict(status='generating',split=split,type=kind,attempts=attempts,accepted_current_type=len(good),target_current_type=target,updated_unix=time.time(),pid=os.getpid()))
                counts[kind]=dict(target=target,accepted=len(good),candidates=attempts);questions+=good
            frozen=dict(target_book_id=book,split=split,questions=questions,by_type=counts,policy=task.TASK_POLICY,
                source_split=file_record(root/'source_split.json'),observed_sources=file_record(base/'observed_sources.json'),human_audited=False)
            frozen['dataset_id']=digest(frozen);write_immutable(output,frozen);result[split]=file_record(output)
        atomic(root/'state.json',dict(status='completed',references=result,pid=os.getpid(),completed_unix=time.time()))
        return result
    finally:runtime.close()


def query_vectors(base,book,reference):
    result={}
    for q in asset(reference)['questions']:
        p=Path(base)/'task_vectors'/book/(q['question_id']+'.npy');record=read_document(p.with_suffix('.npy.meta.json'))['vectors']
        if file_record(p)!=record:raise ValueError('Query vector changed')
        result[q['question_id']]=np.load(p)[0]
    return result


def evaluate_worker(base,job,assets,reference,code,development=False):
    runtime=limited_runtime(runtime_for(asset(assets['cohorts'][job['cohort']]['construction']),Path(base)/'runtime'/('development' if development else 'evaluations')/job['name']),base)
    try:return run.evaluate(job,assets,reference,query_vectors(base,job['target_book_id'],reference),runtime,base,code,development=development)
    finally:runtime.close()


def switch_checks(base,assets,datasets):
    checks=[]
    for cohort,row in assets['cohorts'].items():
        retriever=run.retriever_for(assets,cohort);before=digest(retriever.graph);sequence=row['active_books']+[row['active_books'][0]];first=None
        if any(book not in datasets for book in sequence):continue
        for index,book in enumerate(sequence):
            reference=datasets[book]['development'];qs=asset(reference)['questions']
            if not qs:raise ValueError('No independent development question for switch check')
            q=qs[0];vectors=query_vectors(base,book,reference);selected,trace=retriever.retrieve(q['question'],book,vectors[q['question_id']])
            context=run.pack_context(selected,q['question']);run.validate_context(context,book);fingerprint=digest(context)
            if first is None:first=fingerprint
            checks.append(dict(cohort=cohort,target_book_id=book,concept_graph_unchanged=digest(retriever.graph)==before,
                evidence_book_scope_valid=all(r['book_id']==book for r in selected),structure_load_seconds=trace['structure_load_seconds'],
                repeat_first_book_consistent=fingerprint==first if index==len(sequence)-1 else None))
        if digest(retriever.graph)!=before or fingerprint!=first:raise ValueError('Textbook switching mutated C or changed repeated target result')
    write_immutable(Path(base)/'switch_checks.json',dict(checks=checks,policy='same_C_version_only_S_target_changes; retrieval_only_invariance_not_QA_accuracy_claim'))


def main(base):
    import fcntl
    base=Path(base);lock=(base/'pipeline.lock').open('ab');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    credentials();assets=read_document(base/'assets/result.json');code=dict(files=code_records());code['identity']=digest(code)
    write_immutable(base/'implementation.json',code)
    construction=asset(assets['cohorts']['compAB']['construction']);config=asset(construction['files']['assets']['runtime_config']);check_runtime(config)
    write_immutable(base/'pipeline_contract.json',dict(assets=file_record(base/'assets/result.json'),implementation=file_record(base/'implementation.json'),
        task_policy=task.TASK_POLICY,retrieval=POLICY,methods=list(METHODS),answer_prompt=run.ANSWER,judge_prompt=run.JUDGE,runtime_configuration=config,limits=file_record(base/'limits.json')))
    observed=observed_sources(assets);write_immutable(base/'observed_sources.json',observed)
    runtime=limited_runtime(runtime_for(construction,base/'runtime/observed_task_embeddings'),base)
    try:record=embeddings(runtime,observed['questions'],base/'observed_question_vectors.npy');old_vectors=np.load(record['path'])
    finally:runtime.close()
    datasets={};failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(prepare_book,base,assets,book,observed,old_vectors):book for book in sorted(assets['structures'])}
        for future in concurrent.futures.as_completed(pending):
            book=pending[future]
            try:datasets[book]=future.result()
            except Exception as exc:
                atomic(base/'failures'/('generation_'+book+'.json'),dict(error_type=type(exc).__name__,traceback=traceback.format_exc(),failed_unix=time.time()))
                if account_status(exc) or (base/'halt.json').exists():raise
                failures.append(dict(book=book,error_type=type(exc).__name__))
            atomic(base/'pipeline_state.json',dict(status='generating_target_book_tasks',completed_books=len(datasets),failed_books=failures,total_books=8,pid=os.getpid(),updated_unix=time.time()))
    write_immutable(base/'datasets/index.json',dict(books=datasets,failures=failures))
    switch_checks(base,assets,datasets)
    jobs=run.matched_jobs(assets);dev_jobs=[j for j in jobs if j['cohort']=='compAB' and j['target_book_id']==assets['cohorts']['compAB']['active_books'][0]]
    for job in dev_jobs:
        result=evaluate_worker(base,job,assets,datasets[job['target_book_id']]['development'],file_record(base/'implementation.json'),True)
        if result['status']!='completed':raise ValueError('Development execution gate incomplete')
    write_immutable(base/'development_gate.json',dict(status='passed_execution_scope_and_receipts',job_count=len(dev_jobs),policy=POLICY,
        note='No confirmation evaluation scores observed for parameter selection; default retrieval unchanged'))
    write_immutable(base/'evaluation_matrix.json',dict(jobs=jobs,datasets=file_record(base/'datasets/index.json'),contract=file_record(base/'pipeline_contract.json')))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(evaluate_worker,base,job,assets,datasets[job['target_book_id']]['confirmation'],file_record(base/'implementation.json')):job for job in jobs if job['target_book_id'] in datasets}
        for future in concurrent.futures.as_completed(pending):
            job=pending[future]
            try:results.append(future.result())
            except Exception as exc:
                atomic(base/'failures'/(job['name']+'.json'),dict(error_type=type(exc).__name__,traceback=traceback.format_exc(),failed_unix=time.time()))
                if account_status(exc) or (base/'halt.json').exists():raise
                failures.append(dict(job=job,error_type=type(exc).__name__))
            atomic(base/'pipeline_state.json',dict(status='evaluating_fixed_confirmation_sets',completed_jobs=sum(r['status']=='completed' for r in results),finished_jobs=len(results),expected_jobs=len(jobs),failures=failures,pid=os.getpid(),updated_unix=time.time()))
    report=dict(status='completed' if len(results)==len(jobs) and all(r['status']=='completed' for r in results) and not failures else 'partial',
        results=results,failures=failures,expected_jobs=len(jobs),dataset_index=file_record(base/'datasets/index.json'),contract=file_record(base/'pipeline_contract.json'),
        limitations='Machine-generated and machine-audited tasks and judging; no human labels. Existing unchanged C geometric/KGE results are separate, not newly trained. Native SOTA construction suite not complete.',completed_unix=time.time())
    write_immutable(base/('final_report.json' if report['status']=='completed' else 'partial_report_'+digest(report)+'.json'),report)
    atomic(base/'pipeline_state.json',dict(status=report['status'],expected_jobs=len(jobs),completed_jobs=sum(r['status']=='completed' for r in results),failures=failures,pid=os.getpid(),completed_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);args=parser.parse_args()
    try:main(args.base)
    except Exception as exc:
        atomic(args.base/'pipeline_failure.json',dict(error_type=type(exc).__name__,traceback=traceback.format_exc(),failed_unix=time.time()))
        atomic(args.base/'pipeline_state.json',dict(status='failed_preserved_checkpoints',error_type=type(exc).__name__,pid=os.getpid(),updated_unix=time.time()));raise
