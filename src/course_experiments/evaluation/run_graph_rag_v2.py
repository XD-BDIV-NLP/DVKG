from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ..contributions import write_immutable
from ..manifest import digest, file_record, read_document
from .graph_rag_v2 import BUDGET, EvidenceRetriever, evaluate, pack_context
from .protocol import ROOT, bind_runtime, freeze_phase, runtime_for, validate_phase

LIMITS = dict(max_http_attempts=30000,max_seconds=172800,job_workers=4)


def checked(record):
    if file_record(record['path'])!=record:raise ValueError('Frozen source changed')
    return read_document(record['path'])


def atomic_state(path,value):
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8');temporary.replace(path)


def prepare(base,sources):
    source=read_document(sources);jobs=[];cohorts={}
    expected={(cohort,kind) for cohort in ['compAB','compABCD','osAB','netAB'] for kind in ['F','G0','T','vector','keyword']}
    if len(source)!=20 or {(j['cohort'],j['kind']) for j in source}!=expected or len({j['name'] for j in source})!=20:raise ValueError('Exact unique twenty-unit matrix required')
    if any(j['name']!=j['cohort']+'_'+j['kind'] or j['control']!=('vector' if j['kind']=='vector' else 'keyword' if j['kind']=='keyword' else 'graph') for j in source):raise ValueError('Matrix control/name differs')
    contract=dict(version='evidence_graph_rag_v2_20261019',budget=BUDGET,limits=LIMITS,
                  source_manifest=file_record(sources),raw_text_exported=False)
    write_immutable(base/'contract.json',contract)
    for job in source:
        parent=checked(job['parent_phase']);old=checked(job['parent_result'])
        if old['status']!='completed':raise ValueError('Parent not completed')
        if parent['phase_id']!=digest({k:v for k,v in parent.items() if k!='phase_id'}) or old['input_digest']!=parent['phase_id']:raise ValueError('Parent phase identity differs')
        for key in ['graph','index','reference']:checked(parent['files'][key])
        construction=checked(parent['files']['construction'])
        config_record=parent['files'].get('runtime_config') or construction['files']['assets']['runtime_config']
        checked(config_record)
        paths={key:parent['files'][key]['path'] for key in ['graph','index','reference','construction']}
        paths.update(runtime_config=config_record['path'],parent_phase=job['parent_phase']['path'],
                     parent_result=job['parent_result']['path'],contract=base/'contract.json',source_manifest=sources)
        for folder in ['course_experiments','domain_ckg_workspace','src']:
            for p in (ROOT/folder).rglob('*.py'):
                if '__pycache__' not in p.parts and 'tests' not in p.parts:paths['dependency:'+str(p.relative_to(ROOT))]=p
        for p in (ROOT/'prompts/incremental_v5').glob('*.txt'):paths['runtime_prompt:'+p.name]=p
        phase_path=base/'phases'/(job['name']+'.json')
        frozen=freeze_phase(paths,dict(BUDGET,control=job['control'],active_books=sorted(old['active_books'])),phase_path)
        validate_phase(phase_path)
        reference=read_document(paths['reference']);index=read_document(paths['index']);graph=read_document(paths['graph'])
        if old['graph_id']!=digest(graph) or old['split_id']!=reference['reference_id']:raise ValueError('Parent inputs differ')
        rows=[r for r in index['rows'] if r['book_id'] in old['active_books']]
        available={eid for r in rows for eid in r['evidence_ids']}
        if not all(set(q['evidence_ids'])<=available for q in reference['questions']):raise ValueError('Unavailable reference sources')
        shared=(reference['reference_id'],index['index_id'],sorted(old['active_books']))
        if job['cohort'] in cohorts and cohorts[job['cohort']]!=shared:raise ValueError('Cohort inputs differ')
        cohorts[job['cohort']]=shared
        runtime=runtime_for(construction,base/'preflight_runtime'/job['name'])
        try:bind_runtime(frozen,runtime)
        finally:runtime.db.close()
        diagnostic=dict(model_calls=0,question_count=len(reference['questions']))
        if job['control']!='vector':
            q=reference['questions'][0];retriever=EvidenceRetriever(rows,graph)
            selected,trace=retriever.retrieve(q['question'],job['control'])
            contexts=pack_context(selected,q['question'])
            diagnostic.update(seed_count=len(trace['seed_ids']),path_count=len(trace['paths']),
                              context_rows=len(contexts),context_bytes=len(json.dumps(contexts,ensure_ascii=False).encode('utf-8')))
            write_immutable(base/'diagnostics'/(job['name']+'.json'),dict(trace=trace,context=contexts,model_calls=0))
        jobs.append(dict(job,phase=file_record(phase_path),question_count=len(reference['questions']),diagnostic=diagnostic))
    priority={'F':0,'vector':1,'keyword':2,'G0':3,'T':4}
    jobs.sort(key=lambda j:(priority[j['kind']],['compAB','osAB','netAB','compABCD'].index(j['cohort'])))
    result=dict(status='prepared_zero_model_calls',jobs=jobs,total_questions=sum(j['question_count'] for j in jobs),
                total_jobs=len(jobs),contract=file_record(base/'contract.json'),limits=LIMITS)
    result['plan_id']=digest(result);write_immutable(base/'prepared.json',result)
    db=sqlite3.connect(base/'limits.sqlite');db.executescript('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,created REAL,status TEXT,stage TEXT,code INTEGER);');db.close()
    return dict(status=result['status'],total_jobs=len(jobs),total_questions=result['total_questions'],plan_id=result['plan_id'],model_calls=0)


def frozen_plan(base):
    plan=read_document(base/'prepared.json')
    if plan['plan_id']!=digest({k:v for k,v in plan.items() if k!='plan_id'}):raise ValueError('Plan changed')
    checked(plan['contract'])
    return plan


def limit_runtime(base,runtime):
    original=runtime._record
    def record(key,stage,status,value):
        code=value.get('status_code') if isinstance(value,dict) else None
        db=sqlite3.connect(base/'limits.sqlite',timeout=30)
        try:
            db.execute('BEGIN IMMEDIATE')
            if status=='request_started':
                if (base/'halt.json').exists():raise RuntimeError('Evaluation halted; preserve checkpoints')
                rows=db.execute("SELECT COUNT(*),MIN(created) FROM events WHERE status='request_started'").fetchone()
                if rows[0]>=LIMITS['max_http_attempts'] or (rows[1] and time.time()-rows[1]>LIMITS['max_seconds']):raise RuntimeError('Evaluation budget exhausted; preserve checkpoints')
            db.execute('INSERT INTO events(created,status,stage,code) VALUES(?,?,?,?)',(time.time(),status,stage,code));db.commit()
        finally:db.close()
        if code in {401,402,403}:write_immutable(base/'halt.json',dict(status='account_or_permission_block',status_code=code))
        original(key,stage,status,value)
    runtime._record=record
    return runtime


def worker(base,name,smoke=False):
    plan=frozen_plan(base);job=next(j for j in plan['jobs'] if j['name']==name)
    phase=validate_phase(job['phase']['path']);checked(job['phase'])
    if not os.environ.get('SILICONFLOW_API_KEY'):raise RuntimeError('Set SILICONFLOW_API_KEY before an API run')
    files=phase['files'];output=base/'jobs'/name
    runtime=limit_runtime(base,runtime_for(read_document(files['construction']['path']),output/'runtime'))
    state=base/'job_states'/(name+'.json');state.parent.mkdir(parents=True,exist_ok=True)
    atomic_state(state,dict(status='running_smoke' if smoke else 'running_full',pid=os.getpid(),started_unix=time.time()))
    try:
        result=evaluate(read_document(files['reference']['path']),read_document(files['index']['path']),read_document(files['graph']['path']),
                        phase['parameters']['active_books'],job['control'],job['phase']['path'],output,runtime,stop_after=1 if smoke else None)
        atomic_state(state,dict(status=result['status'],completed_count=result['completed_count'],question_count=result['question_count'],
                               result_id=result['result_id'],updated_unix=time.time(),pid=os.getpid()))
        if smoke and result['status']!='smoke_completed':raise RuntimeError('Real model smoke did not complete')
        if not smoke and result['status']!='completed':raise RuntimeError('Partial unit preserved; failed questions require continuation')
        return result
    except Exception as exc:
        (base/'job_states'/(name+'.failure.log')).write_text(traceback.format_exc(),'utf-8')
        atomic_state(state,dict(status='failed_preserved_checkpoints',error_type=type(exc).__name__,updated_unix=time.time(),pid=os.getpid()))
        raise
    finally:runtime.db.close()


def controller(base):
    import fcntl
    lock=(base/'execution.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=frozen_plan(base);completed=[]
    def run(job,smoke=False):
        command=[sys.executable,'-u','-m','course_experiments.evaluation.run_graph_rag_v2','worker','--base',str(base),'--job',job['name']]
        if smoke:command+=['--smoke']
        with (base/'logs'/(job['name']+('_smoke' if smoke else '')+'.log')).open('a') as log:
            return job['name'],subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
    (base/'logs').mkdir(exist_ok=True)
    try:
        atomic_state(base/'state.json',dict(status='real_model_smoke',updated_unix=time.time(),completed_jobs=[]))
        _,code=run(plan['jobs'][0],True)
        if code:raise RuntimeError('Smoke failed; queue preserved without full launch')
        atomic_state(base/'state.json',dict(status='running_full_matrix',updated_unix=time.time(),completed_jobs=[]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=LIMITS['job_workers']) as pool:
            remaining=iter(plan['jobs']);active={pool.submit(run,j):j for j in [next(remaining) for _ in range(min(LIMITS['job_workers'],len(plan['jobs'])))]}
            while active:
                done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    job=active.pop(future);name,code=future.result();completed.append(dict(name=name,exit_code=code))
                    atomic_state(base/'state.json',dict(status='running_full_matrix',updated_unix=time.time(),completed_jobs=completed))
                    following=next(remaining,None)
                    if following is not None and not (base/'halt.json').exists():active[pool.submit(run,following)]=following
        atomic_state(base/'state.json',dict(status='account_blocked' if (base/'halt.json').exists() else 'matrix_workers_finished',
                                          updated_unix=time.time(),completed_jobs=completed,all_workers_exit_zero=len(completed)==len(plan['jobs']) and all(j['exit_code']==0 for j in completed)))
    except Exception as exc:
        (base/'controller.failure.log').write_text(traceback.format_exc(),'utf-8')
        atomic_state(base/'state.json',dict(status='failed_preserved_checkpoints',error_type=type(exc).__name__,updated_unix=time.time(),completed_jobs=completed));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('step',choices=['prepare','worker','controller']);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--sources',type=Path);p.add_argument('--job');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.step=='prepare':print(json.dumps(prepare(a.base,a.sources)))
    elif a.step=='worker':worker(a.base,a.job,a.smoke)
    else:controller(a.base)

if __name__=='__main__':main()
