from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable
from .protocol import ROOT,runtime_for,check_runtime
from .repaired_full import evaluate,limited_runtime
from .rag_repair_v3 import source_units,POLICY
from . import graph_rag_v2 as v2

LIMITS=dict(max_http_attempts=30000,max_seconds=259200,job_workers=4,source_embedding_batch=32)


def embedding_pieces(text):
    return [text[i:i+2048] for i in range(0,len(text),2048)]


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8');tmp.replace(path)


def checked(record):
    if file_record(record['path'])!=record:raise ValueError('Frozen source changed: '+record['path'])
    return read_document(record['path'])


def credentials():
    if not os.environ.get('SILICONFLOW_API_KEY'):raise RuntimeError('Set SILICONFLOW_API_KEY before an API run')


def setup(base):
    base.mkdir(parents=True,exist_ok=True);write_immutable(base/'limits.json',LIMITS)
    with sqlite3.connect(base/'budget.sqlite',timeout=60) as db:
        db.executescript('CREATE TABLE IF NOT EXISTS events(created REAL,stage TEXT,status TEXT,code INTEGER); CREATE INDEX IF NOT EXISTS event_status ON events(status,created);')
    with sqlite3.connect(base/'request_cache.sqlite',timeout=60) as db:
        db.execute('CREATE TABLE IF NOT EXISTS responses(key TEXT PRIMARY KEY,record TEXT NOT NULL)')


def cache_matches():
    v2.normalize=functools.lru_cache(maxsize=32768)(v2.normalize)
    old=v2.matches
    @functools.lru_cache(maxsize=32768)
    def pattern(name):
        name=v2.normalize(name)
        if not name:return None
        left=r'(?<![a-z0-9_])' if name[0].isascii() and name[0].isalnum() else ''
        right=r'(?![a-z0-9_])' if name[-1].isascii() and name[-1].isalnum() else ''
        return re.compile(left+re.escape(name)+right)
    def matches(name,text):
        p=pattern(name);return p is not None and p.search(v2.normalize(text)) is not None
    for name in ['RAM','Cache','缓存','I/O','']:
        for text in ['program','CACHE缓存','I/O','']:assert matches(name,text)==old(name,text)
    v2.matches=matches


def code_records():
    files=[]
    for folder in ['course_experiments','domain_ckg_workspace','src']:
        files.extend(p for p in (ROOT/folder).rglob('*.py') if 'tests' not in p.parts and '__pycache__' not in p.parts)
    files.extend((ROOT/'prompts/evaluation').glob('*.txt'));files.extend((ROOT/'prompts/incremental_v5').glob('*.txt'))
    return [file_record(p) for p in sorted(set(files))]


def prepare(base,parent):
    setup(base);old=read_document(parent/'prepared.json')
    if old['plan_id']!=digest({k:v for k,v in old.items() if k!='plan_id'}):raise ValueError('Parent plan identity changed')
    if len(old['jobs'])!=20:raise ValueError('Exact twenty-unit parent required')
    records=[]
    for job in old['jobs']:
        phase=checked(job['phase'])
        for key in ['graph','reference','index','construction']:checked(phase['files'][key])
        construction=read_document(phase['files']['construction']['path'])
        config=checked(phase['files'].get('runtime_config') or construction['files']['assets']['runtime_config'])
        check_runtime(config)
        records.append(dict(name=job['name'],cohort=job['cohort'],kind=job['kind'],control=job['control'],
            active_books=phase['parameters']['active_books'],parent_phase=job['phase'],question_count=job['question_count']))
    sources=dict(parent_plan=file_record(parent/'prepared.json'),parent_base=str(parent),jobs=records,implementation=code_records(),
        policy=POLICY,budget=v2.BUDGET,limits=LIMITS,unit_vector_policy='BGE_M3_full_unit_nonoverlapping_2048_character_segments_L2_mean_L2_no_source_text_removed',
        additional_phases='corrected_graphs_and_source_independent_supplement_separate_immutable_manifests')
    sources['contract_id']=digest(sources);write_immutable(base/'contract.json',sources)
    return sources


def frozen(base):
    contract=read_document(base/'contract.json')
    if contract['contract_id']!=digest({k:v for k,v in contract.items() if k!='contract_id'}):raise ValueError('Contract changed')
    for r in contract['implementation']:
        if file_record(r['path'])!=r:raise ValueError('Actual frozen implementation changed')
    if contract['implementation']!=code_records():raise ValueError('Code inventory changed')
    return contract


def build_unit_index(base,contract):
    import numpy as np
    folder=base/'unit_index';folder.mkdir(exist_ok=True)
    if (folder/'index.json').exists():
        index=read_document(folder/'index.json')
        if file_record(index['vectors']['path'])!=index['vectors']:raise ValueError('Unit vector matrix changed')
        return file_record(folder/'index.json')
    by_key={};source_records=[]
    for cohort in ['compABCD','osAB','netAB']:
        job=next(j for j in contract['jobs'] if j['name']==cohort+'_F');phase=checked(job['parent_phase']);record=phase['files']['index']
        index=checked(record);source_records.append(record)
        for r in source_units(index['rows']):
            key=(r['book_id'],r['evidence_ids'][0]);r={k:v for k,v in r.items() if k!='vector'}
            if key in by_key and by_key[key]['text']!=r['text']:raise ValueError('Evidence ID has inconsistent raw source')
            by_key.setdefault(key,r)
    rows=[by_key[k] for k in sorted(by_key)]
    phase=checked(contract['jobs'][0]['parent_phase']);runtime=limited_runtime(runtime_for(read_document(phase['files']['construction']['path']),folder/'runtime'),base)
    parts=[];texts=[];offsets=[]
    for row in rows:
        start=len(texts);texts.extend(embedding_pieces(row['text']));offsets.append((start,len(texts)))
    try:
        for first in range(0,len(texts),LIMITS['source_embedding_batch']):
            batch=texts[first:first+LIMITS['source_embedding_batch']];key=digest(dict(model='BAAI/bge-m3',texts=batch))
            path=folder/'batches'/(key+'.npy');meta=path.with_suffix('.json');path.parent.mkdir(exist_ok=True)
            if meta.exists():
                record=read_document(meta);assert file_record(path)==record['vectors'];part=np.load(path)
            else:
                opts=runtime.config['embedding'];body=dict(model=opts['model'],input=batch,encoding_format='float')
                raw=runtime._raw_cached(key)
                raw=raw if raw is not None else runtime._post('/embeddings',body,opts,key,'source_unit_index')
                data=sorted(raw['data'],key=lambda r:r['index']);part=np.asarray([r['embedding'] for r in data],dtype=np.float32)
                assert part.shape==(len(batch),1024) and np.isfinite(part).all()
                norm=np.linalg.norm(part,axis=1,keepdims=True);assert (norm>0).all();part=part/norm
                if path.exists():path.rename(path.with_suffix('.uncommitted_'+str(time.time_ns())))
                with path.open('xb') as stream:np.save(stream,part)
                write_immutable(meta,dict(input_digest=key,vectors=file_record(path),rows=len(batch)))
            parts.append(part)
            atomic(base/'state.json',dict(status='building_frozen_unit_index',completed_segments=first+len(batch),total_segments=len(texts),total_units=len(rows),pid=os.getpid(),observed_unix=time.time()))
        segments=np.concatenate(parts,axis=0);matrix=np.asarray([segments[a:b].mean(axis=0) for a,b in offsets],dtype=np.float32)
        norms=np.linalg.norm(matrix,axis=1,keepdims=True);assert (norms>0).all();matrix=matrix/norms
        path=folder/'vectors.npy'
        if path.exists():path.rename(path.with_suffix('.uncommitted_'+str(time.time_ns())))
        with path.open('xb') as stream:np.save(stream,matrix)
        index=dict(rows=rows,vectors=file_record(path),source_indices=source_records,embedding_model='BAAI/bge-m3',normalized=True)
        index['index_id']=digest(index);write_immutable(folder/'index.json',index)
        return file_record(folder/'index.json')
    finally:runtime.close()


def prepare_jobs(base,contract,unit_index):
    jobs=[]
    for item in contract['jobs']:
        parent=checked(item['parent_phase']);files={k:parent['files'][k] for k in ['graph','index','reference','construction']}
        files['unit_index']=unit_index;files['vector_matrix']=read_document(unit_index['path'])['vectors'];files['contract']=file_record(base/'contract.json')
        phase=dict(files=files,parameters=dict(v2.BUDGET,repair_policy=POLICY,control=item['control'],active_books=item['active_books']))
        phase['phase_id']=digest(phase);p=base/'phases'/(item['name']+'.json');write_immutable(p,phase)
        job={k:v for k,v in item.items() if k!='parent_phase'};job['phase']=file_record(p);jobs.append(job)
    matrix=dict(jobs=jobs,contract=file_record(base/'contract.json'),question_instances=sum(j['question_count'] for j in jobs))
    matrix['matrix_id']=digest(matrix);write_immutable(base/'main_matrix.json',matrix);return matrix


def worker(base,matrix_path,name,smoke=False):
    credentials();cache_matches();frozen(base)
    matrix=read_document(matrix_path)
    if matrix['matrix_id']!=digest({k:v for k,v in matrix.items() if k!='matrix_id'}):raise ValueError('Matrix changed')
    job=next(j for j in matrix['jobs'] if j['name']==name);phase=checked(job['phase'])
    runtime=limited_runtime(runtime_for(read_document(phase['files']['construction']['path']),base/'jobs'/name/'runtime'),base)
    try:
        atomic(base/'job_states'/(name+'.json'),dict(status='running',pid=os.getpid(),started_unix=time.time()))
        result=evaluate(job,phase,base/'jobs'/name,runtime,base,stop_after=1 if smoke else None)
        atomic(base/'job_states'/(name+'.json'),dict(status=result['status'],pid=os.getpid(),completed_count=result['checkpoint_count'],question_count=result['question_count'],updated_unix=time.time()))
        if smoke and result['checkpoint_count']!=1:raise RuntimeError('Real smoke did not commit')
        if not smoke and result['status']!='completed':raise RuntimeError('Partial results preserved; bounded continuation required')
    finally:runtime.close()


def run_matrix(base,matrix_path):
    matrix=read_document(matrix_path);(base/'logs').mkdir(exist_ok=True);completed=[]
    def run(job,smoke=False):
        command=[sys.executable,'-u','-m','course_experiments.evaluation.run_repaired_full','worker','--base',str(base),'--matrix',str(matrix_path),'--job',job['name']]
        if smoke:command.append('--smoke')
        with (base/'logs'/(job['name']+('_smoke' if smoke else '')+'.log')).open('a') as log:
            return dict(name=job['name'],exit_code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode)
    smoke=run(matrix['jobs'][0],True)
    if smoke['exit_code']:raise RuntimeError('Real smoke failed; do not continue full matrix')
    with concurrent.futures.ThreadPoolExecutor(max_workers=LIMITS['job_workers']) as pool:
        remaining=iter(matrix['jobs']);active={pool.submit(run,j):j for j in [next(remaining) for _ in range(min(LIMITS['job_workers'],len(matrix['jobs'])))]}
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job=active.pop(future);result=future.result();completed.append(result)
                atomic(base/'state.json',dict(status='running_frozen_matrix',matrix=str(matrix_path),completed_jobs=completed,pid=os.getpid(),updated_unix=time.time()))
                following=next(remaining,None)
                if following is not None and not (base/'halt.json').exists():active[pool.submit(run,following)]=following
    success=len(completed)==len(matrix['jobs']) and all(r['exit_code']==0 for r in completed)
    atomic(base/'matrix_states'/(matrix_path.stem+'.json'),dict(status='completed' if success else 'failed_preserved_checkpoints',completed_jobs=completed,updated_unix=time.time()))
    return success


def controller(base,parent):
    import fcntl
    setup(base);stream=(base/'execution.lock').open('a+b');fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        credentials();contract=prepare(base,parent);frozen(base)
        index=build_unit_index(base,contract);prepare_jobs(base,contract,index)
        ok=run_matrix(base,base/'main_matrix.json')
        atomic(base/'state.json',dict(status='main_matrix_completed' if ok else 'main_matrix_failed_preserved',updated_unix=time.time(),pid=os.getpid()))
    except Exception as exc:
        (base/'controller_failure.log').write_text(traceback.format_exc(),'utf-8')
        atomic(base/'state.json',dict(status='failed_preserved_checkpoints',error_type=type(exc).__name__,updated_unix=time.time(),pid=os.getpid()))
        raise
    finally:fcntl.flock(stream,fcntl.LOCK_UN);stream.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('step',choices=['prepare','controller','worker','matrix']);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--matrix',type=Path);p.add_argument('--job');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.step=='prepare':prepare(a.base,a.parent)
    elif a.step=='controller':controller(a.base,a.parent)
    elif a.step=='matrix':credentials();run_matrix(a.base,a.matrix)
    else:worker(a.base,a.matrix,a.job,a.smoke)

if __name__=='__main__':main()
