"""补充题全部真实作答；参考来源覆盖与答案得分分开。"""
from __future__ import annotations
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
import types
from pathlib import Path

POLICY='all_fixed_questions_real_answers_reference_source_coverage_separate_no_forced_zero'


def summarize_coverage(records,expected,coverage):
    from course_experiments.evaluation.repaired_full import summarize
    result=summarize(records,expected)
    result.pop('answerability_coverage',None)
    n=len(expected)
    result['reference_source_coverage']=sum(coverage[k] for k in expected)/n if n else None
    result['questions_missing_selected_reference_sources']=sum(not coverage[k] for k in expected)
    result['answer_attempt_coverage']=sum(r['status']=='completed' for r in records)/n if n else None
    result['denominator_policy']=POLICY
    return result


def evaluate_all(job,phase,output,runtime,base,stop_after=None):
    from course_experiments.evaluation import repaired_full as full
    from course_experiments.manifest import read_document
    if phase['parameters'].get('reference_coverage_policy')!=POLICY:raise ValueError('Supplement coverage policy not frozen')
    reference=read_document(phase['files']['reference']['path']);index=read_document(phase['files']['index']['path'])
    available={e for r in index['rows'] if r['book_id'] in job['active_books'] for e in r['evidence_ids']}
    coverage={q['question_id']:full.answerable(q,available) for q in reference['questions']}
    # 复制调用环境，只替换跳答与覆盖汇总，不修改冻结模块及检索实现。
    environment=dict(full.evaluate.__globals__,answerable=lambda question,available:True,
        summarize=lambda records,expected:summarize_coverage(records,expected,coverage))
    function=types.FunctionType(full.evaluate.__code__,environment,full.evaluate.__name__,full.evaluate.__defaults__,full.evaluate.__closure__)
    function.__kwdefaults__=full.evaluate.__kwdefaults__
    return function(job,phase,output,runtime,base,stop_after=stop_after)


def worker(base,matrix_path,name,smoke=False):
    from course_experiments.manifest import digest,file_record,read_document
    from course_experiments.evaluation.run_repaired_full import credentials,frozen,cache_matches,checked,atomic
    from course_experiments.evaluation.protocol import runtime_for
    from course_experiments.evaluation.repaired_full import limited_runtime
    credentials();frozen(base);cache_matches();matrix=read_document(matrix_path)
    if matrix['matrix_id']!=digest({k:v for k,v in matrix.items() if k!='matrix_id'}):raise ValueError('Supplement matrix changed')
    job=next(j for j in matrix['jobs'] if j['name']==name);phase=checked(job['phase'])
    if phase['files']['fair_evaluator']['sha256']!=file_record(__file__)['sha256']:raise ValueError('Actual supplement evaluator changed')
    runtime=limited_runtime(runtime_for(read_document(phase['files']['construction']['path']),base/'jobs'/name/'runtime'),base)
    try:
        atomic(base/'job_states'/(name+'.json'),dict(status='running',pid=os.getpid(),started_unix=time.time(),evaluation_policy=POLICY))
        result=evaluate_all(job,phase,base/'jobs'/name,runtime,base,stop_after=1 if smoke else None)
        atomic(base/'job_states'/(name+'.json'),dict(status=result['status'],pid=os.getpid(),completed_count=result['checkpoint_count'],question_count=result['question_count'],updated_unix=time.time(),evaluation_policy=POLICY))
        if smoke and result['checkpoint_count']!=1:raise RuntimeError('Real supplement smoke did not commit')
        if not smoke and result['status']!='completed':raise RuntimeError('Supplement partial results preserved')
    finally:runtime.close()


def run_matrix(base,matrix_path):
    from course_experiments.manifest import read_document
    from course_experiments.evaluation.protocol import ROOT
    from course_experiments.evaluation.run_repaired_full import atomic,LIMITS
    matrix=read_document(matrix_path);(base/'logs').mkdir(exist_ok=True);completed=[]
    def run(job,smoke=False):
        if job['name'].startswith('supp_'):
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--base',str(base),'--matrix',str(matrix_path),'--job',job['name']]
        else:
            command=[sys.executable,'-u','-m','course_experiments.evaluation.run_repaired_full','worker','--base',str(base),'--matrix',str(matrix_path),'--job',job['name']]
        if smoke:command.append('--smoke')
        with (base/'logs'/(job['name']+('_smoke' if smoke else '')+'.log')).open('a') as stream:
            return dict(name=job['name'],exit_code=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode)
    smoke=run(matrix['jobs'][0],True)
    if smoke['exit_code']:raise RuntimeError('Real matrix smoke failed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=LIMITS['job_workers']) as pool:
        remaining=iter(matrix['jobs']);active={pool.submit(run,j):j for j in [next(remaining) for _ in range(min(LIMITS['job_workers'],len(matrix['jobs'])))]}
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                active.pop(future);completed.append(future.result())
                atomic(base/'state.json',dict(status='running_frozen_matrix',matrix=str(matrix_path),completed_jobs=completed,pid=os.getpid(),updated_unix=time.time()))
                following=next(remaining,None)
                if following is not None and not (base/'halt.json').exists():active[pool.submit(run,following)]=following
    success=len(completed)==len(matrix['jobs']) and all(r['exit_code']==0 for r in completed)
    atomic(base/'matrix_states'/(matrix_path.stem+'.json'),dict(status='completed' if success else 'failed_preserved_checkpoints',completed_jobs=completed,updated_unix=time.time()))
    return success


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(a.base/'app'));worker(a.base,a.matrix,a.job,a.smoke)

if __name__=='__main__':main()
