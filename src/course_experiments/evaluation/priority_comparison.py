from __future__ import annotations
import time


def acquire_released_lock(acquire,timeout=15.,*,clock=time.monotonic,pause=time.sleep):
    deadline=clock()+timeout
    while True:
        try:
            acquire()
            return
        except BlockingIOError:
            remaining=deadline-clock()
            if remaining<=0:raise TimeoutError('Execution lock not released within handoff budget') from None
            pause(min(.1,remaining))


def ordered_jobs(original, baselines):
    jobs=list(original)+list(baselines)
    if len({j['name'] for j in jobs})!=len(jobs):
        raise ValueError('Duplicate comparison job names')
    cohorts=['compAB','osAB','netAB','compABCD']
    # 原题对照先于补充任务，保留每个任务的原始身份。
    return sorted(jobs,key=lambda j:(0 if j['kind']=='F' else 1 if j['kind'] in {'keyword','vector','K12'} else 2,
                                    cohorts.index(j['cohort']),j['kind']))


def complete_for_handoff(job,result):
    return (result.get('status')=='completed' and result.get('job')==job
            and result.get('question_count')==job['question_count']
            and result.get('checkpoint_count')==job['question_count'])
