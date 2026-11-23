"""固定五阶段的有限依赖队列；默认只检查，--start才执行，不创建心跳。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from filelock import FileLock,Timeout

from course_experiments.manifest import digest,file_record,read_document
from course_experiments.evaluation import run_scope_identity_batch as runner
from course_experiments.evaluation.run_repaired_full import atomic


POLICY=dict(version='finite_identity_dependency_controller_v1',poll_seconds=10,max_model_workers=1,
    deadline='original_formal_provider_started_unix_plus172800_never_reset',
    independent='prior24_development_can_run_while_waiting_for_single119_source_gate',
    terminal='failed_or_exited_source119_stops_source_branch_preserves_independent_prior_development',
    recovery='attach_exact_recorded_live_worker_or_consume_exit_receipt_never_relaunch_completed_or_failed_stage',
    start='explicit_start_flag_only_no_automation_or_heartbeat')


def formal_deadline(root):
    value=read_document(Path(root)/'limits.json')
    return value['started_unix']+value['limits']['max_seconds']


def process_identity(pid,required=()):
    try:
        root=Path('/proc')/str(pid);stat=(root/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
        if tail[0]=='Z':return None
        command=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf-8','replace')
        if any(part not in command for part in required):return None
        return dict(pid=int(pid),start_ticks=tail[19],command_digest=digest(command))
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def source_readiness(root,ids,*,alive):
    root=Path(root);result_path=root/'result.json'
    if result_path.exists():
        result=runner.read_verified_checkpoint(result_path)
        if result.get('status')!='completed_development_gate' or result.get('failed_candidates',0)!=0:
            return 'failed'
        if result.get('candidate_count')!=len(ids):return 'failed'
        for cid in ids:
            path=root/'decisions'/cid/'decision.json'
            if not path.exists():return 'failed'
            row=runner.read_verified_checkpoint(path)
            if row.get('status')!='completed' or row.get('candidate_id')!=cid:return 'failed'
        return 'ready'
    state=root/'state.json'
    if state.exists() and read_document(state).get('status') in {
            'partial_development_gate','halted_preserved_existing_budget','failed','failed_preserved_checkpoints'}:
        return 'failed'
    return 'waiting' if alive else 'failed'


def next_stage(stages,source_status):
    def success(name):return stages.get(name,{}).get('exit_code')==0
    if 'prior24-single' not in stages:return 'prior24-single'
    if success('prior24-single') and 'prior24-batch' not in stages:return 'prior24-batch'
    if source_status=='ready' and 'source32' not in stages:return 'source32'
    if success('source32') and 'source-full' not in stages:return 'source-full'
    if success('source-full') and success('prior24-batch') and 'prior-full' not in stages:return 'prior-full'
    return None


def source_snapshot(state,root,ids,*,alive):
    if state.get('source119_final_status') in {'ready','failed'}:return state['source119_final_status']
    status=source_readiness(root,ids,alive=alive)
    if status in {'ready','failed'}:
        state['source119_final_status']=status
        result=Path(root)/'result.json'
        if result.exists():state['source119_verified_result']=file_record(result)
    return status


def reserve_launch(state,name,path):
    if name in state['stages']:raise ValueError('A persisted stage cannot be launched again')
    state['stages'][name]=dict(status='launch_reserved',reserved_unix=time.time())
    atomic(path,state)


def verify_worker_receipt(value,stage,preparation_id,controller_code):
    if (value.get('stage'),value.get('preparation_id'),value.get('controller_code'))!=(stage,preparation_id,controller_code):
        raise ValueError('Worker receipt belongs to a different frozen controller or population')


def worker(args):
    record=args.queue_root/'stages'/args.worker_stage;record.mkdir(parents=True,exist_ok=True)
    code=1;error=None;identity=dict(stage=args.worker_stage,
        preparation_id=read_document(args.preparation_root/'manifest.json')['preparation_id'],
        controller_code=file_record(Path(__file__))['sha256'])
    atomic(record/'worker_started.json',dict(pid=os.getpid(),identity=process_identity(os.getpid()),
        **identity,created_unix=time.time(),controller_policy=POLICY['version']))
    try:
        runner.main(['--run-root',str(args.run_root),'--config',str(args.config),'--preparation-root',str(args.preparation_root),
            '--output-root',str(args.output_root),'--single119-root',str(args.single119_root),'--stage',args.worker_stage])
        code=0
    except SystemExit as exc:code=exc.code if isinstance(exc.code,int) else 1
    except BaseException as exc:
        error=type(exc).__name__;path=record/'worker_error_server_only.log';path.write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        atomic(record/'exit.json',dict(**identity,exit_code=code,error_type=error,
            finished_unix=time.time(),worker_pid=os.getpid()))
    return code


def common_arguments(args):
    values=[]
    for name in ['run_root','config','preparation_root','output_root','single119_root','queue_root']:
        values+=['--'+name.replace('_','-'),str(getattr(args,name))]
    return values


def run_queue(args,manifest):
    args.queue_root.mkdir(parents=True,exist_ok=True);state_path=args.queue_root/'queue.json';deadline=formal_deadline(args.run_root/'formal_provider')
    frozen=dict(policy=POLICY,preparation_id=manifest['preparation_id'],formal_deadline=deadline,
        single119_pid=args.single119_pid,single119_root=str(args.single119_root),controller_code=file_record(Path(__file__))['sha256'])
    signature=digest(frozen)
    state=read_document(state_path) if state_path.exists() else dict(contract_id=signature,contract=frozen,stages={})
    if state['contract_id']!=signature:raise ValueError('Finite controller contract changed; existing receipts preserved')
    if state.get('source119_verified_result') and file_record(state['source119_verified_result']['path'])['sha256']!=state['source119_verified_result']['sha256']:
        raise ValueError('Previously verified single119 terminal receipt changed')
    plan=read_document(args.preparation_root/'source32_plan.json');ids=plan['single_population_candidate_ids']
    if len(ids)!=119 or len(set(ids))!=119:raise ValueError('Fixed119 prerequisite population required')
    initial=process_identity(args.single119_pid,('run_scope_identity_gate',str(args.single119_root)))
    if 'single119_identity' not in state:state['single119_identity']=initial
    handles={};source_status='waiting'
    def save(status):
        state.update(status=status,updated_unix=time.time(),controller_pid=os.getpid(),
            source119_status=source_status,formal_deadline=deadline);atomic(state_path,state)
    while True:
        if any((args.run_root/'formal_provider'/name).exists() for name in ['halt.json','budget_halt.json']):
            save('existing_global_budget_or_account_halt');return 2
        if time.time()>=deadline:
            save('original_formal_deadline_reached');return 2
        current=process_identity(args.single119_pid,('run_scope_identity_gate',str(args.single119_root)))
        alive=bool(current and current==state.get('single119_identity'))
        source_status=source_snapshot(state,args.single119_root,ids,alive=alive)
        reserved=[(name,row) for name,row in state['stages'].items() if row.get('status')=='launch_reserved']
        if reserved:
            name,row=reserved[0];started_path=args.queue_root/'stages'/name/'worker_started.json'
            if started_path.exists():
                started=read_document(started_path);verify_worker_receipt(started,name,manifest['preparation_id'],frozen['controller_code'])
                row.update(status='running',pid=started['pid'],process_identity=started['identity'])
                save('recovered_reserved_worker');continue
            if time.time()-row['reserved_unix']<10:time.sleep(10);continue
            save('ambiguous_reserved_launch_stopped_without_relaunch');return 2
        running=[(name,row) for name,row in state['stages'].items() if row.get('status')=='running']
        if len(running)>1:raise ValueError('More than one recorded model worker')
        if running:
            name,row=running[0];exit_path=args.queue_root/'stages'/name/'exit.json'
            if exit_path.exists():
                receipt=read_document(exit_path);verify_worker_receipt(receipt,name,manifest['preparation_id'],frozen['controller_code'])
                row.update(status='exited',exit_code=receipt['exit_code'],exit_receipt=file_record(exit_path))
                if name in handles:handles.pop(name).wait()
                save('stage_exited');continue
            identity=process_identity(row['pid'])
            if not identity or identity!=row['process_identity']:
                row.update(status='exited_without_receipt',exit_code=125,finished_unix=time.time())
                save('worker_exit_without_receipt_preserved');continue
            save('one_worker_running');time.sleep(10);continue
        name=next_stage(state['stages'],source_status)
        if name:
            folder=args.queue_root/'stages'/name;folder.mkdir(parents=True,exist_ok=True)
            existing=folder/'worker_started.json'
            if existing.exists():
                started=read_document(existing);exit_path=folder/'exit.json'
                verify_worker_receipt(started,name,manifest['preparation_id'],frozen['controller_code'])
                state['stages'][name]=dict(status='running',pid=started['pid'],process_identity=started['identity'],
                    launch_recovered_from=file_record(existing));save('recovered_previous_worker_launch');continue
            command=[sys.executable,str(Path(__file__).resolve()),*common_arguments(args),'--worker-stage',name,'--start']
            reserve_launch(state,name,state_path)
            log=(folder/'console_server_only.log').open('ab')
            try:child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            finally:log.close()
            identity=process_identity(child.pid)
            state['stages'][name]=dict(status='running',pid=child.pid,process_identity=identity,launched_unix=time.time(),
                command_digest=digest(command),preparation_id=manifest['preparation_id'])
            handles[name]=child;save('stage_launched');continue
        if source_status=='waiting':save('waiting_fixed_single119');time.sleep(10);continue
        completed=all(state['stages'].get(name,{}).get('exit_code')==0 for name in runner.STAGES)
        save('finite_queue_completed' if completed else 'finite_queue_finished_with_failed_or_blocked_stages')
        return 0 if completed else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-root','config','preparation-root','output-root','single119-root','queue-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--single119-pid',type=int,default=825436)
    parser.add_argument('--start',action='store_true');parser.add_argument('--worker-stage',choices=sorted(runner.STAGES))
    args=parser.parse_args(argv)
    if args.worker_stage:
        if not args.start:
            print(dict(status='internal_worker_check_only_not_started',model_calls=0));return 0
        return worker(args)
    manifest=runner.validate_preparation(args.preparation_root)
    if not args.start:
        print(dict(status='check_only_not_started',model_calls=0,policy=POLICY,preparation_id=manifest['preparation_id'],
            formal_deadline=formal_deadline(args.run_root/'formal_provider')));return 0
    args.queue_root.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(args.queue_root/'controller.lock'),timeout=0):return run_queue(args,manifest)
    except Timeout:raise SystemExit('An exact finite identity controller already holds the operation lock')


if __name__=='__main__':raise SystemExit(main())
