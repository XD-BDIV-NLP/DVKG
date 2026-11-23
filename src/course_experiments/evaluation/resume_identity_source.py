"""独立源阶段接续入口；原APP与prior产物不改，默认零API。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback


CONTINUATION=('source32','source-full','prior-full')
POLICY=dict(version='source_resume_duplicate_cap_v3',source_job='exact_original_recovery_source_job',
    changes='remove_duplicate_local_http_cap_from_call_only_restore_identical_contract',
    prior='read_existing_completed_development_never_copy_or_rejudge',
    queue='wait_original_controller_natural_exit_then_single_worker_finite',
    budget='original_global_start_and_all_local_cumulative_caps_unchanged',poll_seconds=10)


def process_identity(pid,required=()):
    from course_experiments.manifest import digest
    try:
        root=Path('/proc')/str(pid);stat=(root/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
        if tail[0]=='Z':return None
        command=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf-8','replace')
        if any(part not in command for part in required):return None
        return dict(pid=int(pid),start_ticks=tail[19],command_digest=digest(command))
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def source_contract_arguments(gate,migration,packets,origins):
    from course_experiments.manifest import digest
    full=migration['source_contract']
    if full.get('local_http_cap')!=300:raise ValueError('Original source300 cap required')
    passed={key:value for key,value in full.items() if key!='local_http_cap'}
    rebuilt=dict(gate.contract_for(packets,origins),**passed,local_http_cap=300)
    if rebuilt!=full or digest(rebuilt)!=migration['source_job']:
        raise ValueError('Source resume must preserve the exact frozen source job')
    return passed


def invoke_source_gate(gate,migration,packets,origins,provider,root,allowance):
    contract=source_contract_arguments(gate,migration,packets,origins)
    return gate.run_gate(packets,origins,provider,root,max_http_attempts=300,contract=contract,allowance_root=allowance)


def verify_preserved(recovery,migration,packets,old,new):
    from course_experiments.evaluation.run_native_full import _read_checkpoint
    from course_experiments.manifest import file_record,digest
    count=pending=0;records=[]
    for packet in packets:
        cid=packet['candidate_id'];source=Path(old)/'decisions'/cid;target=Path(new)/'decisions'/cid
        original=source/'decision.json'
        if not original.exists():continue
        row=recovery.batch.read_verified_checkpoint(original)
        completed=row.get('status')=='completed'
        if completed:count+=1
        if not completed and (source/'primary.json').exists() and not (source/'reverse.json').exists():pending+=1
        for name in ['decision.json','primary.json','reverse.json']:
            path=source/name
            if not path.exists() or (name=='decision.json' and not completed):continue
            prior=recovery.batch.read_verified_checkpoint(path);current=_read_checkpoint(target/name,migration['source_job'])
            value=recovery.payload(current);provenance=value.pop('recovery_from',None)
            if value!=recovery.payload(prior) or provenance!=file_record(path):
                raise ValueError('A successful original source checkpoint changed')
            records.append(dict(original=file_record(path),preserved=file_record(target/name)))
    if count!=migration['source_original_counts']['completed'] or pending!=migration['source_prior_success_primary_count']:
        raise ValueError('Original completed/pending-primary population changed')
    return dict(completed_decisions_preserved=count,pending_primary_preserved=pending,
        checkpoint_records_digest=digest(records),checkpoint_count=len(records))


def completed_stage(root,name,contract):
    from course_experiments.evaluation.run_native_full import _read_checkpoint
    from course_experiments.manifest import digest
    path=Path(root)/name/'result.json'
    if not path.exists():return None
    result=_read_checkpoint(path,digest(contract))
    if result.get('status')!='completed':raise ValueError('Existing incomplete stage retained; not relaunched')
    return result


def check(args,recovery):
    from course_experiments.manifest import file_record,read_document,digest
    migration=recovery.validate_migration(args)
    if file_record(args.trace)['sha256']!='3d28374626b4db20149cce99284347671d6c955b4211e33e850b38d4f075def4':
        raise ValueError('Reviewed source TypeError trace changed')
    packets,origins=recovery.source_arrays(migration['source_contract'])
    source_contract_arguments(recovery.gate,migration,packets,origins)
    old=Path(migration['recovery']['original_source_contract']['path']).parent
    preserved=verify_preserved(recovery,migration,packets,old,args.output_root/'source119')
    recovery.validate_original_preparation(args.preparation_root)
    guard=read_document(args.guard)
    if guard['migration_id']!=migration['migration_id'] or guard['source_job']!=migration['source_job']:
        raise ValueError('Source resume guard belongs to a different migration')
    for path,sha in guard['implementation'].items():
        if file_record(path)['sha256']!=sha:raise ValueError('Immutable source resume implementation changed')
    current=process_identity(guard['legacy_identity']['pid']) if guard.get('legacy_identity') else None
    if current and current!=guard['legacy_identity']:raise ValueError('Original controller PID identity changed')
    return migration,dict(status='check_only_source_resume_v3',source_job=migration['source_job'],
        migration_id=migration['migration_id'],preserved=preserved,legacy_controller_alive=bool(current),
        model_calls=0,policy=POLICY,source_result_exists=(args.output_root/'source119/result.json').exists())


def run(args,recovery,migration):
    from filelock import FileLock,Timeout
    from course_experiments.manifest import read_document,file_record,digest
    from course_experiments.evaluation.run_native_full import _read_checkpoint
    from course_experiments.evaluation.run_repaired_full import atomic
    root=args.output_root;path=root/'source_resume_v3.json';guard=read_document(args.guard)
    identity=digest(dict(policy=POLICY,guard=file_record(args.guard)))
    state=read_document(path) if path.exists() else dict(resume_identity=identity,stages={})
    if state['resume_identity']!=identity:raise ValueError('Source resume contract changed')
    limits=read_document(args.run_root/'formal_provider/limits.json')
    deadline=limits['started_unix']+limits['limits']['max_seconds']
    def save(status):
        state.update(status=status,updated_unix=time.time(),formal_deadline=deadline);atomic(path,state)
    def healthy():
        if time.time()>=deadline:raise TimeoutError('Original formal deadline reached')
        if any((args.run_root/'formal_provider'/n).exists() for n in ['halt.json','budget_halt.json']):
            raise RuntimeError('Existing global halt retained')
    def execute(name):
        provider=None
        try:
            healthy()
            if name=='source119':
                packets,origins=recovery.source_arrays(migration['source_contract']);source=root/'source119'
                if (source/'result.json').exists():
                    result=_read_checkpoint(source/'result.json',migration['source_job'])
                else:
                    provider=recovery.FormatRepairProvider(recovery.provider_for(args,source/'transport'),root/'format_repair_attempts')
                    with recovery.parser_bridge():
                        result=invoke_source_gate(recovery.gate,migration,packets,origins,provider,source,
                            Path(migration['budgets']['source119']['path']))
                success=result.get('status')=='completed_development_gate' and result.get('failed_candidates')==0
                if success:
                    recovery.batch.load_single119(source,[p['candidate_id'] for p in packets])
            else:
                deps=recovery.batch.dependencies(name,args.preparation_root,root/'mapping',root/'source119')
                contract=recovery.stage_contract(name,args.preparation_root,args.config,deps,migration['recovery'])
                result=completed_stage(root/'mapping',name,contract)
                if result is None:
                    packets,_=recovery.batch.stage_inputs(name,args.preparation_root)
                    provider=recovery.provider_for(args,root/'mapping'/name/'transport')
                    with recovery.parser_bridge():
                        result=recovery.batch.run_stage(name,args.preparation_root,root/'mapping',root/'source119',provider,
                            deps=deps,contract=contract,allowance_root=recovery.allowance_for(args.run_root,name,packets))
                success=result.get('status')=='completed'
            state['stages'][name]=dict(status='completed' if success else 'failed',exit_code=0 if success else 2,
                result=file_record((root/'source119' if name=='source119' else root/'mapping'/name)/'result.json'))
            save('stage_finished');return success
        except Exception as exc:
            error=root/(name+'_source_resume_v3_error_server_only.log');error.write_text(traceback.format_exc(),encoding='utf-8')
            state['stages'][name]=dict(status='failed_or_blocked',error_type=type(exc).__name__,error_record=file_record(error),exit_code=2)
            save('stage_failed_preserved');return False
        finally:
            if provider is not None:provider.close()
    with FileLock(str(root/'source_resume_v3.lock'),timeout=0):
        while guard.get('legacy_identity') and process_identity(guard['legacy_identity']['pid'])==guard['legacy_identity']:
            healthy();save('waiting_original_controller_natural_exit');time.sleep(10)
        with FileLock(str(root/'finite_recovery_controller.lock'),timeout=0):
            # 只消费实际结果，不沿用旧队列失败/blocked锁存。
            check(args,recovery)
            if not execute('source119'):save('source_gate_failed_no_downstream_launch');return 2
            for name in CONTINUATION:
                if not execute(name):save('finite_source_resume_failed_or_blocked');return 2
            save('finite_source_resume_completed');return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['app','run-root','config','preparation-root','output-root','guard','trace']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--start',action='store_true');args=parser.parse_args(argv)
    # 延迟导入，确保原APP的绝对模块路径及source_job完全不变。
    sys.path.insert(0,str(args.app))
    from course_experiments.evaluation import run_identity_recovery as recovery
    if Path(recovery.__file__).resolve().parent.parent.parent!=args.app.resolve():
        raise ValueError('Resume must import the exact original recovery app')
    migration,metadata=check(args,recovery)
    if not args.start:print(json.dumps(metadata));return 0
    return run(args,recovery,migration)


if __name__=='__main__':raise SystemExit(main())
