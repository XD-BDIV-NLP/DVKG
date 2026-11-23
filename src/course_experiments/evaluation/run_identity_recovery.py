"""身份事故的独立revision；旧产物不改，成功检查点复用，默认零API。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import traceback

from filelock import FileLock

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import run_scope_identity_gate as gate
from . import run_scope_identity_batch as batch
from . import scope_identity_batch as core
from . import scope_identity_packing as packing
from .identity_response import parser_bridge,POLICY as PARSER_POLICY
from .identity_format_repair import FormatRepairProvider,POLICY as REPAIR_POLICY
from .native_provider import AuditedNativeProvider,NativeAccountUnavailable,NativeBudgetExceeded
from .protocol import check_runtime
from .run_native_full import LIMITS,_checkpoint
from .run_repaired_full import atomic,credentials


PROVIDER_SHA='0d0b7b23ff5c3c12e0fba2f59646a939b9f8b50692e8039583efe033df09b102'
REVIEWED_PROVIDER_SHAS={PROVIDER_SHA,
    '9a7a5c58cfcea10f41d0b4f25d737a92637f38c7480976dd2df9ef47d19665c0',
    'af32fd606f52b78bc2d30d117950148f6207dc820c2819a5463a3394d4e7482f'}
PROBE_SHA='3808a11573e1d519bca23b2c650fe3016afa927286929490742087987d522719'
POLICY=dict(version='identity_incident_and_format_recovery_revision_v2',old_outputs='unchanged_original_receipts_and_decisions_retained',
    replay='same_original_raw_response_strict_validator_nested_boolean_shape_only',source_repair=REPAIR_POLICY,
    parser=PARSER_POLICY,budgets='reuse_original_source300_prior64_all_stage_caps_global30000_start_never_reset',
    queue='source119_then_prior_single_prior_batch_source32_source_full_prior_full_one_worker_finite_no_legacy_failed_latch')
ORDER=('source119','prior24-single','prior24-batch','source32','source-full','prior-full')


def validate_original_preparation(root):
    root=Path(root);manifest=read_document(root/'manifest.json')
    if manifest['preparation_id']!=digest({k:v for k,v in manifest.items() if k!='preparation_id'}):
        raise ValueError('Original preparation manifest digest changed')
    for name,record in manifest['files'].items():
        if file_record(root/name)['sha256']!=record['sha256']:raise ValueError('Original preparation data changed')
    for name,sha in manifest['implementation'].items():
        actual=file_record(Path(__file__).with_name(name))['sha256']
        if name=='native_provider.py' and sha=='5e8661663e476183f401a201971636a0402dfa6ba2a275c6b2ae3fd4ad80e859' and actual in REVIEWED_PROVIDER_SHAS:
            continue
        if actual!=sha:raise ValueError('Unreviewed original mapper implementation substitution')
    packing.OfficialFlashCounter(manifest['token_assets_root']);return manifest


def payload(checkpoint):return {k:copy.deepcopy(v) for k,v in checkpoint.items() if k not in {'job_identity','checkpoint_digest'}}


def rebind_checkpoint(old,new,job):
    value=batch.read_verified_checkpoint(old);value=payload(value);value['recovery_from']=file_record(old)
    return _checkpoint(new,value,job)


def reparse_batch(old,packets):
    rows={p['candidate_id']:p for p in packets};raw=old['raw_response']
    if raw['choices'][0].get('finish_reason')!='stop':raise ValueError('Cannot offline-repair incomplete model output')
    parsed=json.loads(raw['choices'][0]['message']['content'])
    expected=json.loads(packing.messages_for(packets,reverse=old['reverse'])[1]['content'])
    values=parsed.get('results',[]);ids=[r.get('candidate_id') for r in values]
    if parsed.get('batch_id')!=expected['batch_id'] or set(ids)!=set(rows) or len(ids)!=len(set(ids)):
        raise ValueError('Offline parser replay cannot guess batch or candidate identities')
    result=payload(old);decisions={}
    with parser_bridge():
        for value in values:
            packet=rows[value['candidate_id']]
            decisions[value['candidate_id']]=core._validate(packet,core._anonymous(packet),value,old['reverse'])
    result.update(decisions=decisions,parser_bridge=PARSER_POLICY,original_raw_response_digest=digest(raw))
    return result


def source_arrays(contract):
    packets=origins=None
    for record in contract['inputs'].values():
        if file_record(record['path'])!=record:raise ValueError('Original source gate input changed')
        if not str(record['path']).endswith('.json'):continue
        value=read_document(record['path'])
        if isinstance(value,list) and value:
            if 'left' in value[0] and 'right' in value[0]:packets=value
            elif 'graph_entity_id' in value[0]:origins=value
    if not packets or not origins or len(packets)!=119:raise ValueError('Frozen original119 packet/origin population unavailable')
    return packets,origins


def allowance_for(run_root,stage,packets=None,origins=None):
    run_root=Path(run_root)
    if stage=='source119':
        return run_root/'scope_identity_development_allowance'/digest(dict(packets=digest(packets),origins=digest(origins),policy=gate.POLICY['version']))
    return run_root/'identity_mapping_stage_allowance'/digest(dict(stage=stage,policy=batch.POLICY['version'],
        population=digest(sorted(p['candidate_id'] for p in packets)),judge=digest(core.JUDGE),cap=batch.STAGES[stage]['http_cap']))


def stage_contract(stage,preparation,config,deps,recovery):
    manifest=read_document(Path(preparation)/'manifest.json');packets,origins=batch.stage_inputs(stage,Path(preparation))
    implementation=dict(manifest['implementation'],**{'native_provider.py':PROVIDER_SHA})
    return dict(policy=batch.POLICY,stage=stage,limits=batch.STAGES[stage],preparation_id=manifest['preparation_id'],
        packet_digest=digest(packets),origin_digest=digest(origins),config=file_record(config),
        implementation=implementation,max_tokens=8192,input_max_bytes=manifest['input_sizes'][stage],
        dependency_digest=digest(deps),recovery_protocol=recovery)


def prepare(args):
    output=Path(args.output_root);output.mkdir(parents=True,exist_ok=True)
    implementation={name:file_record(Path(__file__).with_name(name)) for name in [
        'run_identity_recovery.py','identity_response.py','identity_format_repair.py','native_provider.py']}
    if implementation['native_provider.py']['sha256']!=PROVIDER_SHA:raise ValueError('Verified0d0 provider required in the new isolated app')
    probe=args.run_root/'incidents/kggen_offline_placeholder_401_20261022/availability_probe.json'
    if file_record(probe)['sha256']!=PROBE_SHA:raise ValueError('Availability incident probe receipt changed')
    old_contract=read_document(args.old_source_root/'contract.json');packets,origins=source_arrays(old_contract)
    validate_original_preparation(args.preparation_root)
    recovery=dict(policy=POLICY,implementation=implementation,probe=file_record(probe),
        original_source_contract=file_record(args.old_source_root/'contract.json'),
        original_mapping_preparation=file_record(args.preparation_root/'manifest.json'))
    recovery['recovery_id']=digest(recovery)
    contract=dict(gate.contract_for(packets,origins),inputs=old_contract['inputs'],provider_implementation=implementation['native_provider.py'],
        recovery_protocol=recovery,local_http_cap=300)
    job=digest(contract);source=output/'source119';counts={};archive=[]
    for packet in packets:
        cid=packet['candidate_id'];old=args.old_source_root/'decisions'/cid;new=source/'decisions'/cid
        decision=old/'decision.json'
        if decision.exists():
            row=batch.read_verified_checkpoint(decision);status=row['status'];counts[status]=counts.get(status,0)+1
            if status=='completed':rebind_checkpoint(decision,new/'decision.json',job)
            elif row.get('error_type') in {'NativeAccountUnavailable','ValueError'}:
                archive.append(dict(candidate_id=cid,classification='account_incident' if row['error_type']=='NativeAccountUnavailable' else 'bounded_format_repair',
                    original_failed_decision=file_record(decision),error_record=row.get('error_record')))
            else:raise ValueError('Unreviewed original source error cannot be reset in recovery')
        for name in ('primary.json','reverse.json'):
            if (old/name).exists():rebind_checkpoint(old/name,new/name,job)
    write_immutable(source/'contract.json',dict(contract,job_identity=job))
    prior='prior24-single';prior_contract=stage_contract(prior,args.preparation_root,args.config,{},recovery);prior_job=digest(prior_contract)
    prior_packets,_=batch.stage_inputs(prior,args.preparation_root);by_id={p['candidate_id']:p for p in prior_packets};replayed=positive=0
    for path in sorted((args.old_mapping_root/prior/'batches').glob('*/result.json')):
        old=batch.read_verified_checkpoint(path);work=[by_id[cid] for cid in old['candidate_ids']]
        result=reparse_batch(old,work);result['recovery_from']=file_record(path)
        _checkpoint(output/'mapping'/prior/'batches'/path.parent.name/'result.json',result,prior_job)
        replayed+=len(result['decisions']);positive+=sum(r['eligible_identity_edge'] for r in result['decisions'].values())
    if replayed!=24:raise ValueError('All original24 primary responses must be reparsed without omission')
    source_allowance=allowance_for(args.run_root,'source119',packets,origins)
    prior_allowance=allowance_for(args.run_root,prior,prior_packets)
    budget_records={}
    for name,folder,maximum in [('source119',source_allowance,300),(prior,prior_allowance,64)]:
        path=folder/'reservations.json';state=read_document(path)
        if state['maximum']!=maximum:raise ValueError('Original cumulative local cap changed')
        budget_records[name]=dict(path=str(folder),maximum=maximum,initial_reserved=state['reserved_http_attempts'],initial_receipt=file_record(path))
    migration=dict(recovery=recovery,source_contract=contract,source_job=job,source_original_counts=counts,
        source_prior_success_primary_count=sum((args.old_source_root/'decisions'/p['candidate_id']/'primary.json').exists() and
            not (args.old_source_root/'decisions'/p['candidate_id']/'reverse.json').exists() and
            (args.old_source_root/'decisions'/p['candidate_id']/'decision.json').exists() and
            batch.read_verified_checkpoint(args.old_source_root/'decisions'/p['candidate_id']/'decision.json').get('status')=='failed' for p in packets),
        source_failures_archived_by_reference=archive,source_success_copy_only_no_original_modified=True,
        prior_raw_primary_replayed=replayed,prior_primary_positive_reverse_pending=positive,
        old_partial_prior_result=file_record(args.old_mapping_root/prior/'result.json'),
        budgets=budget_records,original_formal_limits=read_document(args.run_root/'formal_provider/limits.json'),
        model_calls=0,old_failed_controller_not_reused=True)
    migration['migration_id']=digest(migration);write_immutable(output/'migration.json',migration)
    return migration


def validate_migration(args):
    value=read_document(args.output_root/'migration.json')
    if value['migration_id']!=digest({k:v for k,v in value.items() if k!='migration_id'}):raise ValueError('Recovery migration changed')
    for record in value['recovery']['implementation'].values():
        if file_record(record['path'])!=record:raise ValueError('Recovery implementation changed')
    for record in [value['recovery']['probe'],value['recovery']['original_source_contract'],value['recovery']['original_mapping_preparation']]:
        if file_record(record['path'])!=record:raise ValueError('Original incident/source/preparation changed')
    if read_document(args.run_root/'formal_provider/limits.json')!=value['original_formal_limits']:
        raise ValueError('Original formal start or limits changed')
    for row in value['budgets'].values():
        current=read_document(Path(row['path'])/'reservations.json')
        if current['maximum']!=row['maximum'] or current['reserved_http_attempts']<row['initial_reserved']:
            raise ValueError('An original local cumulative cap was reset')
    return value


def provider_for(args,transport):
    check_runtime(read_document(args.config));credentials()
    from ..construction import PROMPTS
    from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
    runtime=ModelRuntime(transport,PROMPTS,read_document(args.config))
    return AuditedNativeProvider(runtime,args.run_root/'formal_provider',**LIMITS)


def run_queue(args,migration):
    root=args.output_root;state_path=root/'queue_v2.json'
    state=read_document(state_path) if state_path.exists() else dict(migration_id=migration['migration_id'],stages={})
    if state['migration_id']!=migration['migration_id']:raise ValueError('Recovery queue belongs to a different migration')
    for name in ORDER:
        if name in state['stages']:continue
        if any((args.run_root/'formal_provider'/flag).exists() for flag in ('halt.json','budget_halt.json')):
            state['status']='existing_global_halt_preserved';atomic(state_path,state);return 2
        try:
            if name=='source119':
                packets,origins=source_arrays(migration['source_contract']);source=root/'source119'
                provider=FormatRepairProvider(provider_for(args,source/'transport'),root/'format_repair_attempts')
                with parser_bridge():
                    result=gate.run_gate(packets,origins,provider,source,contract=migration['source_contract'],
                        allowance_root=Path(migration['budgets']['source119']['path']))
                success=result['status']=='completed_development_gate' and result['failed_candidates']==0
            else:
                manifest=validate_original_preparation(args.preparation_root)
                deps=batch.dependencies(name,args.preparation_root,root/'mapping',root/'source119')
                contract=stage_contract(name,args.preparation_root,args.config,deps,migration['recovery'])
                packets,_=batch.stage_inputs(name,args.preparation_root)
                provider=provider_for(args,root/'mapping'/name/'transport')
                with parser_bridge():
                    result=batch.run_stage(name,args.preparation_root,root/'mapping',root/'source119',provider,
                        deps=deps,contract=contract,allowance_root=allowance_for(args.run_root,name,packets))
                success=result['status']=='completed'
            state['stages'][name]=dict(status='completed' if success else 'failed',exit_code=0 if success else 2)
        except batch.DependencyNotReady:
            state['stages'][name]=dict(status='blocked_by_uncompleted_qualification',exit_code=3)
        except (NativeAccountUnavailable,NativeBudgetExceeded):
            state['status']='global_account_or_budget_halt_preserved';atomic(state_path,state);raise
        except Exception as exc:
            path=root/(name+'_recovery_error_server_only.log');path.write_text(traceback.format_exc())
            state['stages'][name]=dict(status='failed',exit_code=2,error_type=type(exc).__name__,error_record=file_record(path))
        state['status']='finite_recovery_running';atomic(state_path,state)
    success=all(state['stages'].get(name,{}).get('exit_code')==0 for name in ORDER)
    state['status']='finite_recovery_completed' if success else 'finite_recovery_finished_failed_or_blocked';atomic(state_path,state)
    return 0 if success else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-root','config','preparation-root','old-source-root','old-mapping-root','output-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--start',action='store_true')
    args=parser.parse_args(argv)
    if args.prepare:
        value=prepare(args)
        print(json.dumps(dict(status='prepared_zero_API_recovery',migration_id=value['migration_id'],source_counts=value['source_original_counts'],
            prior_reparsed=value['prior_raw_primary_replayed'],prior_reverse_pending=value['prior_primary_positive_reverse_pending'],
            budget_inheritance=value['budgets'],model_calls=0)));return 0
    value=validate_migration(args)
    if not args.start:
        print(json.dumps(dict(status='check_only',migration_id=value['migration_id'],model_calls=0,
            source_counts=value['source_original_counts'],prior_reparsed=value['prior_raw_primary_replayed'],
            prior_reverse_pending=value['prior_primary_positive_reverse_pending'],policy=POLICY)));return 0
    with FileLock(str(args.output_root/'finite_recovery_controller.lock'),timeout=0):return run_queue(args,value)


if __name__=='__main__':raise SystemExit(main())
