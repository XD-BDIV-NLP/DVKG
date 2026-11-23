"""独立实体身份开发gate；复用正式账本，仅输出新身份投影，不改旧图或旧split。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from pathlib import Path
import time

from filelock import FileLock

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from ..splits import freeze_split
from .native_provider import AuditedNativeProvider,NativeBudgetExceeded,NativeAccountUnavailable
from .protocol import check_runtime
from .run_native_full import LIMITS,ReceiptTracker,_checkpoint,_read_checkpoint,_error,_no_scope
from .run_repaired_full import atomic,credentials
from . import scope_identity as identity


POLICY = dict(version='scope_identity_machine_development_gate_v1',maximum_local_http_reservations=300,
    global_budget='existing_formal_provider_30000_HTTP_172800_seconds_4_slots_never_reset',
    confirmation_inputs=False,rank_inputs=False,primary='source_definition_scope_identity_same_sense_only',
    consistency='reverse_sides_for_every_positive_primary_no_additional_prompt_tuning',
    ambiguity='reject_all_conflicting_targets_for_an_original_graph_entity',
    output='identity_projection_only_no_new_facts_no_original_graph_mutation',
    activation='development_only_frozen_rules_all_applicable_methods_reference_equivalence_components_then_new_split',
    old_training='never_relabel_old_300_runs_as_new_identity_protocol',human_audited=False)
FORMAL_PROVIDER_SHA = '5e8661663e476183f401a201971636a0402dfa6ba2a275c6b2ae3fd4ad80e859'


class LocalHTTPAllowanceExceeded(RuntimeError):
    pass


@contextmanager
def local_http_cap(provider, root, maximum):
    if type(maximum) is not int or not 1<=maximum<=300:
        raise ValueError('Local HTTP allowance must be 1..300; global limits remain frozen')
    runtime = getattr(provider,'runtime',None)
    if runtime is None:
        yield
        return
    root = Path(root);root.mkdir(parents=True,exist_ok=True)
    path = root/'reservations.json'
    original = runtime._record
    def record(key,stage,status,value):
        if status == 'request_started':
            with FileLock(str(root/'reservations.lock'),timeout=30):
                state = read_document(path) if path.exists() else dict(maximum=maximum,reserved_http_attempts=0)
                if state['maximum']!=maximum:
                    raise ValueError('Frozen local HTTP allowance cannot change')
                if state['reserved_http_attempts']>=maximum:
                    raise LocalHTTPAllowanceExceeded('Scope identity development local HTTP allowance exhausted')
                state.update(reserved_http_attempts=state['reserved_http_attempts']+1,
                    last_request_id=key,updated_unix=time.time(),counting='conservative_reservations_before_shared_global_attempt')
                atomic(path,state)
        # 本地文件锁已释放，才进入正式账本和原transport。
        return original(key,stage,status,value)
    runtime._record = record
    try:
        yield
    finally:
        runtime._record = original


def identity_projection(origins, decisions):
    targets = {}
    for row in origins:
        if decisions.get(row['candidate_id'],{}).get('eligible_identity_edge'):
            key = (row['graph_record']['sha256'],row['graph_entity_id'])
            targets.setdefault(key,set()).add(row['reference_entity_id'])
    conflicts = {key for key,values in targets.items() if len(values)>1}
    conflict_ids = {r['candidate_id'] for r in origins if
        (r['graph_record']['sha256'],r['graph_entity_id']) in conflicts}
    mapped = {};rejected = {}
    for row in origins:
        rid = digest(row)
        if row['candidate_id'] in conflict_ids:
            rejected[rid] = dict(row,reason='conflicting_or_polysemous_reference_targets')
        elif decisions.get(row['candidate_id'],{}).get('eligible_identity_edge'):
            mapped[rid] = row
        else:
            rejected[rid] = dict(row,reason='no_positive_two_sided_consistent_source_identity')
    result = dict(mapping=list(mapped.values()),rejections=list(rejected.values()),
        policy=POLICY['output'],activation=POLICY['activation'],original_graphs_modified=False,
        usable_for_KGE=False,reference_identity_components_frozen=False,new_split_required=True)
    return dict(result,identity_projection_id=digest(result))


def freeze_identity_split(reference, frozen_components):
    # 仅接收独立判定并冻结后的reference身份组件，不能从图覆盖或rank推导。
    result = copy.deepcopy(reference)
    entities = {e['entity_id']:e for e in result['entities']}
    mapping = {key:key for key in entities};seen=set()
    for component in frozen_components:
        ids = sorted(set(component))
        if len(ids)<2 or not set(ids)<=entities.keys() or seen & set(ids):
            raise ValueError('Frozen reference identity components must be disjoint known IDs')
        seen.update(ids);target=ids[0]
        mapping.update({key:target for key in ids})
        row = entities[target]
        row['identity_provenance'] = [copy.deepcopy(entities[key]) for key in ids]
        row['scope'] = 'frozen_identity_component_'+digest(ids)
        row['aliases'] = sorted({name for key in ids for name in
            [entities[key]['canonical_name'],*entities[key].get('aliases',[])] if name!=row['canonical_name']})
        row['citations'] = list({digest(c):c for key in ids for c in entities[key].get('citations',[])}.values())
    result['entities'] = [e for key,e in entities.items() if mapping[key]==key]
    for fact in result['facts']:
        fact['head']=mapping[fact['head']];fact['tail']=mapping[fact['tail']]
    result['identity_component_mapping']=mapping
    result['reference_id']=digest({k:v for k,v in result.items() if k!='reference_id'})
    return result,freeze_split(result)


def contract_for(packets, origins):
    ids = [p['candidate_id'] for p in packets]
    if not packets or len(set(ids))!=len(ids) or any(p['candidate_id']!=digest({
            k:v for k,v in p.items() if k!='candidate_id'}) for p in packets):
        raise ValueError('Unique hash-verified development packets required')
    if any(row['candidate_id'] not in set(ids) for row in origins) or set(ids)!={r['candidate_id'] for r in origins}:
        raise ValueError('Every development candidate requires exact server-only origin provenance')
    packets_by_id={p['candidate_id']:p for p in packets}
    if any(row['reference_entity_id']!=packets_by_id[row['candidate_id']]['left']['entity_id'] or
            row['graph_entity_id']!=packets_by_id[row['candidate_id']]['right']['entity_id'] for row in origins):
        raise ValueError('Identity origin endpoints differ from their frozen source packet')
    return dict(policy=POLICY,candidate_ids=sorted(ids),packets_digest=digest(packets),origins_digest=digest(origins),
        implementation={p.name:file_record(p) for p in [Path(__file__),Path(identity.__file__)]},
        judge_prompt_digest=digest(identity.JUDGE),identity_policy=identity.POLICY)


def run_gate(packets, origins, provider, root, *, max_http_attempts=300, contract=None,allowance_root=None):
    contract = dict(contract_for(packets,origins),**(contract or {}),local_http_cap=max_http_attempts)
    job = digest(contract);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    write_immutable(root/'contract.json',dict(contract,job_identity=job))
    decisions = {};tracker=ReceiptTracker(provider) if hasattr(provider,'request_executor') else None
    def state(status):
        atomic(root/'state.json',dict(status=status,job_identity=job,candidate_count=len(packets),
            checkpoint_count=len(decisions),updated_unix=time.time(),model_calls_owned_by_existing_formal_budget=True))
    try:
        with local_http_cap(provider,allowance_root or root/'local_allowance',max_http_attempts):
            for packet in sorted(packets,key=lambda p:digest([42,p['candidate_id']])):
                cid=packet['candidate_id'];folder=root/'decisions'/cid;path=folder/'decision.json'
                if path.exists():
                    decisions[cid]=_read_checkpoint(path,job)
                    continue
                state('running_development_identity_judgment');provider.raise_if_fatal()
                scope=tracker.scope(folder/'requests.jsonl') if tracker else _no_scope()
                try:
                    with scope:
                        primary_path=folder/'primary.json'
                        primary=_read_checkpoint(primary_path,job)['judgment'] if primary_path.exists() else identity.judge_pair(provider,packet)
                        if not primary_path.exists():_checkpoint(primary_path,dict(judgment=primary),job)
                        reverse=None
                        if primary['eligible_identity_edge']:
                            reverse_path=folder/'reverse.json'
                            reverse=_read_checkpoint(reverse_path,job)['judgment'] if reverse_path.exists() else identity.judge_pair(provider,packet,reverse=True)
                            if not reverse_path.exists():_checkpoint(reverse_path,dict(judgment=reverse),job)
                        eligible=bool(primary['eligible_identity_edge'] and reverse and reverse['eligible_identity_edge'])
                        row=dict(status='completed',candidate_id=cid,primary=primary,reverse=reverse,
                            eligible_identity_edge=eligible,human_audited=False)
                except Exception as exc:
                    row=dict(status='failed',candidate_id=cid,eligible_identity_edge=False,**_error(folder,exc))
                    decisions[cid]=_checkpoint(path,row,job)
                    provider.raise_if_fatal()
                    if isinstance(exc,(LocalHTTPAllowanceExceeded,NativeBudgetExceeded,NativeAccountUnavailable)):
                        state('halted_preserved_existing_budget');raise
                    continue
                decisions[cid]=_checkpoint(path,row,job)
            provider.raise_if_fatal()
            projection=identity_projection(origins,decisions)
            accepted={r['candidate_id'] for r in projection['mapping']}
            failed=sum(r['status']=='failed' for r in decisions.values())
            result=dict(status='completed_development_gate' if not failed else 'partial_development_gate',
                candidate_count=len(packets),accepted_candidates=len(accepted),failed_candidates=failed,
                rejected_candidates=len(packets)-len(accepted)-failed,identity_projection=projection,
                human_audited=False,reference_components_or_split_created=False,job_identity=job)
            _checkpoint(root/'identity_projection.json',projection,job)
            _checkpoint(root/'result.json',result,job)
            state(result['status']);return result
    except (LocalHTTPAllowanceExceeded,NativeBudgetExceeded,NativeAccountUnavailable):
        state('halted_preserved_existing_budget');raise
    finally:
        if tracker:tracker.close()
        provider.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-root','config','packets','origins','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--local-http-cap',type=int,default=300)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    packets=read_document(args.packets);origins=read_document(args.origins);config=read_document(args.config)
    check_runtime(config)
    if read_document(args.run_root/'formal_provider/limits.json')['limits']!=LIMITS:
        raise ValueError('Existing formal provider limits differ; no new global budget allowed')
    provider_record=file_record(Path(__file__).with_name('native_provider.py'))
    if provider_record['sha256']!=FORMAL_PROVIDER_SHA:
        raise ValueError('Scope gate requires the frozen 5e formal provider implementation')
    contract=dict(contract_for(packets,origins),inputs={p.name:file_record(p) for p in [args.config,args.packets,args.origins]},
        provider_implementation=provider_record)
    if args.check_only:
        print(digest(contract));return
    credentials()
    from ..construction import PROMPTS
    from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
    runtime=ModelRuntime(args.output/'transport',PROMPTS,config)
    provider=AuditedNativeProvider(runtime,args.run_root/'formal_provider',**LIMITS)
    allowance=args.run_root/'scope_identity_development_allowance'/digest(dict(
        packets=digest(packets),origins=digest(origins),policy=POLICY['version']))
    run_gate(packets,origins,provider,args.output,max_http_attempts=args.local_http_cap,
        contract=contract,allowance_root=allowance)


if __name__=='__main__':main()
