"""有限批量身份门禁与全量身份投影；所有原文和判定只存原服务器。"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time

from filelock import FileLock

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from ..splits import canonical_reference
from . import scope_identity_batch as core
from . import scope_mapping_preflight as preflight
from . import prior_definition_identity as prior
from . import scope_identity_packing as packing
from .native_provider import AuditedNativeProvider,NativeBudgetExceeded,NativeAccountUnavailable
from .protocol import check_runtime
from .run_native_full import LIMITS,ReceiptTracker,_checkpoint,_read_checkpoint,_error,_no_scope
from .run_repaired_full import atomic,credentials
from .run_scope_identity_gate import identity_projection,LocalHTTPAllowanceExceeded,FORMAL_PROVIDER_SHA


STAGES = {
    'source32':dict(http_cap=24,logical_max=12,kind='two_sided_source'),
    'source-full':dict(http_cap=2200,logical_max=1884,kind='two_sided_source'),
    'prior24-single':dict(http_cap=64,logical_max=48,kind='prior_definition_identity'),
    'prior24-batch':dict(http_cap=12,logical_max=6,kind='prior_definition_identity'),
    'prior-full':dict(http_cap=72,logical_max=52,kind='prior_definition_identity')}
POLICY = dict(version='finite_identity_mapping_stages_token_v1',max_tokens=8192,input_truncation=False,
    token_packing=packing.POLICY['version'],
    global_budget='existing_formal_provider_30000_HTTP_172800_seconds_4_slots_never_reset',
    source_and_prior='separate_populations_caps_checkpoints_and_dependency_gates',
    prior_single='anonymous_batch8_prompt_and_validation_with_exactly_one_pair_per_request_new_config',
    failures='fixed_denominator_failed_checkpoints_preserved_no_automatic_prompt_repair',
    full_mapping='identity_projection_only_new_reference_equivalence_split_required_before_KGE',
    single_negatives='original_single_pair_negative_identity_pairs_veto_later_projection',
    queue='finite_dependency_exit_no_polling_no_background_monitor',human_audited=False)
PRIOR_POLICY = dict(version='prior_development24_v1',seed=42,maximum_per_course=8,target_pairs=24,
    selection='course_then_hash_all200_prior_candidate_IDs_no_folds_ranks_coverage_or_single_decisions',
    insufficient_course='report_shortfall_no_backfill_from_another_course',minimum_exact_decision_agreement=23,
    negative_to_positive=0,both_side_exact_quotes=24,all_accepted_positives_reversed=True,
    original_prior3='preserved_not_substituted_for_prior24')


class DependencyNotReady(RuntimeError):pass


def verified_read(record):
    if file_record(record['path'])['sha256']!=record['sha256']:raise ValueError('Frozen input receipt changed')
    return read_document(record['path'])


def prepare_population(inventory):
    references={};sources={};source_packets={};prior_packets={};source_origins=[];prior_origins=[]
    for row in inventory['graphs']:
        course=row['course_id']
        if course not in references:
            references[course]=canonical_reference(read_document(row['reference']['path']))['entities']
        if row.get('evidence_path'):
            index=sources.setdefault(course,{})
            for key,item in read_document(row['evidence_path']).items():
                for eid in [key,item.get('native_id'),item.get('id')]:
                    if eid:index[eid]=item
    for row in inventory['graphs']:
        raw=read_document(row['path']);cap=None
        if row.get('artifact'):
            if raw.get('status')!='complete':raise ValueError('Inventory artifact is not complete')
            graph={key:raw['semantic_view'][key] for key in ['entities','relations']}
            cap=raw['semantic_view'].get('capabilities')
        else:graph=raw
        course=row['course_id'];evidence=sources.get(course,{}) if row['method']!='G0' else {}
        if row.get('artifact'):evidence=preflight.source_evidence_index(evidence,raw.get('evidence',[]))
        summary,packets=preflight.prepare_graph(course,graph,references[course],sources.get(course,{}),evidence,cap)
        if summary['qualification']['status']!='eligible_typed_graph':continue
        graph_record=file_record(row['path'])
        for packet in packets:
            origin=dict(method=row['method'],course_id=course,book_set=row.get('book_set',[]),graph_record=graph_record,
                graph_entity_id=packet['right']['entity_id'],reference_entity_id=packet['left']['entity_id'])
            if packet['both_sides_have_source']:
                source_packets[packet['candidate_id']]=packet
                source_origins.append(dict(origin,candidate_id=packet['candidate_id']))
            else:
                try:alternative=prior.prepare_pair(packet)
                except ValueError:continue
                prior_packets[alternative['candidate_id']]=alternative
                prior_origins.append(dict(origin,candidate_id=alternative['candidate_id']))
    return ([source_packets[key] for key in sorted(source_packets)],
        [prior_packets[key] for key in sorted(prior_packets)],
        sorted(source_origins,key=digest),sorted(prior_origins,key=digest))


def _input_max(packets,*,single=False):
    maximum_user=maximum_messages=0;groups=defaultdict(list)
    def size(group):
        wire=dict(version=core.POLICY['version'],kind=core._kind(group[0]),course_id=group[0]['course_id'],
            pairs=[core._anonymous(packet) for packet in group]);wire['batch_id']=digest(wire)
        user=json.dumps(wire,ensure_ascii=False)
        messages=[dict(role='system',content=core.JUDGE),dict(role='user',content=user)]
        return len(user.encode('utf-8')),len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))
    rows=core.plan_batches(packets)
    if single:rows=[[packet] for group in rows for packet in group]
    for group in rows:
        user,messages=size(group);maximum_user=max(maximum_user,user);maximum_messages=max(maximum_messages,messages)
    primary_user,primary_messages=maximum_user,maximum_messages
    for packet in packets:
        encoded=json.dumps(core._anonymous(packet),ensure_ascii=False)
        groups[(packet['course_id'],core._kind(packet))].append((packet,len(encoded.encode('utf-8')),
            len(json.dumps(encoded,ensure_ascii=False).encode('utf-8'))))
    # 阳性子集反转后会重新组批，上界必须涵盖非相邻的最大输入。
    for group in groups.values():
        for cost in [1,2]:
            largest=[row[0] for row in sorted(group,key=lambda row:row[cost],reverse=True)[:1 if single else 8]]
            user,messages=size(largest);maximum_user=max(maximum_user,user);maximum_messages=max(maximum_messages,messages)
    return dict(maximum_user_json_utf8_bytes=maximum_user,maximum_messages_json_utf8_bytes=maximum_messages,
        primary_observed_user_json_utf8_bytes=primary_user,primary_observed_messages_json_utf8_bytes=primary_messages,
        upper_bound='any_positive_subset_regrouping_same_course_and_branch_at_most8',
        input_not_truncated=True,bytes_are_not_token_counts=True,max_tokens=8192)


def write_preparation(inventory_path,preflight_metadata_path,source32_plan_path,output,token_assets):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    inventory=read_document(inventory_path);audit=read_document(preflight_metadata_path)
    expected={r['graph']['path']:r['graph']['sha256'] for r in audit['rows']}
    input_records={}
    for row in inventory['graphs']:
        if file_record(row['path'])['sha256']!=expected[row['path']]:raise ValueError('Frozen inventory graph changed')
        if file_record(row['reference']['path'])['sha256']!=row['reference']['sha256']:
            raise ValueError('Frozen reference changed')
        for path in [row['path'],row['reference']['path'],row.get('evidence_path')]:
            if path:input_records[path]=file_record(path)
    source,prior_packets,source_origins,prior_origins=prepare_population(inventory)
    if len(source)!=7519 or len(prior_packets)!=200:raise ValueError('Frozen full candidate population changed')
    index=read_document(audit['candidate_index']['path'])
    if file_record(audit['candidate_index']['path'])['sha256']!=audit['candidate_index']['sha256']:
        raise ValueError('Candidate index changed')
    source_ids={r['candidate_id'] for r in index if r['typed_graph_eligible'] and r['identity_judgment_branch']=='two_sided_source'}
    prior_ids={r['candidate_id'] for r in index if r['typed_graph_eligible'] and r['identity_judgment_branch']=='prior_definition_identity'}
    if source_ids!={p['candidate_id'] for p in source} or prior_ids!={p['source_candidate_id'] for p in prior_packets}:
        raise ValueError('Regenerated candidates differ from frozen0API preflight')
    source_plan=read_document(source32_plan_path);prior_plan=prior24_plan(prior_packets)
    original119=verified_read(source_plan['population'])
    if core.development_plan(original119)['development_plan_id']!=source_plan['development_plan_id']:
        raise ValueError('Frozen32 selection or core changed')
    source32=verified_read(source_plan['selected_packets']);origin32=verified_read(source_plan['selected_origins'])
    if {p['candidate_id'] for p in source32}!=set(source_plan['candidate_ids']):raise ValueError('Frozen32 packets changed')
    selected_prior=[p for p in prior_packets if p['candidate_id'] in set(prior_plan['candidate_ids'])]
    groups={'source32':(source32,origin32),'source_full':(source,source_origins),
        'prior24':(selected_prior,[r for r in prior_origins if r['candidate_id'] in set(prior_plan['candidate_ids'])]),
        'prior_full':(prior_packets,prior_origins)}
    values={'source32_plan.json':source_plan,'prior24_plan.json':prior_plan,'input_inventory.json':inventory,
        'input_records.json':input_records,'source119_origins_server_only.json':verified_read(source_plan['origin_population'])}
    for name,(packets,origins) in groups.items():
        values[name+'_packets_server_only.json']=packets;values[name+'_origins_server_only.json']=origins
    files={}
    for name,value in values.items():
        write_immutable(output/name,value);files[name]=file_record(output/name)
    counter=packing.OfficialFlashCounter(token_assets);input_sizes={};requests={};token_audits={}
    for stage in STAGES:
        stem={'source32':'source32','source-full':'source_full','prior24-single':'prior24',
            'prior24-batch':'prior24','prior-full':'prior_full'}[stage]
        packets=groups[stem][0];single=stage=='prior24-single'
        def progress(value):
            if value['completed_original_groups']%16==0 or value['completed_original_groups']==value['total_original_groups']:
                atomic(output/'token_count_progress.json',dict(value,stage=stage,updated_unix=time.time(),model_calls=0))
        planned=packing.plan_batches(packets,counter,single=single,progress=progress)
        audits=[counter.audit(rows) for rows in planned];logical=2*len(planned)
        input_sizes[stage]=_input_max(packets,single=single)
        token_audits[stage]=dict(primary_requests=len(planned),all_positive_reversal_scenario_requests=logical,
            prompt_tokens_max=max(r['prompt_tokens'] for r in audits),
            total_tokens_with8192_output_and1024_wrapper_max=max(r['total_with_reserves'] for r in audits),
            working_context_tokens=262144,official_context_tokens=1048576,all_primary_and_reversed_full_groups_fit=True,
            runtime='every_actual_positive_subset_recounted_and_if_needed_split_before_HTTP',
            subset_independent_hard_request_limit=STAGES[stage]['http_cap'],
            tokenizer_assets=counter.assets,full_audits_server_only=True)
        write_immutable(output/(stage+'_token_audits_server_only.json'),audits)
        files[stage+'_token_audits_server_only.json']=file_record(output/(stage+'_token_audits_server_only.json'))
        requests[stage]=dict(STAGES[stage],actual_pairs=len(packets),
            logical_baseline_without_token_splits=STAGES[stage]['logical_max'],
            all_positive_token_packed_logical_requests=logical,
            logical_max_note='original_baseline_not_unconditional_after_token_packing_HTTP_cap_is_hard')
    names=['run_scope_identity_batch.py','scope_identity_batch.py','scope_identity.py','prior_definition_identity.py','scope_identity_packing.py',
        'scope_mapping_preflight.py','run_scope_identity_gate.py','native_provider.py','run_native_full.py']
    implementation={name:file_record(Path(__file__).with_name(name))['sha256'] for name in names}
    if implementation['scope_identity_batch.py']!='2c55fa3a97618324141b5e524d1675168918264245074422384ebe1304af92a2':
        raise ValueError('Frozen source32 batch core changed')
    manifest=dict(policy=POLICY,prior_policy=PRIOR_POLICY,files=files,implementation=implementation,input_sizes=input_sizes,
        token_packing_policy=packing.POLICY,token_audits=token_audits,token_assets_root=str(Path(token_assets)),
        requests=requests,full_mapping_logical_baseline=1936,development_logical_baseline=66,
        full_mapping_all_positive_packed_scenario=sum(requests[s]['all_positive_token_packed_logical_requests'] for s in ['source-full','prior-full']),
        source_http_caps_total=2224,prior_http_caps_total=148,model_calls=0,single_decisions_or_scores_read=False,
        original_graphs_modified=False,old_splits_modified=False,raw_text_exported=False,
        input_inventory=file_record(inventory_path),frozen_preflight_metadata=file_record(preflight_metadata_path))
    manifest['preparation_id']=digest(manifest);write_immutable(output/'manifest.json',manifest)
    return manifest


def prior24_plan(packets):
    if len({p['candidate_id'] for p in packets})!=len(packets) or any(core._kind(p)!='prior_definition_identity' for p in packets):
        raise ValueError('Unique prior-only candidate population required')
    groups=defaultdict(list)
    for packet in packets:groups[packet['course_id']].append(packet)
    selected=[];counts={}
    for course,rows in sorted(groups.items()):
        chosen=sorted(rows,key=lambda p:digest([42,PRIOR_POLICY['version'],course,p['candidate_id']]))[:8]
        selected.extend(chosen);counts[course]=len(chosen)
    result=dict(policy=PRIOR_POLICY,population_digest=digest(sorted(p['candidate_id'] for p in packets)),
        candidate_ids=sorted(p['candidate_id'] for p in selected),selected_per_course=counts,
        actual_pairs=len(selected),sufficient_for_declared24=len(selected)==24 and len(counts)==3,
        minimum_exact_decision_agreement=23,enabled_for_full_mapping=False)
    return dict(result,prior_plan_id=digest(result))


def compare_prior24(plan,single,batched):
    ids=plan['candidate_ids'];complete=valid=agreement=unsafe=0;reverse_ok=True
    for cid in ids:
        left=single.get(cid,{});right=batched.get(cid,{})
        complete+=left.get('status')=='completed' and right.get('status')=='completed'
        valid+=left.get('citation_valid') is True and right.get('citation_valid') is True
        a=(left.get('primary') or {}).get('decision');b=(right.get('primary') or {}).get('decision')
        agreement+=a in {'same_sense','different_sense','insufficient'} and a==b
        unsafe+=bool(right.get('eligible_identity_edge') and not left.get('eligible_identity_edge'))
        for row in [left,right]:
            if row.get('eligible_identity_edge') and not row.get('reversed_positive_confirmed'):reverse_ok=False
    return dict(prior_plan_id=plan['prior_plan_id'],fixed_denominator=24,actual_pairs=len(ids),
        completed_pairs=complete,valid_both_side_citation_pairs=valid,exact_decision_agreement=agreement,
        single_negative_to_batch_positive=unsafe,every_accepted_positive_reverse_confirmed=reverse_ok,
        passes_declared_gate=plan['sufficient_for_declared24'] and complete==valid==24 and agreement>=23 and unsafe==0 and reverse_ok,
        original_prior3_preserved=True,enabled_for_full_mapping=False,human_audited=False)


@contextmanager
def local_http_cap(provider,root,maximum):
    if type(maximum) is not int or not 1<=maximum<=2200:raise ValueError('Invalid frozen stage HTTP cap')
    runtime=getattr(provider,'runtime',None)
    if runtime is None:
        yield;return
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'reservations.json';original=runtime._record
    def record(key,stage,status,value):
        if status=='request_started':
            with FileLock(str(root/'reservations.lock'),timeout=30):
                state=read_document(path) if path.exists() else dict(maximum=maximum,reserved_http_attempts=0)
                if state['maximum']!=maximum:raise ValueError('Frozen stage HTTP cap changed')
                if state['reserved_http_attempts']>=maximum:raise LocalHTTPAllowanceExceeded('Frozen identity stage HTTP cap exhausted')
                state.update(reserved_http_attempts=state['reserved_http_attempts']+1,last_request_id=key,
                    counting='conservative_before_shared_global_attempt',updated_unix=time.time());atomic(path,state)
        return original(key,stage,status,value)
    runtime._record=record
    try:yield
    finally:runtime._record=original


def apply_negative_veto(origins,decisions,old_origins,old_decisions):
    def key(row):return (row['graph_record']['sha256'],row['graph_entity_id'],row['reference_entity_id'])
    negatives={key(r) for r in old_origins if r['candidate_id'] in old_decisions and
        not old_decisions[r['candidate_id']]['eligible_identity_edge']}
    vetoed={r['candidate_id'] for r in origins if key(r) in negatives}
    result=copy.deepcopy(decisions)
    for cid in vetoed:
        result[cid].update(eligible_identity_edge=False,veto='preserved_frozen_single_pair_negative')
    return result


def validate_stage_packets(stage,packets):
    if stage not in STAGES or not packets or any(core._kind(p)!=STAGES[stage]['kind'] for p in packets):
        raise ValueError('Source/prior stage population boundary changed')
    if len({p['candidate_id'] for p in packets})!=len(packets):raise ValueError('Duplicate stage candidate IDs')


def execute_batches(packets,origins,provider,root,job,*,single=False,tracker=None,counter=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);primary={};reverse={}
    by_id={p['candidate_id']:p for p in packets}
    if len(by_id)!=len(packets) or set(by_id)!={r['candidate_id'] for r in origins} or any(
            r['reference_entity_id']!=by_id[r['candidate_id']]['left']['entity_id'] or
            r['graph_entity_id']!=by_id[r['candidate_id']]['right']['entity_id'] for r in origins):
        raise ValueError('Stage origin endpoints differ from frozen packets')
    def run(rows,reverse_input):
        identity=digest(dict(ids=[p['candidate_id'] for p in rows],reverse=reverse_input,single=single))
        folder=root/'batches'/identity;path=folder/'result.json'
        if path.exists():return _read_checkpoint(path,job)
        provider.raise_if_fatal()
        try:
            audit=counter.audit(rows) if counter else None
            if audit and not audit['fits']:raise packing.OversizeIdentityPair('Actual batch exceeds frozen working context')
            with tracker.scope(folder/'requests.jsonl') if tracker else _no_scope():
                result=core.judge_batch(provider,rows,reverse=reverse_input)
            if audit:result['token_audit']=audit
        except Exception as exc:
            error=_error(folder,exc)
            atomic(folder/'attempt_failure.json',dict(error,job_identity=job))
            provider.raise_if_fatal()
            if isinstance(exc,(LocalHTTPAllowanceExceeded,NativeBudgetExceeded,NativeAccountUnavailable)):raise
            result=dict(candidate_ids=[p['candidate_id'] for p in rows],reverse=reverse_input,**error,
                decisions={p['candidate_id']:dict(status='failed',eligible_identity_edge=False,citation_valid=False,
                    error_type=type(exc).__name__) for p in rows})
        return _checkpoint(path,result,job)
    planned=packing.plan_batches(packets,counter,single=single) if counter else core.plan_batches(packets)
    if single and not counter:planned=[[p] for rows in planned for p in rows]
    for rows in planned:primary.update(run(rows,False)['decisions'])
    for original_group in planned:
        positives=[p for p in original_group if primary[p['candidate_id']]['eligible_identity_edge']]
        reversed_groups=packing.plan_batches(positives,counter,single=single) if counter else core.plan_batches(positives)
        if single and not counter:reversed_groups=[[p] for rows in reversed_groups for p in rows]
        for rows in reversed_groups:reverse.update(run(rows,True)['decisions'])
    decisions={}
    for cid in by_id:
        first=primary[cid];second=reverse.get(cid)
        valid=first['status']=='validated' and (not first['eligible_identity_edge'] or
            second is not None and second['status']=='validated')
        eligible=bool(first['eligible_identity_edge'] and second and second['eligible_identity_edge'])
        decisions[cid]=dict(candidate_id=cid,status='completed' if valid else 'failed',primary=first.get('judgment'),
            reverse=second,primary_record=first,citation_valid=valid,eligible_identity_edge=eligible,
            reversed_positive_confirmed=eligible)
    projection=identity_projection(origins,decisions);accepted={r['candidate_id'] for r in projection['mapping']}
    for cid,row in decisions.items():row['eligible_identity_edge']=cid in accepted
    provider.raise_if_fatal()
    return dict(candidate_count=len(packets),decisions=decisions,identity_projection=projection,
        raw_response_server_only=True,usable_for_KGE=False,original_graphs_modified=False,old_splits_modified=False)


def read_verified_checkpoint(path):
    value=read_document(path)
    if not value.get('job_identity'):raise ValueError('Checkpoint job identity missing')
    return _read_checkpoint(path,value['job_identity'])


def load_single119(root,ids):
    result={}
    for cid in ids:
        path=Path(root)/'decisions'/cid/'decision.json'
        if not path.exists():raise DependencyNotReady('Original119 single-pair decisions are not all present')
        row=read_verified_checkpoint(path)
        if row.get('candidate_id')!=cid or row.get('status')!='completed':
            raise DependencyNotReady('Original119 includes failed/incomplete decisions; no candidate may be skipped')
        result[cid]=row
    return result


def stage_inputs(stage,preparation):
    stem={'source32':'source32','source-full':'source_full','prior24-single':'prior24',
        'prior24-batch':'prior24','prior-full':'prior_full'}[stage]
    packets=read_document(preparation/(stem+'_packets_server_only.json'))
    origins=read_document(preparation/(stem+'_origins_server_only.json'))
    validate_stage_packets(stage,packets);return packets,origins


def dependencies(stage,preparation,output,single119_root):
    source_plan=read_document(preparation/'source32_plan.json');data={}
    if stage in {'source32','source-full'}:
        data['single119']=load_single119(single119_root,source_plan['single_population_candidate_ids'])
    def prerequisite(name):
        path=output/name/'result.json'
        if not path.exists():raise DependencyNotReady('Prerequisite stage missing: '+name)
        result=read_verified_checkpoint(path)
        if result.get('status')!='completed':raise DependencyNotReady('Prerequisite stage is incomplete: '+name)
        manifest=read_document(preparation/'manifest.json')
        if result.get('preparation_id')!=manifest['preparation_id']:raise ValueError('Prerequisite uses a different frozen population')
        return result
    if stage=='source-full':
        if not prerequisite('source32')['comparison']['passes_declared_gate']:raise DependencyNotReady('Source32 frozen threshold not passed')
    if stage=='prior24-batch':data['single_prior24']=prerequisite('prior24-single')['decisions']
    if stage=='prior-full':
        data['source_full']=prerequisite('source-full')
        if not prerequisite('prior24-batch')['comparison']['passes_declared_gate']:raise DependencyNotReady('Prior24 frozen threshold not passed')
    return data


def run_stage(stage,preparation,output,single119_root,provider,*,deps,contract,allowance_root):
    packets,origins=stage_inputs(stage,preparation);folder=output/stage;folder.mkdir(parents=True,exist_ok=True)
    job=digest(contract);write_immutable(folder/'contract.json',dict(contract,job_identity=job))
    tracker=ReceiptTracker(provider) if hasattr(provider,'request_executor') else None
    def state(status):atomic(folder/'state.json',dict(status=status,stage=stage,job_identity=job,
        updated_unix=time.time(),population=len(packets),global_budget_unchanged=True))
    try:
        if (folder/'result.json').exists():return _read_checkpoint(folder/'result.json',job)
        state('running')
        with local_http_cap(provider,allowance_root,STAGES[stage]['http_cap']):
            counter=packing.OfficialFlashCounter(read_document(preparation/'manifest.json')['token_assets_root'])
            result=execute_batches(packets,origins,provider,folder,job,single=stage=='prior24-single',tracker=tracker,counter=counter)
            if stage=='source32':
                result['comparison']=core.compare_development(read_document(preparation/'source32_plan.json'),
                    deps['single119'],result['decisions'])
            elif stage=='prior24-batch':
                result['comparison']=compare_prior24(read_document(preparation/'prior24_plan.json'),
                    deps['single_prior24'],result['decisions'])
            if stage=='source-full':
                old_origins=read_document(preparation/'source119_origins_server_only.json')
                result['decisions']=apply_negative_veto(origins,result['decisions'],old_origins,deps['single119'])
                result['identity_projection']=identity_projection(origins,result['decisions'])
            if stage=='prior-full':
                single=read_verified_checkpoint(output/'prior24-single/result.json')['decisions']
                old_origins=read_document(preparation/'prior24_origins_server_only.json')
                result['decisions']=apply_negative_veto(origins,result['decisions'],old_origins,single)
                result['identity_projection']=identity_projection(origins,result['decisions'])
                source_origins=read_document(preparation/'source_full_origins_server_only.json')
                result['combined_identity_projection']=identity_projection(source_origins+origins,
                    dict(deps['source_full']['decisions'],**result['decisions']))
            provider.raise_if_fatal()
            failed=sum(row['status']!='completed' for row in result['decisions'].values())
            passed=result.get('comparison',{}).get('passes_declared_gate',True)
            result.update(status='completed' if not failed and passed else 'failed_frozen_gate' if not passed else 'partial',
                stage=stage,preparation_id=read_document(preparation/'manifest.json')['preparation_id'],
                failed_pairs=failed,policy=POLICY,local_http_cap=STAGES[stage]['http_cap'],
                statistics=provider.statistics() if hasattr(provider,'statistics') else {},statistics_scope='shared_global_not_stage_cost',
                per_stage_cost='requests.jsonl_actual_receipt_ids_only',new_split_created=False)
            _checkpoint(folder/'result.json',result,job);state(result['status']);return result
    except (LocalHTTPAllowanceExceeded,NativeBudgetExceeded,NativeAccountUnavailable):
        state('halted_preserved_existing_budget');raise
    except Exception:
        state('failed_preserved_checkpoints');raise
    finally:
        if tracker:tracker.close()
        provider.close()


def validate_preparation(root):
    manifest=read_document(root/'manifest.json')
    if manifest['preparation_id']!=digest({k:v for k,v in manifest.items() if k!='preparation_id'}):
        raise ValueError('Frozen preparation manifest digest changed')
    for name,record in manifest['files'].items():
        if file_record(root/name)['sha256']!=record['sha256']:raise ValueError('Frozen preparation input changed: '+name)
    for name,sha in manifest['implementation'].items():
        if file_record(Path(__file__).with_name(name))['sha256']!=sha:raise ValueError('Frozen mapper implementation changed')
    packing.OfficialFlashCounter(manifest['token_assets_root'])
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-root','config','preparation-root','output-root','single119-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=sorted(STAGES),required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv);manifest=validate_preparation(args.preparation_root);config=read_document(args.config);check_runtime(config)
    if read_document(args.run_root/'formal_provider/limits.json')['limits']!=LIMITS:raise ValueError('Existing formal global limits changed')
    if file_record(Path(__file__).with_name('native_provider.py'))['sha256']!=FORMAL_PROVIDER_SHA:raise ValueError('Frozen formal provider required')
    packets,origins=stage_inputs(args.stage,args.preparation_root)
    contract=dict(policy=POLICY,stage=args.stage,limits=STAGES[args.stage],preparation_id=manifest['preparation_id'],
        packet_digest=digest(packets),origin_digest=digest(origins),config=file_record(args.config),
        implementation=manifest['implementation'],max_tokens=8192,input_max_bytes=manifest['input_sizes'][args.stage])
    try:deps=dependencies(args.stage,args.preparation_root,args.output_root,args.single119_root);ready=True
    except DependencyNotReady:deps={};ready=False
    contract['dependency_digest']=digest(deps)
    if args.check_only:
        print(json.dumps(dict(status='check_only',stage=args.stage,dependencies_ready=ready,model_calls=0,
            contract_id=digest(contract),limits=STAGES[args.stage],input_max_bytes=manifest['input_sizes'][args.stage])));return
    if not ready:
        atomic(args.output_root/args.stage/'state.json',dict(status='dependency_not_ready',stage=args.stage,model_calls=0))
        raise SystemExit(3)
    credentials()
    from ..construction import PROMPTS
    from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
    runtime=ModelRuntime(args.output_root/args.stage/'transport',PROMPTS,config)
    provider=AuditedNativeProvider(runtime,args.run_root/'formal_provider',**LIMITS)
    allowance=args.run_root/'identity_mapping_stage_allowance'/digest(dict(stage=args.stage,
        policy=POLICY['version'],population=digest(sorted(p['candidate_id'] for p in packets)),
        judge=digest(core.JUDGE),cap=STAGES[args.stage]['http_cap']))
    result=run_stage(args.stage,args.preparation_root,args.output_root,args.single119_root,provider,
        deps=deps,contract=contract,allowance_root=allowance)
    if result['status']!='completed':raise SystemExit(2)


if __name__=='__main__':main()
