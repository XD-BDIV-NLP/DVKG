"""独立reference↔reference三阶段有限资格；默认check-only，共享原正式账本。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import sys

from filelock import FileLock

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from ..splits import canonical_reference
from . import reference_identity as ref
from . import scope_identity_batch as core
from . import scope_identity_packing as packing
from . import run_scope_identity_batch as batch
from .native_provider import AuditedNativeProvider
from .protocol import check_runtime
from .run_native_full import LIMITS,ReceiptTracker,_checkpoint,_read_checkpoint
from .run_repaired_full import atomic,credentials
from .identity_response import parser_bridge,POLICY as PARSER_POLICY


PROVIDER_SHA='0d0b7b23ff5c3c12e0fba2f59646a939b9f8b50692e8039583efe033df09b102'
MAX_HTTP=500
STAGES=('development-single','development-batch','full')
DependencyNotReady=batch.DependencyNotReady


def preserve_single_negatives(decisions,single):
    result=copy.deepcopy(decisions)
    for cid,row in single.items():
        if not row['eligible_identity_edge'] and cid in result:
            result[cid].update(eligible_identity_edge=False,veto='preserved_reference_development_single_negative')
    return result


class StageProvider:
    def __init__(self,provider,stage):self.provider=provider;self.stage=stage
    def raise_if_fatal(self):return self.provider.raise_if_fatal()
    def chat_response(self,messages,**kwargs):
        kwargs['stage']='reference_identity.'+self.stage+'.'+kwargs.get('stage','judge')
        return self.provider.chat_response(messages,**kwargs)


def execute_reference_batches(packets,origins,provider,root,job,*,single=False,tracker=None,counter=None,stage='offline'):
    result=batch.execute_batches(packets,origins,StageProvider(provider,stage),root,job,
        single=single,tracker=tracker,counter=counter)
    # reference等价是无向关系；原图映射的一对多冲突规则不用于这里。
    result['intermediate_directional_projection_not_used']=result.pop('identity_projection')
    for row in result['decisions'].values():
        primary=row.get('primary_record',{});reverse=row.get('reverse') or {}
        eligible=bool(row['status']=='completed' and primary.get('eligible_identity_edge') and reverse.get('eligible_identity_edge'))
        row.update(eligible_identity_edge=eligible,reversed_positive_confirmed=eligible)
    result.update(identity_components_not_yet_frozen=True,graph_inputs=False,usable_for_KGE=False)
    return result


def stage_dependencies(stage,output,preparation_id):
    def read(name):
        path=Path(output)/name/'result.json'
        if not path.exists():raise DependencyNotReady('Reference development prerequisite missing: '+name)
        row=batch.read_verified_checkpoint(path)
        if row.get('status')!='completed' or row.get('preparation_id')!=preparation_id:
            raise DependencyNotReady('Reference development prerequisite incomplete: '+name)
        return row
    if stage=='development-single':return {}
    single=read('development-single')
    if stage=='development-batch':return dict(single=single)
    if stage!='full':raise ValueError('Unknown reference qualification stage')
    batched=read('development-batch')
    if batched.get('comparison',{}).get('passes_declared_gate') is not True:
        raise DependencyNotReady('Reference batch consistency gate did not pass')
    return dict(single=single,batched=batched)


def write_preparation(mapping_preparation,output,*,expected_count=485):
    mapping_preparation=Path(mapping_preparation);output=Path(output)
    old=read_document(mapping_preparation/'manifest.json')
    if old['preparation_id']!=digest({k:v for k,v in old.items() if k!='preparation_id'}):raise ValueError('Original mapping preparation changed')
    for name in ('input_inventory.json','input_records.json'):
        if file_record(mapping_preparation/name)['sha256']!=old['files'][name]['sha256']:raise ValueError('Frozen reference source inventory changed')
    inputs=read_document(mapping_preparation/'input_records.json');inventory=read_document(mapping_preparation/'input_inventory.json')
    references={};evidence_paths=defaultdict(set)
    for row in inventory['graphs']:
        course=row['course_id'];record=row['reference']
        if course in references and references[course]!=record:raise ValueError('Conflicting course reference records')
        references[course]=record
        if row.get('evidence_path'):evidence_paths[course].add(row['evidence_path'])
    packets=[];origins=[];source_records={}
    for course,record in sorted(references.items()):
        if file_record(record['path'])!=record:raise ValueError('Independent reference changed')
        evidence={}
        for path in sorted(evidence_paths[course]):
            if file_record(path)!=inputs[path]:raise ValueError('Original reference source evidence changed')
            source_records[path]=inputs[path]
            for eid,value in read_document(path).items():
                for key in {eid,value.get('native_id'),value.get('id')} - {None,''}:
                    if key in evidence:
                        from .scope_identity import _text
                        if _text(evidence[key])!=_text(value):raise ValueError('Conflicting original source evidence identity')
                    evidence[key]=value
        canonical=canonical_reference(read_document(record['path']));rows={r['entity_id']:r for r in canonical['entities']}
        for left,right in ref.candidate_pairs(canonical['entities']):
            packet=ref.prepare_pair(course,rows[left],rows[right],evidence);packets.append(packet)
            origins.append(dict(candidate_id=packet['candidate_id'],course_id=course,graph_record=record,
                graph_entity_id=right,reference_entity_id=left,left_reference_entity_id=left,right_reference_entity_id=right,
                provenance='two_independent_reference_source_definitions'))
    packets.sort(key=lambda p:p['candidate_id']);origins.sort(key=digest)
    if len(packets)!=expected_count:raise ValueError('Independent reference candidate population count changed')
    plan=ref.development_plan(packets)
    if not plan['sufficient_for_declared24']:raise ValueError('Cannot silently reduce the declared reference24 development population')
    selected=set(plan['candidate_ids'])
    values={'full_packets_server_only.json':packets,'full_origins_server_only.json':origins,
        'development_packets_server_only.json':[p for p in packets if p['candidate_id'] in selected],
        'development_origins_server_only.json':[r for r in origins if r['candidate_id'] in selected],
        'development24_plan.json':plan,'reference_records.json':references,'source_records.json':source_records}
    for name,value in values.items():write_immutable(output/name,value)
    counter=packing.OfficialFlashCounter(old['token_assets_root']);audits={};logical=0
    for stage in STAGES:
        work=packets if stage=='full' else values['development_packets_server_only.json']
        groups=packing.plan_batches(work,counter,single=stage=='development-single',progress=lambda row:
            atomic(output/('token_progress_'+stage+'.json'),dict(row,model_calls=0)))
        group_audits=[counter.audit(group) for group in groups]
        byte_counts=[]
        for group in groups:
            for reverse in (False,True):
                messages=packing.messages_for(group,reverse=reverse)
                byte_counts.append(len(json.dumps(messages,ensure_ascii=False).encode()))
        audits[stage]=dict(candidate_count=len(work),primary_groups=len(groups),all_positive_primary_and_reverse_logical=2*len(groups),
            maximum_messages_utf8_bytes=max(byte_counts),maximum_prompt_tokens=max(r['prompt_tokens'] for r in group_audits),
            maximum_tokens_with_reserves=max(r['total_with_reserves'] for r in group_audits),all_groups_fit=all(r['fits'] for r in group_audits),
            token_packing_policy=packing.POLICY,max_tokens=8192,truncation=False)
        logical+=2*len(groups)
    implementation={name:file_record(Path(__file__).with_name(name))['sha256'] for name in [
        'run_reference_identity.py','reference_identity.py','scope_identity_batch.py','scope_identity_packing.py',
        'scope_identity.py','run_scope_identity_batch.py','identity_response.py','native_provider.py']}
    manifest=dict(policy=ref.POLICY,files={name:file_record(output/name) for name in values},references=references,
        source_records=source_records,token_assets_root=old['token_assets_root'],token_audits=audits,
        all_positive_three_stage_logical=logical,maximum_actual_shared_local_HTTP=MAX_HTTP,
        development_count=24,full_count=len(packets),implementation=implementation,
        model_calls=0,training_started=False,original_mapping_preparation=file_record(mapping_preparation/'manifest.json'))
    manifest['preparation_id']=digest(manifest);write_immutable(output/'manifest.json',manifest);return manifest


def validate_preparation(preparation):
    preparation=Path(preparation);manifest=read_document(preparation/'manifest.json')
    if manifest['preparation_id']!=digest({k:v for k,v in manifest.items() if k!='preparation_id'}):raise ValueError('Reference preparation manifest changed')
    for name,record in manifest['files'].items():
        if file_record(preparation/name)['sha256']!=record['sha256']:raise ValueError('Reference preparation file changed')
    for record in list(manifest['references'].values())+list(manifest['source_records'].values()):
        if file_record(record['path'])!=record:raise ValueError('Original independent reference/source changed')
    for name,sha in manifest['implementation'].items():
        if file_record(Path(__file__).with_name(name))['sha256']!=sha:raise ValueError('Frozen reference qualification implementation changed')
    if manifest['implementation']['native_provider.py']!=PROVIDER_SHA:raise ValueError('Verified 0d0 native provider required')
    return manifest


def run_stage(stage,preparation,output,provider,*,deps,manifest,allowance_root):
    preparation=Path(preparation);output=Path(output);folder=output/stage
    stem='full' if stage=='full' else 'development'
    packets=read_document(preparation/(stem+'_packets_server_only.json'))
    origins=read_document(preparation/(stem+'_origins_server_only.json'))
    contract=dict(stage=stage,preparation_id=manifest['preparation_id'],policy=ref.POLICY,
        dependencies={k:digest(v) for k,v in deps.items()},maximum_local_actual_HTTP_all_stages=MAX_HTTP)
    job=digest(contract);write_immutable(folder/'contract.json',dict(contract,job_identity=job))
    tracker=ReceiptTracker(provider) if hasattr(provider,'request_executor') else None
    try:
        if (folder/'result.json').exists():return _read_checkpoint(folder/'result.json',job)
        atomic(folder/'state.json',dict(status='running',stage=stage,job_identity=job))
        with batch.local_http_cap(provider,allowance_root,MAX_HTTP):
            counter=packing.OfficialFlashCounter(manifest['token_assets_root'])
            with parser_bridge():
                result=execute_reference_batches(packets,origins,provider,folder,job,single=stage=='development-single',
                    tracker=tracker,counter=counter,stage=stage)
            if stage=='development-batch':result['comparison']=ref.compare_development(
                read_document(preparation/'development24_plan.json'),deps['single']['decisions'],result['decisions'])
            if stage=='full':
                result['decisions']=preserve_single_negatives(result['decisions'],deps['single']['decisions'])
            failed=sum(r['status']!='completed' for r in result['decisions'].values())
            passed=result.get('comparison',{}).get('passes_declared_gate',True)
            if stage=='full' and not failed:result['reference_components']=ref.freeze_components(origins,result['decisions'])
            result.update(status='completed' if not failed and passed else 'partial' if failed else 'failed_frozen_gate',
                stage=stage,preparation_id=manifest['preparation_id'],failed_pairs=failed,policy=ref.POLICY,
                response_shape_parser=PARSER_POLICY,
                shared_local_HTTP_cap=MAX_HTTP,old_mapping_stages_modified=False,
                statistics=provider.statistics() if hasattr(provider,'statistics') else {},statistics_scope='shared_global_not_this_stage_cost')
            provider.raise_if_fatal();_checkpoint(folder/'result.json',result,job)
            atomic(folder/'state.json',dict(status=result['status'],stage=stage,job_identity=job));return result
    except Exception:
        atomic(folder/'state.json',dict(status='failed_preserved_checkpoints',stage=stage,job_identity=job));raise
    finally:
        if tracker:tracker.close()
        provider.close()


def load_qualification(preparation,output):
    from .run_identity_kge import validate_completed_decision
    preparation=Path(preparation);output=Path(output);manifest=validate_preparation(preparation)
    stages={}
    for stage in STAGES:
        path=output/stage/'result.json'
        if not path.exists():raise DependencyNotReady('Independent reference qualification stage missing: '+stage)
        result=batch.read_verified_checkpoint(path)
        if (result.get('status')!='completed' or result.get('stage')!=stage or result.get('failed_pairs')!=0 or
                result.get('preparation_id')!=manifest['preparation_id']):
            raise DependencyNotReady('Independent reference qualification stage incomplete: '+stage)
        stem='full' if stage=='full' else 'development'
        packets=read_document(preparation/(stem+'_packets_server_only.json'))
        if set(result['decisions'])!={p['candidate_id'] for p in packets}:raise ValueError('Reference qualification fixed population differs')
        if result.get('candidate_count')!=len(packets):raise ValueError('Reference qualification candidate count differs')
        for packet in packets:validate_completed_decision(packet,result['decisions'][packet['candidate_id']])
        stages[stage]=result
    comparison=ref.compare_development(read_document(preparation/'development24_plan.json'),
        stages['development-single']['decisions'],stages['development-batch']['decisions'])
    if comparison!=stages['development-batch'].get('comparison') or not comparison['passes_declared_gate']:
        raise DependencyNotReady('Independent reference24 consistency threshold not passed')
    full=stages['full']['decisions'];single=stages['development-single']['decisions']
    if preserve_single_negatives(full,single)!=full:raise ValueError('Reference single-pair negative veto was not retained')
    origins=read_document(preparation/'full_origins_server_only.json');components=ref.freeze_components(origins,full)
    if stages['full'].get('reference_components')!=components:raise ValueError('Frozen independent reference components differ')
    result=dict(status='completed_reference_identity_qualification',policy=ref.POLICY,preparation_id=manifest['preparation_id'],
        components=components['components'],rejected_components=components['rejected_components'],
        original_reference_digests={c:digest(read_document(r['path'])) for c,r in manifest['references'].items()},
        reference_records=manifest['references'],fixed_candidate_count=len(full),failed_pairs=0,
        development_comparison=comparison,reference_component_id=components['reference_component_id'],
        stage_receipts={stage:file_record(output/stage/'result.json') for stage in STAGES},
        identity_projection_only=True,original_facts_removed=0,graph_inputs=False,human_audited=False)
    result['qualification_id']=digest(result);return result


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='prepare':
        parser=argparse.ArgumentParser(description='Original server only; no API/tokenizer-host upload')
        parser.add_argument('--mapping-preparation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
        args=parser.parse_args(argv[1:]);value=write_preparation(args.mapping_preparation,args.output)
        print(json.dumps(dict(status='prepared_no_API',preparation_id=value['preparation_id'],token_audits=value['token_audits'],
            maximum_actual_local_HTTP=MAX_HTTP,all_positive_three_stage_logical=value['all_positive_three_stage_logical'])));return
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-root','config','preparation-root','output-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=STAGES,required=True);parser.add_argument('--start',action='store_true')
    args=parser.parse_args(argv);manifest=validate_preparation(args.preparation_root)
    config=read_document(args.config);check_runtime(config)
    if read_document(args.run_root/'formal_provider/limits.json')['limits']!=LIMITS:raise ValueError('Existing formal budget cannot change')
    try:deps=stage_dependencies(args.stage,args.output_root,manifest['preparation_id']);ready=True
    except DependencyNotReady:deps={};ready=False
    if not args.start:
        print(json.dumps(dict(status='check_only',stage=args.stage,dependencies_ready=ready,maximum_actual_local_HTTP_all_stages=MAX_HTTP,
            preparation_id=manifest['preparation_id'],model_calls=0,training_started=False)));return
    if not ready:raise SystemExit('Independent reference qualification development gate incomplete; no model call')
    allowance=args.run_root/'reference_identity_qualification_allowance'/digest(dict(
        preparation_id=manifest['preparation_id'],policy=ref.POLICY,maximum_actual_HTTP=MAX_HTTP))
    args.output_root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(args.output_root/'single_reference_worker.lock'),timeout=0):
        credentials()
        from ..construction import PROMPTS
        from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
        runtime=ModelRuntime(args.output_root/args.stage/'transport',PROMPTS,config)
        provider=AuditedNativeProvider(runtime,args.run_root/'formal_provider',**LIMITS)
        result=run_stage(args.stage,args.preparation_root,args.output_root,provider,deps=deps,manifest=manifest,allowance_root=allowance)
    if result['status']!='completed':raise SystemExit(2)


if __name__=='__main__':main()
