"""新完整图的真实有界单对核验；旧判定原样继承，只请求新增候选。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager,closing
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import time

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import identity_single_v2 as identity,native_provider
from .identity_source_full_execution import bind_authorization,validate_actual_proofs,PROVIDER_SHA,START
from .run_identity_single_v2 import OfficialMessageCounter,RUNTIME_CONFIG_ID
from .run_native_full import LIMITS,_checkpoint
from .run_repaired_full import credentials
from .protocol import runtime_for
from .identity_single_v2_g0 import _manifest
from .identity_single_v2_full import prepare_full
from .run_native_full import prepare_plan
from .run_identity_source_full import SEMANTIC_SHA


def _payload(row):return {k:copy.deepcopy(v) for k,v in row.items() if k not in {'job_identity','checkpoint_digest'}}


def population_contract(packets,branch,provider,counter,sealed):
    contract=dict(protocol=identity.POLICY,branch=branch,packets_digest=digest(packets),sealed=sealed,
        implementation=file_record(Path(identity.__file__)),runtime_config_digest=digest(provider.runtime.config) if hasattr(provider,'runtime') else None,
        maximum_actual_HTTP_per_exact_body=getattr(provider,'_identity_single_attempt_cap',None),token_assets=counter.assets if counter else None,
        maximum_cumulative_v2_actual_HTTP=getattr(provider,'_identity_single_total_cap',None),worker_count=4,
        partition='frozen_candidate_order_index_mod_count_no_score_or_coverage')
    return dict(contract,job_identity=digest(contract))


def adopt_population(old_packets,new_packets,branch,old_root,new_root,provider,counter):
    old_root=Path(old_root);new_root=Path(new_root);old=identity.validate_saved_population(old_packets,branch,old_root)
    old_contract=read_document(old_root/'contract.json');sealed=old_contract['sealed'];old_index={p['candidate_id']:p for p in old_packets};new_index={p['candidate_id']:p for p in new_packets}
    if (not old_index.keys()<=new_index.keys() or len(new_index)!=len(new_packets) or
            any(new_index[cid]!=p for cid,p in old_index.items()) or any(p['candidate_id']!=digest({k:v for k,v in p.items() if k!='candidate_id'}) for p in new_packets)):
        raise ValueError('Every inherited original candidate and exact body must remain unchanged')
    contract=population_contract(new_packets,branch,provider,counter,sealed);write_immutable(new_root/'contract.json',contract);job=contract['job_identity']
    for cid in old_index:
        source=old_root/'decisions'/cid;destination=new_root/'decisions'/cid
        for name in ['primary','reverse']:
            path=source/(name+'.json')
            if path.exists():
                stored=identity._read_checkpoint(path,old_contract['job_identity'])
                if stored['body_digest']!=digest(identity.request_body(new_index[cid],branch,reverse=name=='reverse')):raise ValueError('Inherited original response body changed')
                _checkpoint(destination/(name+'.json'),dict(_payload(stored),inherited_raw=file_record(path)),job)
        row=old['decisions'][cid]
        _checkpoint(destination/'decision.json',dict(_payload(row),inherited_decision=file_record(source/'decision.json'),
            inherited_classification_unchanged=True,old_UNKNOWN_never_reasked=True),job)
    receipt=dict(branch=branch,old_result=file_record(old_root/'result.json'),old_contract=file_record(old_root/'contract.json'),
        new_contract=file_record(new_root/'contract.json'),old_candidate_count=len(old_packets),new_total_candidates=len(new_packets),
        additional_candidates=len(new_packets)-len(old_packets),inherited_packet_digest=digest(old_packets),all_old_classifications_unchanged=True,
        original_UNKNOWN_not_reasked=True,old_counts={k:old[k] for k in ['accepted_count','negative_count','insufficient_count','unknown_count']},model_calls=0)
    receipt['inherited_body_actual_attempts']={}
    if hasattr(provider,'runtime'):
        with closing(sqlite3.connect('file:'+str(provider.database)+'?mode=ro',uri=True)) as db:
            attempts=dict(db.execute("SELECT key,COUNT(*) FROM events WHERE status='request_started' GROUP BY key"))
        for packet in old_packets:
            for reverse in [False,True]:
                key=digest(dict(endpoint='/chat/completions',body=identity.request_body(packet,branch,reverse=reverse),config=provider.runtime.config))
                receipt['inherited_body_actual_attempts'][key]=attempts.get(key,0)
    write_immutable(new_root/'adoption_receipt.json',receipt);return receipt


def validate_adoption(old_packets,branch,root,*,database=None,config=None):
    root=Path(root);receipt=read_document(root/'adoption_receipt.json');contract=read_document(root/'contract.json');old_root=Path(receipt['old_result']['path']).parent
    for record in [receipt['old_result'],receipt['old_contract'],receipt['new_contract']]:
        if file_record(record['path'])!=record:raise ValueError('Inherited original result/contract lineage changed')
    if receipt['inherited_packet_digest']!=digest(old_packets) or receipt['old_candidate_count']!=len(old_packets):raise ValueError('Inherited original population changed')
    old_contract=read_document(old_root/'contract.json')
    for packet in old_packets:
        cid=packet['candidate_id'];old_path=old_root/'decisions'/cid/'decision.json';new_path=root/'decisions'/cid/'decision.json'
        previous=identity._read_checkpoint(old_path,old_contract['job_identity']);current=identity._read_checkpoint(new_path,contract['job_identity']);payload=_payload(current)
        if payload.pop('inherited_decision',None)!=file_record(old_path):raise ValueError('Inherited original decision receipt changed')
        if payload.pop('inherited_classification_unchanged',None) is not True or payload.pop('old_UNKNOWN_never_reasked',None) is not True or payload!=_payload(previous):
            raise ValueError('Original classification or UNKNOWN was changed after adoption')
        for name in ['primary','reverse']:
            old=old_path.parent/(name+'.json');new=new_path.parent/(name+'.json')
            if old.exists()!=new.exists():raise ValueError('An original unresolved direction was reasked or an original raw response removed')
            if old.exists():
                payload=_payload(identity._read_checkpoint(new,contract['job_identity']))
                if payload.pop('inherited_raw',None)!=file_record(old) or payload!=_payload(identity._read_checkpoint(old,old_contract['job_identity'])):
                    raise ValueError('Inherited original raw changed')
    counts=receipt['inherited_body_actual_attempts']
    if contract['runtime_config_digest'] is not None:
        if database is None or config is None or digest(config)!=contract['runtime_config_digest']:raise ValueError('Complete original-body actual-attempt audit required')
        keys={digest(dict(endpoint='/chat/completions',body=identity.request_body(p,branch,reverse=r),config=config)) for p in old_packets for r in [False,True]}
        if set(counts)!=keys:raise ValueError('Inherited original-body actual-attempt denominator changed')
    if database is not None:
        with closing(sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)) as db:
            actual=dict(db.execute("SELECT key,COUNT(*) FROM events WHERE status='request_started' GROUP BY key"))
        if any(actual.get(key,0)!=count for key,count in counts.items()):raise ValueError('An inherited original body received additional actual HTTP')
    return dict(branch=branch,inherited_candidates=len(old_packets),all_original_classifications_and_raw_unchanged=True,original_HTTP_not_reasked=True,receipt=file_record(root/'adoption_receipt.json'))


@contextmanager
def append_attempt_guard(provider):
    original=provider.request_executor
    def request(endpoint,body,stage):
        if stage not in {'identity_single_v2.'+b+'.'+d for b in ['source','prior'] for d in ['primary','reverse']}:
            raise ValueError('Append controller cannot spend an unrelated allocation')
        provider.raise_if_fatal();return original(endpoint,body,stage)
    with identity.actual_attempt_cap(provider,2):
        provider._identity_single_total_cap=33000;provider.request_executor=request
        try:yield
        finally:provider.request_executor=original


def _remaining(provider,watermark):
    with closing(sqlite3.connect('file:'+str(provider.database)+'?mode=ro',uri=True)) as db:
        total=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started'").fetchone()[0]
        local=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND id>? AND stage LIKE 'identity_single_v2.%'",(watermark,)).fetchone()[0]
    return max(0,min(33000-local,200000-total)),local


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','worker','finalize'])
    for key in ['run-root','append-plan','previous-preparation','mapping-preparation','reference-preparation','previous-processed','processed','parent-package','output','tokenizer-manifest']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker-index',type=int,default=0);args=parser.parse_args(argv)
    plan=read_document(args.append_plan);manifest=_manifest(args.mapping_preparation);old_manifest=_manifest(args.previous_preparation)
    if file_record(args.mapping_preparation/'manifest.json')!=plan['preparation'] or manifest['previous_preparation']!=file_record(args.previous_preparation/'manifest.json'):
        raise ValueError('Exact frozen append plan and original preparation required')
    if plan['original_package']!=file_record(args.parent_package):raise ValueError('Inherited parent package changed')
    if file_record(Path(native_provider.__file__))['sha256']!=PROVIDER_SHA:raise ValueError('Exact approved provider required')
    if file_record(Path(identity.__file__))['sha256']!=SEMANTIC_SHA:raise ValueError('Original frozen semantic judge implementation changed')
    native=prepare_plan(args.run_root,'hipporag','book_49bb8d31ddb9');config=read_document(native['construction']['files']['assets']['runtime_config']['path'])
    if digest(config)!=RUNTIME_CONFIG_ID:raise ValueError('Original runtime changed')
    packets={b:read_document(args.mapping_preparation/(b+'_full_packets_server_only.json')) for b in ['source','prior']}
    counter=OfficialMessageCounter(args.tokenizer_manifest,Path(manifest['token_assets_root'])/'encoding/encoding_dsv4.py')
    credentials();provider=native_provider.AuditedNativeProvider(runtime_for(native['construction'],args.processed/'transport'/str(args.worker_index)),args.run_root/'formal_provider',**LIMITS)
    try:
        allocation=bind_authorization(provider,args.run_root);provider.raise_if_fatal()
        if args.mode=='prepare':
            remaining,spent=_remaining(provider,allocation['watermark_event_id']);required=plan['required_cap2_upper']
            if required>remaining:
                print(json.dumps(dict(status='append_budget_shortfall_no_HTTP',required_cap2_upper=required,available_actual_HTTP=remaining,actual_HTTP_shortfall=required-remaining,model_calls=0,raw_text_exported=False)));return 2
            adoptions={}
            with append_attempt_guard(provider):
                for branch in ['source','prior']:
                    old_packets=read_document(args.previous_preparation/(branch+'_full_packets_server_only.json'))
                    validate_actual_proofs(old_packets,args.previous_processed/branch,config,provider.database,read_document(args.previous_processed/branch/'contract.json'),branch=branch)
                    adoptions[branch]=adopt_population(old_packets,packets[branch],branch,args.previous_processed/branch,args.processed/branch,provider,counter)
            destination=args.processed/'reference'
            if not destination.exists():shutil.copytree(args.previous_processed/'reference',destination)
            reference_packets=read_document(args.reference_preparation/'full_packets_server_only.json');identity.validate_saved_population(reference_packets,'reference',destination)
            sealed=read_document(args.previous_processed/'sealed_unknown.json');write_immutable(args.processed/'sealed_unknown.json',sealed)
            authorized=dict(version='identity_append_authorized_execution_v1',allocation=allocation,source_candidate_count=len(packets['source']),
                source_packets_digest=digest(packets['source']),prior_packets_digest=digest(packets['prior']),worker_count=4,
                previous_package=file_record(args.parent_package),previous_preparation=file_record(args.previous_preparation/'manifest.json'),
                new_preparation=file_record(args.mapping_preparation/'manifest.json'),append_plan=file_record(args.append_plan),adoptions=adoptions,
                executor=file_record(Path(__file__)),original_semantic_protocol=file_record(Path(identity.__file__)),reference_result=file_record(destination/'result.json'))
            authorized['execution_identity']=digest(authorized);write_immutable(args.processed/'source_execution_authorization.json',authorized)
            print(json.dumps(dict(status='append_prepared_all_old_classifications_and_original_raw_inherited',old_candidate_counts={b:r['old_candidate_count'] for b,r in adoptions.items()},model_calls=0,raw_text_exported=False)));return 0
        authorized=read_document(args.processed/'source_execution_authorization.json')
        if authorized['execution_identity']!=digest({k:v for k,v in authorized.items() if k!='execution_identity'}) or authorized['append_plan']!=file_record(args.append_plan):raise ValueError('Append authorized execution changed')
        if args.mode=='worker':
            with append_attempt_guard(provider):
                result={b:identity.run_population(packets[b],b,provider,args.processed/b,sealed=read_document(args.processed/'sealed_unknown.json') if b=='source' else {},
                    counter=counter,worker_index=args.worker_index,worker_count=4) for b in ['source','prior']}
            print(json.dumps(dict(status='append_partition_completed',worker_index=args.worker_index,partition_counts={b:r['partition_processed'] for b,r in result.items()},raw_text_exported=False)));return 0
        proofs={}
        for branch in ['source','prior']:
            proofs[branch]=validate_actual_proofs(packets[branch],args.processed/branch,config,provider.database,read_document(args.processed/branch/'contract.json'),branch=branch)
            identity.finalize_population(packets[branch],branch,args.processed/branch)
        write_immutable(args.processed/'append_complete_actual_proofs.json',dict(proofs,execution=file_record(args.processed/'source_execution_authorization.json')))
        package=prepare_full(args.mapping_preparation,args.reference_preparation,args.processed,args.output,database=provider.database,config=config,parent_package=args.parent_package)
        print(json.dumps(dict(status='append_complete_qualified_current_inventory',package_id=package['package_id'],package=file_record(args.output/'package.json'),
            candidate_counts=package['original_population_counts'],training_jobs=package['existing_training_matrix']['total_jobs'],new_candidates_qualified=True,raw_text_exported=False)));return 0
    finally:provider.close()


if __name__=='__main__':raise SystemExit(main())
