"""Append the first execution of a zero-HTTP slot timeout; its UNKNOWN stays sealed."""
from __future__ import annotations

import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from urllib.error import URLError
from filelock import FileLock

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import identity_single_v2 as identity
from .run_native_full import _checkpoint,_read_checkpoint,_error,ReceiptTracker


POLICY=dict(version='zero_HTTP_slot_first_execution_preserved_UNKNOWN_v1',
    eligibility='only_original_600_second_slot_timeout_with_zero_actual_HTTP_and_no_raw',
    execution='same_original_primary_body_runtime_and_global_stage_budget',
    classification='original_UNKNOWN_and_error_receipt_unchanged_even_if_first_response_valid',
    identities='original_complete_population_contract_and_job_no_reset',
    retries='no_existing_actual_model_attempt_or_semantic_failure_reasked',
    accepted_identity_edges=False)


def actual_count(provider,key):
    with closing(sqlite3.connect('file:'+str(provider.database)+'?mode=ro',uri=True)) as db:
        return db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND key=?",(key,)).fetchone()[0]


def saved_native_response(provider,key,body):
    with closing(sqlite3.connect('file:'+str(provider.database)+'?mode=ro',uri=True)) as db:
        stored=db.execute('SELECT record FROM receipts WHERE key=?',(key,)).fetchone()
    if stored is None:raise ValueError('Paid first execution lacks a published native receipt; preserve evidence without reasking')
    record=json.loads(stored[0])
    if file_record(record['path'])!=record:raise ValueError('Original paid native receipt changed')
    value=read_document(record['path'])
    if (value['request_id']!=key or value['endpoint']!='/chat/completions' or value['request']!=body or
            value['runtime_config']!=provider.runtime.config):raise ValueError('Original paid native response identity changed')
    return value['response'],record


def complete_unattempted_as_unknown(packets,branch,root,candidate_id,provider,*,counter,execute=True):
    if branch not in {'prior','reference'}:raise ValueError('Only the existing bounded prior/reference populations are authorized')
    root=Path(root);contract=read_document(root/'contract.json');job=contract['job_identity'];folder=root/'decisions'/candidate_id
    if (job!=digest({k:v for k,v in contract.items() if k!='job_identity'}) or contract['packets_digest']!=digest(packets) or
            contract['branch']!=branch or contract['protocol']!=identity.POLICY or file_record(contract['implementation']['path'])!=contract['implementation'] or
            file_record(Path(identity.__file__))['sha256']!=contract['implementation']['sha256']):
        raise ValueError('Original full population, implementation or job identity changed')
    if (digest(provider.runtime.config)!=contract['runtime_config_digest'] or counter.assets!=contract['token_assets'] or
            getattr(provider,'_identity_single_attempt_cap',None)!=2 or
            getattr(provider,'_identity_single_total_cap',None)!=contract['maximum_cumulative_v2_actual_HTTP']):
        raise ValueError('Original runtime, token assets and cumulative caps required')
    packet=next((p for p in packets if p['candidate_id']==candidate_id),None)
    if packet is None or candidate_id!=digest({k:v for k,v in packet.items() if k!='candidate_id'}):raise ValueError('Unknown or changed original candidate')
    with FileLock(str(folder/'.infrastructure_completion.lock'),timeout=10):
        decision=_read_checkpoint(folder/'decision.json',job);decision_record=file_record(folder/'decision.json')
        if decision['classification']!='UNKNOWN' or decision.get('sealed_unknown') or decision.get('error_type')!='TimeoutError':
            raise ValueError('Only an original pre-execution infrastructure UNKNOWN can be completed')
        error=decision.get('error_record')
        if not error or file_record(error['path'])!=error or 'Native global request slots remained occupied for 600 seconds' not in Path(error['path']).read_text():
            raise ValueError('Original exact slot timeout proof required')
        body=identity.request_body(packet,branch);key=digest(dict(endpoint='/chat/completions',body=body,config=provider.runtime.config))
        history=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
        if not history or any(row['request_id']!=key for row in history):raise ValueError('Original exact primary request identity changed')
        receipt=folder/'infrastructure_completion.json'
        if receipt.exists():
            value=read_document(receipt)
            if (value.get('completion_id')!=digest({k:v for k,v in value.items() if k!='completion_id'}) or
                    value['original_decision']!=decision_record or value['original_contract']!=file_record(root/'contract.json') or
                    value['policy']!=POLICY or value['classification']!='UNKNOWN' or value['eligible_identity_edge'] is not False or
                    value['request_id']!=key or value['body_digest']!=digest(body) or value['original_error']!=error or
                    file_record(value['implementation']['path'])!=value['implementation'] or
                    file_record(value['pre_execution_proof']['path'])!=value['pre_execution_proof'] or
                    value['actual_HTTP_before']!=0 or value['actual_HTTP_after']!=actual_count(provider,key) or not 1<=value['actual_HTTP_after']<=2):
                raise ValueError('First-execution infrastructure receipt changed')
            for record in value['appended_artifacts'].values():
                if file_record(record['path'])!=record:raise ValueError('Appended original raw or failure receipt changed')
            if (folder/'primary.json').exists():_read_checkpoint(folder/'primary.json',job)
            return value
        formatted=copy.deepcopy(body['messages']);formatted[0]['response_format']=body['response_format']
        tokens=max(counter.count(body['messages']),counter.count(formatted));total=tokens+body['max_tokens']+1024
        if total>262144:raise ValueError('Original complete messages exceed fixed context; no truncation')
        proof=dict(policy=POLICY,original_decision=decision_record,original_contract=file_record(root/'contract.json'),
            original_error=error,candidate_id=candidate_id,request_id=key,actual_HTTP_before=0,
            classification='UNKNOWN',eligible_identity_edge=False,original_decision_changed=False,
            body_digest=digest(body),implementation=file_record(Path(__file__)))
        before=actual_count(provider,key);recover=before>0;preproof=folder/'infrastructure_first_execution_proof.json'
        if (folder/'reverse.json').exists():raise ValueError('An original model response must never be reasked')
        if recover:
            if before>2 or not preproof.exists() or read_document(preproof)!=proof:
                raise ValueError('An existing actual model attempt without this immutable zero-attempt proof must never be reasked')
            raw,native_receipt=saved_native_response(provider,key,body)
        elif (folder/'primary.json').exists():raise ValueError('A response without its actual attempt proof cannot be completed')
        if not execute:return dict(proof,status='check_only_saved_first_execution_recovery' if recover else 'check_only_zero_actual_slot_timeout',model_API_calls=0,raw_text_exported=False)
        write_immutable(folder/'infrastructure_first_execution_proof.json',proof)
        tracker=ReceiptTracker(provider);artifacts={}
        try:
            provider.raise_if_fatal()
            if recover:hit=True
            else:
                with tracker.scope(folder/'requests.jsonl'):
                    raw,hit=provider.request_executor('/chat/completions',body,'identity_single_v2.'+branch+'.primary')
            if actual_count(provider,key)<1:raise RuntimeError('No actual first execution; qualification remains blocked')
            native_raw,native_receipt=saved_native_response(provider,key,body)
            if native_raw!=raw:raise ValueError('Published native raw differs from first execution')
            if (folder/'primary.json').exists():
                existing=_read_checkpoint(folder/'primary.json',job)
                if existing['body_digest']!=digest(body) or existing['raw_response']!=raw:raise ValueError('Saved first execution raw differs from original native receipt')
            else:
                _checkpoint(folder/'primary.json',dict(body_digest=digest(body),raw_response=raw,cache_hit=hit,raw_server_only=True,
                    token_audit=dict(actual_messages_tokens=tokens,total_with_output_and1024_reserve=total,working_context=262144,truncation=False)),job)
            artifacts['primary']=file_record(folder/'primary.json')
            artifacts['native_receipt']=native_receipt
            try:identity.validate_raw(packet,branch,raw);validation='validated_kept_UNKNOWN_no_edge'
            except (ValueError,TypeError,KeyError,IndexError):validation='failed_kept_UNKNOWN'
        except (identity.RequestAttemptsExhausted,URLError,TimeoutError,ConnectionError) as exc:
            provider.raise_if_fatal()
            if actual_count(provider,key)==0:raise
            failure=_error(folder/'infrastructure_failure',exc);artifacts['failure']=failure['error_record'];validation='bounded_transport_failure_kept_UNKNOWN'
        finally:tracker.close()
        after=actual_count(provider,key)
        if not 1<=after<=2:raise RuntimeError('Actual cumulative attempts must remain within the original two-attempt cap')
        value=dict(proof,actual_HTTP_after=after,strict_validation=validation,appended_artifacts=artifacts,recovered_without_HTTP=recover,
            pre_execution_proof=file_record(folder/'infrastructure_first_execution_proof.json'),raw_text_exported=False)
        value['completion_id']=digest(value);write_immutable(receipt,value)
        if file_record(folder/'decision.json')!=decision_record:raise ValueError('Original UNKNOWN decision must stay byte-identical')
        return value


def main(argv=None):
    import argparse,time
    from types import SimpleNamespace
    from . import run_identity_single_v2 as thin
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-root','mapping-preparation','reference-preparation','output-root','tokenizer-manifest']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['prior','reference'],required=True);parser.add_argument('--candidate-id',required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--check-only',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv);packets,_,manifests=thin.original_inputs(args.mapping_preparation,args.reference_preparation)
    native=thin.prepare_plan(args.run_root,'hipporag','book_49bb8d31ddb9');config=read_document(native['construction']['files']['assets']['runtime_config']['path']);thin.check_runtime(config)
    if digest(config)!=thin.RUNTIME_CONFIG_ID:raise ValueError('Original runtime changed')
    counter=thin.OfficialMessageCounter(args.tokenizer_manifest,Path(manifests[0]['token_assets_root'])/'encoding/encoding_dsv4.py')
    snapshot,_,_=thin.ledger_snapshot(args.run_root)
    contract=read_document(args.output_root/args.stage/'contract.json')
    if contract['maximum_cumulative_v2_actual_HTTP']!=3000:raise ValueError('Only the existing reference/prior 3000 actual HTTP reservation is authorized')
    if not args.execute:
        provider=SimpleNamespace(runtime=SimpleNamespace(config=config),root=args.run_root/'formal_provider',database=args.run_root/'formal_provider/audit.sqlite',
            _identity_single_attempt_cap=2,_identity_single_total_cap=3000)
        value=complete_unattempted_as_unknown(packets[args.stage],args.stage,args.output_root/args.stage,args.candidate_id,provider,counter=counter,execute=False)
    else:
        if snapshot['halts'] or time.time()>=snapshot['deadline_unix'] or snapshot['remaining_80k']<=0:raise RuntimeError('Original global halt, cap or deadline prevents execution')
        thin.credentials();provider=thin.AuditedNativeProvider(thin.runtime_for(native['construction'],args.output_root/'infrastructure_transport'/args.candidate_id),args.run_root/'formal_provider',**thin.LIMITS)
        try:
            with identity.actual_attempt_cap(provider,2,total_http_cap=3000):
                value=complete_unattempted_as_unknown(packets[args.stage],args.stage,args.output_root/args.stage,args.candidate_id,provider,counter=counter)
        finally:provider.close()
    print(json.dumps({k:value[k] for k in ['status','candidate_id','completion_id','actual_HTTP_before','actual_HTTP_after','classification','eligible_identity_edge','strict_validation','raw_text_exported'] if k in value}))


if __name__=='__main__':main()
