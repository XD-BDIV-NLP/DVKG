"""原始source7519单对核验：新授权薄入口，默认零API，四分区只由主进程汇总。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import identity_single_v2 as identity,identity_source_full_execution as execution,native_provider
from .run_identity_single_v2 import original_inputs,OfficialMessageCounter,RUNTIME_CONFIG_ID
from .run_native_full import prepare_plan,LIMITS
from .run_repaired_full import credentials
from .protocol import runtime_for,check_runtime


PREFLIGHT_SHA='c22082f56197134965e03f33aa1d1c8355e9e8da432c1dbc9fe62d0c07772024'
SEMANTIC_SHA='96d7f43115e42a90ee1d06d1b6b0ee90402fd47f9da8529743b1d70f7351a6b8'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-root','mapping-preparation','reference-preparation','output-root','tokenizer-manifest','preflight']:
        parser.add_argument('--'+name,type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--check-only',action='store_true');mode.add_argument('--start',action='store_true');mode.add_argument('--finalize',action='store_true')
    parser.add_argument('--worker-index',type=int,default=0)
    args=parser.parse_args(argv)
    if file_record(Path(native_provider.__file__))['sha256']!=execution.PROVIDER_SHA:raise ValueError('Root-approved immutable provider bytes required')
    if file_record(Path(identity.__file__))['sha256']!=SEMANTIC_SHA:raise ValueError('Original single-pair semantic protocol changed')
    if file_record(args.preflight)['sha256']!=PREFLIGHT_SHA:raise ValueError('Reviewed zero-API complete source cache/token audit changed')
    preflight=read_document(args.preflight)
    if (preflight['model_calls']!=0 or preflight['branches']['source']['candidate_count']!=7519 or
            preflight['branches']['source']['total_with_output_and1024_reserve_max']>262144 or
            preflight['execution_actual_HTTP_cap2_upper']>33000):raise ValueError('Complete original source preflight not within approved execution bound')
    packets,origins,manifests=original_inputs(args.mapping_preparation,args.reference_preparation)
    native=prepare_plan(args.run_root,'hipporag','book_49bb8d31ddb9');config=read_document(native['construction']['files']['assets']['runtime_config']['path']);check_runtime(config)
    if digest(config)!=RUNTIME_CONFIG_ID:raise ValueError('Original runtime changed')
    sealed=read_document(args.output_root/'sealed_unknown.json')
    if set(sealed)!={'4a8e483da824d7334f9a37997ae8c4d325e2cc94963369072153f6ac732480d1'}:raise ValueError('Exact original exhausted source case must remain sealed')
    for cid,evidence in sealed.items():
        if identity.sealed_unknown(evidence['failed_decision'],evidence['repair_attempts'])!=evidence:raise ValueError('Original exhausted case proof changed')
    counter=OfficialMessageCounter(args.tokenizer_manifest,Path(manifests[0]['token_assets_root'])/'encoding/encoding_dsv4.py')
    credentials();provider=native_provider.AuditedNativeProvider(runtime_for(native['construction'],args.output_root/'source_transport'/str(args.worker_index)),args.run_root/'formal_provider',**LIMITS)
    try:
        allocation=execution.bind_authorization(provider,args.run_root);provider.raise_if_fatal()
        operational=dict(version='identity_source_full_authorized_execution_v1',allocation=allocation,
            semantic_protocol=file_record(Path(identity.__file__)),provider=file_record(Path(native_provider.__file__)),
            executor=file_record(Path(__file__)),guard=file_record(Path(execution.__file__)),
            historical_policy_budget_field='preserved_history_overridden_only_for_operational_execution_by_new_explicit_human_authorization',
            mapping_manifest=file_record(args.mapping_preparation/'manifest.json'),reference_manifest=file_record(args.reference_preparation/'manifest.json'),
            tokenizer=counter.assets,preflight=file_record(args.preflight),sealed=file_record(args.output_root/'sealed_unknown.json'),
            source_candidate_count=7519,source_packets_digest=digest(packets['source']),worker_count=4,
            old_reference_prior_753_HTTP_inherited=True,reference_only_split_unchanged=True)
        operational['execution_identity']=digest(operational)
        write_immutable(args.output_root/'source_execution_authorization.json',operational)
        if args.start:
            with execution.source_execution_guard(provider,allocation):
                result=identity.run_population(packets['source'],'source',provider,args.output_root/'source',sealed=sealed,counter=counter,worker_index=args.worker_index,worker_count=4)
            print(json.dumps(dict(raw_text_exported=False,status=result['status'],partition_processed=result['partition_processed'],worker_index=args.worker_index,training_started=False)));return
        if args.finalize:
            source_root=args.output_root/'source';contract=read_document(source_root/'contract.json')
            if contract['worker_count']!=4 or contract['maximum_cumulative_v2_actual_HTTP']!=33000:raise ValueError('Original four-worker source contract changed')
            proof=execution.validate_actual_proofs(packets['source'],source_root,config,provider.database,contract)
            result=identity.finalize_population(packets['source'],'source',source_root)
            write_immutable(source_root/'actual_HTTP_completion_proof.json',dict(proof,execution=file_record(args.output_root/'source_execution_authorization.json'),result=file_record(source_root/'result.json')))
            print(json.dumps(dict(raw_text_exported=False,status=result['status'],**{k:result[k] for k in ['candidate_count','accepted_count','negative_count','insufficient_count','unknown_count','all_candidates_processed']},actual_proofs=proof,training_started=False)));return
        with sqlite3.connect('file:'+str(provider.database)+'?mode=ro',uri=True) as db:
            actual=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started'").fetchone()[0]
            used=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND stage LIKE 'identity_single_v2.%' AND id>?",(allocation['watermark_event_id'],)).fetchone()[0]
        print(json.dumps(dict(raw_text_exported=False,status='check_only_no_API_new_authorization_bound',execution_identity=operational['execution_identity'],
            original_started_unix=provider.started,deadline_unix=provider.started+provider.effective_max_seconds,effective_global_cap=provider.effective_http_cap,
            global_actual_HTTP=actual,identity_additional_actual_HTTP=used,identity_remaining=33000-used,
            source_candidate_count=7519,execution_actual_HTTP_cap2_upper=preflight['execution_actual_HTTP_cap2_upper'],model_calls=0,training_started=False)))
    finally:provider.close()


if __name__=='__main__':main()
