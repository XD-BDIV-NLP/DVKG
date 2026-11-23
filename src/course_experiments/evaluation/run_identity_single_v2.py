"""原服务器逐对身份执行薄入口；默认零API预检，分区不发布全量result。"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import identity_single_v2 as identity
from .native_provider import AuditedNativeProvider
from .run_native_full import prepare_plan,LIMITS
from .run_repaired_full import credentials
from .protocol import runtime_for,check_runtime


TOKENIZER_MANIFEST_SHA='521f3a97e638b33ccb0969cec10b006d905310f0ab4fcae67368bba93d6312db'
SERIALIZER_SHA='abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6'
RUNTIME_CONFIG_ID='a3ca111ac68dd4037849f1e4a26af213281a3a375cb3c963e1b969764d8fe4bd'


class OfficialMessageCounter:
    def __init__(self,manifest_path,serializer_path):
        manifest_path=Path(manifest_path);serializer_path=Path(serializer_path);manifest=read_document(manifest_path)
        if (file_record(manifest_path)['sha256']!=TOKENIZER_MANIFEST_SHA or manifest.get('model')!='deepseek-ai/DeepSeek-V4-Flash' or
                manifest.get('add_special_tokens') is not False or manifest.get('package_version')!='0.22.2'):
            raise ValueError('Reviewed official tokenizer manifest changed')
        for record in manifest['files'].values():
            if file_record(record['path'])!=record:raise ValueError('Frozen official tokenizer asset changed')
        if file_record(serializer_path)['sha256']!=SERIALIZER_SHA:raise ValueError('Frozen official complete-message serializer changed')
        sys.path.insert(0,manifest['vendor'])
        import tokenizers
        if tokenizers.__version__!=manifest['package_version']:raise ValueError('Frozen tokenizers package version differs')
        self.encoder=tokenizers.Tokenizer.from_file(manifest['files']['tokenizer.json']['path'])
        spec=importlib.util.spec_from_file_location('identity_v2_official_chat_encoding',serializer_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.serialize=module.encode_messages;self.cache={}
        self.assets=dict(manifest=file_record(manifest_path),serializer=file_record(serializer_path),**manifest['files'])
    def count(self,messages):
        key=digest(messages)
        if key not in self.cache:self.cache[key]=len(self.encoder.encode(self.serialize(messages,thinking_mode='chat'),add_special_tokens=False).ids)
        return self.cache[key]


def original_inputs(mapping_preparation,reference_preparation):
    mapping_preparation=Path(mapping_preparation);reference_preparation=Path(reference_preparation);manifests=[];packets={};origins={}
    for folder in [mapping_preparation,reference_preparation]:
        manifest=read_document(folder/'manifest.json')
        if manifest['preparation_id']!=digest({k:v for k,v in manifest.items() if k!='preparation_id'}):raise ValueError('Frozen preparation changed')
        for record in manifest['files'].values():
            if file_record(record['path'])!=record:raise ValueError('Frozen original candidate/source input changed')
        manifests.append(manifest)
    for record in [*read_document(mapping_preparation/'input_records.json').values(),*manifests[1]['references'].values(),*manifests[1]['source_records'].values()]:
        if file_record(record['path'])!=record:raise ValueError('Frozen original graph/reference/source changed')
    for branch,folder,stem,count in [('source',mapping_preparation,'source_full',7519),('prior',mapping_preparation,'prior_full',200),('reference',reference_preparation,'full',485)]:
        packets[branch]=read_document(folder/(stem+'_packets_server_only.json'));origins[branch]=read_document(folder/(stem+'_origins_server_only.json'))
        if len(packets[branch])!=count or len({p['candidate_id'] for p in packets[branch]})!=count or {p['candidate_id'] for p in packets[branch]}!={r['candidate_id'] for r in origins[branch]}:
            raise ValueError('Original full candidate denominator changed')
        if any(p['candidate_id']!=digest({k:v for k,v in p.items() if k!='candidate_id'}) for p in packets[branch]):raise ValueError('Original candidate identity changed')
    return packets,origins,manifests


def scan_exact_cache(populations,config,records,*,attempts=None,sealed=None,counter=None):
    attempts=attempts or {};sealed=sealed or {};all_keys=set();existing=set();execution_missing=set();branches={};audits=[]
    for branch,packets in populations.items():
        counts=Counter();counts['accepted_before_conflict']=0;maximum_tokens=maximum_total=0
        for packet in packets:
            directions={};keys={};present={}
            for reverse in [False,True]:
                label='reverse' if reverse else 'primary';body=identity.request_body(packet,branch,reverse=reverse)
                key=digest(dict(endpoint='/chat/completions',body=body,config=config));keys[label]=key;all_keys.add(key)
                if counter:
                    formatted=[dict(m) for m in body['messages']];formatted[0]=dict(formatted[0],response_format=body['response_format'])
                    tokens=max(counter.count(body['messages']),counter.count(formatted));total=tokens+body['max_tokens']+1024
                    maximum_tokens=max(maximum_tokens,tokens);maximum_total=max(maximum_total,total)
                    if total>262144:raise ValueError('Complete original messages exceed frozen context; no truncation or paid stage allowed')
                present[label]=key in records
                if key not in records:counts[label+'_missing']+=1;continue
                existing.add(key);record=records[key]
                if file_record(record['path'])!=record:raise ValueError('Existing exact receipt SHA changed')
                receipt=read_document(record['path'])
                if receipt['request']!=body or receipt['runtime_config']!=config:raise ValueError('Existing receipt is not the exact original body/config')
                try:directions[label]=identity.validate_raw(packet,branch,receipt['response'],reverse=reverse);counts[label+'_strict_valid']+=1
                except (ValueError,TypeError,KeyError,IndexError):counts[label+'_invalid_UNKNOWN']+=1
            first=directions.get('primary');second=directions.get('reverse')
            if packet['candidate_id'] in sealed:counts['sealed_UNKNOWN']+=1;classification='sealed_UNKNOWN'
            elif not present['primary']:
                counts['unprocessed_primary']+=1;classification='unprocessed'
                execution_missing.add(keys['primary'])
                if not present['reverse']:execution_missing.add(keys['reverse'])
            elif not first:counts['cached_UNKNOWN']+=1;classification='UNKNOWN'
            elif not first['eligible_identity_edge']:
                classification=first['judgment']['decision'];counts['cached_'+classification]+=1
            elif not present['reverse']:
                counts['positive_needs_reverse']+=1;classification='positive_reverse_unprocessed';execution_missing.add(keys['reverse'])
            elif second and second['eligible_identity_edge']:counts['accepted_before_conflict']+=1;classification='accepted_before_conflict'
            else:counts['cached_UNKNOWN']+=1;classification='UNKNOWN'
            audits.append(dict(branch=branch,candidate_id=packet['candidate_id'],classification=classification,primary_request_id=keys['primary'],reverse_request_id=keys['reverse']))
        branches[branch]=dict(candidate_count=len(packets),**counts,actual_messages_tokens_max=maximum_tokens,total_with_output_and1024_reserve_max=maximum_total)
    missing=all_keys-existing
    result=dict(protocol=identity.POLICY['version'],branches=branches,all_positive_exact_body_count=len(all_keys),existing_exact_receipts=len(existing),missing_exact_body_count=len(missing),
        first_attempt_upper=len(missing),actual_HTTP_cap2_upper=sum(max(0,2-attempts.get(key,0)) for key in missing),
        execution_missing_body_upper=len(execution_missing),execution_actual_HTTP_cap2_upper=sum(max(0,2-attempts.get(key,0)) for key in execution_missing),
        cached_invalid_is_UNKNOWN_not_negative=True,all_candidates_processed=False,qualification_created=False,model_calls=0)
    return result


def ledger_snapshot(run_root):
    root=Path(run_root)/'formal_provider';limits=read_document(root/'limits.json')
    if limits['limits']!=LIMITS:raise ValueError('Existing formal limits changed')
    extension=read_document(root/'budget_extension_authorization_20261022.json')
    if extension.get('approved_cumulative_http_attempts')!=80000 or extension.get('original_started_unix')!=limits['started_unix'] or extension.get('deadline_reset') is not False:
        raise ValueError('Original inherited 80k authorization changed')
    with sqlite3.connect('file:'+str(root/'audit.sqlite')+'?mode=ro',uri=True) as db:
        records={key:json.loads(value) for key,value in db.execute('SELECT key,record FROM receipts')}
        attempts=dict(db.execute("SELECT key,COUNT(*) FROM events WHERE status='request_started' GROUP BY key"))
        spent_v2=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND stage LIKE 'identity_single_v2.%'").fetchone()[0]
    return dict(limits=limits,extension=file_record(root/'budget_extension_authorization_20261022.json'),actual_HTTP=sum(attempts.values()),v2_actual_HTTP=spent_v2,
        remaining_80k=max(0,80000-sum(attempts.values())),deadline_unix=limits['started_unix']+LIMITS['max_seconds'],halts=[p.name for p in root.glob('*halt*.json')]),records,attempts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run-root','mapping-preparation','reference-preparation','output-root','tokenizer-manifest']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['all','source','prior','reference'],default='all')
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--check-only',action='store_true');modes.add_argument('--start',action='store_true');modes.add_argument('--finalize',action='store_true');modes.add_argument('--qualify',action='store_true')
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--worker-index',type=int,default=0);parser.add_argument('--http-cap',type=int)
    parser.add_argument('--sealed-unknown',type=Path)
    args=parser.parse_args(argv);packets,origins,manifests=original_inputs(args.mapping_preparation,args.reference_preparation)
    native=prepare_plan(args.run_root,'hipporag','book_49bb8d31ddb9');config=read_document(native['construction']['files']['assets']['runtime_config']['path']);check_runtime(config)
    if digest(config)!=RUNTIME_CONFIG_ID:raise ValueError('Original full runtime config changed')
    sealed=read_document(args.sealed_unknown) if args.sealed_unknown else {}
    for cid,evidence in sealed.items():
        if identity.sealed_unknown(evidence['failed_decision'],evidence['repair_attempts'])!=evidence or evidence['candidate_id']!=cid:
            raise ValueError('Sealed exhausted UNKNOWN proof changed')
    exhausted='4a8e483da824d7334f9a37997ae8c4d325e2cc94963369072153f6ac732480d1'
    if args.start and args.stage in {'all','source'} and exhausted not in sealed:raise ValueError('Original exhausted source case must be sealed UNKNOWN before full execution')
    branches=list(packets) if args.stage=='all' else [args.stage]
    if args.qualify:
        value=identity.qualify_all(args.mapping_preparation,args.reference_preparation,args.output_root)
        print(json.dumps(dict(raw_text_exported=False,status=value['status'],qualification_id=value['qualification_id'],branch_counts=value['branch_counts'],model_calls=0)));return
    if args.finalize:
        values={branch:identity.finalize_population(packets[branch],branch,args.output_root/branch) for branch in branches}
    else:
        counter=OfficialMessageCounter(args.tokenizer_manifest,Path(manifests[0]['token_assets_root'])/'encoding/encoding_dsv4.py')
        snapshot,records,attempts=ledger_snapshot(args.run_root)
        if not args.start:
            value=scan_exact_cache({b:packets[b] for b in branches},config,records,attempts=attempts,sealed=sealed,counter=counter)
            value.update(raw_text_exported=False,status='check_only_no_API',ledger=snapshot,tokenizer=counter.assets,
                original_reference_facts={course:dict(fact_count=len(read_document(rec['path'])['facts']),record=rec) for course,rec in manifests[1]['references'].items()},
                all_reference_facts_retained=True,training_started=False,code=file_record(Path(__file__)))
            args.output_root.mkdir(parents=True,exist_ok=True);path=args.output_root/('check_only_'+digest(value)+'.json');write_immutable(path,value)
            print(json.dumps(value));return
        if args.http_cap is None or not 1<=args.http_cap<=32438:raise ValueError('Explicit jointly reserved total v2 HTTP cap required before paid execution')
        if snapshot['halts'] or time.time()>=snapshot['deadline_unix'] or snapshot['remaining_80k']<=0:raise RuntimeError('Existing global halt/budget/deadline prevents execution')
        credentials();provider=AuditedNativeProvider(runtime_for(native['construction'],args.output_root/'transport'/str(args.worker_index)),args.run_root/'formal_provider',**LIMITS)
        try:
            with identity.actual_attempt_cap(provider,2,total_http_cap=args.http_cap):
                values={branch:identity.run_population(packets[branch],branch,provider,args.output_root/branch,sealed=sealed if branch=='source' else {},counter=counter,
                    worker_index=args.worker_index,worker_count=args.workers) for branch in branches}
        finally:provider.close()
    print(json.dumps(dict(raw_text_exported=False,protocol=identity.POLICY['version'],results={branch:{k:value[k] for k in ['status','candidate_count','accepted_count','negative_count','insufficient_count','unknown_count','all_candidates_processed','partition_processed'] if k in value} for branch,value in values.items()},training_started=False)))


if __name__=='__main__':main()
