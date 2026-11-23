"""既有严格单对判定的source全量执行守卫；语义协议与参考拆分不变。"""
from __future__ import annotations

from contextlib import closing,contextmanager
from pathlib import Path
import sqlite3
import time

from ..manifest import digest,file_record
from . import identity_single_v2 as identity


START=1792604536.907112
PREFIX='identity_single_v2.source.'
PROVIDER_SHA='af32fd606f52b78bc2d30d117950148f6207dc820c2819a5463a3394d4e7482f'
AUTHORIZATION_SHA='59e4516b5a2157dc4fb7b023234c5a6af6af5e6e570b137de4f73285d1cd079b'
ALLOCATION_SHA='e3270b28db9a5909c2dd6c47ca7fad6fe9225b6008247d5468917090c487f055'


class SourceAllocationExhausted(RuntimeError):pass
class SourceDeadlineReached(RuntimeError):pass


def bind_authorization(provider,run_root):
    root=Path(run_root)/'formal_provider'
    if (file_record(root/'budget_extension_authorization_20261023.json')['sha256']!=AUTHORIZATION_SHA or
            file_record(root/'bounded_allocations_20261023.json')['sha256']!=ALLOCATION_SHA or
            provider.started!=START or provider.effective_http_cap!=200000 or provider.effective_max_seconds!=345600 or
            (provider.budget_extension or {}).get('sha256')!=AUTHORIZATION_SHA or not provider.allocation or
            provider.allocation.get('watermark_event_id')!=102848 or provider.allocation.get('inherited_actual_http')!=51430 or
            provider.allocation.get('caps')!={'identity':33000,'echa':60000,'native':45000,'blind64':5000}):
        raise ValueError('Exact root-approved provider authorization/allocation and inherited clock required')
    return dict(stage_prefix=PREFIX,watermark_event_id=102848,maximum_additional_actual_HTTP=33000,
        authorized_cumulative_HTTP=200000,authorized_total_seconds=345600,original_started_unix=START,
        deadline_unix=START+345600,authorization_sha256=AUTHORIZATION_SHA,allocation_sha256=ALLOCATION_SHA)


def validate_actual_proofs(packets,root,config,database,contract,*,branch='source'):
    if branch not in {'source','prior'}:raise ValueError('Only graph identity populations have append HTTP proofs')
    root=Path(root);counts=dict(candidate_count=len(packets),sealed_UNKNOWN=0,primary_raw=0,reverse_raw=0,unresolved_with_actual=0)
    with closing(sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True)) as db:
        attempts=dict(db.execute("SELECT key,COUNT(*) FROM events WHERE status='request_started' GROUP BY key"))
    for packet in packets:
        folder=root/'decisions'/packet['candidate_id'];row=identity._read_checkpoint(folder/'decision.json',contract.get('job_identity'))
        if row.get('sealed_unknown'):
            evidence=row['sealed_unknown']
            if (packet['candidate_id'] not in contract.get('sealed',{}) or evidence!=contract['sealed'][packet['candidate_id']] or
                    identity.sealed_unknown(evidence['failed_decision'],evidence['repair_attempts'])!=evidence):
                raise ValueError('Original sealed UNKNOWN proof changed')
            counts['sealed_UNKNOWN']+=1;continue
        for reverse,name in [(False,'primary'),(True,'reverse')]:
            path=folder/(name+'.json');body=identity.request_body(packet,branch,reverse=reverse)
            if path.exists():
                counts[name+'_raw']+=1
                if not reverse:
                    try:positive=identity.validate_raw(packet,branch,identity._read_checkpoint(path,contract.get('job_identity'))['raw_response'])['eligible_identity_edge']
                    except (ValueError,TypeError,KeyError,IndexError):positive=False
                    if not positive:break
                continue
            key=digest(dict(endpoint='/chat/completions',body=body,config=config))
            if row.get('classification')!='UNKNOWN' or attempts.get(key,0)<1:
                raise ValueError('Missing original response and zero actual HTTP cannot be completed or qualified')
            counts['unresolved_with_actual']+=1;break
    return dict(counts,zero_HTTP_without_raw=0,all_original_raw_revalidated_at_finalization=True)


def _actual_count(provider,key):
    with closing(sqlite3.connect('file:'+str(provider.database)+'?mode=ro',uri=True)) as db:
        return db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND key=?",(key,)).fetchone()[0]


def _validate_allocation_numbers(value):
    if (value.get('stage_prefix')!=PREFIX or value.get('maximum_additional_actual_HTTP')!=33000 or
            value.get('authorized_cumulative_HTTP')!=200000 or value.get('authorized_total_seconds')!=345600 or
            value.get('original_started_unix')!=START or value.get('deadline_unix')!=START+345600 or
            type(value.get('watermark_event_id')) is not int or value['watermark_event_id']<0):
        raise ValueError('Exact approved source allocation, original start and new 96-hour deadline required')


@contextmanager
def source_execution_guard(provider,allocation,*,now=time.time,on_wait=lambda value:None):
    _validate_allocation_numbers(allocation);original_request=provider.request_executor
    def deadline():
        provider.raise_if_fatal()
        if now()>=allocation['deadline_unix']:raise SourceDeadlineReached('Original-start authorized 96-hour deadline reached')
    with identity.actual_attempt_cap(provider,2):
        body_bounded=provider._write_transaction;provider._identity_single_total_cap=33000
        def bounded(fn,*,operation,request_id=None):
            def transaction(db):
                if operation=='event_request_started':
                    deadline()
                    spent=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND stage LIKE ? AND id>?",(PREFIX+'%',allocation['watermark_event_id'])).fetchone()[0]
                    if spent>=33000:raise SourceAllocationExhausted('Original-watermark source allocation of 33000 actual HTTP exhausted')
                return fn(db)
            return body_bounded(transaction,operation=operation,request_id=request_id)
        def request(endpoint,body,stage):
            if stage not in {PREFIX+'primary',PREFIX+'reverse'}:raise ValueError('Source controller cannot spend another stage allocation')
            key=digest(dict(endpoint=endpoint,body=body,config=provider.runtime.config))
            while True:
                deadline();before=_actual_count(provider,key)
                try:return original_request(endpoint,body,stage)
                except TimeoutError as exc:
                    after=_actual_count(provider,key)
                    if before or after or str(exc)!='Native global request slots remained occupied for 600 seconds':raise
                    on_wait(dict(request_id=key,stage=stage,actual_HTTP=0,error_type='TimeoutError',reason='waiting_for_first_original_execution'))
        provider._write_transaction=bounded;provider.request_executor=request
        try:yield
        finally:provider.request_executor=original_request;provider._write_transaction=body_bounded
