from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest

from course_experiments.manifest import digest
from course_experiments.evaluation import identity_single_v2 as identity
from unittest.mock import patch


START=1792604536.907112


def allocation(watermark):
    return dict(stage_prefix='identity_single_v2.source.',watermark_event_id=watermark,
        maximum_additional_actual_HTTP=33000,authorized_cumulative_HTTP=200000,
        original_started_unix=START,authorized_total_seconds=345600,deadline_unix=START+345600)


class Provider:
    def __init__(self,root):
        self.database=Path(root)/'audit.sqlite';self.runtime=SimpleNamespace(config={});self.lock=threading.Lock();self.calls=0;self.zero_waits=0
        with closing(sqlite3.connect(self.database)) as db,db:db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,key TEXT,stage TEXT,status TEXT)')
    def raise_if_fatal(self):pass
    def _write_transaction(self,fn,*,operation,request_id=None):
        with self.lock,closing(sqlite3.connect(self.database)) as db,db:return fn(db)
    def request_executor(self,endpoint,body,stage):
        self.calls+=1
        if self.zero_waits:
            self.zero_waits-=1;raise TimeoutError('Native global request slots remained occupied for 600 seconds')
        key=digest(dict(endpoint=endpoint,body=body,config=self.runtime.config))
        self._write_transaction(lambda db:db.execute('INSERT INTO events(key,stage,status) VALUES(?,?,?)',(key,stage,'request_started')),operation='event_request_started',request_id=key)
        return {'original_raw':'synthetic'},False
    def count(self,key=None):
        with closing(sqlite3.connect(self.database)) as db:
            return db.execute("SELECT COUNT(*) FROM events WHERE status='request_started'"+(' AND key=?' if key else ''),(key,) if key else ()).fetchone()[0]
    def fill(self,n,stage,key='old'):
        with closing(sqlite3.connect(self.database)) as db,db:db.executemany('INSERT INTO events(key,stage,status) VALUES(?,?,?)',[(key,stage,'request_started')]*n)


class SourceExecutionTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_source_full_execution
        return identity_source_full_execution
    def guard(self,provider,watermark=0,**kwargs):
        return self.api().source_execution_guard(provider,allocation(watermark),now=lambda:START+200000,**kwargs)

    def test_watermark_and_stage_scope_do_not_charge_other_stages_or_reset_body_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Provider(tmp);p.fill(2,'old_source_recovery','used');watermark=2;p.fill(32999,'identity_single_v2.source.primary');p.fill(5,'echa.unrelated')
            with self.guard(p,watermark):
                p.request_executor('/chat/completions',{'new':1},'identity_single_v2.source.primary')
                with self.assertRaises(self.api().SourceAllocationExhausted):p.request_executor('/chat/completions',{'new':2},'identity_single_v2.source.reverse')
                with self.assertRaises(identity.RequestAttemptsExhausted):p._write_transaction(lambda db:db.execute("INSERT INTO events(key,stage,status) VALUES('used','identity_single_v2.source.primary','request_started')"),operation='event_request_started',request_id='used')
            self.assertEqual(p.count(),33007)

    def test_four_workers_cannot_overrun_remaining_single_actual_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Provider(tmp);p.fill(32999,'identity_single_v2.source.primary')
            def one(i):
                try:p.request_executor('/chat/completions',{'new':i},'identity_single_v2.source.primary');return True
                except self.api().SourceAllocationExhausted:return False
            with self.guard(p),ThreadPoolExecutor(4) as pool:results=list(pool.map(one,range(4)))
            self.assertEqual(sum(results),1);self.assertEqual(p.count(),33000)

    def test_zero_actual_slot_timeout_waits_for_first_execution_without_reasking_a_judgment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Provider(tmp);p.zero_waits=1;waits=[]
            with self.guard(p,on_wait=waits.append):p.request_executor('/chat/completions',{'new':1},'identity_single_v2.source.primary')
            self.assertEqual(p.calls,2);self.assertEqual(p.count(),1);self.assertEqual(len(waits),1);self.assertEqual(waits[0]['actual_HTTP'],0)

    def test_existing_actual_or_network_timeout_is_never_reasked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Provider(tmp);body={'new':1};key=digest(dict(endpoint='/chat/completions',body=body,config={}));p.fill(1,'old_stage',key);p.zero_waits=1
            with self.guard(p):
                with self.assertRaises(TimeoutError):p.request_executor('/chat/completions',body,'identity_single_v2.source.primary')
            self.assertEqual(p.calls,1);self.assertEqual(p.count(),1)

    def test_non_source_requests_and_changed_budget_identity_rejected_before_HTTP(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Provider(tmp)
            with self.guard(p):
                with self.assertRaises(ValueError):p.request_executor('/chat/completions',{},'identity_single_v2.prior.primary')
            bad=allocation(0);bad['original_started_unix']+=1
            with self.assertRaises(ValueError):
                with self.api().source_execution_guard(p,bad):pass
            self.assertEqual(p.calls,0)

    def test_new_96_hour_deadline_stops_zero_HTTP_wait_without_fake_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Provider(tmp);p.zero_waits=1
            with self.assertRaises(self.api().SourceDeadlineReached):
                with self.api().source_execution_guard(p,allocation(0),now=lambda:START+345600):p.request_executor('/chat/completions',{},'identity_single_v2.source.primary')
            self.assertEqual(p.calls,0)

    def test_authorization_binding_requires_exact_provider_sidecars_and_original_clock(self):
        api=self.api()
        p=SimpleNamespace(started=START,effective_http_cap=200000,effective_max_seconds=345600,
            budget_extension={'sha256':api.AUTHORIZATION_SHA},allocation=dict(watermark_event_id=102848,
                inherited_actual_http=51430,caps={'identity':33000,'echa':60000,'native':45000,'blind64':5000}))
        with patch.object(api,'file_record',side_effect=lambda path:{'sha256':api.AUTHORIZATION_SHA if str(path).endswith('authorization_20261023.json') else api.ALLOCATION_SHA}):
            self.assertEqual(api.bind_authorization(p,Path('/synthetic'))['watermark_event_id'],102848)
            p.started+=1
            with self.assertRaises(ValueError):api.bind_authorization(p,Path('/synthetic'))
        p.started=START
        with patch.object(api,'file_record',return_value={'sha256':'tampered'}):
            with self.assertRaises(ValueError):api.bind_authorization(p,Path('/synthetic'))

    def test_no_raw_and_no_actual_cannot_publish_full_source_completion(self):
        api=self.api();packet={'candidate_id':'candidate'}
        with tempfile.TemporaryDirectory() as tmp:
            p=Provider(tmp);root=Path(tmp)/'source';folder=root/'decisions/candidate';folder.mkdir(parents=True)
            def checkpoint(path,job):return {'classification':'UNKNOWN','error_record':{}}
            with patch.object(api.identity,'_read_checkpoint',side_effect=checkpoint),patch.object(api.identity,'request_body',return_value={'synthetic':1}):
                with self.assertRaises(ValueError):api.validate_actual_proofs([packet],root,{},p.database,{})
                body={'synthetic':1};key=digest(dict(endpoint='/chat/completions',body=body,config={}))
                p.fill(1,'identity_single_v2.source.primary',key)
                self.assertEqual(api.validate_actual_proofs([packet],root,{},p.database,{})['unresolved_with_actual'],1)


if __name__=='__main__':unittest.main()
