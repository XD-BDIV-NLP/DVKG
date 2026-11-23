from __future__ import annotations

import concurrent.futures
from contextlib import closing
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from course_experiments.evaluation.native_provider import (
    AuditedNativeProvider, NativeBudgetExceeded, NativeAccountUnavailable,
)
from course_experiments.manifest import file_record


CONFIG = dict(llm=dict(base_url='https://api.siliconflow.cn/v1',
    model='deepseek-ai/DeepSeek-V4-Flash', extra_body=dict(enable_thinking=False)),
    embedding=dict(base_url='https://api.siliconflow.cn/v1', model='BAAI/bge-m3'))


class FakeRuntime:
    def __init__(self):
        self.config = copy.deepcopy(CONFIG)
        self.calls = []
        self.code = None
        self.active = self.maximum = 0
        self.lock = threading.Lock()

    def _record(self, *args):
        pass

    def _raw_cached(self, key):
        return None

    def _post(self, endpoint, body, options, key, stage):
        self._record(key, stage, 'request_started', dict(attempt=1))
        with self.lock:
            self.calls.append(copy.deepcopy(body))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        time.sleep(.025)
        with self.lock:
            self.active -= 1
        if self.code:
            self._record(key, stage, 'http_error', dict(status_code=self.code))
            raise RuntimeError(f'Model service returned HTTP {self.code}')
        if endpoint == '/embeddings':
            result = dict(data=[dict(index=i, embedding=[float(i+1)]*1024)
                for i in reversed(range(len(body['input'])))], usage=dict(total_tokens=1))
        else:
            result = dict(choices=[dict(message=dict(content='answer'), finish_reason='stop')],
                usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        self._record(key, stage, 'response', result)
        return result


def process_worker(root):
    p = AuditedNativeProvider(FakeRuntime(), root, max_http_attempts=20,
                             max_seconds=3600, max_concurrency=2)
    p.chat([dict(role='user',content='process-shared')], max_tokens=100)


def distinct_process_worker(root, worker):
    p = AuditedNativeProvider(FakeRuntime(), root, max_http_attempts=100,
                             max_seconds=3600, max_concurrency=2)
    try:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda i:p.chat([dict(role='user',content=f'worker-{worker}-item-{i}')],
                max_tokens=100),range(12)))
    finally:
        p.close()


def distinct_real_runtime_worker(root, worker):
    import io
    from unittest.mock import patch
    from course_experiments.construction import PROMPTS
    from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
    runtime=ModelRuntime(Path(root)/('transport_'+str(worker)),PROMPTS,dict(copy.deepcopy(CONFIG),service_retries=0))
    p=AuditedNativeProvider(runtime,root,max_http_attempts=100,max_seconds=3600,max_concurrency=2)
    raw=dict(choices=[dict(message=dict(content='synthetic'),finish_reason='stop')],
        usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2))
    def response(*args,**kwargs):
        time.sleep(.01)
        return io.BytesIO(json.dumps(raw).encode())
    try:
        with patch.dict('os.environ',{'SILICONFLOW_API_KEY':'offline-test-only'}), \
             patch('urllib.request.urlopen',side_effect=response), \
             concurrent.futures.ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda i:p.chat([dict(role='user',content=f'real-runtime-{worker}-item-{i}')],
                max_tokens=100),range(12)))
    finally:
        p.close()


class NativeProviderTests(unittest.TestCase):
    def authorize_extension(self, provider, **changes):
        value = dict(version='human_authorized_cumulative_extension_v1',
            question_call='call_ec6a116c31f8441cb621cded7ca170b9',
            answer='允许按8万累计上限继续',
            original_limits_sha256=file_record(self.root/'limits.json')['sha256'],
            original_started_unix=provider.started, original_limits=provider.limits,
            approved_cumulative_http_attempts=80000, inherited_spend=True,
            deadline_reset=False, edc_excluded=True)
        value.update(changes)
        (self.root/'budget_extension_authorization_20261022.json').write_text(
            json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def test_explicit_extension_preserves_old_bytes_spend_cache_clock_and_slots(self):
        old = self.provider(max_http_attempts=1)
        self.ask(old)
        prior = (self.root/'limits.json').read_bytes()
        self.authorize_extension(old)
        old.close()
        extended = self.provider(FakeRuntime(), max_http_attempts=1)
        self.assertTrue(self.ask(extended)['_native_cache_hit'])
        self.ask(extended, 'second')
        stats = extended.statistics()
        self.assertEqual(stats['http_attempts'], 2)
        self.assertEqual(stats['effective_http_cap'], 80000)
        self.assertEqual(stats['limits']['max_http_attempts'], 1)
        self.assertEqual(stats['started_unix'], old.started)
        self.assertEqual(stats['limits']['max_seconds'], 3600)
        self.assertEqual(stats['limits']['max_concurrency'], 2)
        self.assertEqual((self.root/'limits.json').read_bytes(), prior)

    def test_extension_rejects_changed_clock_original_record_cap_or_approval(self):
        p = self.provider()
        for changes in [dict(original_started_unix=p.started+1),
                        dict(original_limits_sha256='changed'),
                        dict(approved_cumulative_http_attempts=80001),
                        dict(question_call='unapproved'), dict(inherited_spend=False),
                        dict(deadline_reset=True), dict(edc_excluded=False)]:
            self.authorize_extension(p, **changes)
            with self.assertRaisesRegex(ValueError, 'authorization'):
                self.provider(FakeRuntime())
        self.assertEqual(p.statistics()['http_attempts'], 0)

    def test_extension_never_clears_existing_halt_or_extends_time(self):
        from unittest.mock import patch
        p = self.provider()
        self.authorize_extension(p)
        extended = self.provider(FakeRuntime())
        with patch('course_experiments.evaluation.native_provider.time.time',
                   return_value=p.started+3601):
            with self.assertRaises(NativeBudgetExceeded):
                self.ask(extended, 'expired')
        (self.root/'budget_halt.json').write_text('{}')
        with self.assertRaises(NativeBudgetExceeded):
            self.ask(extended, 'halted')
        self.assertTrue((self.root/'budget_halt.json').exists())
        self.assertEqual(extended.statistics()['http_attempts'], 0)

    def test_internal_empty_string_embedding_is_exact_but_empty_batch_or_nontext_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = FakeRuntime()
            provider = AuditedNativeProvider(runtime, root, max_http_attempts=5,
                max_seconds=3600, max_concurrency=1)
            try:
                provider.request_executor('/embeddings', dict(model='BAAI/bge-m3',
                    input=[''], encoding_format='float'), 'synthetic.native_empty')
                self.assertEqual(runtime.calls[0]['input'], [''])
                for values in ([], [None], [3]):
                    with self.assertRaises(ValueError):
                        provider.request_executor('/embeddings', dict(model='BAAI/bge-m3',
                            input=values, encoding_format='float'), 'synthetic.invalid')
                self.assertEqual(len(runtime.calls), 1)
            finally:
                provider.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = FakeRuntime()

    def provider(self, runtime=None, **kwargs):
        provider = AuditedNativeProvider(runtime or self.runtime, self.root,
            max_http_attempts=kwargs.pop('max_http_attempts', 20), max_seconds=3600,
            max_concurrency=kwargs.pop('max_concurrency', 2), **kwargs)
        self.addCleanup(provider.close)
        return provider

    def ask(self, p, text='same', **kwargs):
        return p.chat_response([dict(role='user', content=text)], max_tokens=100, **kwargs)

    def test_same_exact_request_single_flight_and_cross_instance_reuse(self):
        p = self.provider()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            out = list(pool.map(lambda _: self.ask(p), range(4)))
        self.assertEqual(len(self.runtime.calls), 1)
        self.assertEqual(sum(not r['_native_cache_hit'] for r in out), 1)
        second = FakeRuntime()
        self.assertTrue(self.ask(self.provider(second))['_native_cache_hit'])
        self.assertEqual(second.calls, [])

    def test_exact_native_format_and_limit_are_preserved(self):
        p = self.provider()
        self.ask(p)
        self.ask(p, response_format=dict(type='json_object'))
        p.chat_response([dict(role='user', content='same')], max_tokens=101)
        self.assertEqual(len(self.runtime.calls), 3)
        self.assertNotIn('response_format', self.runtime.calls[0])
        self.assertEqual(self.runtime.calls[1]['response_format'], dict(type='json_object'))
        self.assertEqual(self.runtime.calls[2]['max_tokens'], 101)
        self.assertFalse(self.runtime.calls[0]['enable_thinking'])

    def test_quota_counts_real_attempts_but_does_not_block_cache(self):
        p = self.provider(max_http_attempts=1)
        self.ask(p)
        with self.assertRaises(NativeBudgetExceeded):
            self.ask(p, 'another')
        with self.assertRaises(NativeBudgetExceeded):
            p.raise_if_fatal()
        self.assertTrue(self.ask(p)['_native_cache_hit'])
        self.assertEqual(p.statistics()['http_attempts'], 1)

    def test_account_failure_stops_other_instances(self):
        p = self.provider()
        self.runtime.code = 402
        with self.assertRaises(NativeAccountUnavailable):
            self.ask(p)
        second = FakeRuntime()
        with self.assertRaises(NativeAccountUnavailable):
            self.ask(self.provider(second), 'other')
        self.assertEqual(second.calls, [])
        self.assertEqual(p.statistics()['http_attempts'], 1)

    def test_endpoint_and_model_override_rejected_before_network(self):
        p = self.provider()
        with self.assertRaises(ValueError):
            p.request_executor('https://other.invalid', {}, 'bad')
        with self.assertRaises(ValueError):
            p.request_executor('/chat/completions', dict(model='other'), 'bad')
        self.assertEqual(self.runtime.calls, [])

    def test_raw_receipt_tampering_is_not_silent_cache_miss(self):
        p = self.provider()
        self.ask(p)
        path = next((self.root/'receipts').glob('*.json'))
        value = json.loads(path.read_text())
        value['response']['choices'][0]['message']['content'] = 'changed'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'receipt'):
            self.ask(p)
        self.assertEqual(len(self.runtime.calls), 1)

    def test_embedding_order_normalization_and_shape(self):
        values = self.provider().embed(['a', 'b'])
        self.assertEqual(values.shape, (2, 1024))
        self.assertAlmostEqual(float((values[0]**2).sum()), 1, places=5)

    def test_global_slots_and_closed_sqlite_transactions(self):
        p = self.provider()
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            list(pool.map(lambda i: self.ask(p, str(i)), range(8)))
        self.assertLessEqual(self.runtime.maximum, 2)
        self.assertEqual(p.statistics()['http_attempts'], 8)
        import sqlite3
        with closing(sqlite3.connect(self.root/'audit.sqlite', timeout=.1)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 16)

    def test_separate_processes_share_one_actual_request(self):
        import multiprocessing
        self.provider()
        context = multiprocessing.get_context('spawn')
        jobs = [context.Process(target=process_worker,args=(str(self.root),)) for _ in range(2)]
        for job in jobs:
            job.start()
        for job in jobs:
            job.join(20)
            self.assertEqual(job.exitcode, 0)
        self.assertEqual(self.provider(FakeRuntime()).statistics()['http_attempts'], 1)

    def test_busy_begin_retries_only_sqlite_before_one_actual_request(self):
        import sqlite3
        from unittest.mock import patch
        p = self.provider()
        self.addCleanup(p.close)
        original = sqlite3.connect
        busy = []
        class BusyOnce:
            def __init__(self, connection):self.connection=connection
            def __getattr__(self,name):return getattr(self.connection,name)
            def __enter__(self):self.connection.__enter__();return self
            def __exit__(self,*args):return self.connection.__exit__(*args)
            def execute(self, sql, *args):
                if sql == 'BEGIN IMMEDIATE' and not busy:
                    busy.append(True)
                    raise sqlite3.OperationalError('database is locked')
                return self.connection.execute(sql,*args)
        with patch('course_experiments.evaluation.native_provider.sqlite3.connect',
                side_effect=lambda *args,**kwargs:BusyOnce(original(*args,**kwargs))):
            out = self.ask(p)
        self.assertEqual(out['choices'][0]['message']['content'],'answer')
        self.assertEqual(len(self.runtime.calls),1)
        self.assertEqual(p.statistics()['http_attempts'],1)
        self.assertTrue(self.ask(p)['_native_cache_hit'])
        retries=[json.loads(line) for line in (self.root/'storage_contention.jsonl').read_text('utf-8').splitlines()]
        self.assertEqual(len(retries),1)
        self.assertEqual(retries[0]['status'],'sqlite_busy_retry')

    def test_idle_audit_connection_preserves_wal_inode_without_holding_transaction(self):
        import sqlite3
        p=self.provider()
        identities=[]
        for i in range(4):
            p._record(str(i),'offline_wal_probe','response',{})
            p.statistics()
            wal=self.root/'audit.sqlite-wal'
            shm=self.root/'audit.sqlite-shm'
            self.assertTrue(wal.exists() and shm.exists(),'Last short connection closed and removed WAL/SHM')
            identities.append((wal.stat().st_ino,shm.stat().st_ino))
        self.assertEqual(len(set(identities)),1)
        self.assertIsNone(p._audit_keepalive.isolation_level)
        self.assertFalse(p._audit_keepalive.in_transaction)
        with closing(sqlite3.connect(p.database,timeout=.1)) as db:
            self.assertEqual(db.execute('PRAGMA synchronous').fetchone()[0],2)
            db.execute('BEGIN IMMEDIATE')
            db.rollback()
        p.close()
        self.assertFalse(wal.exists())
        self.assertFalse(shm.exists())

    @unittest.skipUnless(os.name=='posix','POSIX flock regression requires Linux')
    def test_busy_slot_wait_does_not_truncate_or_modify_lock_inode(self):
        import fcntl
        p=self.provider()
        paths=[self.root/'locks'/f'slot_{i}.lock' for i in range(2)]
        held=[]
        for path in paths:
            descriptor=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
            os.write(descriptor,b'preexisting slot marker')
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            held.append(descriptor)
        before=[(path.read_bytes(),path.stat().st_mtime_ns,path.stat().st_ctime_ns) for path in paths]
        entered=threading.Event()
        def wait_slot():
            with p._slot():entered.set()
        worker=threading.Thread(target=wait_slot)
        worker.start()
        try:
            time.sleep(.35)
            after=[(path.read_bytes(),path.stat().st_mtime_ns,path.stat().st_ctime_ns) for path in paths]
            self.assertFalse(entered.is_set())
        finally:
            for descriptor in held:
                fcntl.flock(descriptor,fcntl.LOCK_UN)
                os.close(descriptor)
            worker.join(3)
        self.assertEqual(before,after)
        self.assertTrue(entered.is_set())

    def test_distinct_requests_multiple_processes_runtimes_keep_exact_budget_and_slots(self):
        import multiprocessing
        import sqlite3
        seed = self.provider(max_http_attempts=100)
        seed.close()
        context=multiprocessing.get_context('spawn')
        jobs=[context.Process(target=distinct_process_worker,args=(str(self.root),i)) for i in range(3)]
        for job in jobs:job.start()
        for job in jobs:
            job.join(30)
            self.assertEqual(job.exitcode,0)
        p=self.provider(FakeRuntime(),max_http_attempts=100)
        self.addCleanup(p.close)
        self.assertEqual(p.statistics()['http_attempts'],36)
        self.assertEqual(p.statistics()['receipt_count'],36)
        with closing(sqlite3.connect(self.root/'audit.sqlite')) as db:
            self.assertEqual(db.execute('PRAGMA journal_mode').fetchone()[0],'wal')
            active=maximum=0
            for status, in db.execute('SELECT status FROM events ORDER BY id'):
                active += 1 if status=='request_started' else -1 if status=='response' else 0
                maximum=max(maximum,active)
            self.assertEqual(active,0)
            self.assertLessEqual(maximum,2)

    def test_real_transport_databases_remain_distinct_in_concurrent_processes(self):
        import multiprocessing
        import sqlite3
        seed=self.provider(max_http_attempts=100);seed.close()
        context=multiprocessing.get_context('spawn')
        jobs=[context.Process(target=distinct_real_runtime_worker,args=(str(self.root),i)) for i in range(3)]
        for job in jobs:job.start()
        for job in jobs:
            job.join(30)
            self.assertEqual(job.exitcode,0)
        p=self.provider(FakeRuntime(),max_http_attempts=100);self.addCleanup(p.close)
        self.assertEqual(p.statistics()['http_attempts'],36)
        self.assertEqual(p.statistics()['receipt_count'],36)
        for worker in range(3):
            path=self.root/('transport_'+str(worker))/'model_cache.sqlite'
            self.assertNotEqual(path.resolve(),p.database.resolve())
            with closing(sqlite3.connect(path)) as db:
                counts=dict(db.execute('SELECT status,COUNT(*) FROM attempts GROUP BY status'))
            self.assertEqual(counts.get('request_started'),12)
            self.assertEqual(counts.get('response'),12)

    def test_frozen_budget_cannot_be_reset_by_new_instance(self):
        self.provider()
        with self.assertRaisesRegex(ValueError, 'limits'):
            self.provider(FakeRuntime(), max_http_attempts=21)

    def test_elapsed_deadline_is_fatal_without_another_http_call(self):
        p = self.provider()
        p.started = time.time()-3601
        with self.assertRaises(NativeBudgetExceeded):
            p.raise_if_fatal()
        self.assertEqual(self.runtime.calls, [])

    def test_recovered_transport_response_is_reported_as_cache(self):
        raw = dict(choices=[dict(message=dict(content='recovered'),finish_reason='stop')])
        self.runtime._raw_cached = lambda key: raw
        p = self.provider()
        out = self.ask(p)
        self.assertTrue(out['_native_cache_hit'])
        self.assertEqual(self.runtime.calls, [])

    def test_real_runtime_http_retries_are_each_charged_and_account_errors_halt(self):
        from unittest.mock import patch
        from urllib.error import HTTPError
        import io
        from course_experiments.construction import PROMPTS
        from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
        config = dict(copy.deepcopy(CONFIG),service_retries=1)
        runtime = ModelRuntime(self.root/'transport',PROMPTS,config)
        p = self.provider(runtime)
        self.addCleanup(p.close)
        raw = dict(choices=[dict(message=dict(content='real transport mock'),finish_reason='stop')])
        with patch.dict('os.environ',{'SILICONFLOW_API_KEY':'offline-test-only'}), \
             patch('urllib.request.urlopen',side_effect=[HTTPError('https://api.siliconflow.cn/v1',503,'busy',{},None),
                 io.BytesIO(json.dumps(raw).encode())]), \
             patch('domain_ckg_workspace.ckg_incremental.runtime.time.sleep'):
            self.ask(p)
        self.assertEqual(p.statistics()['http_attempts'],2)
        with patch.dict('os.environ',{'SILICONFLOW_API_KEY':'offline-test-only'}), \
             patch('urllib.request.urlopen',side_effect=HTTPError('https://api.siliconflow.cn/v1',402,'account',{},None)):
            with self.assertRaises(NativeAccountUnavailable):
                self.ask(p,'account-failure')
        self.assertEqual(p.statistics()['http_attempts'],3)


if __name__ == '__main__':
    unittest.main()
