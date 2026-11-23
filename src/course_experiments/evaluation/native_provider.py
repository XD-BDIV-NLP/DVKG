"""复用原传输层的原生模型访问器；原文收据仅保留在实验服务器。"""
from __future__ import annotations

import asyncio
from contextlib import closing, contextmanager
import errno
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from filelock import FileLock, Timeout
import numpy as np

from ..manifest import digest, file_record, read_document
from .protocol import check_runtime


class NativeBudgetExceeded(RuntimeError):
    pass


class NativeAccountUnavailable(RuntimeError):
    pass


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


class AuditedNativeProvider:
    def __init__(self, runtime, root, *, max_http_attempts=1200, max_seconds=86400,
                 max_concurrency=4):
        check_runtime(runtime.config)
        self.runtime = runtime
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ('locks', 'receipts'):
            (self.root/name).mkdir(exist_ok=True)
        self.limits = dict(max_http_attempts=max_http_attempts, max_seconds=max_seconds,
                           max_concurrency=max_concurrency)
        if any(type(v) is not int or v < 1 for v in self.limits.values()):
            raise ValueError('Positive integer native limits required')
        self.database = self.root/'audit.sqlite'
        with FileLock(str(self.root/'initialize.lock')):
            limit_path = self.root/'limits.json'
            if limit_path.exists():
                stored = read_document(limit_path)
                if stored['limits'] != self.limits:
                    raise ValueError('Frozen native limits cannot be changed or reset')
            else:
                stored = dict(limits=self.limits, started_unix=time.time())
                _atomic(limit_path, stored)
            self.started = stored['started_unix']
            self.effective_http_cap = self.limits['max_http_attempts']
            self.effective_max_seconds = self.limits['max_seconds']
            self.budget_extension = None
            extension_path = self.root/'budget_extension_authorization_20261022.json'
            if extension_path.exists():
                extension = read_document(extension_path)
                expected = dict(version='human_authorized_cumulative_extension_v1',
                    question_call='call_ec6a116c31f8441cb621cded7ca170b9',
                    answer='允许按8万累计上限继续',
                    original_limits_sha256=file_record(limit_path)['sha256'],
                    original_started_unix=self.started, original_limits=self.limits,
                    approved_cumulative_http_attempts=80000, inherited_spend=True,
                    deadline_reset=False, edc_excluded=True)
                if (any(extension.get(k) != v or type(extension.get(k)) is not type(v)
                        for k,v in expected.items()) or self.effective_http_cap >= 80000):
                    raise ValueError('Cumulative budget extension authorization differs from original limits or human approval')
                self.effective_http_cap = 80000
                self.budget_extension = file_record(extension_path)
            extension_path = self.root/'budget_extension_authorization_20261023.json'
            if extension_path.exists():
                extension = read_document(extension_path)
                expected = dict(version='human_authorized_cumulative_extension_v2',
                    question_call='call_6b80ba3cee9a4340b6b8331f6f92d189',
                    answer='允许累计20万次、原起点96小时，按有界子额度执行',
                    original_limits_sha256=file_record(limit_path)['sha256'],
                    previous_authorization_sha256=(self.budget_extension or {}).get('sha256'),
                    original_started_unix=self.started, original_limits=self.limits,
                    approved_cumulative_http_attempts=200000, approved_total_seconds=345600,
                    inherited_spend=True, deadline_reset=False, edc_excluded=True)
                if (not self.budget_extension or self.limits != dict(max_http_attempts=30000,
                        max_seconds=172800, max_concurrency=4) or self.started != 1792604536.907112
                        or any(extension.get(k) != v or type(extension.get(k)) is not type(v)
                               for k, v in expected.items())):
                    raise ValueError('200000/96h authorization differs from human approval or inherited ledger')
                self.effective_http_cap = 200000
                self.effective_max_seconds = 345600
                self.budget_extension = file_record(extension_path)
            with self._writer_lock(30):
                with closing(sqlite3.connect(self.database, timeout=30)) as db:
                    db.execute('PRAGMA journal_mode=WAL')
                    db.executescript('''
                        CREATE TABLE IF NOT EXISTS events(
                          id INTEGER PRIMARY KEY,created REAL,key TEXT,stage TEXT,status TEXT,code INTEGER);
                        CREATE INDEX IF NOT EXISTS event_status ON events(status,id);
                        CREATE TABLE IF NOT EXISTS receipts(key TEXT PRIMARY KEY,record TEXT NOT NULL);
                    ''')
        self.original_record = runtime._record
        if getattr(runtime, '_native_auditor', None) is not None:
            raise RuntimeError('A runtime can have only one native auditing owner')
        self.allocation = None
        allocation_path = self.root/'bounded_allocations_20261023.json'
        if self.effective_http_cap == 200000 and not allocation_path.exists():
            raise ValueError('200000/96h authorization requires a bounded allocation')
        if allocation_path.exists():
            allocation = read_document(allocation_path)
            caps = allocation.get('caps', {})
            marker = allocation.get('watermark_event_id')
            inherited = allocation.get('inherited_actual_http')
            if (self.effective_http_cap != 200000 or allocation.get('version') != 'human_authorized_bounded_allocations_v1'
                    or allocation.get('authorization_sha256') != self.budget_extension['sha256']
                    or type(marker) is not int or marker < 0 or type(inherited) is not int or inherited < 0
                    or set(caps) != {'echa', 'identity', 'native', 'blind64'}
                    or any(type(v) is not int or v < 1 for v in caps.values())
                    or sum(caps.values())+inherited > self.effective_http_cap):
                raise ValueError('Bounded allocation authorization exceeds the inherited cumulative budget')
            with closing(sqlite3.connect(self.database, timeout=30)) as db:
                maximum = db.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
                if marker > maximum:
                    raise ValueError('Bounded allocation watermark exceeds the existing ledger boundary')
                actual = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND id<=?", (marker,)).fetchone()[0]
            if actual != inherited:
                raise ValueError('Bounded allocation inherited spending differs from shared ledger')
            self.allocation = allocation
        # 保留无事务连接，避免短连接反复触发末连接WAL清理和重建。
        self._audit_keepalive = sqlite3.connect(self.database, timeout=30,
            isolation_level=None, check_same_thread=False)
        with closing(self._audit_keepalive.execute('SELECT COUNT(*) FROM events')) as cursor:
            cursor.fetchone()
        runtime._native_auditor = self
        runtime._record = self._record

    def _record(self, key, stage, status, value):
        code = value.get('status_code') if isinstance(value, dict) else None
        def record(db):
            if status == 'request_started':
                self.raise_if_fatal()
                count = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started'").fetchone()[0]
                if count >= self.effective_http_cap or time.time()-self.started >= self.effective_max_seconds:
                    _atomic(self.root/'budget_halt.json', dict(status='budget_exhausted',
                        http_attempts=count, created_unix=time.time()))
                    raise NativeBudgetExceeded('Frozen native gate budget exhausted; preserve receipts')
                if self.allocation:
                    prefixes = [('echa_v1/', 'echa'), ('identity_single_v2.', 'identity'), ('blind64_20261022_v1/', 'blind64')]
                    group = next((g for prefix, g in prefixes if stage.startswith(prefix)), 'native')
                    cases = ' '.join('WHEN substr(stage,1,?)=? THEN ?' for _ in prefixes)
                    params = [v for prefix, g in prefixes for v in (len(prefix), prefix, g)]
                    used = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND id>? AND (CASE "
                        +cases+" ELSE 'native' END)=?", [self.allocation['watermark_event_id'], *params, group]).fetchone()[0]
                    if used >= self.allocation['caps'][group]:
                        raise NativeBudgetExceeded('Bounded '+group+' allocation exhausted; preserve checkpoints')
            if status == 'http_error' and code in {401, 402, 403}:
                _atomic(self.root/'halt.json', dict(status='account_or_permission',
                    status_code=code, created_unix=time.time(), request_id=key))
            db.execute('INSERT INTO events(created,key,stage,status,code) VALUES(?,?,?,?,?)',
                       (time.time(), key, stage, status, code))
        self._write_transaction(record, operation='event_' + status, request_id=key)
        self.original_record(key, stage, status, value)
        if status == 'http_error' and code in {401, 402, 403}:
            raise NativeAccountUnavailable(f'Official model returned HTTP {code}; task halted')

    def _write_transaction(self, operation_fn, *, operation, request_id=None):
        # 只串行短账本事务；网络和原transport写入必须在这把锁之外。
        started = time.monotonic()
        deadline = started + 30
        attempt = 0
        while True:
            attempt += 1
            remaining = max(.001, deadline-time.monotonic())
            try:
                with self._writer_lock(remaining):
                    with closing(sqlite3.connect(self.database, timeout=min(.25,remaining))) as db:
                        try:
                            db.execute('BEGIN IMMEDIATE')
                            result = operation_fn(db)
                            db.commit()
                            return result
                        except BaseException:
                            db.rollback()
                            raise
            except sqlite3.OperationalError as exc:
                code = getattr(exc,'sqlite_errorcode',None)
                busy = code is not None and code & 255 in {sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED}
                if not busy and str(exc) not in {'database is locked','database table is locked'}:
                    raise
                self._storage_contention(dict(status='sqlite_busy_retry', operation=operation,
                    request_id=request_id, attempt=attempt, sqlite_errorcode=code,
                    elapsed_seconds=time.monotonic()-started))
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(.05,deadline-time.monotonic()))
            except Timeout:
                self._storage_contention(dict(status='sqlite_writer_coordination_timeout', operation=operation,
                    request_id=request_id, attempt=attempt, elapsed_seconds=time.monotonic()-started))
                raise TimeoutError('Native accounting storage writer did not become available within 30 seconds') from None

    def _storage_contention(self, value):
        value = dict(value, created_unix=time.time(), pid=os.getpid())
        with FileLock(str(self.root/'storage_contention.lock'), timeout=1):
            with (self.root/'storage_contention.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(value,ensure_ascii=False)+'\n')

    def raise_if_fatal(self):
        halt = self.root/'halt.json'
        if halt.exists():
            code = read_document(halt)['status_code']
            raise NativeAccountUnavailable(f'Official model HTTP {code} halt is still active')
        if (self.root/'budget_halt.json').exists():
            raise NativeBudgetExceeded('Frozen native budget halt is still active')
        if time.time()-self.started >= self.effective_max_seconds:
            raise NativeBudgetExceeded('Frozen native total deadline exhausted')

    @contextmanager
    def _try_slot_lock(self, path):
        if os.name == 'posix':
            import fcntl
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            acquired = False
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                yield acquired
            finally:
                if acquired:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
        else:
            lock = FileLock(str(path))
            acquired = False
            try:
                try:
                    lock.acquire(timeout=0)
                    acquired = True
                except Timeout:
                    pass
                yield acquired
            finally:
                if acquired:
                    lock.release()

    @contextmanager
    def _writer_lock(self, timeout):
        # 与原FileLock使用同一inode；争用只读flock，不反复截断或改权限。
        path = self.root/'database_write.lock'
        deadline = time.monotonic() + timeout
        while True:
            with self._try_slot_lock(path) as acquired:
                if acquired:
                    yield
                    return
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise Timeout(str(path))
            time.sleep(min(.1, remaining))

    @contextmanager
    def _slot(self):
        started = time.monotonic()
        while True:
            self.raise_if_fatal()
            for i in range(self.limits['max_concurrency']):
                # 每次独立fd，不截断/改权限，避免争用轮询制造磁盘日志写入。
                with self._try_slot_lock(self.root/'locks'/f'slot_{i}.lock') as acquired:
                    if acquired:
                        yield
                        return
            if self.effective_max_seconds == self.limits['max_seconds'] and time.monotonic()-started > 600:
                raise TimeoutError('Native global request slots remained occupied for 600 seconds')
            time.sleep(.1)

    def _cached(self, key, endpoint, body):
        with closing(sqlite3.connect(self.database, timeout=30)) as db:
            found = db.execute('SELECT record FROM receipts WHERE key=?', (key,)).fetchone()
        if found is None:
            return None
        record = json.loads(found[0])
        if file_record(record['path']) != record:
            raise ValueError('Native receipt file changed')
        value = read_document(record['path'])
        if value['request_id'] != key or value['endpoint'] != endpoint or value['request'] != body:
            raise ValueError('Native receipt identity changed')
        return value['response']

    def _validate_request(self, endpoint, body):
        check_runtime(self.runtime.config)
        if endpoint not in {'/chat/completions', '/embeddings'}:
            raise ValueError('Only frozen official chat/embedding endpoint paths allowed')
        opts = self.runtime.config['llm' if endpoint == '/chat/completions' else 'embedding']
        if body.get('model') != opts['model']:
            raise ValueError('Native model override rejected')
        if endpoint == '/chat/completions':
            if (body.get('enable_thinking') is not False or body.get('temperature') != 0 or
                    body.get('seed') != 42 or type(body.get('max_tokens')) is not int or body['max_tokens'] < 1):
                raise ValueError('Native generation must preserve frozen model controls')
            allowed = {'model', 'messages', 'max_tokens', 'temperature', 'seed',
                       'enable_thinking', 'response_format', 'stop'}
            if set(body)-allowed or not body.get('messages'):
                raise ValueError('Unsupported native chat request parameters')
        else:
            if set(body)-{'model','input','encoding_format'} or not isinstance(body.get('input'), list) or not body['input']:
                raise ValueError('Unsupported native embedding request parameters')
            if any(not isinstance(t,str) for t in body['input']):
                raise ValueError('Native embedding inputs must be strings')
        return opts

    def request_executor(self, endpoint, body, stage):
        opts = self._validate_request(endpoint, body)
        key = digest(dict(endpoint=endpoint, body=body, config=self.runtime.config))
        with FileLock(str(self.root/'locks'/(key+'.lock')), timeout=1800):
            cached = self._cached(key, endpoint, body)
            if cached is not None:
                return cached, True
            self.raise_if_fatal()
            # 原传输层的每次实际重试均经过_record，不能绕过总额度。
            with self._slot():
                raw = self.runtime._raw_cached(key)
                recovered = raw is not None
                if raw is None:
                    raw = self.runtime._post(endpoint, body, opts, key, stage)
            path = self.root/'receipts'/(key+'.json')
            _atomic(path, dict(request_id=key, endpoint=endpoint, request=body,
                               response=raw, runtime_config=self.runtime.config,
                               response_source='runtime_recovery' if recovered else 'new_http',
                               completed_unix=time.time()))
            rec = file_record(path)
            self._write_transaction(lambda db:db.execute('INSERT INTO receipts(key,record) VALUES(?,?)',
                (key,json.dumps(rec))), operation='receipt_publish', request_id=key)
            return raw, recovered

    def chat_response(self, messages, *, max_tokens=4096, response_format=None, stage='native_chat'):
        body = dict(model=self.runtime.config['llm']['model'], messages=messages,
                    temperature=0, seed=42, enable_thinking=False, max_tokens=max_tokens)
        if response_format is not None:
            body['response_format'] = response_format
        raw, hit = self.request_executor('/chat/completions', body, stage)
        return dict(raw, _native_cache_hit=hit)

    def chat(self, messages, **kwargs):
        raw = self.chat_response(messages, **kwargs)
        choice = raw['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Native completion failed or was truncated; inspect preserved receipt')
        value = choice['message']['content']
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Native completion content is empty')
        return value

    def embed(self, texts, *, stage='native_embedding'):
        texts = list(texts)
        if not texts:
            return np.empty((0,1024), dtype=np.float32)
        batches = []
        for first in range(0,len(texts),16):
            batch = texts[first:first+16]
            body = dict(model=self.runtime.config['embedding']['model'], input=batch, encoding_format='float')
            raw, _ = self.request_executor('/embeddings', body, stage)
            rows = raw['data']
            if len(rows) != len(batch) or {r.get('index') for r in rows} != set(range(len(batch))):
                raise ValueError('Native embedding response indices differ')
            vectors = np.asarray([r['embedding'] for r in sorted(rows,key=lambda r:r['index'])], dtype=np.float32)
            if vectors.shape != (len(batch),1024) or not np.isfinite(vectors).all():
                raise ValueError('Native BGE-M3 requires finite (N,1024) vectors')
            norm = np.linalg.norm(vectors,axis=1,keepdims=True)
            if (norm <= 0).any():
                raise ValueError('Native embedding contains zero vector')
            batches.append(vectors/norm)
        return np.concatenate(batches)

    async def achat(self, messages, **kwargs):
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    async def aembed(self, texts, **kwargs):
        return await asyncio.to_thread(self.embed, texts, **kwargs)

    def statistics(self):
        with closing(sqlite3.connect(self.database, timeout=30)) as db:
            counts = dict(db.execute('SELECT status,COUNT(*) FROM events GROUP BY status'))
            receipt_count = db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0]
        return dict(http_attempts=counts.get('request_started',0),
                    successful_responses=counts.get('response',0), receipt_count=receipt_count,
                    status_counts=counts, limits=self.limits, started_unix=self.started,
                    effective_max_seconds=self.effective_max_seconds,
                    deadline_unix=self.started+self.effective_max_seconds,
                    effective_http_cap=self.effective_http_cap, budget_extension=self.budget_extension)

    def close(self):
        self.runtime._record = self.original_record
        self.runtime._native_auditor = None
        close = getattr(self.runtime,'close',None)
        try:
            if close is not None:
                close()
        finally:
            if self._audit_keepalive is not None:
                self._audit_keepalive.close()
                self._audit_keepalive = None
