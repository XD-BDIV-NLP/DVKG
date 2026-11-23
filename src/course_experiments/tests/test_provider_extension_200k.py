from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import sqlite3
from contextlib import contextmanager, closing
from unittest.mock import patch

from course_experiments.evaluation.native_provider import AuditedNativeProvider, NativeBudgetExceeded
from course_experiments.manifest import file_record
from course_experiments.tests.test_native_provider import FakeRuntime


class Approved200kTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.started = 1792604536.907112
        self.limits = dict(max_http_attempts=30000, max_seconds=172800, max_concurrency=4)
        (self.root/'limits.json').write_text(json.dumps(dict(limits=self.limits, started_unix=self.started)))
        old = dict(version='human_authorized_cumulative_extension_v1',
            question_call='call_ec6a116c31f8441cb621cded7ca170b9', answer='允许按8万累计上限继续',
            original_limits_sha256=file_record(self.root/'limits.json')['sha256'],
            original_started_unix=self.started, original_limits=self.limits,
            approved_cumulative_http_attempts=80000, inherited_spend=True, deadline_reset=False, edc_excluded=True)
        self.previous = self.root/'budget_extension_authorization_20261022.json'
        self.previous.write_text(json.dumps(old))

    def extension(self, **changes):
        value = dict(version='human_authorized_cumulative_extension_v2',
            question_call='call_6b80ba3cee9a4340b6b8331f6f92d189',
            answer='允许累计20万次、原起点96小时，按有界子额度执行',
            original_limits_sha256=file_record(self.root/'limits.json')['sha256'],
            previous_authorization_sha256=file_record(self.previous)['sha256'],
            original_started_unix=self.started, original_limits=self.limits,
            approved_cumulative_http_attempts=200000, approved_total_seconds=345600,
            inherited_spend=True, deadline_reset=False, edc_excluded=True)
        value.update(changes)
        auth = self.root/'budget_extension_authorization_20261023.json'
        auth.write_text(json.dumps(value))
        marker = inherited = 0
        if (self.root/'audit.sqlite').exists():
            with closing(sqlite3.connect(self.root/'audit.sqlite')) as db:
                marker = db.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
                inherited = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started'").fetchone()[0]
        allocation = dict(version='human_authorized_bounded_allocations_v1',authorization_sha256=file_record(auth)['sha256'],
            watermark_event_id=marker,inherited_actual_http=inherited,caps=dict(echa=60000,identity=33000,native=45000,blind64=5000))
        (self.root/'bounded_allocations_20261023.json').write_text(json.dumps(allocation))

    def provider(self):
        p = AuditedNativeProvider(FakeRuntime(), self.root, **self.limits)
        self.addCleanup(p.close); return p

    def ask(self, provider, text):
        return provider.chat_response([dict(role='user', content=text)], max_tokens=10)

    def test_approval_inherits_original_spend_bytes_and_cache_beyond_old_deadline(self):
        original = (self.root/'limits.json').read_bytes(); previous = self.previous.read_bytes()
        with patch('course_experiments.evaluation.native_provider.time.time', return_value=self.started+1):
            old = self.provider(); self.ask(old, 'first')
        self.extension()
        with patch('course_experiments.evaluation.native_provider.time.time', return_value=self.started+200000):
            p = self.provider()
            self.assertTrue(self.ask(p, 'first')['_native_cache_hit'])
            self.ask(p, 'second')
            stats = p.statistics()
        self.assertEqual(stats['http_attempts'], 2)
        self.assertEqual(stats['effective_http_cap'], 200000)
        self.assertEqual(stats['effective_max_seconds'], 345600)
        self.assertEqual(stats['deadline_unix'], self.started+345600)
        self.assertEqual(stats['started_unix'], self.started)
        self.assertEqual(stats['limits'], self.limits)
        self.assertEqual((self.root/'limits.json').read_bytes(), original)
        self.assertEqual(self.previous.read_bytes(), previous)

    def test_extension_rejects_changed_predecessor_clock_limit_or_answer(self):
        for changes in [dict(previous_authorization_sha256='changed'),
                dict(original_started_unix=self.started+1), dict(approved_total_seconds=345601),
                dict(approved_cumulative_http_attempts=200001), dict(inherited_spend=False),
                dict(deadline_reset=True), dict(edc_excluded=False), dict(answer='unapproved'),
                dict(approved_total_seconds=True)]:
            self.extension(**changes)
            with self.assertRaisesRegex(ValueError, 'authorization'):
                self.provider()

    def test_expiry_and_existing_halt_still_prevent_new_http(self):
        self.extension(); p = self.provider()
        with patch('course_experiments.evaluation.native_provider.time.time', return_value=self.started+345601):
            with self.assertRaises(NativeBudgetExceeded): self.ask(p, 'expired')
        (self.root/'budget_halt.json').write_text('{}')
        with patch('course_experiments.evaluation.native_provider.time.time', return_value=self.started+100):
            with self.assertRaises(NativeBudgetExceeded): self.ask(p, 'halted')
        self.assertEqual(p.statistics()['http_attempts'], 0)

    def test_bounded_allocations_count_from_shared_watermark_without_halting_other_groups(self):
        self.extension(); old = self.provider(); self.ask(old, 'inherited')
        with closing(sqlite3.connect(self.root/'audit.sqlite')) as db:
            marker = db.execute('SELECT MAX(id) FROM events').fetchone()[0]
        authorization = file_record(self.root/'budget_extension_authorization_20261023.json')
        allocation = dict(version='human_authorized_bounded_allocations_v1', authorization_sha256=authorization['sha256'],
            watermark_event_id=marker, inherited_actual_http=1,
            caps=dict(echa=2, identity=2, native=1, blind64=2))
        (self.root/'bounded_allocations_20261023.json').write_text(json.dumps(allocation))
        p = self.provider(); self.ask(p, 'native one')
        with self.assertRaisesRegex(NativeBudgetExceeded, 'allocation'):
            self.ask(p, 'native second')
        self.assertFalse((self.root/'budget_halt.json').exists())
        p.chat_response([dict(role='user', content='echa request')], max_tokens=10, stage='echa_v1/evaluate')
        self.assertEqual(p.statistics()['http_attempts'], 3)
        self.assertTrue(self.ask(p, 'native one')['_native_cache_hit'])

    def test_new_authorization_waits_past_old_slot_timeout_without_adding_slots(self):
        self.extension(); p = self.provider(); counter = []
        @contextmanager
        def attempt(path):
            counter.append(str(path)); yield len(counter)>4
        with patch.object(p, '_try_slot_lock', side_effect=attempt), \
                patch('course_experiments.evaluation.native_provider.time.monotonic', side_effect=[0,601,602]), \
                patch('course_experiments.evaluation.native_provider.time.sleep'):
            with p._slot(): pass
        self.assertEqual(len(counter), 5)
        self.assertTrue(all('slot_'+str(i)+'.lock' in counter[i] for i in range(4)))

    def test_new_authorization_without_bounded_allocation_is_rejected(self):
        self.extension(); (self.root/'bounded_allocations_20261023.json').unlink()
        with self.assertRaisesRegex(ValueError, 'allocation'):
            self.provider()

    def test_future_watermark_cannot_hide_new_http_from_allocations(self):
        self.extension(); path = self.root/'bounded_allocations_20261023.json'
        value = json.loads(path.read_text()); value['watermark_event_id'] = 1000000
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'watermark'):
            self.provider()


if __name__ == '__main__': unittest.main()
