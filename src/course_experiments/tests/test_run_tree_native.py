from __future__ import annotations

import copy
from contextlib import closing
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from course_experiments.manifest import digest, file_record
from course_experiments.tests.test_tree_kg import CONFIG, FakeProvider, structure


class Provider(FakeProvider):
    def __init__(self):
        super().__init__();self.closed=False
    def statistics(self):return dict(http_attempts=len(self.calls))
    def close(self):self.closed=True


class TreeNativeCLITests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.evaluation.run_tree_native'),
            'Tree native corpus CLI is not implemented')
        self.run=importlib.import_module('course_experiments.evaluation.run_tree_native')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.parent=self.root/'parent';self.out=self.root/'out'
        self.books=list(self.run.AUTHORIZED_BOOKS)
        assets=dict(unit_assets={},structures={},cohorts={})
        def save(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False),'utf-8');return file_record(path)
        config=save(self.parent/'runtime.json',CONFIG)
        construction=save(self.parent/'construction.json',dict(files=dict(assets=dict(runtime_config=config))))
        for book in self.books:
            s=structure(book)
            rows=[dict(book_id=book,chunk_id=r['chunk_id'],text=r['text'],evidence_ids=[eid]) for eid,r in s['evidence'].items()]
            assets['unit_assets'][book]=save(self.parent/(book+'_units.json'),dict(rows=rows))
            assets['structures'][book]=save(self.parent/(book+'_structure.json'),s)
        for key,books in self.run.COHORT_BOOKS.items():
            assets['cohorts'][key]=dict(active_books=list(books),construction=construction)
        save(self.parent/'assets/result.json',assets)
        # 如果准备器偷读问题/答案，下面的非法JSON会使测试失败。
        (self.parent/'datasets').mkdir();(self.parent/'datasets/index.json').write_text('NEVER READ REFERENCE ANSWERS','utf-8')

    def plan(self,**kwargs):
        return self.run.prepare_plan(self.out,parent=self.parent,**kwargs)

    def test_full_preparation_has_four_graph_groups_eight_unique_books_and_exact_quote_positions(self):
        plan=self.plan(scope='full',cohort='all')
        self.assertEqual(set(plan['jobs']),set(self.run.COHORT_BOOKS))
        self.assertEqual(len(plan['book_inputs']),8)
        self.assertEqual(plan['contract']['unique_book_count'],8)
        self.assertFalse(plan['contract']['uses_questions_or_answers'])
        for book,item in plan['book_inputs'].items():
            self.assertTrue(item['audit']['all_evidence_covered'])
            self.assertEqual(item['audit']['source_unit_count'],2)
            for eid,row in item['structure']['evidence'].items():
                position=row['original_source_position'];passage=item['passages'][position['passage_index']]
                self.assertEqual(passage['text'][position['start']:position['end']],row['text'])
                self.assertEqual(position['evidence_id'],eid)

    def test_gate_selection_is_answer_independent_and_partial_is_never_full(self):
        book=self.books[0]
        first=self.plan(scope='gate',cohort='all',book=book,sample_size=1)
        second=self.plan(scope='gate',cohort='all',book=book,sample_size=1)
        self.assertEqual(first['contract'],second['contract'])
        self.assertEqual(first['contract']['scope'],'gate')
        self.assertEqual(len(first['book_inputs'][book]['passages']),1)
        self.assertFalse(first['book_inputs'][book]['audit']['all_original_book_evidence_covered'])
        self.assertEqual(len(first['book_inputs'][book]['structure']['evidence']),1)

    def test_full_rejects_one_book_and_gate_rejects_foreign_book(self):
        with self.assertRaises(ValueError):self.plan(scope='full',book=self.books[0])
        with self.assertRaises(ValueError):self.plan(scope='gate',book='foreign')

    def test_source_tampering_and_missing_evidence_fail_before_provider_creation(self):
        path=self.parent/(self.books[0]+'_units.json')
        value=json.loads(path.read_text('utf-8'));value['rows'].pop();path.write_text(json.dumps(value),'utf-8')
        with self.assertRaises(ValueError):self.plan(scope='full')

    def test_shared_formal_provider_must_already_exist_and_cannot_reset_limits(self):
        with self.assertRaises(ValueError):self.run.require_formal_provider(self.out)
        formal=self.out/'formal_provider';formal.mkdir(parents=True)
        (formal/'limits.json').write_text(json.dumps(dict(limits=self.run.LIMITS,started_unix=1)),'utf-8')
        with self.assertRaises(ValueError):self.run.require_formal_provider(self.out)
        (formal/'audit.sqlite').write_bytes(b'')
        with self.assertRaises(ValueError):self.run.require_formal_provider(self.out)
        with closing(sqlite3.connect(formal/'audit.sqlite')) as db:
            db.execute('CREATE TABLE events(id INTEGER)');db.execute('CREATE TABLE receipts(key TEXT)')
            db.commit()
        self.run.require_formal_provider(self.out)
        altered=dict(self.run.LIMITS,max_http_attempts=30001)
        (formal/'limits.json').write_text(json.dumps(dict(limits=altered,started_unix=1)),'utf-8')
        with self.assertRaises(ValueError):self.run.require_formal_provider(self.out)

    def test_injected_gate_pipeline_preserves_sources_NA_KGE_and_resume(self):
        plan=self.plan(scope='gate',book=self.books[0],sample_size=2)
        provider=Provider();first=self.run.run_plan(plan,self.out,provider)
        count=len(provider.calls);second=self.run.run_plan(plan,self.out,provider)
        self.assertEqual(first,second);self.assertEqual(count,len(provider.calls))
        self.assertEqual(first['status'],'complete')
        self.assertFalse(first['full_benchmark_completed'])
        self.assertFalse(provider.closed)
        row=next(iter(first['jobs'].values()))
        self.assertIsNone(row['capabilities']['kge'])
        self.assertFalse(row['unmodified_native_reproduction'])
        artifact=json.loads(Path(row['artifact']['path']).read_text('utf-8'))
        self.assertTrue(all(e.get('original_source_position') for e in artifact['evidence']))

    def test_budget_exception_keeps_no_complete_summary_and_provider_owned_by_caller(self):
        from course_experiments.evaluation.native_provider import NativeBudgetExceeded
        plan=self.plan(scope='gate',book=self.books[0],sample_size=1)
        provider=Provider();provider.fatal=True
        with self.assertRaises(NativeBudgetExceeded):self.run.run_plan(plan,self.out,provider)
        self.assertFalse(list(self.out.glob('tree_kg/gate/*/complete.json')))
        self.assertFalse(provider.closed)

    def test_full_requires_same_identity_real_gate_before_running_then_covers_all_eight_books(self):
        plan=self.plan(scope='full',cohort='all');provider=Provider()
        with self.assertRaisesRegex(ValueError,'gate'):self.run.run_plan(plan,self.out,provider)
        self.assertFalse(provider.calls)
        gate=self.plan(scope='gate',book=self.books[0],sample_size=1)
        self.run.run_plan(gate,self.out,provider)
        full=self.run.run_plan(plan,self.out,provider)
        self.assertTrue(full['full_eight_book_construction_completed'])
        self.assertFalse(full['full_benchmark_completed']);self.assertFalse(full['kge_completed'])
        self.assertEqual(set(full['jobs']),set(self.run.COHORT_BOOKS))
        found=set()
        for row in full['jobs'].values():
            artifact=json.loads(Path(row['artifact']['path']).read_text('utf-8'))
            found.update(e['id'] for e in artifact['evidence'])
        self.assertEqual(len(found),16)

    def test_cli_check_only_has_no_credentials_runtime_or_model_calls(self):
        with patch.object(self.run,'credentials',side_effect=AssertionError('no secrets in check-only')),patch.object(
                self.run,'runtime_for',side_effect=AssertionError('no runtime in check-only')):
            self.assertEqual(self.run.main(['--run-root',str(self.out),'--parent',str(self.parent),
                '--scope','full','--cohort','all','--check-only']),0)


if __name__=='__main__':unittest.main()
