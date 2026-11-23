from __future__ import annotations

import copy
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from course_experiments.construction import PROMPTS, ROOT
from course_experiments.contributions import export_contribution
from course_experiments.manifest import digest, file_record
from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
from domain_ckg_workspace.ckg_incremental.semantic_merge import SemanticMerger


class ExactCacheTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.semantic_cache'),
                             'Exact cache adapter is missing')
        from course_experiments import semantic_cache
        return semantic_cache

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = dict(workers=1, llm=dict(model='offline'), embedding=dict(model='offline'))
        self.runtime = ModelRuntime(self.root/'target', PROMPTS, self.config)
        self.addCleanup(self.runtime.close)
        self.calls = []
        def call(stage, payload, validate):
            self.calls.extend(p['pair_id'] for p in payload['pairs'])
            result = dict(decisions={p['pair_id']:dict(decision='separate', reason='fresh', evidence_ids=[])
                                    for p in payload['pairs']})
            validate(result)
            return result
        self.runtime.call = call
        self.runtime.embed = lambda texts: [np.array([0.9, 0.0], dtype=np.float32) for _ in texts]
        self.graph = dict(nodes={}, claims={}, evidence={}, books={'b':dict(id='b', title='Book')}, events=[])
        for name in ('a', 'z'):
            eid = 'e'+name
            self.graph['evidence'][eid] = dict(id=eid, book_id='b', text='Definition '+name, title_path=['Chapter'])
            self.graph['nodes'][name] = dict(id=name, name='Concept '+name, entity_type='Concept', aliases=['shared','alias '+name],
                definition='Definition '+name, definition_status='explicit', scope='scope',
                evidence_ids=[eid], definition_evidence_ids=[eid],
                definitions=[dict(value='Definition '+name, evidence_ids=[eid], conditions='condition'),
                             dict(value='Other',evidence_ids=['ea' if name=='z' else 'ez'],conditions='other')],
                scope_variants=['scope'])
        self.native_pair = SemanticMerger(None, self.graph['evidence'], self.graph['books'])._pair(
            self.graph['nodes']['z'], self.graph['nodes']['a'])
        self.decision = dict(pair_id=self.native_pair['pair_id'], decision='merge', reason='same',
                             evidence_ids=['ez', 'ea'], left_qualifier='left', right_qualifier='right', extra={'keep':True})
        self.book = export_contribution(self.graph, 'c', 'b', 'full', 'input')
        self.g0 = export_contribution(dict(nodes={}, claims={}, evidence={}, books={}, events=[]), 'c', None, 'full', 'g0')

    def freeze(self, decision=True, asset_changes=None):
        api = self.api()
        corpus = dict(books=[dict(id='b')], chunks=[dict(id='chunk',book_id='b')], evidence=list(self.graph['evidence'].values()))
        corpus['fingerprint'] = digest(corpus)
        corpus_path = self.root/'corpus.json'
        corpus_path.write_text(json.dumps(corpus), 'utf-8')
        config_path = self.root/'config.json'
        config_path.write_text(json.dumps(self.config), 'utf-8')
        assets = {p:file_record(ROOT/p) for p in api.CODE_PATHS}
        assets.update({name:file_record(PROMPTS/name) for pair in ModelRuntime.FILES.values() for name in pair})
        assets['runtime_config'] = file_record(config_path)
        if asset_changes is not None:
            asset_changes(assets)
        manifest = dict(courses=dict(courses=[dict(course_id='c',books=[dict(book_id='b')])]),
                        files=dict(assets=assets,sources={'b':file_record(corpus_path)}))
        manifest['manifest_id'] = digest(manifest)
        manifest_path = self.root/'construction.json'
        manifest_path.write_text(json.dumps(manifest), 'utf-8')
        scoped = dict(corpus, construction_scope='full', source_corpus_fingerprint=corpus['fingerprint'], smoke_chunks=None)
        scoped.pop('fingerprint')
        self.book['input_id'] = digest(dict(manifest=manifest['manifest_id'], book_id='b', corpus=digest(scoped)))
        self.book['artifact_id'] = digest({k:v for k,v in self.book.items() if k!='artifact_id'})
        contribution_path = self.root/'contribution.json'
        contribution_path.write_text(json.dumps(self.book), 'utf-8')
        identity = {k:self.book[k] for k in ('input_id','scope','course_id','book_id')}
        identity_path = self.root/'construction_identity.json'
        identity_path.write_text(json.dumps(identity), 'utf-8')
        cache_path = self.root/'frozen.sqlite'
        with closing(sqlite3.connect(cache_path)) as db, db:
            db.execute('CREATE TABLE semantic_decisions(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            if decision:
                db.execute('INSERT INTO semantic_decisions VALUES (?,?)',
                    (digest([self.runtime.semantic_fingerprint,self.native_pair['pair_id']]),json.dumps(self.decision)))
        return dict(source_kind='completed_independent_full_book', **identity, artifact_id=self.book['artifact_id'],
                    contribution=file_record(contribution_path), cache=file_record(cache_path),
                    construction_identity=file_record(identity_path), construction_manifest=file_record(manifest_path),
                    semantic_fingerprint=self.runtime.semantic_fingerprint, runtime_fingerprint=self.runtime.fingerprint,
                    code_hashes={p:file_record(ROOT/p)['sha256'] for p in api.CODE_PATHS})

    def run_prefill(self, source):
        return self.api().prefill_exact_cache(self.g0,[self.book],['b'],runtime=self.runtime,sources=[source])

    def current_pair(self):
        graph = self.book['graph']
        return SemanticMerger(None,graph['evidence'],graph['books'])._pair(graph['nodes']['c/b/node/z'],graph['nodes']['c/b/node/a'])

    def test_exact_hit_roundtrip_preserves_all_fields_and_normal_judge_uses_cache(self):
        source = self.freeze()
        original = Path(source['cache']['path']).read_bytes()
        report = self.run_prefill(source)
        pair = self.current_pair()
        expected = dict(self.decision,pair_id=pair['pair_id'], evidence_ids=['c/b/evidence/ez','c/b/evidence/ea'])
        self.assertEqual(self.runtime.semantic_decision(pair['pair_id']), expected)
        merger = SemanticMerger(self.runtime)
        merger._judge([pair])
        self.assertEqual(merger.decisions[pair['pair_id']],expected)
        self.assertEqual(self.calls,[])
        self.assertEqual(report['counts']['hit'],1)
        self.assertEqual(report['hits'][0]['source_pair_id'],self.native_pair['pair_id'])
        self.assertEqual(report['hits'][0]['rebound_decision_hash'],digest(expected))
        self.assertTrue(all('SEARCH' in s and 'INDEX' in s for s in report['sources'][0]['query_plan']))
        self.assertEqual(Path(source['cache']['path']).read_bytes(),original)

    def test_miss_remains_normal_request(self):
        report = self.run_prefill(self.freeze(False))
        pair = self.current_pair()
        self.assertEqual(report['counts']['nohit'],1)
        self.assertIsNone(self.runtime.semantic_decision(pair['pair_id']))
        SemanticMerger(self.runtime)._judge([pair])
        self.assertEqual(self.calls,[pair['pair_id']])
        self.assertEqual(self.runtime.semantic_decision(pair['pair_id'])['reason'],'fresh')

    def test_duplicate_role_prompt_records_allow_exact_reuse(self):
        def aliases(assets):
            assets['g0_prompt']=copy.deepcopy(assets[ModelRuntime.FILES['prior'][1]])
            assets['textbook_prompt']=copy.deepcopy(assets[ModelRuntime.FILES['extract_graph'][1]])
        source=self.freeze(asset_changes=aliases)
        before=Path(source['construction_manifest']['path']).read_bytes()
        try:
            report=self.run_prefill(source)
        except ValueError as exc:
            self.fail(f'Identical frozen prompt records under different roles were rejected: {exc}')
        self.assertEqual(report['counts']['hit'],1)
        self.assertEqual(self.runtime.semantic_decision(self.current_pair()['pair_id'])['decision'],'merge')
        self.assertEqual(Path(source['construction_manifest']['path']).read_bytes(),before)
        self.assertEqual(self.calls,[])

    def test_same_prompt_basename_at_distinct_path_is_ambiguous_before_write(self):
        def aliases(assets):
            name=ModelRuntime.FILES['prior'][1]
            other=self.root/'other'/name
            other.parent.mkdir()
            other.write_bytes(Path(assets[name]['path']).read_bytes())
            assets['g0_prompt']=file_record(other)
        source=self.freeze(asset_changes=aliases)
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            self.run_prefill(source)
        self.assertEqual(self.runtime.db.execute('SELECT count(*) FROM semantic_decisions').fetchone()[0],0)

    def test_same_prompt_path_with_distinct_hash_is_ambiguous_before_write(self):
        def aliases(assets):
            name=ModelRuntime.FILES['prior'][1]
            assets['g0_prompt']=dict(assets[name],sha256='0'*64)
        source=self.freeze(asset_changes=aliases)
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            self.run_prefill(source)
        self.assertEqual(self.runtime.db.execute('SELECT count(*) FROM semantic_decisions').fetchone()[0],0)

    def test_duplicate_prompt_records_still_require_file_hash_integrity(self):
        def aliases(assets):
            name=ModelRuntime.FILES['prior'][1]
            assets[name]=dict(assets[name],sha256='0'*64)
            assets['g0_prompt']=dict(assets[name])
        source=self.freeze(asset_changes=aliases)
        with self.assertRaisesRegex(ValueError,'file hash mismatch'):
            self.run_prefill(source)
        self.assertEqual(self.runtime.db.execute('SELECT count(*) FROM semantic_decisions').fetchone()[0],0)

    def test_existing_target_preserved(self):
        source = self.freeze()
        pair = self.current_pair()
        existing = dict(self.decision, pair_id=pair['pair_id'],decision='separate',evidence_ids=[])
        self.runtime.save_semantic_decisions({pair['pair_id']:existing})
        report = self.run_prefill(source)
        self.assertEqual(report['counts']['preservedexisting'],1)
        self.assertEqual(self.runtime.semantic_decision(pair['pair_id']),existing)

    def test_invalid_decision_is_audited_without_repair_or_write(self):
        self.decision['evidence_ids'] = ['not-supplied']
        report = self.run_prefill(self.freeze())
        self.assertEqual(report['counts']['invalid'],1)
        self.assertIsNone(self.runtime.semantic_decision(self.current_pair()['pair_id']))
        self.assertEqual(self.calls,[])

    def test_all_judge_contract_invalid_fields_are_rejected(self):
        for change in [dict(pair_id='wrong'),dict(decision='yes'),dict(reason=' '),dict(evidence_ids='ea'),
                       dict(evidence_ids=[42]),dict(left_qualifier='x'*61),dict(right_qualifier=None)]:
            with self.subTest(change=change):
                self.assertFalse(self.api()._valid_decision(dict(self.decision,**change), self.native_pair))

    def test_changed_fingerprint_or_code_rejects_before_target_write(self):
        source = self.freeze()
        for field in ('semantic_fingerprint','runtime_fingerprint','code_hashes'):
            changed = copy.deepcopy(source)
            changed[field] = {} if field=='code_hashes' else 'different'
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_prefill(changed)
        self.assertIsNone(self.runtime.semantic_decision(self.current_pair()['pair_id']))

    def test_snapshot_and_artifact_hash_tamper_rejected(self):
        source = self.freeze()
        for field in ('cache','contribution','construction_manifest','construction_identity'):
            changed = copy.deepcopy(source);changed[field]['sha256']='0'*64
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_prefill(changed)

    def test_live_cache_and_sidecars_rejected(self):
        source = self.freeze()
        changed = copy.deepcopy(source);changed['cache']=file_record(self.runtime.root/'model_cache.sqlite')
        with self.assertRaises(ValueError):self.run_prefill(changed)
        for suffix in ('-wal','-journal','-shm'):
            path=Path(source['cache']['path']+suffix);path.touch()
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):self.run_prefill(source)
            path.unlink()

    def test_bijection_and_foreign_course_book_scope_rejected(self):
        source = self.freeze()
        for field,value in [('course_id','other'),('book_id','other'),('scope','smoke'),('source_kind','global')]:
            changed=copy.deepcopy(source);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_prefill(changed)
        self.book['native_id_map']['nodes']['extra']='c/b/node/a'
        self.book['artifact_id']=digest({k:v for k,v in self.book.items() if k!='artifact_id'})
        with self.assertRaises(ValueError):self.run_prefill(source)

    def test_changed_current_content_fails_closed(self):
        source=self.freeze()
        self.book['graph']['nodes']['c/b/node/z']['scope']='changed'
        self.book['artifact_id']=digest({k:v for k,v in self.book.items() if k!='artifact_id'})
        with self.assertRaises(ValueError):self.run_prefill(source)

    def test_source_connection_cannot_write(self):
        source=self.freeze()
        with self.api()._open_snapshot(source['cache'],self.runtime) as db:
            with self.assertRaises(sqlite3.OperationalError):db.execute('DELETE FROM semantic_decisions')

    def test_snapshot_changed_after_open_is_rejected(self):
        import os
        source=self.freeze()
        with self.api()._Source(source,self.book,self.runtime) as frozen:
            path=Path(source['cache']['path'])
            stat=path.stat()
            os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+10000000))
            with self.assertRaises(ValueError):frozen.lookup(self.native_pair)

    def test_unindexed_snapshot_is_rejected_before_any_target_write(self):
        source=self.freeze()
        path=self.root/'unindexed.sqlite'
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE semantic_decisions(key TEXT,value TEXT)')
        source['cache']=file_record(path)
        with self.assertRaises(ValueError):self.run_prefill(source)
        self.assertIsNone(self.runtime.semantic_decision(self.current_pair()['pair_id']))

    def test_nonunique_cache_key_schema_is_rejected(self):
        source=self.freeze()
        path=self.root/'ambiguous.sqlite'
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE semantic_decisions(key TEXT,value TEXT)')
            db.execute('CREATE INDEX lookup ON semantic_decisions(key)')
        source['cache']=file_record(path)
        with self.assertRaises(ValueError):self.run_prefill(source)

    def test_runtime_model_changed_after_initialization_is_rejected(self):
        source=self.freeze()
        self.runtime.config=copy.deepcopy(self.runtime.config)
        self.runtime.config['llm']['model']='different'
        with self.assertRaises(ValueError):self.run_prefill(source)

    def test_unchanged_build_view_consumes_prefill_and_inputs_stay_immutable(self):
        from course_experiments.fusion import build_view
        source=self.freeze()
        before=copy.deepcopy([self.g0,self.book])
        report=self.run_prefill(source)
        view=build_view(self.g0,[self.book],['b'],runtime=self.runtime)
        self.assertEqual(report['counts']['hit'],1)
        self.assertEqual(view['alignment_mapping']['c/b/node/z'],'c/b/node/a')
        self.assertEqual(view['alignment_audit'][0]['reason'],'same')
        self.assertEqual(self.calls,[])
        self.assertEqual([self.g0,self.book],before)

    def test_payload_roundtrip_detects_direction_definition_scope_conditions_and_order(self):
        source=self.freeze()
        with self.api()._Source(source,self.book,self.runtime) as frozen:
            pair=self.current_pair()
            self.assertIsNotNone(frozen.rebind(pair,'c/b/node/z','c/b/node/a'))
            mutations=[]
            switched=copy.deepcopy(pair);switched['left'],switched['right']=switched['right'],switched['left'];mutations.append(switched)
            for field,value in [('definition','changed'),('scope','changed'),('aliases',['changed']),('entity_type','Metric')]:
                changed=copy.deepcopy(pair);changed['left'][field]=value;mutations.append(changed)
            changed=copy.deepcopy(pair);changed['left']['definition_variants'][0]['conditions']='changed';mutations.append(changed)
            changed=copy.deepcopy(pair);changed['left']['evidence'][0]['text']='changed';mutations.append(changed)
            for field in ('aliases','definition_variants','evidence'):
                changed=copy.deepcopy(pair);changed['left'][field].reverse();mutations.append(changed)
            for changed in mutations:
                changed['pair_id']=digest({k:v for k,v in changed.items() if k!='pair_id'})
                self.assertIsNone(frozen.rebind(changed,'c/b/node/z','c/b/node/a'))

    def test_g0_crossbook_candidates_are_not_reused(self):
        source=self.freeze()
        graph=copy.deepcopy(self.graph)
        graph['books']={'other':dict(id='other',title='Book')}
        for row in graph['evidence'].values():row['book_id']='other'
        other=export_contribution(graph,'c','other','full','other-input')
        prior=dict(nodes={'p':dict(id='p',name='prior',aliases=['shared'],entity_type='Concept',definition='prior',evidence_ids=[])},
                   claims={},evidence={},books={},events=[])
        g0=export_contribution(prior,'c',None,'full','g0')
        report=self.api().prefill_exact_cache(g0,[self.book,other],['b','other'],runtime=self.runtime,sources=[source])
        self.assertGreater(report['counts']['noneligible'],0)
        self.assertEqual(report['counts']['hit'],1)
        self.assertEqual(self.calls,[])


if __name__ == '__main__':
    unittest.main()
