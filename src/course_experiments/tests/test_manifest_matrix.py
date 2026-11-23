from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class ManifestMatrixTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.manifest'), 'manifest module missing')
        from course_experiments import manifest, matrix
        return manifest, matrix

    def fixture(self, root):
        courses = {'courses': [{'course_id': c, 'description': c, 'g0_inputs': ['description', 'schema'], 'books': [{'alias': a, 'book_id': c + a, 'source_fingerprint': hashlib.sha256((c + a).encode()).hexdigest()} for a in 'AB']} for c in ['one', 'two']]}
        config = {'seed': 42, 'models': {'construction': 'deepseek-ai/DeepSeek-V4-Flash', 'embedding': 'BAAI/bge-m3'}, 'book_sets': [['A'], ['B'], ['A', 'B']], 'primary_states': ['fixed_g0', 'fused'], 'additional_states': ['textbook_only'], 'evaluations': ['rag', 'topology', 'kge'], 'methods': [{'method_id': 'course_g0_textbook_fusion'}, {'method_id': 'baseline', 'commit': None, 'status': 'pending'}], 'budgets': {
            'g0': {'max_output_tokens': 12000, 'temperature': 0},
            'textbook': {'chunk_chars': 4500, 'evidence_chars': 650, 'max_output_tokens': 12000, 'temperature': 0, 'max_retries': 6},
            'rag': {'retrieval_top_k': 5, 'context_token_limit': 1000, 'max_output_tokens': 1000, 'temperature': 0},
            'kge': {'models': ['TransE'], 'epochs': 1, 'batch_size': 4, 'negative_samples': 1, 'learning_rate': 0.01, 'split_ratios': [0.8, 0.1, 0.1]}}}
        assets = {}
        for role in ['schema', 'g0_prompt', 'textbook_prompt', 'rag_prompt', 'questions', 'reference_facts', 'fact_split', 'gpu_environment', 'evaluator_rag', 'evaluator_topology', 'evaluator_kge']:
            path = root / (role + '.txt')
            path.write_text(role, encoding='utf-8')
            assets[role] = str(path)
        sources = {}
        for course in courses['courses']:
            for book in course['books']:
                p = root / (book['book_id'] + '.json')
                corpus = {'books': [{'id': book['book_id'], 'source_fingerprint': book['source_fingerprint']}], 'chunks': [{'id': 'chunk1', 'book_id': book['book_id'], 'evidence_ids': ['e1']}], 'evidence': [{'id': 'e1', 'book_id': book['book_id'], 'text': 'fixture'}]}
                corpus['fingerprint'] = hashlib.sha256(json.dumps(corpus, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                p.write_text(json.dumps(corpus), encoding='utf-8')
                sources[book['book_id']] = str(p)
        return courses, config, sources, assets

    def test_missing_module_and_changed_source_invalidates_manifest(self):
        manifest, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            frozen = manifest.freeze(*args, output=root / 'frozen.json')
            self.assertEqual(manifest.validate(root / 'frozen.json')['manifest_id'], frozen['manifest_id'])
            changed_path = Path(next(iter(args[2].values())))
            changed = json.loads(changed_path.read_text(encoding='utf-8'))
            changed['evidence'][0]['text'] = 'changed'
            del changed['fingerprint']
            changed['fingerprint'] = hashlib.sha256(json.dumps(changed, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            changed_path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'digest'):
                manifest.validate(root / 'frozen.json')

    def test_immutable_freeze_and_g0_excludes_books(self):
        manifest, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            first = manifest.freeze(*args, output=root / 'one.json')
            with self.assertRaises(FileExistsError):
                manifest.freeze(*args, output=root / 'one.json')
            changed_path = Path(next(iter(args[2].values())))
            changed_path.write_text(changed_path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            second = manifest.freeze(*args, output=root / 'two.json')
            self.assertNotEqual(first['manifest_id'], second['manifest_id'])
            self.assertEqual(first['g0_identities'], second['g0_identities'])

    def test_matrix_covers_methods_and_blocks_pending_baseline(self):
        manifest, matrix = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = manifest.freeze(*self.fixture(root), output=root / 'f.json')
            rows = matrix.generate(frozen)
            self.assertEqual(len(rows), 72)
            self.assertEqual({r['state'] for r in rows}, {'fixed_g0', 'fused', 'textbook_only', 'native'})
            native = [r for r in rows if r['state'] == 'native']
            self.assertTrue(all(r['status'] == 'blocked' and r['score'] is None for r in native))
            fixed = [r for r in rows if r['course_id'] == 'one' and r['state'] == 'fixed_g0' and r['evaluation'] == 'topology']
            self.assertEqual(len({r['execution_id'] for r in fixed}), 1)
            rag = [r for r in rows if r['course_id'] == 'one' and r['state'] == 'fixed_g0' and r['evaluation'] == 'rag']
            self.assertEqual(len({r['execution_id'] for r in rag}), 3)
            with self.assertRaisesRegex(ValueError, 'blocked'):
                matrix.require_executable(native[0], root / 'f.json')

    def test_missing_budget_or_evaluator_refuses_freeze(self):
        manifest, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            courses, config, sources, assets = self.fixture(root)
            valid_rag = config['budgets']['rag']
            config['budgets']['rag'] = None
            with self.assertRaisesRegex(ValueError, 'budget'):
                manifest.freeze(courses, config, sources, assets, output=root / 'f.json')
            config['budgets']['rag'] = valid_rag
            del assets['evaluator_kge']
            with self.assertRaisesRegex(ValueError, 'evaluator_kge'):
                manifest.freeze(courses, config, sources, assets, output=root / 'f.json')

    def test_changed_evaluator_and_tampered_row_cannot_execute(self):
        manifest, matrix = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            frozen = manifest.freeze(*args, output=root / 'f.json')
            row = matrix.generate(frozen)[0]
            self.assertEqual(matrix.require_executable(row, root / 'f.json'), row)
            altered = dict(row, status='blocked')
            with self.assertRaisesRegex(ValueError, 'differs'):
                matrix.require_executable(altered, root / 'f.json')
            Path(args[3]['evaluator_rag']).write_text('new evaluator', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'digest'):
                matrix.require_executable(row, root / 'f.json')

    def test_verified_baseline_capabilities_and_all_declared_methods(self):
        manifest, matrix = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            args[1]['methods'].append({'method_id': 'other', 'status': 'pending'})
            args[1]['methods'][1].update(commit='a' * 40, status='verified', repository='https://example.org/repo', capability_scope=['rag'])
            frozen = manifest.freeze(*args, output=root / 'f.json')
            rows = matrix.generate(frozen)
            self.assertEqual(len(rows), 90)
            self.assertEqual({r['method_id'] for r in rows}, {'course_g0_textbook_fusion', 'baseline', 'other'})
            native = [r for r in rows if r['method_id'] == 'baseline']
            self.assertEqual({r['evaluation'] for r in native if r['status'] == 'ready'}, {'rag'})
            self.assertTrue(all(r['score'] is None for r in rows))

    def test_incomplete_matrix_protocol_refuses_freeze(self):
        manifest, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            args[1]['evaluations'] = ['rag']
            with self.assertRaisesRegex(ValueError, 'matrix protocol'):
                manifest.freeze(*args, output=root / 'f.json')

    def test_construction_freeze_does_not_require_unprepared_evaluation_assets(self):
        manifest, matrix = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            courses, config, sources, assets = self.fixture(root)
            config['budgets']['rag'] = None
            config['budgets']['kge'] = None
            assets = {k: assets[k] for k in ['schema', 'g0_prompt', 'textbook_prompt']}
            frozen = manifest.freeze(courses, config, sources, assets, output=root / 'c.json', stage='construction')
            rows = matrix.generate(frozen)
            self.assertTrue(all(r['status'] == 'blocked' and r['score'] is None for r in rows))
            self.assertEqual(manifest.require_construction(root / 'c.json')['stage'], 'construction')
            with self.assertRaisesRegex(ValueError, 'evaluation'):
                matrix.require_executable(rows[0], root / 'c.json')

    def test_partial_budget_fields_refuse_freeze(self):
        manifest, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            args[1]['budgets']['g0'] = {'temperature': 0}
            with self.assertRaisesRegex(ValueError, 'budget'):
                manifest.freeze(*args, output=root / 'f.json')

    def test_misassigned_corpus_is_rejected(self):
        manifest, _ = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            args[2]['oneA'] = args[2]['oneB']
            with self.assertRaisesRegex(ValueError, 'book'):
                manifest.freeze(*args, output=root / 'f.json')

    def test_budget_value_types_and_bounds(self):
        manifest, _ = self.api()
        for role, field, value in [('g0', 'max_output_tokens', -1), ('g0', 'temperature', 'pending'), ('textbook', 'max_retries', True), ('rag', 'retrieval_top_k', 0), ('kge', 'split_ratios', [0.8, 0.2, 0.2])]:
            with self.subTest(role=role, field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.fixture(root)
                args[1]['budgets'][role][field] = value
                with self.assertRaisesRegex(ValueError, 'budget'):
                    manifest.freeze(*args, output=root / 'f.json')


if __name__ == '__main__':
    unittest.main()
