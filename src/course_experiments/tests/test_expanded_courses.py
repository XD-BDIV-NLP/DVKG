from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from course_experiments import manifest, matrix
from course_experiments.tests import test_manifest_matrix


def expanded_fixture(root):
    courses, config, sources, assets = test_manifest_matrix.ManifestMatrixTests().fixture(root)
    courses['version'] = config['version'] = 2
    courses['courses'][0]['books'] += [
        dict(alias=a, book_id='one' + a, source_fingerprint='source-' + a) for a in 'CD']
    third = copy.deepcopy(courses['courses'][1])
    third.update(course_id='three', description='three')
    for book in third['books']:
        book['book_id'] = 'three' + book['alias']
    courses['courses'].append(third)
    for course in courses['courses']:
        for book in course['books']:
            if book['book_id'] in sources:
                continue
            corpus = dict(books=[dict(id=book['book_id'], source_fingerprint=book['source_fingerprint'])],
                          chunks=[dict(id='c1', book_id=book['book_id'], evidence_ids=['e1'])],
                          evidence=[dict(id='e1', book_id=book['book_id'], text='fixture')])
            corpus['fingerprint'] = manifest.digest(corpus)
            path = root / (book['book_id'] + '.json')
            path.write_text(json.dumps(corpus), encoding='utf-8')
            sources[book['book_id']] = str(path)
    config['course_book_sets'] = {'one': [['A'], ['B'], ['A', 'B'], ['A', 'B', 'C', 'D']]}
    return courses, config, sources, assets


class ExpandedCoursesTests(unittest.TestCase):
    def test_expanded_configuration_preserves_courses_budgets_and_exact_ab_abcd_comparison(self):
        config_root = Path(__file__).resolve().parents[1] / 'config'
        courses = manifest.read_document(config_root / 'courses.expanded_20261016.yaml')
        config = manifest.read_document(config_root / 'experiments.expanded_20261016.yaml')
        old_courses = manifest.read_document(config_root / 'courses.yaml')
        old_config = manifest.read_document(config_root / 'experiments.yaml')
        network = manifest.read_document(config_root / 'courses.networks_20261016.yaml')
        self.assertEqual((courses['version'], config['version']), (2, 2))
        self.assertEqual(len(courses['courses']), 3)
        comp = courses['courses'][0]
        self.assertEqual({k: v for k, v in comp.items() if k != 'books'},
                         {k: v for k, v in old_courses['courses'][0].items() if k != 'books'})
        self.assertEqual(comp['books'][:2], old_courses['courses'][0]['books'])
        self.assertEqual([b['book_id'] for b in comp['books']],
                         ['book_49bb8d31ddb9', 'book_25f718562685', 'book_efaf8522b95d', 'book_bef9d155e6d8'])
        self.assertEqual([b['source_fingerprint'] for b in comp['books'][2:]],
                         ['c08fbee62d6077b25d04ab913aa6bbf2fd16192437323bd590600f765ebe5112',
                          'da8f5e2254c51cee36279c59ff6d0580b4b1170666bc923357d07e2035ba7a0a'])
        self.assertEqual(courses['courses'][1], old_courses['courses'][1])
        self.assertEqual(courses['courses'][2], network['courses'][0])
        for key in ('models', 'methods', 'budgets', 'schema'):
            self.assertEqual(config[key], old_config[key])
        self.assertEqual(config['course_book_sets'],
                         {'computer_organization': [['A'], ['B'], ['A', 'B'], ['A', 'B', 'C', 'D']]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, assets = expanded_fixture(root)
            sources = {}
            for course in courses['courses']:
                for book in course['books']:
                    corpus = dict(books=[dict(id=book['book_id'], source_fingerprint=book['source_fingerprint'])],
                                  chunks=[dict(book_id=book['book_id'], evidence_ids=['e1'])],
                                  evidence=[dict(id='e1', book_id=book['book_id'], text='fixture')])
                    corpus['fingerprint'] = manifest.digest(corpus)
                    path = root / (book['book_id'] + '.json')
                    path.write_text(json.dumps(corpus), encoding='utf-8')
                    sources[book['book_id']] = str(path)
            frozen = manifest.freeze(courses, config, sources, assets, output=root / 'f.json', stage='construction')
            self.assertEqual(len(matrix.generate(frozen)), 210)
            self.assertTrue(all(r['status'] == 'blocked' for r in matrix.generate(frozen)))
            with self.assertRaisesRegex(ValueError, 'budget'):
                manifest.freeze(courses, config, sources, assets, output=root / 'eval.json')

    def test_version_two_freezes_three_courses_eight_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = manifest.freeze(*expanded_fixture(root), output=root / 'f.json')
            self.assertEqual(len(frozen['files']['sources']), 8)
            self.assertEqual(len(frozen['g0_identities']), 3)
            self.assertEqual(manifest.validate(root / 'f.json'), frozen)

    def test_matrix_uses_course_override_and_keeps_default_ab_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            courses, config, _, _ = expanded_fixture(Path(tmp))
            frozen = dict(courses=courses, experiments=config, stage='evaluation', manifest_id='fixture')
            rows = matrix.generate(frozen)
            self.assertEqual(len(rows), 120)
            for course, wanted in [('one', {'A', 'B', 'A+B', 'A+B+C+D'}),
                                   ('two', {'A', 'B', 'A+B'}), ('three', {'A', 'B', 'A+B'})]:
                self.assertEqual({r['book_set'] for r in rows if r['course_id'] == course}, wanted)
            four = [r for r in rows if r['book_set'] == 'A+B+C+D']
            self.assertTrue(four)
            self.assertTrue(all(r['book_ids'] == ['oneA', 'oneB', 'oneC', 'oneD'] for r in four))
            fixed = [r for r in rows if r['course_id'] == 'one' and r['state'] == 'fixed_g0'
                     and r['evaluation'] == 'topology']
            self.assertEqual(len({r['execution_id'] for r in fixed}), 1)

    def test_cli_accepts_c_and_d_book_aliases(self):
        from course_experiments.construction import main
        for alias in 'CD':
            with self.subTest(alias=alias), patch('sys.argv', ['construction', 'textbook',
                    '--manifest', 'f.json', '--course-id', 'one', '--book-alias', alias,
                    '--output-root', 'out']), patch('course_experiments.construction.construct',
                    return_value={'artifact_id': 'accepted'}) as construct, redirect_stdout(StringIO()) as output:
                main()
                self.assertEqual(construct.call_args.args[4], alias)
                self.assertEqual(output.getvalue().strip(), 'accepted')

    def test_version_two_rejects_invalid_course_book_sets(self):
        cases = [({'unknown': [['A']]}, 'unknown'),
                 ({'two': [['A', 'C']]}, 'declared'),
                 ({'two': [['D']]}, 'declared'),
                 ({'one': []}, 'book sets'),
                 ({'one': [[]]}, 'book set'),
                 ({'one': [['A', 'A']]}, 'book set'),
                 ({'one': [['A', 'B'], ['B', 'A'], ['A', 'B', 'C', 'D']]}, 'duplicate'),
                 ({'one': [['A', 'B', 'C', 'D']]}, 'A/B'),
                 ({'one': [['A', 'B']]}, 'A/B')]
        for override, message in cases:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = expanded_fixture(root)
                args[1]['course_book_sets'].update(override)
                with self.assertRaisesRegex(ValueError, message):
                    manifest.freeze(*args, output=root / 'f.json')

    def test_version_two_rejects_invalid_declared_aliases(self):
        for aliases in ['ABCC', 'ABCE', 'ABDDD', 'AC']:
            with self.subTest(aliases=aliases), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = expanded_fixture(root)
                books = args[0]['courses'][0]['books']
                args[0]['courses'][0]['books'] = [dict(books[i % 4], alias=a, book_id='bad' + str(i))
                                                 for i, a in enumerate(aliases)]
                with self.assertRaisesRegex(ValueError, 'aliases'):
                    manifest.freeze(*args, output=root / 'f.json')

    def test_version_two_keeps_source_model_and_budget_checks(self):
        for mutation, message in [('source', 'source_fingerprint'), ('model', 'model'), ('budget', 'budget')]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = expanded_fixture(root)
                if mutation == 'source':
                    args[0]['courses'][0]['books'][2]['source_fingerprint'] = 'forged'
                elif mutation == 'model':
                    args[1]['models']['construction'] = 'unapproved'
                else:
                    args[1]['budgets']['textbook']['max_retries'] = True
                with self.assertRaisesRegex(ValueError, message):
                    manifest.freeze(*args, output=root / 'f.json')

    def test_version_one_does_not_enable_new_book_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = expanded_fixture(root)
            args[0]['version'] = args[1]['version'] = 1
            with self.assertRaisesRegex(ValueError, 'A/B'):
                manifest.freeze(*args, output=root / 'f.json')

    def test_version_two_four_books_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = expanded_fixture(root)
            del args[1]['course_book_sets']
            with self.assertRaisesRegex(ValueError, 'explicit'):
                manifest.freeze(*args, output=root / 'f.json')

    def test_legacy_two_courses_keep_three_global_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = test_manifest_matrix.ManifestMatrixTests().fixture(root)
            args[1]['course_book_sets'] = {'one': [['A', 'B']]}
            frozen = manifest.freeze(*args, output=root / 'f.json')
            self.assertEqual(frozen['version'], 1)
            rows = matrix.generate(frozen)
            self.assertEqual(len(rows), 72)
            self.assertEqual({r['book_set'] for r in rows}, {'A', 'B', 'A+B'})

    def test_constructor_uses_declared_four_book_alias_and_rejects_other_course(self):
        from course_experiments.construction import construct, implementation_assets
        from course_experiments.tests.test_construction import ConstructionExecutionTests
        from tests.test_incremental_graph_extraction import ExtractionRuntime
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / 'legacy'
            legacy.mkdir()
            old_path, _ = ConstructionExecutionTests().fixture(legacy)
            old = manifest.read_document(old_path)
            courses, config, sources, _ = expanded_fixture(root)
            for path in sources.values():
                corpus = manifest.read_document(path)
                corpus['chunks'][0].update(title_path=['存储'], section_id='s')
                corpus['fingerprint'] = manifest.digest({k: v for k, v in corpus.items() if k != 'fingerprint'})
                Path(path).write_text(json.dumps(corpus), encoding='utf-8')
            assets = {k: v['path'] for k, v in old['files']['assets'].items()}
            assets.update({k: str(v) for k, v in implementation_assets().items()})
            frozen_path = root / 'expanded.json'
            manifest.freeze(courses, config, sources, assets, output=frozen_path, stage='construction')
            runtime = ExtractionRuntime()
            runtime.stats = lambda: {}
            runtime.close = lambda: None
            with patch('course_experiments.construction.ModelRuntime', return_value=runtime):
                for alias in 'CD':
                    result = construct(frozen_path, 'one', 'textbook', root / 'out', alias)
                    self.assertEqual(result['book_id'], 'one' + alias)
                    self.assertTrue(result['graph']['nodes'])
                    with self.assertRaisesRegex(ValueError, 'declared'):
                        construct(frozen_path, 'two', 'textbook', root / 'out', alias)


if __name__ == '__main__':
    unittest.main()
