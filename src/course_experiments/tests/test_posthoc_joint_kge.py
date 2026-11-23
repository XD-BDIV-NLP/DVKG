"""双书后合并保持谓词、来源和重新分折。"""
import importlib
import unittest


class PosthocJointKGETests(unittest.TestCase):
    def merge(self, books):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.evaluation.posthoc_joint_kge'),
                             '双书后合并尚未实现')
        module = importlib.import_module('course_experiments.evaluation.posthoc_joint_kge')
        return module.merge_book_triples(books)

    def books(self, a, b):
        return {'book_49bb8d31ddb9': a, 'book_25f718562685': b}

    def test_identical_names_align_and_common_fact_keeps_both_sources(self):
        joint = self.merge(self.books([('缓存', '加速', '访存')],
                                      [('缓存', '加速', '访存'), ('CPU', '包含', '缓存')]))
        self.assertEqual(joint['triples'], [('CPU', '包含', '缓存'), ('缓存', '加速', '访存')])
        self.assertEqual(joint['counts']['shared_entities'], 2)
        self.assertEqual(joint['counts']['duplicates_removed'], 1)
        record = next(r for r in joint['triple_sources'] if r['triple'][0] == '缓存')
        self.assertEqual(set(record['books']), set(self.books([], [])))

    def test_no_alias_fuzzy_case_or_predicate_normalization_is_invented(self):
        joint = self.merge(self.books([('CPU', 'has', '缓存')],
                                      [('cpu', 'contains', 'Cache')]))
        self.assertEqual(joint['counts']['entities'], 4)
        self.assertEqual(joint['counts']['relations'], 2)
        self.assertEqual(joint['counts']['shared_entities'], 0)

    def test_only_boundary_whitespace_and_unicode_nfc_are_normalized(self):
        joint = self.merge(self.books([(' e\u0301 ', 'r', 'x')], [('é', 'r', 'x')]))
        self.assertEqual(joint['triples'], [('é', 'r', 'x')])

    def test_reverse_edges_and_parallel_predicates_survive_until_pair_split(self):
        a = [(str(i), 'r', str(j)) for i in range(10) for j in range(i + 1, 10)]
        b = [(str(j), 'r', str(i)) for i in range(10) for j in range(i + 1, 10)]
        joint = self.merge(self.books(a, b + [('1', 'other', '2')]))
        core = importlib.import_module('course_experiments.evaluation.native_graph_kge')
        bundle = core.validate_bundle(core.prepare_native_graph(joint['triples']))
        self.assertEqual(bundle['counts']['unique_triples'], 91)
        self.assertEqual(bundle['status'], 'ready')

    def test_single_book_extra_book_empty_book_and_bad_predicate_are_rejected(self):
        bad = [self.books([('a', 'r', 'b')], []),
               {'book_49bb8d31ddb9': [('a', 'r', 'b')]},
               dict(self.books([('a', 'r', 'b')], [('a', 'r', 'b')]), other=[('a', 'r', 'b')]),
               self.books([('a', '', 'b')], [('a', 'r', 'b')])]
        for books in bad:
            with self.subTest(books=books), self.assertRaises(ValueError):
                self.merge(books)

    def test_book_and_triple_order_does_not_change_joint_identity(self):
        books = self.books([('a', 'r', 'b'), ('c', 'r', 'a')], [('a', 'r', 'b')])
        self.assertEqual(self.merge(books), self.merge({k: list(reversed(v)) for k, v in reversed(list(books.items()))}))

    def test_book_local_ids_are_resolved_to_names_before_alignment(self):
        self.merge(self.books([('a', 'r', 'b')], [('a', 'r', 'b')]))
        module = importlib.import_module('course_experiments.evaluation.posthoc_joint_kge')
        self.assertTrue(hasattr(module, 'name_native_graph'), '原生ID到名称的映射尚未实现')
        doc = {'native_graph': {'nodes': [{'id': '0', 'name': 'CPU'}, {'id': '1', 'name': '缓存'}],
                                'edges': [{'source': '0', 'target': '1', 'predicate': '包含'}], 'directed': True}}
        graph = module.name_native_graph(doc)['native_graph']
        self.assertEqual(graph['edges'], [{'source': 'CPU', 'target': '缓存', 'predicate': '包含'}])
        self.assertTrue(graph['directed'])
        doc['native_graph']['edges'][0]['target'] = 'unknown'
        with self.assertRaises(ValueError):
            module.name_native_graph(doc)


if __name__ == '__main__':
    unittest.main()
