"""原生三元组划分、脱敏和结果汇总的回归。"""
import importlib
import math
import unittest


class NativeGraphKGETests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.evaluation.native_graph_kge'),
                             'Native graph KGE preparation is not implemented')
        return importlib.import_module('course_experiments.evaluation.native_graph_kge')

    def fixture(self):
        return [(str(i), 'open relation', str(j)) for i in range(10) for j in range(10) if i != j]

    def test_open_predicates_and_untyped_entities_are_not_dropped(self):
        m = self.module()
        value = m.prepare_native_graph([('a', 'arbitrary predicate', 'b'),
                                        ('a', 'arbitrary predicate', 'b')])
        self.assertEqual(value['counts']['unique_triples'], 1)
        self.assertEqual(value['counts']['relations'], 1)
        self.assertEqual(value['status'], 'insufficient_holdout')

    def test_empty_or_malformed_triple_is_rejected_instead_of_invented(self):
        m = self.module()
        for triples in [[], [('a', '', 'b')], [('a', 'r')], [('a', None, 'b')]]:
            with self.subTest(triples=triples), self.assertRaises(ValueError):
                m.prepare_native_graph(triples)

    def test_json_graph_keeps_native_predicates_without_ontology_filter(self):
        m = self.module()
        value = m.graph_triples({'native_graph': {'directed': True, 'edges': [
            {'source': 'a', 'target': 'b', 'predicate': 'arbitrary predicate', 'type':'unmapped'}]}})
        self.assertEqual(value['triples'], [('a', 'arbitrary predicate', 'b')])
        self.assertEqual(value['relation_semantics'], 'native_predicates')

    def test_untyped_undirected_graph_does_not_invent_relation_from_description(self):
        m = self.module()
        value = m.graph_triples({'native_graph': {'directed': False, 'edges': [
            {'source': 'a', 'target': 'b', 'description': 'some prose', 'keywords':'topic'}]}})
        self.assertEqual(set(value['triples']), {('a','__native_related__','b'), ('b','__native_related__','a')})
        self.assertEqual(value['relation_semantics'], 'adjacency_only_generic_relation')

    def test_our_entity_ids_are_preferred_over_ambiguous_display_names(self):
        m = self.module()
        value = m.graph_triples({'relations':[{'head_entity_id':'id1','tail_entity_id':'id2',
            'head':'same name','tail':'same name','relation':'requires'}]})
        self.assertEqual(value['triples'], [('id1','requires','id2')])

    def test_reverse_and_parallel_facts_stay_in_same_fold(self):
        m = self.module()
        value = m.prepare_native_graph(self.fixture() + [('1', 'second relation', '2')])
        where = {}
        for name, triples in value['splits'].items():
            for h, r, t in triples:
                pair = tuple(sorted((h, t)))
                self.assertIn(where.setdefault(pair, name), [name])
        self.assertEqual(value['status'], 'ready')
        self.assertGreater(len(value['splits']['test']), 0)
        self.assertGreater(len(value['splits']['validation']), 0)

    def test_input_order_does_not_change_frozen_split(self):
        m = self.module()
        a = m.prepare_native_graph(self.fixture())
        b = m.prepare_native_graph(list(reversed(self.fixture())))
        self.assertEqual(a, b)

    def test_heldout_entities_and_relations_are_present_in_training(self):
        m = self.module()
        triples = self.fixture() + [('rare', 'singleton relation', '1')]
        value = m.prepare_native_graph(triples)
        train = value['splits']['train']
        entities = {e for h, r, t in train for e in (h, t)}
        relations = {r for h, r, t in train}
        for split in ['validation', 'test']:
            for h, r, t in value['splits'][split]:
                self.assertTrue(h in entities and t in entities and r in relations)
        self.assertEqual(sum(len(v) for v in value['splits'].values()), 91)

    def test_gpu_export_contains_only_opaque_graph_labels(self):
        m = self.module()
        value = m.prepare_native_graph([('原教材概念', '原关系文字', '另一个概念')])
        labels = [v for split in value['splits'].values() for triple in split for v in triple]
        self.assertTrue(labels)
        self.assertTrue(all(len(v) == 64 and set(v) <= set('0123456789abcdef') for v in labels))
        self.assertNotIn('原教材概念', str(value))

    def test_incomplete_seed_group_is_not_published_as_final_mean(self):
        m = self.module()
        rows = [dict(model='TransE', seed=42, status='completed', metrics={'mrr': .2})]
        summary = m.summarize_seeds(rows)
        self.assertEqual(summary['status'], 'partial')
        self.assertNotIn('metrics', summary)

    def test_three_seed_summary_uses_sample_standard_deviation(self):
        m = self.module()
        rows = [dict(model='TransE', seed=seed, status='completed',
                     metrics={k:score for k in ['mrr', 'hits_at_1', 'hits_at_3', 'hits_at_10']})
                for seed, score in [(42, .2), (43, .4), (44, .6)]]
        summary = m.summarize_seeds(rows)
        self.assertEqual(summary['status'], 'completed')
        self.assertAlmostEqual(summary['metrics']['mrr']['mean'], .4)
        self.assertAlmostEqual(summary['metrics']['mrr']['std'], .2)


if __name__ == '__main__':
    unittest.main()
