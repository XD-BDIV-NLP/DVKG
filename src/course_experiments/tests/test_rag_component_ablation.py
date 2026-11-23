"""预算消融只移除路由成本约束，保留区域数和原始成本披露。"""
import importlib
import unittest

from course_experiments.tests.test_evidence_hierarchical_alignment import fixture
from course_experiments.evaluation import evidence_hierarchical_alignment as core


class RAGComponentAblationTests(unittest.TestCase):
    def selector(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.evaluation.rag_component_ablation'),
                             '严格组件消融尚未实现')
        module=importlib.import_module('course_experiments.evaluation.rag_component_ablation')
        return module.without_token_budget(core.select_regions)

    def test_overbudget_region_can_be_selected_without_losing_region_limit(self):
        _,s=fixture()
        coverage={k:{'concept':1.} for k,n in s['nodes'].items() if n['type']!='Book'}
        costs={e:100 for e in s['evidence']}
        self.assertFalse(core.select_regions(s,coverage,{'concept':1},costs,budget=10,max_regions=1)['region_ids'])
        result=self.selector()(s,coverage,{'concept':1},costs,budget=10,max_regions=1)
        self.assertEqual(len(result['region_ids']),1)
        self.assertGreater(result['tokens'],10)
        self.assertEqual(result['tokens'],sum(costs[e] for e in result['evidence_ids']))
        self.assertFalse(result['routing_token_budget_applied'])

    def test_input_token_counts_are_not_modified_or_reported_as_zero(self):
        _,s=fixture()
        leaves={k:{k:1.} for k,n in s['nodes'].items() if n['type']=='Evidence'}
        costs={e:30 for e in s['evidence']};before=dict(costs)
        result=self.selector()(s,leaves,{k:1 for k in leaves},costs,budget=1,max_regions=2)
        self.assertEqual(len(result['region_ids']),2)
        self.assertEqual(result['tokens'],60)
        self.assertEqual(costs,before)

    def test_unverified_unknown_or_negative_token_counts_are_rejected(self):
        _,s=fixture();costs={e:1 for e in s['evidence']}
        for bad in [{},dict(costs,a=-1)]:
            with self.subTest(costs=bad),self.assertRaises(ValueError):
                self.selector()(s,{}, {},bad,budget=1,max_regions=1)


if __name__=='__main__':
    unittest.main()
