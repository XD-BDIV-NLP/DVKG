import copy
import unittest

from course_experiments.baselines.tree_kg import semantic_input, target_update_view


class TargetScopeTests(unittest.TestCase):
    def value(self):
        return dict(nodes=[dict(id='a', neighbors=[dict(id='context')]),
                           dict(id='b', neighbors=[])])

    def test_requested_updates_complete_and_only_given_neighbor_extras_ignored(self):
        raw = dict(updates=[dict(id='a', description='A'),
                            dict(id='context', description='context'),
                            dict(id='b', description='B')])
        before = copy.deepcopy(raw)
        value, audit = target_update_view(self.value(), raw)
        self.assertEqual([row['id'] for row in value['updates']], ['a', 'b'])
        self.assertEqual(raw, before)
        self.assertEqual(audit['ignored_context_IDs'], ['context'])

    def test_missing_duplicate_and_foreign_updates_still_rejected(self):
        for ids in [['a', 'context'], ['a', 'b', 'a'], ['a', 'b', 'foreign']]:
            with self.assertRaises(ValueError):
                target_update_view(self.value(), dict(updates=[
                    dict(id=key, description='description') for key in ids]))

    def test_recursive_projection_preserves_all_semantics_and_raw_graph(self):
        raw = dict(nodes=[dict(id='a', description='source', evidence_ids=['p'] * 100,
                    neighbors=[dict(id='b', description='meaning', evidence_ids=['p'],
                                    source_support='retained')],
                    relations=[dict(source='a', target='b', predicate='related_to',
                                    description='relationship', evidence_ids=['p'],
                                    origin='retained')])])
        before = copy.deepcopy(raw)
        value = semantic_input(raw)
        self.assertEqual(raw, before)
        self.assertEqual(value['nodes'][0]['neighbors'][0]['source_support'], 'retained')
        self.assertEqual(value['nodes'][0]['relations'][0]['origin'], 'retained')
        self.assertNotIn('evidence_ids', value['nodes'][0]['neighbors'][0])
        self.assertEqual(value['nodes'][0]['relations'][0]['description'], 'relationship')


if __name__ == '__main__':
    unittest.main()
