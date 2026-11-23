import copy,unittest
from course_experiments.baselines.k12_native_projection import flat_native_edges, model_schema_view
class NativeEdgesTests(unittest.TestCase):
 def test_official_aggregated_targets_are_all_preserved(self):
  raw=dict(nodes=[dict(id=x) for x in ['exercise','concept1','concept2']],edges=[dict(source='exercise',type='tests_concept',target='stale_raw_id',target_name_to_ids=[dict(target='concept1',target_name='a'),dict(target='concept2',target_name='b')],properties=dict(source_marker='actual'))])
  before=copy.deepcopy(raw);edges=flat_native_edges(raw)
  self.assertEqual([e['target'] for e in edges],['concept1','concept2']);self.assertEqual([e['predicate'] for e in edges],['tests_concept']*2);self.assertEqual(raw,before);self.assertEqual(edges[0]['properties'],dict(source_marker='actual'))
 def test_foreign_and_empty_aggregated_targets_fail_closed(self):
  for targets in [[],[dict(target='foreign')]]:
   with self.assertRaises(ValueError):flat_native_edges(dict(nodes=[dict(id='exercise')],edges=[dict(source='exercise',type='tests_concept',target_name_to_ids=targets)]))
 def test_standard_edges_preserved_and_endpoint_checked(self):
  raw=dict(nodes=[dict(id='a'),dict(id='b')],edges=[dict(source='a',target='b',type='relates_to',properties=dict(relations='source-derived'))]);self.assertEqual(flat_native_edges(raw)[0]['properties'],raw['edges'][0]['properties'])
 def test_exercise_name_is_its_existing_native_stem_only_in_validation_view(self):
  raw=dict(nodes=[dict(id='ex',label='Exercise',stem='existing source exercise')],edges=[]);before=copy.deepcopy(raw)
  view=model_schema_view(raw);self.assertEqual(view['nodes'][0]['name'],'existing source exercise');self.assertEqual(raw,before)
  bad=dict(nodes=[dict(id='c',label='Concept',stem='cannot supply concept name')],edges=[]);self.assertNotIn('name',model_schema_view(bad)['nodes'][0])
if __name__=='__main__':unittest.main()
