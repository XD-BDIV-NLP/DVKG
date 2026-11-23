import copy,unittest
from course_experiments.baselines.tree_kg import relation_input
class RelationInputTests(unittest.TestCase):
 def test_complete_semantics_without_repeated_audit_ids(self):
  nodes=[dict(id=str(i),name='concept',description='source summary',aliases=['alias'],type='Concept',evidence_ids=['p']*10000,book_ids=['book'],section_ids=['section'],origin='initial_extraction',source_support='section_provenance_only_not_sentence_verified',type_status='native_untyped_candidate') for i in range(4)]
  raw=dict(nodes=nodes,context='all original context');before=copy.deepcopy(raw);v=relation_input(raw)
  self.assertEqual(raw,before);self.assertEqual(v['context'],raw['context']);self.assertEqual([x['id'] for x in v['nodes']],[x['id'] for x in nodes]);self.assertEqual(len(v['nodes']),4)
  for old,new in zip(nodes,v['nodes']):
   for key in ['id','name','description','aliases','type']:self.assertEqual(old[key],new[key])
   self.assertNotIn('evidence_ids',new)
 def test_unknown_semantic_fields_retained(self):
  value=dict(nodes=[dict(id='section',parent='p',title='chapter',description='summary',custom_fact='retained')],context='context');self.assertEqual(relation_input(value),value)
if __name__=='__main__':unittest.main()
