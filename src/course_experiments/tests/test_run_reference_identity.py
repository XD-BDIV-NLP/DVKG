import copy
import tempfile
import unittest
from pathlib import Path


class RunReferenceIdentityTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import run_reference_identity
        return run_reference_identity

    def triangle(self):
        from course_experiments.evaluation import reference_identity as ref
        evidence={'e':dict(text='Alpha is the same independently quoted source definition.')}
        entities=[dict(entity_id=key,canonical_name='Alpha',scope='source scope '+key,aliases=[],
            citations=[dict(evidence_id='e',quote=evidence['e']['text'])]) for key in 'abc']
        packets=[];origins=[]
        for left,right in [(0,1),(0,2),(1,2)]:
            p=ref.prepare_pair('c',entities[left],entities[right],evidence);packets.append(p)
            origins.append(dict(candidate_id=p['candidate_id'],course_id='c',graph_record=dict(sha256='reference_sha'),
                graph_entity_id=p['right']['entity_id'],reference_entity_id=p['left']['entity_id'],
                left_reference_entity_id=p['left']['entity_id'],right_reference_entity_id=p['right']['entity_id']))
        return packets,origins

    def test_reference_equivalence_does_not_apply_directional_graph_conflict_rule(self):
        api=self.api();packets,origins=self.triangle()
        from course_experiments.tests.test_scope_identity_batch import Provider
        from course_experiments.evaluation.reference_identity import freeze_components
        with tempfile.TemporaryDirectory() as root:
            result=api.execute_reference_batches(packets,origins,Provider(),Path(root),'job')
            self.assertEqual(sum(r['eligible_identity_edge'] for r in result['decisions'].values()),3)
            self.assertEqual(freeze_components(origins,result['decisions'])['components'],{'c':[['a','b','c']]})
            provider=Provider();again=api.execute_reference_batches(packets,origins,provider,Path(root),'job')
            self.assertEqual(provider.calls,[]);self.assertEqual(again['decisions'],result['decisions'])

    def test_single_negative_veto_is_preserved_before_component_freeze(self):
        api=self.api();packets,origins=self.triangle()
        decisions={p['candidate_id']:dict(eligible_identity_edge=True) for p in packets}
        single={packets[0]['candidate_id']:dict(status='completed',eligible_identity_edge=False)}
        result=api.preserve_single_negatives(decisions,single)
        self.assertFalse(result[packets[0]['candidate_id']]['eligible_identity_edge'])
        self.assertTrue(decisions[packets[0]['candidate_id']]['eligible_identity_edge'])

    def test_full_stage_cannot_skip_independent_development_gate(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(api.DependencyNotReady):api.stage_dependencies('full',Path(root),'preparation')
            self.assertEqual(api.stage_dependencies('development-single',Path(root),'preparation'),{})
        self.assertEqual(api.MAX_HTTP,500)
        self.assertEqual(api.PROVIDER_SHA,'0d0b7b23ff5c3c12e0fba2f59646a939b9f8b50692e8039583efe033df09b102')


if __name__=='__main__':unittest.main()
