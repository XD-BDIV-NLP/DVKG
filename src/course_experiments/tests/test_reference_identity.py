import copy
import unittest

from course_experiments.manifest import digest


class ReferenceIdentityTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import reference_identity
        return reference_identity

    def packet(self,number,course='c'):
        return dict(candidate_id=digest([course,number]),course_id=course,
            left=dict(entity_id='a'+str(number)),right=dict(entity_id='b'+str(number)))

    def test_development24_selection_fixed_course_hash_and_no_backfill(self):
        api=self.api();packets=[self.packet(i,c) for c in ['c','o','n'] for i in range(12)]
        first=api.development_plan(packets)
        self.assertEqual(first,api.development_plan(list(reversed(packets))))
        self.assertEqual(len(first['candidate_ids']),24)
        self.assertEqual(set(first['selected_per_course'].values()),{8})
        self.assertFalse(api.development_plan(packets[:5])['sufficient_for_declared24'])

    def row(self,eligible=True,label='same_sense'):
        return dict(status='completed',citation_valid=True,eligible_identity_edge=eligible,
            reversed_positive_confirmed=eligible,primary=dict(decision=label))

    def test_declared_comparison_rejects_negative_to_positive_and_two_label_changes(self):
        api=self.api();plan=api.development_plan([self.packet(i,c) for c in ['c','o','n'] for i in range(8)])
        rows={cid:self.row(False,'insufficient') for cid in plan['candidate_ids']}
        self.assertTrue(api.compare_development(plan,rows,rows)['passes_declared_gate'])
        changed=copy.deepcopy(rows);changed[plan['candidate_ids'][0]]=self.row(True)
        self.assertFalse(api.compare_development(plan,rows,changed)['passes_declared_gate'])
        for cid in plan['candidate_ids'][:2]:changed[cid]=self.row(False,'different_sense')
        self.assertFalse(api.compare_development(plan,rows,changed)['passes_declared_gate'])

    def test_positive_clique_only_no_negative_or_missing_transitive_edges(self):
        api=self.api();origins=[dict(candidate_id='ab',course_id='c',left_reference_entity_id='a',right_reference_entity_id='b'),
            dict(candidate_id='ac',course_id='c',left_reference_entity_id='a',right_reference_entity_id='c'),
            dict(candidate_id='bc',course_id='c',left_reference_entity_id='b',right_reference_entity_id='c')]
        rows={r['candidate_id']:self.row() for r in origins}
        good=api.freeze_components(origins,rows)
        self.assertEqual(good['components'],{'c':[['a','b','c']]})
        rows['bc']=self.row(False,'different_sense')
        bad=api.freeze_components(origins,rows)
        self.assertEqual(bad['components'],{})
        self.assertEqual(len(bad['rejected_components']),1)
        missing=api.freeze_components(origins[:2],{key:value for key,value in rows.items() if key!='bc'})
        self.assertEqual(missing['components'],{})

    def test_failed_candidate_never_vanishes_from_reference_denominator(self):
        api=self.api();origin=[dict(candidate_id='ab',course_id='c',left_reference_entity_id='a',right_reference_entity_id='b')]
        with self.assertRaises(ValueError):api.freeze_components(origin,{'ab':dict(status='failed')})

    def test_ref_pairs_validate_both_sides_against_original_source_not_graphs(self):
        api=self.api();evidence={'e':dict(text='Alpha is a source definition.')}
        entity=dict(entity_id='a',canonical_name='Alpha',aliases=[],scope='chapter1',citations=[dict(evidence_id='e',quote='Alpha is a source definition.')])
        other=dict(entity,entity_id='b',scope='same course technical scope')
        packet=api.prepare_pair('c',entity,other,evidence)
        self.assertTrue(packet['both_sides_have_source'])
        self.assertEqual(packet['right']['source_quotes'],packet['left']['source_quotes'])
        bad=dict(other,citations=[dict(evidence_id='e',quote='absent')])
        with self.assertRaises(ValueError):api.prepare_pair('c',entity,bad,evidence)


if __name__=='__main__':unittest.main()
