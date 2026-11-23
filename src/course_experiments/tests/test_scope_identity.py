from __future__ import annotations

import copy
import unittest

from course_experiments.evaluation import scope_identity as scope
from course_experiments.manifest import digest
from course_experiments.splits import entity_lookup,resolve


def fixture():
    evidence={'s':dict(text='Alpha is a scheduling method.',book_id='book_a')}
    left=dict(entity_id='reference_alpha',canonical_name='Alpha',aliases=[],scope='operating systems',
        citations=[dict(evidence_id='s',quote='Alpha is a scheduling method.')])
    right=dict(entity_id='graph_alpha',canonical_name='Alpha',aliases=[],scope='',
        definition='A scheduling method.',source_node_ids=['s'])
    return left,right,evidence


class ScopeIdentityTests(unittest.TestCase):
    def test_contract_mismatch_is_reproduced_without_automatically_merging(self):
        left,right,evidence=fixture()
        self.assertIsNone(resolve('Alpha','scheduling',entity_lookup([left])))
        original=copy.deepcopy((left,right))
        packet=scope.prepare_pair('os',left,right,evidence,evidence)
        self.assertEqual(packet['status'],'pending_machine_identity_judgment')
        self.assertEqual(packet['scope_category'],'graph_unqualified_reference_technical_domain')
        self.assertEqual((left,right),original)
        self.assertNotIn('mapping',packet)

    def test_candidate_requires_same_course_and_exact_explicit_names(self):
        left,right,evidence=fixture()
        right['canonical_name']='unrelated term'
        with self.assertRaises(ValueError):scope.prepare_pair('os',left,right,evidence,evidence)
        right['aliases']=['Alpha']
        packet=scope.prepare_pair('os',left,right,evidence,evidence)
        self.assertEqual(packet['name_match'],'explicit_name_or_alias')
        with self.assertRaises(ValueError):scope.prepare_pair('',left,right,evidence,evidence)

    def test_scope_strings_are_descriptive_categories_not_semantic_equivalence(self):
        self.assertEqual(scope.scope_category('CPU cache','CPU'),'literal_scope_containment_needs_semantic_review')
        self.assertEqual(scope.scope_category('ＣＰＵ','cpu'),'syntax_normalization_only')
        self.assertEqual(scope.scope_category('network protocol','network topology'),'different_scope_wording_unresolved')

    def test_source_quotes_are_verified_and_missing_definition_evidence_stays_visible(self):
        left,right,evidence=fixture()
        left['citations'][0]['quote']='invented source'
        with self.assertRaises(ValueError):scope.prepare_pair('os',left,right,evidence,evidence)
        left,right,evidence=fixture();right['source_node_ids']=[]
        packet=scope.prepare_pair('os',left,right,evidence,evidence)
        self.assertFalse(packet['both_sides_have_source'])
        self.assertEqual(packet['right']['source_quotes'],[])

    def test_development_sample_uses_all_reference_ids_without_graph_or_test_input(self):
        rows=[dict(entity_id=str(i),canonical_name=str(i)) for i in range(20)]
        selected=scope.development_ids('os',rows,6)
        self.assertEqual(selected,scope.development_ids('os',list(reversed(rows)),6))
        self.assertEqual(len(selected),6)
        self.assertEqual(selected,sorted((e['entity_id'] for e in rows),
            key=lambda x:digest([42,'scope_identity_development_v1','os',x]))[:6])

    def test_same_sense_requires_two_sided_exact_source_and_no_homonym(self):
        left,right,evidence=fixture();packet=scope.prepare_pair('os',left,right,evidence,evidence)
        judgment=dict(candidate_id=packet['candidate_id'],decision='same_sense',
            scope_consistent=True,source_definition_sufficient=True,homonym_excluded=True,
            left_citations=[dict(evidence_id='s',quote='Alpha is a scheduling method.')],
            right_citations=[dict(evidence_id='s',quote='Alpha is a scheduling method.')],reason='synthetic')
        self.assertTrue(scope.validate_judgment(packet,judgment)['eligible_identity_edge'])
        judgment['homonym_excluded']=False
        self.assertFalse(scope.validate_judgment(packet,judgment)['eligible_identity_edge'])
        judgment['right_citations'][0]['quote']='invented'
        with self.assertRaises(ValueError):scope.validate_judgment(packet,judgment)


if __name__=='__main__':unittest.main()
