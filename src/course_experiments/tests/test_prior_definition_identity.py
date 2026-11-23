from __future__ import annotations

import copy
import unittest

from course_experiments.evaluation import prior_definition_identity as prior
from course_experiments.evaluation.scope_identity import prepare_pair
from course_experiments.tests.test_scope_identity import fixture


class PriorDefinitionIdentityTests(unittest.TestCase):
    def packet(self):
        left,right,evidence=fixture();right['source_node_ids']=[]
        return prepare_pair('os',left,right,evidence,evidence)

    def test_prior_definition_is_separate_exact_provenance_not_book_evidence(self):
        packet=prior.prepare_pair(self.packet())
        self.assertEqual(packet['right']['definition_quotes'][0]['provenance'],'prior_generated_definition')
        self.assertFalse(packet['right_has_textbook_source'])
        self.assertFalse(packet['textbook_relation_evidence_supported'])
        self.assertEqual(packet['kind'],'prior_definition_identity')

    def test_missing_original_definition_and_already_anchored_candidates_are_not_relabelled(self):
        packet=self.packet();packet['right']['model_claimed_definitions']=[]
        with self.assertRaises(ValueError):prior.prepare_pair(packet)
        packet=self.packet();packet['right']['source_quotes']=[dict(evidence_id='s',quote='source')]
        with self.assertRaises(ValueError):prior.prepare_pair(packet)

    def test_positive_identity_requires_exact_prior_quote_scope_and_homonym_guards(self):
        packet=prior.prepare_pair(self.packet())
        value=dict(candidate_id=packet['candidate_id'],decision='same_sense',
            definition_identity_sufficient=True,necessary_scope_consistent=True,homonym_excluded=True,
            left_citations=copy.deepcopy(packet['left']['source_quotes']),right_citations=copy.deepcopy(packet['right']['definition_quotes']),reason='synthetic')
        self.assertTrue(prior.validate_judgment(packet,value)['eligible_identity_edge'])
        value['necessary_scope_consistent']=False
        self.assertFalse(prior.validate_judgment(packet,value)['eligible_identity_edge'])
        value['right_citations'][0]['quote']='invented prior definition'
        with self.assertRaises(ValueError):prior.validate_judgment(packet,value)


if __name__=='__main__':unittest.main()
