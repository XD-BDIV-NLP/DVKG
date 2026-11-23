from __future__ import annotations

import copy
import unittest

from course_experiments.evaluation import scope_mapping_preflight as preflight
from course_experiments.manifest import digest


class ScopeMappingPreflightTests(unittest.TestCase):
    def test_native_unit_evidence_is_resolved_without_promoting_sentence_support(self):
        unit=dict(id='native-unit',source_node_id='native-unit',text='Alpha is a method.',
            text_digest=digest('Alpha is a method.'),
            evidence_status='source_unit_provenance_not_verified_sentence')
        base={'reference-evidence':'Alpha'}
        index=preflight.source_evidence_index(base,[unit])
        self.assertEqual(index['native-unit']['text'],unit['text'])
        self.assertEqual(index['native-unit']['evidence_status'],unit['evidence_status'])
        self.assertEqual(base,{'reference-evidence':'Alpha'})
        with self.assertRaises(ValueError):
            preflight.source_evidence_index(base,[dict(unit,text='changed text')])
        with self.assertRaises(ValueError):
            preflight.source_evidence_index({'native-unit':'different original'},[unit])

    def test_existing_source_alias_fields_are_preserved_without_inventing_evidence(self):
        entity=dict(entity_id='e',canonical_name='Alpha',aliases=['A'],scope='necessary scope',
            evidence_ids=['s1'],merged_source_node_ids=['s2'],source_node_ids=['s3'])
        original=copy.deepcopy(entity)
        normalized=preflight.entity_sources(entity)
        self.assertEqual(normalized['source_node_ids'],['s1','s2','s3'])
        self.assertEqual(normalized['aliases'],['A'])
        self.assertEqual(normalized['scope'],'necessary scope')
        self.assertEqual(entity,original)

    def test_nullable_native_alias_is_empty_not_an_inferred_synonym(self):
        entity=dict(entity_id='e',canonical_name='Alpha',aliases=None,scope='',evidence_ids=None)
        normalized=preflight.entity_sources(entity)
        self.assertEqual(normalized['aliases'],[])
        self.assertEqual(normalized['source_node_ids'],[])
        self.assertIsNone(entity['aliases'])
        self.assertEqual(normalized['input_syntax_normalizations'],['aliases_null_to_empty'])

    def test_untyped_and_empty_typed_graphs_are_explicitly_ineligible(self):
        graph=dict(entities=[dict(entity_id='a',canonical_name='A',type='Concept',type_status='native_untyped_candidate')],relations=[])
        self.assertEqual(preflight.qualification(graph)['status'],'not_applicable_untyped_concept_identity')
        graph['entities'][0].pop('type_status')
        self.assertEqual(preflight.qualification(graph)['status'],'insufficient_no_supported_typed_edges')
        graph['relations']=[dict(head_entity_id='a',tail_entity_id='a',relation='is_a')]
        self.assertEqual(preflight.qualification(graph)['status'],'eligible_typed_graph')

    def test_same_course_candidates_count_without_facts_or_method_scores(self):
        left=dict(entity_id='r',canonical_name='Alpha',scope='domain',aliases=[],
            citations=[dict(evidence_id='s',quote='Alpha is a method.')])
        graph=dict(entities=[dict(entity_id='g',canonical_name='Alpha',scope='',aliases=[],type='Concept',
            source_node_ids=['s'])],relations=[dict(head_entity_id='g',tail_entity_id='g',relation='is_a')])
        result,packets=preflight.prepare_graph('os',graph,[left],{'s':'Alpha is a method.'},{'s':'Alpha is a method.'})
        self.assertEqual(result['candidate_pairs'],1)
        self.assertEqual(result['two_sided_source_pairs'],1)
        self.assertEqual(result['maximum_initial_and_reverse_logical_judgments'],2)
        self.assertEqual(len(packets),1)

    def test_prior_definition_without_book_evidence_is_visible_not_fabricated(self):
        left=dict(entity_id='r',canonical_name='Alpha',scope='domain',citations=[dict(evidence_id='s',quote='Alpha')])
        graph=dict(entities=[dict(entity_id='g',canonical_name='Alpha',scope='',type='Concept',definition='a method')],
            relations=[dict(head_entity_id='g',tail_entity_id='g',relation='is_a')])
        result,packets=preflight.prepare_graph('os',graph,[left],{'s':'Alpha'}, {})
        self.assertEqual(result['prior_or_missing_graph_source_pairs'],1)
        self.assertEqual(result['two_sided_source_pairs'],0)
        self.assertEqual(result['maximum_initial_and_reverse_logical_judgments'],0)
        self.assertFalse(packets[0]['both_sides_have_source'])


if __name__=='__main__':unittest.main()
