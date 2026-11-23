import unittest


class ReferenceIdentityAuditTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import reference_identity_audit
        return reference_identity_audit

    def reference(self):
        return dict(course_id='course',entities=[
            dict(entity_id='a',canonical_name='Alpha',aliases=[],scope='chapter one',citations=[dict(evidence_id='1',quote='Alpha')]),
            dict(entity_id='b',canonical_name='ＡＬＰＨＡ',aliases=[],scope='a free technical scope',citations=[dict(evidence_id='2',quote='ＡＬＰＨＡ')]),
            dict(entity_id='c',canonical_name='Container',aliases=[],scope='course',citations=[])],
            facts=[dict(fact_id='f1',head='a',relation='part_of',tail='c',conditions='x'),
                dict(fact_id='f2',head='c',relation='consists_of',tail='b',conditions='y')])

    def test_cross_scope_names_generate_audit_candidates_never_merge(self):
        api=self.api();reference=self.reference();result=api.audit_reference(reference)
        self.assertEqual(result['summary']['cross_scope_reference_pairs'],1)
        self.assertEqual(result['summary']['potential_equivalent_fact_clusters_across_actual_groups'],1)
        self.assertEqual(result['summary']['semantic_equivalence_confirmed'],False)
        self.assertEqual(len(result['candidates']),1)
        self.assertEqual(len(reference['entities']),3)
        self.assertTrue(result['summary']['new_split_created'] is False)

    def test_same_scope_explicit_aliases_are_already_grouped(self):
        api=self.api();reference=self.reference();reference['entities'][1].update(scope='chapter one',canonical_name='other',aliases=['Alpha'])
        result=api.audit_reference(reference)
        self.assertEqual(result['summary']['cross_scope_reference_pairs'],0)
        self.assertEqual(result['summary']['potential_equivalent_fact_clusters_across_actual_groups'],0)

    def test_metadata_never_contains_names_scopes_quotes_or_fact_text(self):
        api=self.api();result=api.audit_reference(self.reference())
        import json
        metadata=json.dumps(result['summary'])
        for value in ['Alpha','chapter one','Container','ＡＬＰＨＡ']:
            self.assertNotIn(value,metadata)
        self.assertEqual(result['summary']['minimum_primary_pair_judgments'],1)
        self.assertEqual(result['summary']['maximum_single_primary_and_positive_reverse_logical'],2)
        self.assertEqual(len(result['summary']['candidate_population_digest']),64)


if __name__=='__main__':unittest.main()
