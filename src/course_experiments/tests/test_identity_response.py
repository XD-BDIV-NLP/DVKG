import copy
import unittest


class IdentityResponseTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_response
        return identity_response

    def test_complete_nested_boolean_guards_losslessly_expand(self):
        api=self.api();value=dict(candidate_id='unchanged',decision='insufficient',left_citations=[],right_citations=[],
            flags=dict(necessary_scope_consistent=True,definition_identity_sufficient=False,homonym_excluded=True))
        before=copy.deepcopy(value);result=api.normalize_flags(value,'prior_definition_identity')
        self.assertEqual(value,before)
        self.assertFalse(result['definition_identity_sufficient'])
        for key in ['candidate_id','decision','left_citations','right_citations','flags']:self.assertEqual(result[key],value[key])

    def test_conflicting_missing_unknown_or_nonboolean_nested_guards_rejected(self):
        api=self.api();flags=dict(scope_consistent=True,source_definition_sufficient=True,homonym_excluded=True)
        variants=[dict(flags=flags,scope_consistent=False),dict(flags=dict(flags,unexpected=True)),
            dict(flags=dict(scope_consistent=True)),dict(flags=dict(flags,homonym_excluded='true'))]
        for value in variants:
            with self.assertRaises(ValueError):api.normalize_flags(value,'two_sided_source')

    def test_old_flat_judgments_remain_exactly_unchanged(self):
        api=self.api();value=dict(candidate_id='same',scope_consistent=True,source_definition_sufficient=False,homonym_excluded=True)
        self.assertEqual(api.normalize_flags(value,'two_sided_source'),value)

    def test_real_validator_and_raw_response_preserved_for_source_and_prior(self):
        api=self.api()
        from course_experiments.evaluation import scope_identity_batch as core
        from course_experiments.tests.test_scope_identity_batch import Provider,packet
        for is_prior in [False,True]:
            def nest(wire,result,count):
                keys=['necessary_scope_consistent','definition_identity_sufficient','homonym_excluded'] if is_prior else ['scope_consistent','source_definition_sufficient','homonym_excluded']
                for row in result['results']:row['flags']={key:row.pop(key) for key in keys}
            work=[packet(1,is_prior=is_prior)]
            original=core.judge_batch(Provider(nest),work)
            self.assertFalse(next(iter(original['decisions'].values()))['eligible_identity_edge'])
            with api.parser_bridge():result=core.judge_batch(Provider(nest),work)
            self.assertTrue(next(iter(result['decisions'].values()))['eligible_identity_edge'])
            self.assertEqual(result['raw_response'],original['raw_response'])
            again=core.judge_batch(Provider(nest),work)
            self.assertEqual(again['decisions'],original['decisions'])


if __name__=='__main__':unittest.main()
