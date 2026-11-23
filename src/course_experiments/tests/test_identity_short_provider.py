import copy
import json
import unittest

from course_experiments.tests.test_scope_identity_batch import packet
from course_experiments.evaluation import scope_identity as source


class IdentityShortProviderTests(unittest.TestCase):
    def test_receipt_tracker_executor_replacement_reaches_actual_provider(self):
        from course_experiments.evaluation.identity_short_provider import ShortLabelProvider
        class Provider:
            request_executor='original'
        original=Provider();provider=ShortLabelProvider(original)
        provider.request_executor='tracked'
        self.assertEqual(original.request_executor,'tracked')

    def test_original_validator_receives_exact_restored_ids_and_quotes(self):
        from course_experiments.evaluation.identity_short_provider import ShortLabelProvider
        class Provider:
            def chat_response(self,messages,**kwargs):
                self.messages=messages;self.kwargs=kwargs
                wire=json.loads(messages[-1]['content'])
                value=dict(candidate_id=wire['candidate_id'],decision='different_sense',reason='different',
                    scope_consistent=False,source_definition_sufficient=True,homonym_excluded=True,
                    left_citations=copy.deepcopy(wire['left']['source_quotes']),
                    right_citations=copy.deepcopy(wire['right']['source_quotes']))
                self.raw=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))])
                return self.raw
        original=Provider();provider=ShortLabelProvider(original)
        result=source.judge_pair(provider,packet(3))
        self.assertEqual(result['candidate_id'],packet(3)['candidate_id'])
        self.assertFalse(result['eligible_identity_edge'])
        self.assertEqual(json.loads(original.raw['choices'][0]['message']['content'])['candidate_id'],'P0')
        self.assertEqual(original.kwargs['max_tokens'],1024)
        self.assertEqual(original.messages[0]['content'],source.JUDGE)

    def test_cross_side_label_fails_without_deleting_bad_citation(self):
        from course_experiments.evaluation.identity_short_provider import ShortLabelProvider
        class Provider:
            def chat_response(self,messages,**kwargs):
                wire=json.loads(messages[-1]['content'])
                value=dict(candidate_id='P0',left_citations=[dict(evidence_id='R0',quote='x')],right_citations=[])
                return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))])
        with self.assertRaises(ValueError):source.judge_pair(ShortLabelProvider(Provider()),packet(3))


if __name__=='__main__':unittest.main()
