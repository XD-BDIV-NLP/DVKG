import copy
import unittest

from course_experiments.manifest import digest
from course_experiments.tests.test_scope_identity_batch import packet


class IdentityShortSingleTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_short_single
        return identity_short_single

    def test_source_uses_original_single_judge_and_exact_controls(self):
        api=self.api()
        from course_experiments.evaluation import scope_identity as source
        body,table=api.request_body(packet(3))
        self.assertEqual(body['messages'][0],dict(role='system',content=source.JUDGE))
        self.assertEqual(body['max_tokens'],1024)
        self.assertEqual((body['temperature'],body['seed'],body['enable_thinking']),(0,42,False))
        self.assertEqual(body['response_format'],dict(type='json_object'))
        self.assertEqual(table['pairs']['P0']['candidate_id'],packet(3)['candidate_id'])

    def test_prior_uses_frozen_original_prior24_single_prompt_and_budget(self):
        api=self.api()
        from course_experiments.evaluation import scope_identity_batch as core
        body,table=api.request_body(packet(3,is_prior=True))
        self.assertEqual(body['messages'][0]['content'],core.JUDGE)
        self.assertEqual(body['max_tokens'],8192)
        self.assertTrue(table['batched'])

    def test_graph_or_packet_identity_alone_no_longer_splits_exact_cache(self):
        api=self.api();first=packet(3);second=copy.deepcopy(first)
        second['left']['entity_id']='other_reference';second['right']['entity_id']='other_graph'
        second['candidate_id']=digest({k:v for k,v in second.items() if k!='candidate_id'})
        self.assertEqual(api.request_body(first)[0],api.request_body(second)[0])
        self.assertNotEqual(api.request_body(first)[1],api.request_body(second)[1])

    def test_text_scope_quote_and_order_differences_are_not_normalized_for_dedup(self):
        api=self.api();first=packet(3);changed=copy.deepcopy(first)
        changed['right']['scope']='new necessary scope'
        self.assertNotEqual(api.request_body(first)[0],api.request_body(changed)[0])
        changed=copy.deepcopy(first);changed['left']['source_quotes'].append(copy.deepcopy(changed['left']['source_quotes'][0]))
        self.assertNotEqual(api.request_body(first)[0],api.request_body(changed)[0])


if __name__=='__main__':unittest.main()
