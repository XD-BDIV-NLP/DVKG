import json
import unittest

from course_experiments.evaluation import scope_identity_packing as packing
from course_experiments.tests.test_scope_identity_batch import packet


class Counter:
    def audit(self,rows):
        tokens=sum(len(p['right']['source_quotes'][0]['quote']) for p in rows)
        return dict(prompt_tokens=tokens,total_with_reserves=tokens+10,working_context_tokens=100,
            fits=tokens+10<=100,max_tokens=8192)


class PackingTests(unittest.TestCase):
    def test_complete_group_that_fits_is_counted_once_not_all_prefixes(self):
        class Recorder(Counter):
            def __init__(self):self.calls=[]
            def audit(self,rows):self.calls.append(len(rows));return super().audit(rows)
        work=[packet(i) for i in range(5)]
        for p in work:p['right']['source_quotes'][0]['quote']='x'*10
        counter=Recorder();self.assertEqual(len(packing.plan_batches(work,counter)),1)
        self.assertEqual(counter.calls,[5])

    def test_counted_packing_splits_complete_pairs_deterministically_without_truncation(self):
        work=[packet(i) for i in range(9)]
        for p in work:p['right']['source_quotes'][0]['quote']='x'*40
        planned=packing.plan_batches(work,Counter())
        self.assertTrue(all(len(rows)<=2 for rows in planned))
        self.assertEqual(packing.plan_batches(list(reversed(work)),Counter()),planned)
        self.assertEqual(sum(map(len,planned)),9)
        self.assertTrue(all(p['right']['source_quotes'][0]['quote']=='x'*40 for rows in planned for p in rows))

    def test_one_oversize_pair_is_rejected_before_any_model_call(self):
        work=[packet(1)];work[0]['right']['source_quotes'][0]['quote']='x'*91
        with self.assertRaises(packing.OversizeIdentityPair):packing.plan_batches(work,Counter())

    def test_counter_message_builder_matches_frozen_batch_wire(self):
        from course_experiments.tests.test_scope_identity_batch import Provider
        from course_experiments.evaluation import scope_identity_batch as core
        rows=[packet(1),packet(2)];provider=Provider();core.judge_batch(provider,rows,reverse=True)
        messages=packing.messages_for(rows,reverse=True)
        self.assertEqual(json.loads(messages[1]['content']),provider.calls[0][0])
        self.assertEqual(messages[0]['content'],core.JUDGE)


if __name__=='__main__':unittest.main()
