import unittest

try:
    from course_experiments.evaluation import repaired_full as full
except ImportError:
    full=None


class FullEvaluationTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(full,'Missing frozen full reevaluation implementation')

    def test_unanswerable_stays_in_full_denominator(self):
        rows=[dict(question_id='a',status='completed',scores=dict(f1=.8,em=0,evidence_recall=1.,citation_count=1,valid_citation_count=1),machine_judge=dict(score=1.)),
              dict(question_id='b',status='unanswerable_active_sources')]
        result=full.summarize(rows,['a','b'])
        self.assertEqual(result['metrics']['f1'],.4)
        self.assertEqual(result['metrics']['machine_judge_mean'],.5)
        self.assertEqual(result['unanswerable_count'],1)
        self.assertEqual(result['answerability_coverage'],.5)

    def test_duplicate_checkpoints_are_not_completion(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            full.summarize([dict(question_id='a',status='unanswerable_active_sources')]*2,['a','b'])

    def test_request_identity_binds_reference_for_judge_but_not_graph_label(self):
        a=full.request_identity('system',dict(question='q',context=[]),10,dict(model='frozen'))
        b=full.request_identity('system',dict(question='q',context=[]),10,dict(model='changed'))
        self.assertNotEqual(a,b)

    def test_all_reference_sources_required_for_coverage(self):
        self.assertFalse(full.answerable(dict(evidence_ids=['a','b']),{'a'}))
        self.assertTrue(full.answerable(dict(evidence_ids=['a','b']),{'a','b'}))

    def test_long_source_embedding_keeps_all_text(self):
        from course_experiments.evaluation.run_repaired_full import embedding_pieces,LIMITS
        text='原文缓存条件。'*1800
        pieces=embedding_pieces(text)
        self.assertEqual(''.join(pieces),text)
        self.assertTrue(all(len(p)<=2048 for p in pieces))
        self.assertLessEqual(LIMITS['source_embedding_batch'],32)


if __name__=='__main__':unittest.main()
