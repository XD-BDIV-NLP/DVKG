import unittest
from course_experiments.evaluation.followup_reporting import paired


class FollowupReportTests(unittest.TestCase):
    def test_pairing_uses_question_identity_not_order(self):
        left={'a':1.,'b':0.};right={'b':.2,'a':.8}
        result=paired(left,right)
        self.assertAlmostEqual(result['mean_difference'],0.)
        self.assertEqual(result['paired_questions'],2)

    def test_no_silent_question_drop(self):
        with self.assertRaisesRegex(ValueError,'same fixed questions'):paired({'a':1.},{'b':1.})

if __name__=='__main__':unittest.main()
