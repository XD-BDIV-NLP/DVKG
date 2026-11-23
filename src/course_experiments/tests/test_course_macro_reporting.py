import unittest

from course_experiments.evaluation.followup_reporting import paired_course_macro, holm_adjust


class CourseMacroReportingTests(unittest.TestCase):
    def test_courses_equal_weight_and_shared_book_resampling(self):
        courses={'c1':'comp','c2':'comp','c3':'comp','c4':'comp','o1':'os','o2':'os','n1':'net','n2':'net'}
        left={book:{book+'_q':1.0 if course=='comp' else 0.0} for book,course in courses.items()}
        right={book:{book+'_q':0.0} for book in courses}
        row=paired_course_macro(left,right,courses,replicates=100)
        self.assertAlmostEqual(row['mean_difference'],1/3)
        self.assertEqual(row['paired_questions'],8)
        self.assertEqual(row['percentile_interval_95'],[1/3,1/3])
        self.assertEqual(row['bootstrap_unit'],'paired_whole_book_within_each_fixed_course')

    def test_mismatched_question_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_course_macro({'a':{'q1':1}}, {'a':{'q2':0}}, {'a':'comp'})

    def test_empty_book_is_missing_coverage_not_zero_or_dropped(self):
        row=paired_course_macro({'a':{'q':1},'b':{}},{'a':{'q':0},'b':{}},{'a':'comp','b':'os'})
        self.assertFalse(row['complete_book_coverage'])
        self.assertEqual(row['empty_books'],['b'])
        self.assertIsNone(row['mean_difference'])
        self.assertIsNone(row['percentile_interval_95'])

    def test_duplicate_question_across_books_and_nonfinite_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_course_macro({'a':{'q':1},'b':{'q':1}}, {'a':{'q':0},'b':{'q':0}}, {'a':'comp','b':'os'})
        with self.assertRaises(ValueError):
            paired_course_macro({'a':{'q':float('nan')}},{'a':{'q':0}},{'a':'comp'})

    def test_failure_zero_stays_in_fixed_book_denominator(self):
        row=paired_course_macro({'a':{'q1':1,'q2':0}},{'a':{'q1':0,'q2':0}},{'a':'comp'},replicates=100)
        self.assertEqual(row['paired_questions'],2)
        self.assertEqual(row['mean_difference'],.5)
        self.assertEqual(row['book_difference'],{'a':.5})

    def test_bootstrap_reproducible(self):
        left={'a':{'q1':1},'b':{'q2':0}};right={'a':{'q1':0},'b':{'q2':.5}};courses={'a':'comp','b':'comp'}
        self.assertEqual(paired_course_macro(left,right,courses,seed=42,replicates=100),
                         paired_course_macro(left,right,courses,seed=42,replicates=100))

    def test_holm_monotone_with_original_method_labels(self):
        self.assertEqual(holm_adjust({'b':.04,'a':.01,'c':.03}),{'b':.06,'a':.03,'c':.06})
        self.assertEqual(holm_adjust({}),{})
        with self.assertRaises(ValueError):holm_adjust({'bad':1.1})


if __name__=='__main__':unittest.main()
