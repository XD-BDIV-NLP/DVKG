import unittest
from course_experiments.evaluation.priority_comparison import ordered_jobs, complete_for_handoff, acquire_released_lock


class PriorityComparisonTests(unittest.TestCase):
    def test_preserves_original_jobs_and_prioritizes_comparators(self):
        jobs=[dict(name=c+'_'+k,cohort=c,kind=k,phase={'sha256':c+k},question_count=2)
              for k in ['F','G0','T','keyword','vector'] for c in ['compAB','compABCD','osAB','netAB']]
        baseline=[dict(jobs[0],name='compAB_K12',kind='K12')]
        result=ordered_jobs(jobs,baseline)
        self.assertEqual(len(result),21)
        self.assertEqual({j['name']:j for j in result},{j['name']:j for j in jobs+baseline})
        self.assertLess(result.index(baseline[0]),result.index(jobs[4]))
        self.assertLess(next(i for i,j in enumerate(result) if j['name']=='compAB_keyword'),result.index(jobs[4]))

    def test_duplicate_names_rejected(self):
        j=dict(name='compAB_F',cohort='compAB',kind='F')
        with self.assertRaises(ValueError):ordered_jobs([j],[j])

    def test_handoff_requires_full_results_and_exact_jobs(self):
        j=dict(name='compAB_F',question_count=2,phase={'sha256':'old'})
        r=dict(status='completed',job=j,question_count=2,checkpoint_count=2)
        self.assertTrue(complete_for_handoff(j,r))
        self.assertFalse(complete_for_handoff(j,dict(r,status='partial')))
        self.assertFalse(complete_for_handoff(j,dict(r,checkpoint_count=1)))
        self.assertFalse(complete_for_handoff(j,dict(r,job=dict(j,phase={'sha256':'new'}))))

    def test_lock_handoff_waits_for_delayed_release(self):
        now=[0.];calls=[]
        def acquire():
            calls.append(now[0])
            if len(calls)<3:raise BlockingIOError('old owner exiting')
        acquire_released_lock(acquire,timeout=1.,clock=lambda:now[0],pause=lambda n:now.__setitem__(0,now[0]+n))
        self.assertEqual(len(calls),3)

    def test_busy_lock_has_bounded_wait(self):
        now=[0.]
        def acquire():raise BlockingIOError('other owner still active')
        with self.assertRaises(TimeoutError):
            acquire_released_lock(acquire,timeout=.25,clock=lambda:now[0],pause=lambda n:now.__setitem__(0,now[0]+n))
        self.assertLessEqual(now[0],.25)


if __name__=='__main__':unittest.main()
