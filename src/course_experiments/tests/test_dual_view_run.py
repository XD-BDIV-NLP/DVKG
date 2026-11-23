import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from course_experiments.evaluation.dual_view_run import validate_context,reference_metrics,summarize,matched_jobs
from course_experiments.evaluation import dual_view_run as run
from course_experiments.tests.test_dual_view_rag import retriever


class DualViewRunTests(unittest.TestCase):
    def test_foreign_book_context_rejected(self):
        rows=[dict(book_id='A',evidence_ids=['a'],text='缓存降低延迟')]
        validate_context(rows,'A')
        with self.assertRaises(ValueError):validate_context(rows,'B')

    def test_candidate_and_window_misses_are_separate(self):
        q=dict(evidence_ids=['a','b'],citations=[dict(evidence_id='a',quote='缓存降低'),dict(evidence_id='b',quote='地址转换')],chapter_ids=['c1','c2'])
        structure=dict(evidence={'a':dict(chapter_id='c1'),'b':dict(chapter_id='c2')})
        context=[dict(book_id='A',evidence_ids=['a'],text='缺少引句')]
        score=reference_metrics(q,context,['a'],['a','b'],structure)
        self.assertEqual(score['coarse_evidence_recall'],1)
        self.assertEqual(score['selected_evidence_recall'],.5)
        self.assertEqual(score['evidence_recall'],0)
        self.assertEqual(score['selected_chapter_recall'],.5)

    def test_missing_questions_keep_full_denominator(self):
        record=dict(status='completed',question_id='q',question_type='definition',scores=dict(f1=.6,em=0,evidence_recall=.4,citation_count=0,valid_citation_count=0),machine_judge=dict(score=.5),retrieval=dict(fallback=False,candidate_count=2,total_target_units=10,retrieval_seconds=.1,structure_load_seconds=0))
        result=summarize([record],['q','missing'])
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['metrics']['f1'],.3)
        self.assertEqual(result['completed_count'],1)
        with self.assertRaises(ValueError):summarize([record,record],['q'])

    def test_ab_and_abcd_compare_same_target_books(self):
        assets=dict(cohorts={'compAB':dict(active_books=['A','B']),'compABCD':dict(active_books=['A','B','C','D']),'osAB':dict(active_books=['E','F']),'netAB':dict(active_books=['G','H'])})
        jobs=matched_jobs(assets)
        self.assertEqual(len(jobs),80)
        self.assertEqual({j['target_book_id'] for j in jobs if j['cohort']=='compAB'},{'A','B'})
        self.assertEqual({j['target_book_id'] for j in jobs if j['cohort']=='compABCD'},{'A','B','C','D'})

    def test_resume_reuses_retrieval_after_answer_format_failures(self):
        q=dict(question_id='q',question='缓存如何降低延迟？',answer='缓存降低访存延迟。',type='definition',target_book_id='A',evidence_ids=['e1'],citations=[dict(evidence_id='e1',quote='缓存降低访存延迟。')],chapter_ids=['c1'])
        job=dict(name='fixture',cohort='fixture',target_book_id='A',method='dual')
        args=(job,dict(contract={}),dict(questions=[q]),dict(q=np.array([1.,0.])),SimpleNamespace(config={}))
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'retriever_for',side_effect=lambda *a:retriever()):
            with patch.object(run,'shared_call',side_effect=ValueError('bad format')):
                first=run.evaluate(*args,Path(folder),{})
                self.assertEqual(first['status'],'partial')
            def model(runtime,prompt,*a):
                return (dict(answer=q['answer'],citations=q['citations']) if prompt==run.ANSWER else dict(score=1.,grounding=1.)),{}
            with patch.object(run,'shared_call',side_effect=model):
                second=run.evaluate(*args,Path(folder),{})
                self.assertEqual(second['status'],'completed')
                self.assertEqual(second['metrics']['evidence_recall'],1)


if __name__=='__main__':unittest.main()
