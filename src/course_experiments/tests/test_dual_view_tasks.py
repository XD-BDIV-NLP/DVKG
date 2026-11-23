import unittest
from course_experiments.dual_view import build_structure
from course_experiments.tests.test_dual_view import corpus
from course_experiments.evaluation.dual_view_tasks import split_sources,validate_task,near_duplicate


class DualViewTaskTests(unittest.TestCase):
    def test_development_and_confirmation_chapters_do_not_overlap(self):
        structure=build_structure(corpus());splits=split_sources(structure,{'e1'})
        self.assertNotIn('e1',splits['development']+splits['confirmation'])
        self.assertFalse(set(splits['development'])&set(splits['confirmation']))

    def test_question_is_bound_to_target_book_and_exact_source_quote(self):
        source=build_structure(corpus())['evidence']
        q=dict(question='缓存能够降低哪一种延迟？',answer='缓存降低访存延迟。',citations=[dict(evidence_id='e1',quote='缓存降低访存延迟。')])
        result=validate_task(q,source,'A','definition')
        self.assertEqual(result['target_book_id'],'A')
        with self.assertRaises(ValueError):validate_task(q,source,'B','definition')
        q['citations'][0]['quote']='书中没有的句子'
        with self.assertRaises(ValueError):validate_task(q,source,'A','definition')

    def test_cross_chapter_requires_two_real_chapters_in_same_book(self):
        source=build_structure(corpus())['evidence']
        q=dict(question='缓存与虚拟内存分别如何影响访存？',answer='缓存降低延迟，虚拟内存使用地址映射。',citations=[dict(evidence_id='e1',quote='缓存降低访存延迟。')])
        with self.assertRaises(ValueError):validate_task(q,source,'A','cross_chapter')
        q['citations'].append(dict(evidence_id='e2',quote='虚拟内存使用地址映射。'))
        self.assertEqual(len(validate_task(q,source,'A','cross_chapter')['citations']),2)

    def test_condition_task_requires_both_cases(self):
        q=dict(question='在何种条件下可以使用缓存？',answer='仅依据原文回答。',citations=[dict(evidence_id='e1',quote='缓存降低访存延迟。')])
        with self.assertRaises(ValueError):validate_task(q,build_structure(corpus())['evidence'],'A','conditions')

    def test_duplicates_are_rejected_before_model_evaluation(self):
        self.assertTrue(near_duplicate('缓存 如何 工作？',['缓存如何工作？']))
        self.assertFalse(near_duplicate('解释虚拟内存地址转换',['网络拥塞是什么']))


if __name__=='__main__':unittest.main()
