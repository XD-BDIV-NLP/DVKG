import unittest
from course_experiments import supplement_tasks as s


class SupplementTests(unittest.TestCase):
    def source(self):
        return [dict(book_id='A',chunk_id='a',evidence=[dict(id='x',text='缓存仅在命中时返回。')]),
                dict(book_id='B',chunk_id='b',evidence=[dict(id='y',text='未命中需要访问主存。')])]

    def task(self):
        return dict(question='分别解释命中和未命中。',answer='命中返回，未命中访问主存。',
            citations=[dict(evidence_id='x',quote='命中时返回'),dict(evidence_id='y',quote='访问主存')])

    def test_cross_source_cannot_pass_with_one_chunk(self):
        q=self.task();q['citations']=q['citations'][:1]
        with self.assertRaisesRegex(ValueError,'two source chunks'):s.validate(q,self.source(),'cross_source')

    def test_holdout_excludes_chunk_and_duplicate_evidence(self):
        self.assertEqual(s.eligible(self.source(),{'a'},set()),self.source()[1:])
        self.assertEqual(s.eligible(self.source(),set(),{'y'}),self.source()[:1])

    def test_conditions_need_explicit_contrast(self):
        with self.assertRaisesRegex(ValueError,'condition'):s.validate(self.task(),self.source(),'conditions')

    def test_exact_citations_and_full_sources_are_preserved(self):
        q=s.validate(self.task(),self.source(),'cross_source')
        self.assertEqual(q['evidence_ids'],['x','y'])
        self.assertEqual(q['source_chunk_ids'],['a','b'])

if __name__=='__main__':unittest.main()
