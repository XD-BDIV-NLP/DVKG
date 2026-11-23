import unittest
import numpy as np
from course_experiments.evaluation.dvkg_development import dense_fine_order, occurrence_evidence_ids


class DenseFineTests(unittest.TestCase):
    def test_fine_ranking_cannot_escape_coarse_or_target_book(self):
        rows=[dict(book_id='b',evidence_ids=['e0']),dict(book_id='b',evidence_ids=['e1']),dict(book_id='b',evidence_ids=['e2'])]
        order=dense_fine_order(rows,np.asarray([[1.,0.],[0.,1.],[.8,.2]]),{1,2},np.array([1.,0.]))
        self.assertEqual(order,[2,1])

    def test_equal_dense_scores_have_deterministic_source_tie_break(self):
        rows=[dict(book_id='b',evidence_ids=['z']),dict(book_id='b',evidence_ids=['a'])]
        order=dense_fine_order(rows,np.ones((2,2)),{0,1},np.array([1.,0.]))
        self.assertEqual(order,[1,0])

    def test_occurrence_links_are_exact_names_and_never_single_character_aliases(self):
        rows=[dict(book_id='b',evidence_ids=['1'],text='用CPU处理缓存。'),
              dict(book_id='b',evidence_ids=['2'],text='CPUX 不应误匹配，其他位。')]
        self.assertEqual(occurrence_evidence_ids(rows,'b',['CPU','位']),['1'])
        with self.assertRaises(ValueError):occurrence_evidence_ids(rows,'foreign',['CPU'])


if __name__=='__main__':unittest.main()
