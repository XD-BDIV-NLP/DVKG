import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from course_experiments.evaluation.dual_view_pipeline import embeddings,candidate_bundles,store_vectors
from course_experiments.evaluation.dual_view_tasks import split_sources
from course_experiments.tests.test_dual_view import corpus
from course_experiments.dual_view import build_structure


class PipelineTests(unittest.TestCase):
    def test_embedding_batches_and_index_metadata_are_separate_and_resumable(self):
        calls=[]
        runtime=SimpleNamespace(config=dict(embedding=dict(model='BAAI/bge-m3')),embed=lambda texts:(calls.append(texts) or np.ones((len(texts),1024))))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'query.npy';record=embeddings(runtime,['one','two'],p)
            self.assertEqual(embeddings(runtime,['one','two'],p),record)
            self.assertEqual(len(calls),1)
            self.assertEqual(np.load(p).shape,(2,1024))
            with self.assertRaises(ValueError):embeddings(runtime,['different'],p)

    def test_cross_chapter_candidates_stay_in_split(self):
        structure=build_structure(corpus())
        rows=[dict(evidence_ids=[eid]) for eid in structure['evidence']]
        with tempfile.TemporaryDirectory() as folder:
            record=store_vectors(Path(folder)/'units.npy',np.eye(2))
            units=dict(rows=rows,vectors=record)
            self.assertEqual(list(candidate_bundles(structure,units,['e1'],'cross_chapter')),[])
            bundles=list(candidate_bundles(structure,units,['e1','e2'],'cross_chapter'))
            self.assertEqual(bundles,[['e1','e2']])

    def test_development_has_two_chapters_for_independent_cross_chapter_check(self):
        evidence={str(i):dict(chapter_id=str(i)) for i in range(10)}
        structure=dict(book_id='A',evidence=evidence);splits=split_sources(structure,set())
        self.assertEqual(len(splits['development']),2)
        self.assertEqual(len(splits['confirmation']),8)


if __name__=='__main__':unittest.main()
