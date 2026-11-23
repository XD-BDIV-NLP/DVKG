import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from course_experiments.baselines.tree_kg import TreeKG, nearest_pairs
from course_experiments.tests.test_tree_kg import FakeProvider


class EmbeddingResumeTests(unittest.TestCase):
    def test_cached_embedding_preserves_exact_first_use_vectors_and_pair_distances(self):
        provider = FakeProvider()
        vectors = np.random.default_rng(5).normal(size=(4, 1024)).astype(np.float32)
        provider.embed = Mock(return_value=vectors)
        nodes = [dict(id=str(i), name=str(i), description=f'description {i}') for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            first = TreeKG(provider, Path(directory))._embed(nodes)
            resumed = TreeKG(provider, Path(directory))._embed(nodes)
        self.assertTrue(np.array_equal(first, resumed), 'Resume must not renormalize frozen float32 vectors')
        self.assertEqual(nearest_pairs(nodes, first), nearest_pairs(nodes, resumed))
        provider.embed.assert_called_once()


if __name__ == '__main__':
    unittest.main()
