import unittest

from sherloque.retrieve import RetrieverResult, reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_combines_rankings_for_duplicate_documents(self) -> None:
        first = [
            RetrieverResult(doc_id=1, doc_title="One", score=0.9),
            RetrieverResult(doc_id=2, doc_title="Two", score=0.8),
        ]
        second = [
            RetrieverResult(doc_id=2, doc_title="Two", score=0.95),
            RetrieverResult(doc_id=3, doc_title="Three", score=0.7),
        ]

        results = reciprocal_rank_fusion([first, second])

        self.assertEqual([result.doc_id for result in results], [2, 1, 3])
        self.assertAlmostEqual(results[0].score, 1 / 62 + 1 / 61)

    def test_empty_rankings_return_no_results(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])
