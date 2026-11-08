import unittest

from sherloque.document import Document
from sherloque.retrieve import reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_combines_rankings_for_duplicate_documents(self) -> None:
        first = [
            Document(id="one", title="One", text="First document", score=0.9),
            Document(id="two", title="Two", text="Second document", score=0.8),
        ]
        second = [
            Document(id="two", title="Two", text="Second document", score=0.95),
            Document(id="three", title="Three", text="Third document", score=0.7),
        ]

        results = reciprocal_rank_fusion([first, second])

        self.assertEqual([result.id for result in results], ["two", "one", "three"])
        self.assertAlmostEqual(results[0].score, 1 / 62 + 1 / 61)
        self.assertEqual(results[0].title, "Two")
        self.assertEqual(results[0].text, "Second document")
        self.assertIsNot(results[0], first[1])
        self.assertEqual(first[1].score, 0.8)
        self.assertEqual(second[0].score, 0.95)

    def test_unscored_and_zero_scored_documents_are_fused_by_position(self) -> None:
        documents = [
            Document(id="one", text="Unscored document"),
            Document(id="two", text="Zero scored document", score=0.0),
        ]

        results = reciprocal_rank_fusion([documents])

        self.assertEqual([result.id for result in results], ["one", "two"])
        self.assertAlmostEqual(results[0].score, 1 / 61)
        self.assertAlmostEqual(results[1].score, 1 / 62)
        self.assertIsNone(documents[0].score)
        self.assertEqual(documents[1].score, 0.0)

    def test_empty_rankings_return_no_results(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])
