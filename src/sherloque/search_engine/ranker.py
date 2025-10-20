import psycopg
from psycopg import sql

from config import manage_db_cursor
from sherloque.models import Score, URLMatchDetailModel


class ScoreSuite:
    @staticmethod
    async def _normalize_scores(scores: list[Score], lower_is_better: bool = False) -> list[Score]:
        eps = 0.00001
        if lower_is_better:
            min_score = min(scores, key=lambda x: x.score)
            return [Score(score=min_score.score / (score.score + eps), url=score.url) for score in scores]
        else:
            max_score = max(scores, key=lambda x: x.score)
            return [Score(score=(score.score + eps) / (max_score.score + eps), url=score.url) for score in scores]

    @classmethod
    @manage_db_cursor()
    async def score_word_frequency(
            cls,
            cursor: psycopg.AsyncCursor,
            query: str,
            url_matches: URLMatchDetailModel
    ) -> list[Score]:
        token_ids = url_matches.token_ids
        cur = await cursor.execute(
            sql.SQL("""
                    SELECT tl.url_id, COUNT(tl.token_id)
                    FROM token_location tl
                    WHERE tl.token_id in ({token_ids})
                      AND tl.url_id IN ({url_ids})
                    GROUP BY tl.url_id
                    """).format(
                token_ids=sql.SQL(", ").join([sql.Literal(token_id) for token_id in token_ids]),
                url_ids=sql.SQL(", ").join([sql.Literal(url_match.url_id) for url_match in url_matches.url_matches]),
            )
        )
        rows = [record async for record in cur]
        if not rows:
            return []

        url_by_id = {m.url_id: m.url for m in url_matches.url_matches}
        scores = [Score(score=row[1], url=url_by_id[row[0]]) for row in rows]
        normalized_scores = await cls._normalize_scores(scores, lower_is_better=False)
        return normalized_scores

    async def run_scoring(
            self,
            query: str,
            url_matches: URLMatchDetailModel
    ) -> list[Score]:
        score_results = [
            (1.0, await self.score_word_frequency(query, url_matches)),
        ]
        weighted_scores = []
        for weight, scores in score_results:
            for score in scores:
                weighted_scores.append(Score(score=weight * score.score, url=score.url))

        return weighted_scores


class Ranker:
    def __init__(self, score_suite: ScoreSuite):
        self.score_suite = score_suite

    async def rank(self, query: str, url_matches: URLMatchDetailModel):
        scores = await self.score_suite.run_scoring(query, url_matches)
        ranked_urls = sorted(scores, key=lambda x: x.score, reverse=True)
        return ranked_urls


__all__ = [
    "ScoreSuite",
    "Ranker",
]
