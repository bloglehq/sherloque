from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio.engine import AsyncConnection, AsyncEngine

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
    async def score_word_frequency(
            cls,
            url_matches: URLMatchDetailModel,
            conn: AsyncConnection,
    ) -> list[Score]:
        token_ids = url_matches.token_ids
        cur = await conn.execute(
            text("""
                 SELECT tl.url_id, COUNT(tl.token_id)
                 FROM token_location tl
                 WHERE tl.token_id = ANY(:token_ids)
                   AND tl.url_id = ANY(:url_ids)
                 GROUP BY tl.url_id
                 """),
            {
                "token_ids": token_ids,
                "url_ids": [url_match.url_id for url_match in url_matches.url_matches],
            }
        )
        rows = [record for record in cur]
        if not rows:
            return []

        url_by_id = {m.url_id: m.url for m in url_matches.url_matches}
        scores = [Score(score=row[1], url=url_by_id[row[0]]) for row in rows]
        normalized_scores = await cls._normalize_scores(scores, lower_is_better=False)
        return normalized_scores

    @classmethod
    async def score_token_location_start_bias(
            cls,
            url_matches: URLMatchDetailModel,
            conn: AsyncConnection,
    ) -> list[Score]:
        cur = await conn.execute(
            text("""
                 SELECT tl.url_id, tl.token_id, MIN(tl.location)
                 FROM token_location tl
                 WHERE token_id = ANY(:token_ids)
                   AND url_id = ANY(:url_ids)
                 GROUP BY tl.url_id, tl.token_id
                 """),
            {
                "token_ids": url_matches.token_ids,
                "url_ids": [url_match.url_id for url_match in url_matches.url_matches],
            }
        )
        rows = [record for record in cur]
        if not rows:
            return []

        url_by_id = {m.url_id: m.url for m in url_matches.url_matches}
        pos_sum_per_url = {
            out_row[0]: sum([in_row[2] for in_row in rows if in_row[0] == out_row[0]])
            for out_row in rows
        }
        scores = [Score(score=pos_sum_per_url[row[0]], url=url_by_id[row[0]]) for row in rows]
        normalized_scores = await cls._normalize_scores(scores, lower_is_better=True)
        return normalized_scores

    async def run_scoring(
            self,
            conn: AsyncConnection,
            url_matches: URLMatchDetailModel
    ) -> list[Score]:
        score_results = [
            (1.0, await self.score_word_frequency(url_matches=url_matches, conn=conn)),
            (1.5, await self.score_token_location_start_bias(url_matches=url_matches, conn=conn)),
        ]
        weighted_scores = defaultdict(float)
        for weight, scores in score_results:
            for score in scores:
                weighted_scores[score.url] += weight * score.score
        return [Score(score=score / sum([w for w, _ in score_results]), url=url) for url, score in
                weighted_scores.items()]


class Ranker:
    def __init__(self, score_suite: ScoreSuite, engine: AsyncEngine):
        self.score_suite = score_suite
        self.engine = engine

    async def rank(self, url_matches: URLMatchDetailModel):
        async with self.engine.connect() as conn:
            scores = await self.score_suite.run_scoring(conn=conn, url_matches=url_matches)
        ranked_urls = sorted(scores, key=lambda x: x.score, reverse=True)
        return ranked_urls


__all__ = [
    "ScoreSuite",
    "Ranker",
]
