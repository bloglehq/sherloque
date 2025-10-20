import logging

import psycopg
from psycopg import sql
from pydantic import BaseModel

import log_config
from config import get_settings, manage_db_cursor
from sherloque.utils import preprocess_text

log_config.setup()
settings = get_settings()

LOG = logging.getLogger(__name__)


class Score(BaseModel):
    score: float
    url: str


class ScoreDetailModel(BaseModel):
    name: str
    query: str
    weighted_scores: list[Score]


class URLMatch(BaseModel):
    url: str
    url_id: int


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
            url_matches: list[URLMatch]
    ) -> list[Score]:
        tokens = await preprocess_text(query)
        if not tokens:
            return []
        cur = await cursor.execute(
            sql.SQL("SELECT _id FROM token_list WHERE token IN ({})").format(
                sql.SQL(", ").join([sql.Literal(token) for token in tokens])
            ),
        )
        token_ids = [record[0] async for record in cur]
        if not token_ids:
            return []

        cur = await cursor.execute(
            sql.SQL("""
                SELECT tl.url_id, COUNT(tl.token_id)
                FROM token_location tl
                WHERE tl.token_id in ({token_ids})
                  AND tl.url_id IN ({url_ids})
                GROUP BY tl.url_id
            """).format(
                token_ids=sql.SQL(", ").join([sql.Literal(token_id) for token_id in token_ids]),
                url_ids=sql.SQL(", ").join([sql.Literal(url_match.url_id) for url_match in url_matches]),
            )
        )
        rows = [record async for record in cur]
        if not rows:
            return []
        
        url_by_id = {m.url_id: m.url for m in url_matches}
        scores = [Score(score=row[1], url=url_by_id[row[0]]) for row in rows]
        normalized_scores = await cls._normalize_scores(scores, lower_is_better=False)
        return normalized_scores

    async def run_scoring(
            self,
            query: str,
            url_matches: list[URLMatch]
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

    async def rank(self, query: str, url_matches: list[URLMatch]):
        scores = await self.score_suite.run_scoring(query, url_matches)
        ranked_urls = sorted(scores, key=lambda x: x.score, reverse=True)
        return ranked_urls


class QueryEngine:
    def __init__(self, ranker: Ranker):
        self.ranker = ranker

    @manage_db_cursor()
    async def get_matched_urls(self, cursor: psycopg.AsyncCursor, query: str) -> list[URLMatch]:
        tokens = await preprocess_text(query)
        if not tokens:
            return []
        cur = await cursor.execute(
            sql.SQL("SELECT _id FROM token_list WHERE token IN ({})").format(
                sql.SQL(", ").join([sql.Literal(token) for token in tokens])
            ),
        )
        token_ids = [record[0] async for record in cur]
        if not token_ids:
            LOG.debug("No tokens were found in the database matching the query.")
            return []

        uid_cur = await cursor.execute(
            sql.SQL("""
                    SELECT url_id
                    FROM token_location
                    WHERE token_id IN ({token_ids})
                    GROUP BY url_id
                    HAVING COUNT(DISTINCT token_id) = {num_tokens}
                    """).format(
                token_ids=sql.SQL(", ").join(
                    [sql.Literal(token_id) for token_id in token_ids]
                ),
                num_tokens=sql.Literal(len(token_ids)),
            ),
        )
        url_ids = [record[0] async for record in uid_cur]
        if not url_ids:
            LOG.debug("No URLs were found in the database matching the query.")
            return []
        url_cur = await cursor.execute(
            sql.SQL("""
                    SELECT url
                    FROM url_list
                    WHERE _id IN ({url_ids})
                    """).format(
                url_ids=sql.SQL(", ").join([sql.Literal(url_id) for url_id in url_ids]),
            ),
        )
        urls = [record[0] async for record in url_cur]
        return (
            [URLMatch(url=url, url_id=url_id) for url, url_id in zip(urls, url_ids)]
            if urls
            else []
        )

    async def query(self, query: str):  # -> list[tuple[float, str]]:
        matches: list[URLMatch] = await self.get_matched_urls(query=query)
        if not matches:
            return []
        ranked_urls = await self.ranker.rank(query=query, url_matches=matches)
        return ranked_urls


__all__ = [
    "ScoreSuite",
    "Ranker",
    "QueryEngine",
]
