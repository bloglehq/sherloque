import logging

import psycopg
from psycopg import sql

import log_config
from config import get_settings, manage_db_cursor
from sherloque.models import URLMatch, Score, URLMatchDetailModel
from sherloque.search_engine.ranker import Ranker
from sherloque.utils import preprocess_text

log_config.setup()
settings = get_settings()

LOG = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, ranker: Ranker):
        self.ranker = ranker

    @manage_db_cursor()
    async def get_matched_urls(self, cursor: psycopg.AsyncCursor, query: str) -> URLMatchDetailModel:
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
            return URLMatchDetailModel(url_matches=[], token_ids=[])

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
            return URLMatchDetailModel(url_matches=[], token_ids=[])
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
            URLMatchDetailModel(url_matches=[URLMatch(url=url, url_id=url_id) for url, url_id in zip(urls, url_ids)], token_ids=token_ids)
            if urls
            else URLMatchDetailModel(url_matches=[], token_ids=token_ids)
        )

    async def query(self, query: str) -> list[Score]:
        matches: URLMatchDetailModel = await self.get_matched_urls(query=query)
        if not matches:
            return []
        ranked_urls = await self.ranker.rank(query=query, url_matches=matches)
        return ranked_urls


__all__ = [
    "QueryEngine",
]
