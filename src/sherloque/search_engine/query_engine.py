import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio.engine import AsyncEngine

from app.config import get_async_engine
from sherloque import log_config
from sherloque.models import URLMatch, Score, URLMatchDetailModel
from sherloque.search_engine.ranker import Ranker
from sherloque.utils import preprocess_text

log_config.setup()

LOG = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, ranker: Ranker, engine: AsyncEngine):
        self.ranker = ranker
        self.engine = engine

    async def get_matched_urls(self, query: str) -> URLMatchDetailModel:
        tokens = await preprocess_text(query)
        if not tokens:
            return URLMatchDetailModel(url_matches=[], token_ids=[])

        async with self.engine.connect() as conn:
            cur = await conn.execute(
                text("SELECT _id FROM token_list WHERE token = ANY(:tokens)"),
                {"tokens": tokens},
            )
            token_ids = [record[0] for record in cur]
            if not token_ids:
                LOG.debug("No tokens were found in the database matching the query.")
                return URLMatchDetailModel(url_matches=[], token_ids=[])

            cur = await conn.execute(
                text("""
                     SELECT url_id, COUNT(DISTINCT token_id)
                     FROM token_location
                     WHERE token_id = ANY(:token_ids)
                     GROUP BY url_id
                     """),
                # -- HAVING COUNT(DISTINCT token_id) = :num_tokens
                {
                    "token_ids": token_ids,
                    # "num_tokens": len(token_ids),
                },
            )
            urls_tok_counts = [record for record in cur]
            if not urls_tok_counts:
                LOG.debug("No URLs were found in the database matching the query.")
                return URLMatchDetailModel(url_matches=[], token_ids=[])
            cur = await conn.execute(
                text("""
                     SELECT url
                     FROM url_list
                     WHERE _id = ANY(:url_ids)
                     """),
                {"url_ids": [url_id for url_id, _ in urls_tok_counts]},
            )
            urls = [record[0] for record in cur]

        return (
            URLMatchDetailModel(
                url_matches=[
                    URLMatch(
                        url=url, url_id=url_id, num_matched_tokens=num_matched_tokens
                    ) for url, (url_id, num_matched_tokens) in zip(urls, urls_tok_counts)
                ],
                token_ids=token_ids
            )
            if urls
            else URLMatchDetailModel(url_matches=[], token_ids=token_ids)
        )

    async def query(self, query: str) -> list[Score]:
        matches: URLMatchDetailModel = await self.get_matched_urls(query=query)
        if not matches.url_matches:
            return []
        ranked_urls = await self.ranker.rank(url_matches=matches)
        return ranked_urls


__all__ = [
    "QueryEngine",
]
