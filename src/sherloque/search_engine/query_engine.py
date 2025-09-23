import logging

import psycopg

import log_config
from config import get_settings
from config import manage_db_cursor
from utils import preprocess_text

log_config.setup()
settings = get_settings()

LOG = logging.getLogger(__name__)


class QueryEngine:

    async def get_scored_list(self, urls: list[str]):
        total_scores = {url:0 for url in urls}
        weights = []
        for (weight, scores) in weights:
            for url in total_scores:
                total_scores[url] += weight * scores[url]
        return total_scores

    @manage_db_cursor()
    async def get_matched_urls(self, cursor: psycopg.AsyncCursor, query: str) -> list[tuple[int, str]]:
        tokens = await preprocess_text(query)
        if not tokens:
            return []
        token_placeholders = ", ".join(["%s"] * len(tokens))
        cur = await cursor.execute(
            f"SELECT _id FROM token_list WHERE token IN ({token_placeholders})",  # noqa
            tokens
        )
        token_ids = [record[0] for record in await cur.fetchall()]
        if not token_ids:
            LOG.debug("None of the tokens were found in the database.")

        num_tokens = len(tokens)
        token_id_placeholders = ", ".join(["%s"] * num_tokens)
        uid_cur = await cursor.execute(
            f"""
            SELECT url_id FROM token_location
            WHERE token_id IN ({token_id_placeholders})
            GROUP BY url_id
            HAVING COUNT(DISTINCT token_id) = %s
            """,
            (token_ids + [num_tokens])
        )
        url_ids = [row[0] async for row in uid_cur]
        if not url_ids:
            return []
        uid_placeholders = ", ".join(["%s"] * len(url_ids))
        url_cur = await cursor.execute(
            f"""
            SELECT _id, url FROM url_list
            WHERE _id in ({uid_placeholders})
            """,
            url_ids,
        )
        return [row async for row in url_cur]

    async def query(self, q: str):
        matches = await self.get_matched_urls(q)
        if not matches:
            return []
        url_ids, urls = [], []
        for match in matches:
            url_ids.append(match[0])
            urls.append(match[1])
        scores = await self.get_scored_list(urls)
        ranked_urls = sorted([(score, url) for (url, score) in scores.items()], reverse=True)
        return ranked_urls


__all__ = [
    "QueryEngine",
]
