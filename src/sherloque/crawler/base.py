import logging
from abc import abstractmethod

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.engine import AsyncConnection

from sherloque.utils import ALLOWED_TABLE_FIELDS, preprocess_text

LOG = logging.getLogger(__name__)


class CrawlerBase:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    def __repr__(self):
        return "Crawler base."

    async def preprocess(self, text: str) -> list[str]:
        return await preprocess_text(text)

    async def _get_entry_id(self, conn: AsyncConnection, table: str, field: str, value: str) -> int:
        """Helper function for getting an entry id and adding it if it's not present"""
        if table not in ALLOWED_TABLE_FIELDS or \
                field not in ALLOWED_TABLE_FIELDS[table]:
            raise ValueError("Invalid table or field name provided")

        cur = await conn.execute(
            sa.text(f"SELECT _id FROM {table} WHERE {field} = :value"),
            {"value": value},
        )
        row = cur.fetchone()
        if row is not None:
            return row[0]

        cur = await conn.execute(
            sa.text(f"INSERT INTO {table} ({field}) VALUES (:value) RETURNING _id"),
            {"value": value},
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to insert into {table} ({field})")
        return row[0]

    async def add_to_index(self, conn: AsyncConnection, url: str, text: str):
        """Index an individual page"""
        LOG.info(f"Indexing URL: {url}")
        processed_toks = await self.preprocess(text)
        url_id = await self._get_entry_id(conn, "url_list", "url", url)

        for i, token in enumerate(processed_toks):
            token_id = await self._get_entry_id(conn, "token_list", "token", token)
            await conn.execute(
                sa.text("""
                        INSERT INTO token_location(url_id, token_id, location)
                        VALUES (:url_id, :token_id, :location)
                        """),
                {"url_id": url_id, "token_id": token_id, "location": i}
            )
        LOG.info(f"Finished indexing URL: {url}")

    async def is_indexed(self, conn: AsyncConnection, url: str) -> bool:
        """Return True if this URL is already indexed"""
        cur = await conn.execute(
            sa.text("SELECT _id FROM url_list WHERE url = :url LIMIT 1"),
            {"url": url}
        )
        url_id = cur.fetchone()
        if url_id is None:
            return False

        cur = await conn.execute(
            sa.text("SELECT 1 FROM token_location WHERE url_id = :url_id LIMIT 1"),
            {"url_id": url_id[0]}
        )
        tok_row = cur.fetchone()
        return bool(tok_row)

    async def add_link_ref(self, conn: AsyncConnection, url_from: str, url_to: str, link_text):
        """Add a link between two pages"""
        pass

    @abstractmethod
    async def crawl(self, pages: list[str], *args, **kwargs):
        pass


__all__ = [
    "CrawlerBase",
]
