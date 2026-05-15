import logging
from abc import abstractmethod
from collections import Counter
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.engine import AsyncConnection

from sherloque.utils import ALLOWED_TABLE_FIELDS, preprocess_text

LOG = logging.getLogger(__name__)


class CrawlerBase:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def preprocess(self, text: str) -> list[str]:
        return await preprocess_text(text)

    async def _get_entry_id(
        self,
        conn: AsyncConnection,
        table: str,
        field: str,
        value: str,
    ) -> int:
        """Helper function for getting an entry id and adding it if it's not present"""
        if (
            table not in ALLOWED_TABLE_FIELDS
            or field not in ALLOWED_TABLE_FIELDS[table]
        ):
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

    async def add_to_index(
        self,
        conn: AsyncConnection,
        url: str,
        text: str,
        title: Optional[str] = None,
    ) -> int:
        """Index an individual page"""
        LOG.info(f"Indexing URL: {url}")
        cur = await conn.execute(
            sa.text("""
                    SELECT _id
                    FROM document
                    WHERE url = :url
                    """),
            {"url": url},
        )
        row = cur.fetchone()
        if row is not None:
            LOG.info(f"URL {url} already indexed. Skipping ...")
            return row[0]

        processed_toks = await self.preprocess(text)
        doc_len = len(processed_toks)
        cur = await conn.execute(
            sa.text("""
                    INSERT INTO document (full_text, url, title, len)
                    VALUES (:full_text, :url, :title, :len)
                    RETURNING _id
                    """),
            {"full_text": text, "url": url, "title": title, "len": doc_len},
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to insert document for URL: {url}")
        doc_id = row[0]

        for i, token in enumerate(processed_toks):
            token_id = await self._get_entry_id(conn, "token", "token", token)
            await conn.execute(
                sa.text("""
                        INSERT INTO token_position(doc_id, token_id, position)
                        VALUES (:doc_id, :token_id, :position)
                        """),
                {"doc_id": doc_id, "token_id": token_id, "position": i},
            )

        for token, tf in Counter(processed_toks).items():
            token_id = await self._get_entry_id(conn, "token", "token", token)
            await conn.execute(
                sa.text("""
                        INSERT INTO term_doc_stats(token_id, doc_id, tf)
                        VALUES (:token_id, :doc_id, :tf)
                        """),
                {"token_id": token_id, "doc_id": doc_id, "tf": tf},
            )
            await conn.execute(
                sa.text("""
                        UPDATE token
                        SET doc_freq = doc_freq + 1
                        WHERE _id = :token_id
                """),
                {"token_id": token_id},
            )
        LOG.info(f"Finished indexing URL: {url}")
        return doc_id

    async def add_link_ref(
        self, conn: AsyncConnection, url_from: str, url_to: str, link_text
    ):
        """Add a link between two pages"""
        pass

    @abstractmethod
    async def crawl(self, pages: list[str], *args, **kwargs):
        pass


__all__ = [
    "CrawlerBase",
]
