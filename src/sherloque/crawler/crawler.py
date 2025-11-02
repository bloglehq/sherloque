import logging
from urllib import request

import sqlalchemy as sa
import nltk
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.engine import AsyncConnection

from sherloque import log_config as log_config
from sherloque.utils import ALLOWED_TABLE_FIELDS, preprocess_text

log_config.setup()

LOG = logging.getLogger(__name__)


class LIEnggBlogCrawler:
    def __init__(self, engine: AsyncEngine):
        nltk.download("punkt_tab")
        nltk.download("wordnet")

        self.engine = engine

    def __repr__(self):
        return "LinkedIn Engineering Blog crawler."

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

    async def add_to_index(self, conn: AsyncConnection, url: str, soup: BeautifulSoup):
        """Index an individual page"""
        LOG.info(f"Indexing URL: {url}")
        text = await self.get_text_only(soup)
        processed_toks = await preprocess_text(text)
        url_id = await self._get_entry_id(conn, "url_list", "url", url)

        for i in range(len(processed_toks)):
            token = processed_toks[i]
            # handle this better using nltk
            # if tok in ignore_words: continue
            token_id = await self._get_entry_id(conn, "token_list", "token", token)
            await conn.execute(
                sa.text("""
                        INSERT INTO token_location(url_id, token_id, location)
                        VALUES (:url_id, :token_id, :location)
                        """),
                {"url_id": url_id, "token_id": token_id, "location": i}
            )
        LOG.info(f"Finished indexing URL: {url}")

    async def get_text_only(self, soup: BeautifulSoup) -> str:
        """Extract the text from an HTML page (no tags)"""
        return soup.get_text().replace("\n\n", '')

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

    @staticmethod
    async def _get_related_articles_soups(soup: BeautifulSoup) -> list[BeautifulSoup]:
        """
        From a engineering blog, parse and return the blog links present in 'Related Articles'
        """
        links = []
        i = 0
        while True:
            o = soup.select_one(
                f"#postList0FocusPoint > ul > li:nth-child({i + 1}) > div.list-post__content-container > div.list-post__content-container__title > a"
            )
            if o is None:
                break
            links.append(o)
            i += 1
        return links

    async def crawl(self, pages: list[str], depth: int = 2):
        """Starting with a list of pages, do a breadth first search to the given depth, indexing pages as we go"""
        async with self.engine.connect() as conn:
            for i in range(depth):
                new_pages = set()
                LOG.info(f"Starting {i} level of crawling with {len(new_pages) if new_pages else len(pages)} new pages")
                for page in pages:
                    LOG.info(f"Starting crawling for page: {page}")
                    if await self.is_indexed(conn=conn, url=page):
                        LOG.info(f"URL {page} already indexed. Skipping ...")
                        continue
                    try:
                        c = request.urlopen(page)
                    except Exception as e:
                        LOG.error(f"Could not open page: {str(e)}")
                        continue
                    LOG.debug(f"Retrieved page: {page}")

                    soup = BeautifulSoup(c.read(), "html.parser")
                    await self.add_to_index(conn=conn, url=page, soup=soup)
                    links = await self._get_related_articles_soups(soup)
                    LOG.info(f"Retrieved {len(links)} related articles for page: {page}")
                    for link in links:
                        url = link.attrs["href"]
                        new_pages.add(url)
                        link_text = await self.get_text_only(link)
                        await self.add_link_ref(conn, page, url, link_text)

                    await conn.commit()
                LOG.info(f"Finished crawling page: {page}")
            pages = list(new_pages)


__all__ = [
    "LIEnggBlogCrawler",
]
