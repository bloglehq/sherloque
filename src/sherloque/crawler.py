import logging
from urllib import request

import nltk
import psycopg
from bs4 import BeautifulSoup
from nltk import word_tokenize, WordNetLemmatizer
from psycopg import sql

import log_config as log_config
from config import get_settings, manage_db_cursor
from utils import create_index_tables, ALLOWED_TABLE_FIELDS

log_config.setup()
settings = get_settings()

LOG = logging.getLogger(__name__)


class LIEnggBlogCrawler:
    def __init__(self):
        nltk.download("punkt_tab")
        nltk.download("wordnet")

    def __repr__(self):
        return "LinkedIn Engineering Blog crawler."

    @classmethod
    async def create(cls):
        await create_index_tables()
        return cls()

    @manage_db_cursor()
    async def _get_entry_id(self, cursor: psycopg.AsyncCursor, table: str, field: str, value: str) -> int:
        """Helper function for getting an entry id and adding it if it's not present"""
        if table not in ALLOWED_TABLE_FIELDS or \
                field not in ALLOWED_TABLE_FIELDS[table]:
            raise ValueError("Invalid table or field name provided")

        cur = await cursor.execute(
            sql.SQL("SELECT _id FROM {} WHERE {} = %s").format(
                sql.Identifier(table), sql.Identifier(field)
            ),
            (value,),
        )

        row = await cur.fetchone()
        if row is not None:
            return row[0]

        cur = await cursor.execute(
            sql.SQL("INSERT INTO {} ({}) VALUES (%s) RETURNING _id").format(
                sql.Identifier(table), sql.Identifier(field)
            ),
            (value,),
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("INSERT did not return an id")
        return row[0]

    @manage_db_cursor()
    async def add_to_index(self, cursor: psycopg.AsyncCursor, url: str, soup: BeautifulSoup):
        """Index an individual page"""
        LOG.info(f"Indexing URL: {url}")
        text = await self.get_text_only(soup)
        processed_toks = await self.preprocess_text(text)
        url_id = await self._get_entry_id("url_list", "url", url)

        for i in range(len(processed_toks)):
            token = processed_toks[i]
            # handle this better using nltk
            # if tok in ignore_words: continue
            token_id = await self._get_entry_id("token_list", "token", token)
            await cursor.execute(
                "INSERT INTO token_location(url_id, token_id, location) VALUES (%s, %s, %s)",
                (url_id, token_id, i)
            )
        LOG.info(f"Finished indexing URL: {url}")

    async def get_text_only(self, soup: BeautifulSoup) -> str:
        """Extract the text from an HTML page (no tags)"""
        return soup.get_text().replace("\n\n", '')

    async def preprocess_text(self, text: str) -> list[str]:
        """Tokenize and stem the text using NLTK"""
        toks = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(tok) for tok in toks]

    @manage_db_cursor()
    async def is_indexed(self, cursor: psycopg.AsyncCursor, url: str) -> bool:
        """Return True if this URL is already indexed"""
        cur = await cursor.execute(
            "SELECT _id FROM url_list WHERE url = %s",
            (url,),
        )
        row = await cur.fetchone()
        if row is None:
            return False

        url_id = row[0]
        cur = await cursor.execute(
            "SELECT 1 FROM token_location WHERE url_id = %s LIMIT 1",
            (url_id,),
        )

        tok_row = await cur.fetchone()
        if tok_row is not None:
            return True
        else:
            return False

    async def add_link_ref(self, url_from: str, url_to: str, link_text):
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

    @manage_db_cursor(commit=True)
    async def crawl(self, cursor: psycopg.AsyncCursor, pages: list[str], depth: int = 2):
        """Starting with a list of pages, do a breadth first search to the given depth, indexing pages as we go"""
        for i in range(depth):
            new_pages = set()
            LOG.info(f"Starting {i} level of crawling with {len(new_pages) if new_pages else len(pages)} new pages")
            for page in pages:
                LOG.info(f"Starting crawling for page: {page}")
                if await self.is_indexed(page):
                    LOG.info(f"URL {page} already indexed. Skipping ...")
                    continue
                try:
                    c = request.urlopen(page)
                except Exception as e:
                    LOG.error(f"Could not open page: {str(e)}")
                    continue
                LOG.debug(f"Retrieved page: {page}")

                soup = BeautifulSoup(c.read(), "html.parser")
                await self.add_to_index(page, soup)
                links = await self._get_related_articles_soups(soup)
                LOG.info(f"Retrieved {len(links)} related articles for page: {page}")
                for link in links:
                    url = link.attrs["href"]
                    new_pages.add(url)
                    link_text = await self.get_text_only(link)
                    await self.add_link_ref(page, url, link_text)

                await cursor.connection.commit()
                LOG.info(f"Finished crawling page: {page}")
            pages = list(new_pages)


__all__ = [
    "LIEnggBlogCrawler",
]
