from urllib import request
from urllib.request import HTTPError

import aiosqlite
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize, PorterStemmer

from ..config import get_settings, manage_db_cursor
from ..utils import create_index_tables, ALLOWED_TABLE_FIELDS

settings = get_settings()


class LIEnggBlogCrawler:
    def __init__(self):
        nltk.download("punkt_tab")

    def __repr__(self):
        return f"LinkedIn Engineering Blog crawler."

    @classmethod
    async def create(cls):
        await create_index_tables(2)
        return cls()

    @manage_db_cursor()
    async def _get_entry_id(self, cursor: aiosqlite.Cursor, table: str, field: str, value: str) -> int:
        """Helper function for getting an entry id and adding it if it's not present"""
        if table not in ALLOWED_TABLE_FIELDS or \
                field not in ALLOWED_TABLE_FIELDS[table]:
            raise ValueError("Invalid table or field name provided")

        cur = await cursor.execute(
            f"SELECT ROWID from {table} WHERE {field} = ?",
            (value,),
        )

        row_id = await cur.fetchone()
        if row_id is not None:
            return row_id[0]

        cur = await cursor.execute(
            f"INSERT INTO {table} ({field}) VALUES (?)",
            (value,),
        )

        return cur.lastrowid

    @manage_db_cursor()
    async def add_to_index(self, cursor: aiosqlite.Cursor, url: str, soup: BeautifulSoup):
        """Index an individual page"""
        if await self.is_indexed(url): return
        print("Indexing ", url)

        text = await self.get_text_only(soup)
        processed_toks = await self.preprocess_text(text)
        url_id = await self._get_entry_id("url_list", "url", url)
        for i in range(len(processed_toks)):
            token = processed_toks[i]
            # handle this better using nltk
            # if tok in ignore_words: continue
            token_id = await self._get_entry_id("token_list", "token", token)
            await cursor.execute(
                "INSERT INTO token_location(url_id, token_id, location) VALUES (?, ?, ?)",
                (url_id, token_id, i)
            )

    async def get_text_only(self, soup: BeautifulSoup) -> str:
        """Extract the text from an HTML page (no tags)"""
        return soup.get_text().replace("\n\n", '')

    async def preprocess_text(self, text: str) -> list[str]:
        """Tokenize and stem the text using NLTK"""
        toks = word_tokenize(text)
        stemmer = PorterStemmer()
        return [stemmer.stem(tok) for tok in toks]

    @manage_db_cursor()
    async def is_indexed(self, cursor: aiosqlite.Cursor, url: str) -> bool:
        """Return True if this URL is already indexed"""
        cur = await cursor.execute(
            "SELECT ROWID FROM url_list WHERE url = ?",
            (url,),
        )
        url_rid = await cur.fetchone()
        if url_rid is None:
            return False

        cur = await cursor.execute(
            "SELECT * FROM token_location WHERE url_id = ?",
            (url_rid,),
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
            if o is None: break
            links.append(o)
            i += 1
        return links

    @manage_db_cursor()
    async def crawl(self, cursor: aiosqlite.Cursor, pages: list[str], depth: int = 2):
        """Starting with a list of pages, do a breadth first search to the given depth, indexing pages as we go"""
        for i in range(depth):
            new_pages = set()
            for page in pages:
                try:
                    c = request.urlopen(page)
                except HTTPError as e:
                    print("Could not open page: ", page)
                    print("Error: ", str(e))
                    continue

                soup = BeautifulSoup(c.read(), "html.parser")
                await self.add_to_index(page, soup)
                links = await self._get_related_articles_soups(soup)
                for link in links:
                    url = link.attrs["href"]
                    new_pages.add(url)
                    link_text = await self.get_text_only(link)
                    await self.add_link_ref(page, url, link_text)

                cursor.connection.commit()
            pages = new_pages


__all__ = [
    "LIEnggBlogCrawler",
]
