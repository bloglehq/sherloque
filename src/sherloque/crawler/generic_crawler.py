import logging
from urllib import request

import nltk
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncEngine

from sherloque import log_config as log_config
from sherloque.crawler.base import CrawlerBase

log_config.setup()

LOG = logging.getLogger(__name__)


class GenericCrawler(CrawlerBase):
    def __init__(self, engine: AsyncEngine):
        nltk.download("punkt_tab")
        nltk.download("wordnet")

        super().__init__(engine=engine)

    async def get_text_only(self, soup: BeautifulSoup) -> str:
        """Extract the text from an HTML page (no tags)"""
        return soup.get_text().replace("\n\n", '')

    @staticmethod
    def get_title(soup: BeautifulSoup) -> str | None:
        if soup.title is None or soup.title.string is None:
            return None
        title = soup.title.string.strip()
        return title or None

    async def crawl(self, pages: list[str], *args, **kwargs):
        """Fetch each page, extract text, and index it"""
        async with self.engine.connect() as conn:
            for page in pages:
                LOG.info(f"Starting crawling for page: {page}")
                try:
                    c = request.urlopen(page)
                except Exception as e:
                    LOG.error(f"Could not open page: {str(e)}")
                    continue
                LOG.debug(f"Retrieved page: {page}")

                soup = BeautifulSoup(c.read(), "html.parser")
                text = await self.get_text_only(soup)
                title = self.get_title(soup)
                LOG.info(f"Text obtained for page: {page}")
                await self.add_to_index(conn=conn, url=page, text=text, title=title)
                await conn.commit()


__all__ = [
    "GenericCrawler",
]
