import logging
from urllib import request

import nltk
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncEngine

from sherloque import log_config as log_config
from sherloque.crawler.base import CrawlerBase

log_config.setup()

LOG = logging.getLogger(__name__)


class LIEnggBlogCrawler(CrawlerBase):
    def __init__(self, engine: AsyncEngine):
        nltk.download("punkt_tab")
        nltk.download("wordnet")

        super().__init__(engine=engine)

    def __repr__(self):
        return "LinkedIn Engineering Blog crawler."

    async def get_text_only(self, soup: BeautifulSoup) -> str:
        """Extract the text from an HTML page (no tags)"""
        return soup.get_text().replace("\n\n", "")

    @staticmethod
    def get_title(soup: BeautifulSoup) -> str | None:
        if soup.title is None or soup.title.string is None:
            return None
        title = soup.title.string.strip()
        return title or None

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

    async def crawl(self, pages: list[str], depth: int = 2, *args, **kwargs):
        """Starting with a list of pages, do a breadth first search and index pages"""
        async with self.engine.connect() as conn:
            for i in range(depth):
                new_pages = set()
                LOG.info(
                    f"Starting {i} level of crawling with {len(new_pages) if new_pages else len(pages)} new pages"
                )
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
                    await self.add_to_index(conn=conn, url=page, text=text, title=title)
                    links = await self._get_related_articles_soups(soup)
                    LOG.info(
                        f"Retrieved {len(links)} related articles for page: {page}"
                    )
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
