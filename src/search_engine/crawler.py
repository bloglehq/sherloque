import sqlite3
from urllib import request
from urllib.request import HTTPError

from bs4 import BeautifulSoup


class LIEnggBlogCrawler:

    def __init__(self, dbname):
        self.conn = sqlite3.connect(dbname)

    def __del__(self):
        self.conn.close()

    def db_commit(self):
        self.conn.commit()

    def get_entry_id(self, table, field, value, create_new=True):
        """Auxiliary function for getting an entry id and adding it if it's not present"""
        return None

    def add_to_index(self, url: str, soup: BeautifulSoup):
        """Index an individual page"""
        print("Indexing ", url)

    def get_text_only(self, url: str):
        """Extract the text from an HTML page (no tags)"""
        return None

    def separate_words(self, text):
        """Separate the words by any non-whitespace character"""
        return None

    def is_indexed(self, url):
        """Return True if this URL is already indexed"""
        return False

    def add_link_ref(self, url_from: str, url_to: str, link_text):
        """Add a link between two pages"""
        pass

    @staticmethod
    def _get_related_articles_urls(soup: BeautifulSoup) -> list[str]:
        """
        From a engineering blog, parse and return the blog links present in 'Related Articles'
        """
        links = []
        i = 0
        while True:
            try:
                o = soup.select_one(
                    f"#postList0FocusPoint > ul > li:nth-child({i + 1}) > div.list-post__content-container > div.list-post__content-container__title > a"
                )
                links.append(o.attrs["href"])
            except AttributeError:
                break
            i += 1
        return links

    def crawl(self, pages: list[str], depth: int = 2):
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
                self.add_to_index(page, soup)

                urls = self._get_related_articles_urls(soup)
                for url in urls:
                    new_pages.add(url)
                    link_text = self.get_text_only(url)
                    self.add_link_ref(page, url, link_text)

                self.db_commit()
            pages = new_pages

    def create_index_tables(self):
        """Create the database tables"""
        pass


__all__ = [
    "LIEnggBlogCrawler",
]
