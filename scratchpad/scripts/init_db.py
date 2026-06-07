"""Create the lexical index tables and optionally seed a tiny smoke corpus.

This script is for fast inner-loop sanity checks: does the schema exist, does
the indexer write something, do a handful of obvious queries rank the obvious
documents at the top? It is NOT a retrieval-quality benchmark.

For real evaluation (nDCG@10, MAP, Recall@100 against a standard IR
benchmark), use `eval_bm25.py` against BEIR/SciFact or similar.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import nltk
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio.engine import AsyncEngine

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# load_dotenv(SRC_DIR / ".env")
print(load_dotenv(Path(__file__).parent.parent.parent / "src" / "sherloque" / ".env"))

from sherloque.config import get_async_engine  # noqa: E402
from sherloque.crawler.base import CrawlerBase  # noqa: E402

SMOKE_DOCUMENTS = (
    {
        "url": "test://bm25/python-bm25-exact",
        "title": "python bm25 exact",
        "text": "python bm25 retrieval ranking search guide",
    },
    {
        "url": "test://bm25/python-retrieval-heavy",
        "title": "python retrieval heavy",
        "text": (
            "python retrieval retrieval retrieval retrieval bm25 search system "
            "latency throughput indexing evaluation monitoring operations"
        ),
    },
    {
        "url": "test://bm25/bm25-length-normalization",
        "title": "bm25 length normalization",
        "text": (
            "bm25 ranking formula idf document length normalization saturation"
        ),
    },
    {
        "url": "test://bm25/dense-retrieval",
        "title": "dense retrieval semantic search",
        "text": "dense retrieval vector embedding semantic search neural reranking",
    },
    {
        "url": "test://bm25/python-data-analysis",
        "title": "python data analysis",
        "text": "python data analysis dataframe notebook tutorial",
    },
    {
        "url": "test://bm25/italian-pasta",
        "title": "italian pasta recipe",
        "text": "italian pasta tomato basil olive oil recipe kitchen",
    },
    {
        "url": "test://bm25/marathon-running",
        "title": "marathon running plan",
        "text": "marathon running training weekly pacing endurance race plan",
    },
    {
        "url": "test://bm25/long-noisy-python-bm25",
        "title": "long noisy python bm25",
        "text": (
            "python bm25 retrieval observability pipeline service deployment "
            "incident capacity planning metrics alerting dashboard storage "
            "compute scheduler orchestration reliability backlog maintenance"
        ),
    },
)

SMOKE_QUERY_EXPECTATIONS = (
    {
        "query": "python bm25 retrieval",
        "expected_top_urls": (
            "test://bm25/python-bm25-exact",
            "test://bm25/python-retrieval-heavy",
        ),
        "reason": "all query terms match, with the shorter exact-match document expected to beat the longer tf-heavy variant",
    },
    {
        "query": "retrieval retrieval retrieval",
        "expected_top_urls": ("test://bm25/python-retrieval-heavy",),
        "reason": "the repeated retrieval term should make the tf-heavy document the obvious winner",
    },
    {
        "query": "dense retrieval semantic search",
        "expected_top_urls": ("test://bm25/dense-retrieval",),
        "reason": "this is the only document containing the dense and semantic terms together",
    },
    {
        "query": "italian pasta basil",
        "expected_top_urls": ("test://bm25/italian-pasta",),
        "reason": "the cooking document should cleanly separate from the search-related corpus",
    },
    {
        "query": "bm25 length normalization",
        "expected_top_urls": ("test://bm25/bm25-length-normalization",),
        "reason": "the normalization-focused document is the only one containing all three terms",
    },
)

NLTK_RESOURCES = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
)


class SeedIndexer(CrawlerBase):
    async def crawl(self, pages: list[str], *args, **kwargs):
        raise NotImplementedError("SeedIndexer does not support crawling.")


def ensure_nltk_resources() -> None:
    for resource_path, package_name in NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            downloaded = nltk.download(package_name, quiet=True)
            if not downloaded:
                raise RuntimeError(
                    f"Failed to download required NLTK package: {package_name}"
                )


async def create_index_tables(engine: AsyncEngine):
    async with engine.connect() as conn:
        # pgvector must exist before the document.embedding column references
        # the `vector` type.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS document
                 (
                     _id       BIGSERIAL PRIMARY KEY,
                     full_text TEXT    NOT NULL,
                     url       TEXT,
                     title     TEXT,
                     len       INTEGER NOT NULL CHECK (len >= 0),
                     embedding VECTOR(768),
                     CHECK (url IS NOT NULL OR title IS NOT NULL)
                 )
                 """),
        )
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS token
                 (
                     _id      BIGSERIAL PRIMARY KEY,
                     token    TEXT   UNIQUE NOT NULL,
                     doc_freq BIGINT NOT NULL DEFAULT 0 CHECK (doc_freq >= 0)
                 )
                 """),
        )
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS token_position
                 (
                     _id      BIGSERIAL PRIMARY KEY,
                     doc_id   BIGINT  NOT NULL REFERENCES document (_id) ON DELETE CASCADE,
                     token_id BIGINT  NOT NULL REFERENCES token (_id) ON DELETE CASCADE,
                     position INTEGER NOT NULL CHECK (position >= 0),
                     UNIQUE (doc_id, token_id, position)
                 )
                 """),
        )
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS term_doc_stats
                 (
                     _id      BIGSERIAL PRIMARY KEY,
                     token_id BIGINT NOT NULL REFERENCES token (_id) ON DELETE CASCADE,
                     doc_id   BIGINT NOT NULL REFERENCES document (_id) ON DELETE CASCADE,
                     tf       BIGINT NOT NULL CHECK (tf >= 0),
                     UNIQUE (token_id, doc_id)
                 )
                 """),
        )
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS link
                 (
                     _id         BIGSERIAL PRIMARY KEY,
                     from_doc_id BIGINT NOT NULL REFERENCES document (_id) ON DELETE CASCADE,
                     to_doc_id   BIGINT NOT NULL REFERENCES document (_id) ON DELETE CASCADE,
                     UNIQUE (from_doc_id, to_doc_id)
                 )
                 """),
        )

        await conn.execute(text("CREATE INDEX IF NOT EXISTS document_url_idx ON document(url)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS document_title_idx ON document(title)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS token_token_idx ON token(token)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS token_position_token_id_idx ON token_position(token_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS token_position_doc_id_idx ON token_position(doc_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS term_doc_stats_token_id_idx ON term_doc_stats(token_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS term_doc_stats_doc_id_idx ON term_doc_stats(doc_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS link_to_doc_id_idx ON link(to_doc_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS link_from_doc_id_idx ON link(from_doc_id)"))

        # HNSW ANN index for vector retrieval. Cosine opclass must match the
        # `<=>` operator the VectorRetriever queries with.
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS document_embedding_hnsw "
            "ON document USING hnsw (embedding vector_cosine_ops)"
        ))

        await conn.commit()


async def reset_index_tables(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(
            text("""
                 TRUNCATE TABLE
                   link,
                   token_position,
                   term_doc_stats,
                   token,
                   document
                 RESTART IDENTITY CASCADE
                 """),
        )
        await conn.commit()


async def seed_smoke_data(engine: AsyncEngine) -> tuple[int, int]:
    ensure_nltk_resources()
    indexer = SeedIndexer(engine=engine)
    urls = [doc["url"] for doc in SMOKE_DOCUMENTS]

    async with engine.connect() as conn:
        cur = await conn.execute(
            text("""
                 SELECT url
                 FROM document
                 WHERE url = ANY(:urls)
                 """),
            {"urls": urls},
        )
        existing_urls = {row[0] for row in cur}

        for doc in SMOKE_DOCUMENTS:
            await indexer.add_to_index(
                conn=conn,
                url=doc["url"],
                title=doc["title"],
                text=doc["text"],
            )
        await conn.commit()

    seeded_count = len(SMOKE_DOCUMENTS) - len(existing_urls)
    skipped_count = len(existing_urls)
    return seeded_count, skipped_count


def print_smoke_query_expectations() -> None:
    print("seeded bm25 smoke queries (sanity check only, not a quality benchmark)")
    print("note: keep queries lowercase for now because preprocessing is case-sensitive today")
    for query_spec in SMOKE_QUERY_EXPECTATIONS:
        expected = ", ".join(query_spec["expected_top_urls"])
        print(f"- query: {query_spec['query']}")
        print(f"  expected: {expected}")
        print(f"  reason: {query_spec['reason']}")


async def async_main(args: argparse.Namespace) -> None:
    engine = get_async_engine()
    await create_index_tables(engine)
    print("index tables are ready")

    if args.reset:
        await reset_index_tables(engine)
        print("index tables were reset")

    if args.seed_smoke_data:
        seeded_count, skipped_count = await seed_smoke_data(engine)
        print(
            f"seeded {seeded_count} smoke documents"
            f"{f' and skipped {skipped_count} existing documents' if skipped_count else ''}"
        )
        print_smoke_query_expectations()
    elif args.show_smoke_queries:
        print_smoke_query_expectations()

    await engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the lexical index tables and optionally seed a tiny smoke "
            "corpus. For retrieval-quality evaluation, use eval_bm25.py."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="truncate the lexical index tables before any seeding",
    )
    parser.add_argument(
        "--seed-smoke-data",
        action="store_true",
        help="insert the tiny smoke corpus used for sanity checks (NOT for quality evaluation)",
    )
    parser.add_argument(
        "--show-smoke-queries",
        action="store_true",
        help="print the smoke query expectations without inserting data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
