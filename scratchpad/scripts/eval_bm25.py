"""Evaluate the lexical BM25 index against a standard IR benchmark.

Loads a BEIR dataset via ir_datasets, ingests the corpus through the existing
CrawlerBase pipeline, runs each test query against a multi-term BM25 SQL query,
and scores the resulting runs with ranx (nDCG@10, Recall@100, MAP, MRR@10, P@10).

Doc identity in the database: BEIR doc ids are stored in `document.url` because
url is the existing unique key in the schema. This keeps the eval loop schema-
compatible with the production crawler path.

Default dataset is BEIR/SciFact (~5K docs, 300 queries, binary qrels). A tuned
BM25 should land around nDCG@10 ~ 0.66; use that as a sanity check.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import ir_datasets
from dotenv import load_dotenv
from ranx import Qrels, Run, evaluate
from sqlalchemy import text
from sqlalchemy.ext.asyncio.engine import AsyncEngine

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(SRC_DIR / "sherloque" / ".env")

from config import get_async_engine  # noqa: E402
from sherloque.crawler.base import CrawlerBase  # noqa: E402
from sherloque.utils import preprocess_text  # noqa: E402

LOG = logging.getLogger("eval_bm25")

DEFAULT_DATASET = "beir/scifact/test"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_TOP_K = 1000
METRICS = ["ndcg@10", "recall@100", "map", "mrr@10", "precision@10"]

BM25_SQL = text(
    """
    WITH q_tokens AS (
        SELECT t._id AS token_id, t.doc_freq
        FROM token t
        WHERE t.token = ANY(:tokens)
    ),
    stats AS (
        SELECT
            COUNT(*)::float AS total_docs,
            AVG(len)::float AS avgdl
        FROM document
    )
    SELECT
        d.url AS doc_id,
        SUM(
            LN((stats.total_docs - q.doc_freq + 0.5) / (q.doc_freq + 0.5) + 1)
            * (tds.tf * (:k1 + 1))
            / (
                tds.tf
                + :k1 * (1 - :b + :b * d.len / stats.avgdl)
            )
        )::float AS score
    FROM term_doc_stats tds
    JOIN q_tokens q ON tds.token_id = q.token_id
    JOIN document d ON tds.doc_id = d._id
    CROSS JOIN stats
    GROUP BY d.url
    ORDER BY score DESC
    LIMIT :top_k
    """
)


class BEIRIndexer(CrawlerBase):
    async def crawl(self, pages, *args, **kwargs):
        raise NotImplementedError("BEIRIndexer ingests pre-fetched corpora.")


async def reset_index(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(
            text(
                """
                TRUNCATE TABLE
                  link,
                  token_position,
                  term_doc_stats,
                  token,
                  document
                RESTART IDENTITY CASCADE
                """
            )
        )
        await conn.commit()


async def ingest_corpus(
    engine: AsyncEngine,
    dataset_name: str,
    limit: int | None,
) -> int:
    dataset = ir_datasets.load(dataset_name)
    indexer = BEIRIndexer(engine=engine)

    total = 0
    started = time.perf_counter()
    batch_size = 100

    async with engine.connect() as conn:
        for doc in dataset.docs_iter():
            body = _doc_body(doc)
            title = getattr(doc, "title", None) or None
            await indexer.add_to_index(
                conn=conn,
                url=doc.doc_id,
                title=title,
                text=body,
            )
            total += 1
            if total % batch_size == 0:
                await conn.commit()
                elapsed = time.perf_counter() - started
                LOG.info(
                    "ingested %d docs (%.1f docs/s)", total, total / max(elapsed, 1e-6)
                )
            if limit is not None and total >= limit:
                break
        await conn.commit()

    return total


def _doc_body(doc) -> str:
    parts = []
    title = getattr(doc, "title", None)
    if title:
        parts.append(title)
    text_attr = getattr(doc, "text", None) or getattr(doc, "body", None)
    if text_attr:
        parts.append(text_attr)
    return "\n".join(parts) if parts else ""


async def run_queries(
    engine: AsyncEngine,
    dataset_name: str,
    k1: float,
    b: float,
    top_k: int,
    concurrency: int,
) -> dict[str, dict[str, float]]:
    dataset = ir_datasets.load(dataset_name)
    queries = []
    for query in dataset.queries_iter():
        text_attr = getattr(query, "text", None) or getattr(query, "title", None)
        if text_attr:
            queries.append((query.query_id, text_attr))

    sem = asyncio.Semaphore(concurrency)

    async def score_one(qid: str, qtext: str) -> tuple[str, dict[str, float]]:
        tokens = await preprocess_text(qtext)
        if not tokens:
            return qid, {}
        async with sem, engine.connect() as conn:
            cur = await conn.execute(
                BM25_SQL,
                {"tokens": tokens, "k1": k1, "b": b, "top_k": top_k},
            )
            return qid, {row.doc_id: float(row.score) for row in cur}

    results = await asyncio.gather(*(score_one(qid, qtext) for qid, qtext in queries))
    return dict(results)


def load_qrels(dataset_name: str) -> dict[str, dict[str, int]]:
    dataset = ir_datasets.load(dataset_name)
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance <= 0:
            continue
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    return qrels


async def async_main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Silence the per-URL chatter from CrawlerBase during bulk ingest.
    logging.getLogger("sherloque.crawler.base").setLevel(logging.WARNING)

    engine = get_async_engine()
    try:
        if args.reset:
            LOG.info("truncating index tables")
            await reset_index(engine)

        if args.ingest:
            LOG.info("ingesting corpus from %s", args.dataset)
            total = await ingest_corpus(engine, args.dataset, args.limit)
            LOG.info("ingested %d documents total", total)

        if args.skip_eval:
            return

        LOG.info("running queries against %s", args.dataset)
        run_dict = await run_queries(
            engine, args.dataset, args.k1, args.b, args.top_k, args.concurrency
        )
        LOG.info("ran %d queries", len(run_dict))

        qrels_dict = load_qrels(args.dataset)
        LOG.info("loaded qrels for %d queries", len(qrels_dict))

        # ranx only scores queries present in both qrels and run.
        run_dict = {q: run_dict.get(q, {}) for q in qrels_dict}
        # ranx rejects empty dicts; drop queries with no retrieved docs.
        run_dict = {q: d for q, d in run_dict.items() if d}
        qrels_dict = {q: qrels_dict[q] for q in run_dict}

        if not run_dict:
            LOG.error("no queries produced any results; check ingest and tokenizer")
            return

        qrels = Qrels(qrels_dict)
        run = Run(run_dict)
        scores = evaluate(qrels, run, METRICS)

        print()
        print(f"dataset: {args.dataset}")
        print(f"params: k1={args.k1} b={args.b} top_k={args.top_k}")
        print(f"scored queries: {len(run_dict)}")
        print("-" * 40)
        for metric in METRICS:
            print(f"{metric:>14}: {scores[metric]:.4f}")
    finally:
        await engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 against a BEIR dataset using ranx."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"ir_datasets id (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="truncate index tables before ingesting",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="ingest the dataset corpus before evaluating",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of documents ingested (useful for dry runs)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="ingest only; do not run queries or compute metrics",
    )
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="number of queries to run in parallel against the DB",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
