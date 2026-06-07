"""Compare three retrieval modes on a BEIR IR benchmark, side by side.

Modes evaluated:
  1. BM25-only      -> sherloque.retrieve.BM25Retriever (lexical, SQL BM25)
  2. Vector-only    -> sherloque.retrieve.VectorRetriever (dense, pgvector cosine)
  3. RRF fusion     -> reciprocal_rank_fusion([bm25, vector]) (hybrid)

Each query is run through the REAL production retrievers (not a self-contained
copy of the SQL) so the eval exercises the same code path the app uses. Results
are scored with ranx on the same metric set as eval_bm25.py:
  nDCG@10, Recall@100, MAP, MRR@10, P@10.

Sanity check (beir/scifact/test): tuned BM25 lands around nDCG@10 ~ 0.66. The
point of this script is to see whether dense vectors and especially the RRF
hybrid beat BM25 alone. Low overlap between BM25 and vector hits is expected and
fine -- what matters is whether the *fused* nDCG@10 exceeds both singles.

----------------------------------------------------------------------------
Design decisions (called out because they bit us before):

* doc_id <-> url mapping.
  The production retrievers return `doc_id` = the integer `document._id`, but
  ranx needs the BEIR doc id, which we store in `document.url`. So we load a
  one-time `_id -> url` map from the DB up front and translate every result's
  doc_id to its url when building the ranx run. (Chosen over joining url into
  every retrieval query so we don't have to fork the production SQL.)

* qwen3-embedding is ASYMMETRIC via INSTRUCTIONS, not prefixes. Documents were
  embedded raw (no prefix); queries must be wrapped as
  `Instruct: {task}\nQuery: {query}`. VectorRetriever.retrieve() now applies
  that wrapper internally, so we hand it the RAW query (same as BM25) -- no
  prefixing here. (The old nomic `search_query: ` prefix is gone.)

* Vector score direction. VectorRetriever now returns `score` = cosine
  *similarity* (`1 - (embedding <=> qvec)`, higher is better) and orders results
  best-first. ranx Run sorts by score *descending*, so the score is used as-is.
  RRF only uses rank order, so the raw retriever lists are fed to it unchanged.

* Embeddings must already exist. This script does NOT embed the corpus. Run the
  backfill once first (or pass --backfill here, which shells out to it):
      uv run python scratchpad/scripts/backfill_embeddings.py
  If any `document.embedding` is NULL the script warns and the vector/RRF
  numbers will be wrong.
----------------------------------------------------------------------------

Examples:
  # Corpus already ingested + embedded (the common case): just evaluate.
  uv run python scratchpad/scripts/eval_retrieval.py

  # From scratch: reset, ingest BM25 index, backfill embeddings, then evaluate.
  uv run python scratchpad/scripts/eval_retrieval.py --reset --ingest --backfill
"""

import argparse
import asyncio
import logging
import subprocess
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

# get_settings() reads src/.env (for FIREWORKS_API_KEY); load it here too so the
# embedding client is configured regardless of import order.
load_dotenv(SRC_DIR / ".env")

from sherloque.config import get_async_engine  # noqa: E402
from sherloque.crawler.base import CrawlerBase  # noqa: E402
from sherloque.retrieve import (  # noqa: E402
    BM25Retriever,
    BM25RetrieverConfig,
    VectorRetriever,
    VectorRetrieverConfig,
    reciprocal_rank_fusion,
)

LOG = logging.getLogger("eval_retrieval")

DEFAULT_DATASET = "beir/scifact/test"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_TOP_K = 100  # candidate pool depth fed to each retriever and to RRF
DEFAULT_RRF_K = 60
EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
METRICS = ["ndcg@10", "recall@100", "map", "mrr@10", "precision@10"]

BACKFILL_SCRIPT = Path(__file__).resolve().parent / "backfill_embeddings.py"


class BEIRIndexer(CrawlerBase):
    """Ingests a pre-fetched BEIR corpus through the production index pipeline."""

    async def crawl(self, pages, *args, **kwargs):
        raise NotImplementedError("BEIRIndexer ingests pre-fetched corpora.")


# --------------------------------------------------------------------------- #
# Ingest / setup (mirrors eval_bm25.py so the two stay schema-compatible)
# --------------------------------------------------------------------------- #
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


def _doc_body(doc) -> str:
    parts = []
    title = getattr(doc, "title", None)
    if title:
        parts.append(title)
    text_attr = getattr(doc, "text", None) or getattr(doc, "body", None)
    if text_attr:
        parts.append(text_attr)
    return "\n".join(parts) if parts else ""


async def ingest_corpus(
    engine: AsyncEngine, dataset_name: str, limit: int | None
) -> int:
    dataset = ir_datasets.load(dataset_name)
    indexer = BEIRIndexer(engine=engine)

    total = 0
    started = time.perf_counter()
    batch_size = 100

    async with engine.connect() as conn:
        for doc in dataset.docs_iter():
            await indexer.add_to_index(
                conn=conn,
                url=doc.doc_id,  # BEIR doc id -> document.url (our unique key)
                title=getattr(doc, "title", None) or None,
                text=_doc_body(doc),
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


def run_backfill() -> None:
    """Shell out to the resumable embedding backfill (only fills NULL rows)."""
    LOG.info("running embedding backfill: %s", BACKFILL_SCRIPT)
    subprocess.run(
        ["uv", "run", "python", str(BACKFILL_SCRIPT)],
        cwd=str(ROOT_DIR),
        check=True,
    )


async def count_null_embeddings(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT COUNT(*) FROM document WHERE embedding IS NULL")
            )
        ).scalar_one()


async def load_id_to_url(engine: AsyncEngine) -> dict[int, str]:
    """One-time `document._id -> document.url` map for translating retriever
    doc_ids (integers) into the BEIR doc ids that ranx/qrels are keyed by."""
    async with engine.connect() as conn:
        cur = await conn.execute(text("SELECT _id, url FROM document"))
        return {row._id: row.url for row in cur}


# --------------------------------------------------------------------------- #
# Retrieval: run all three modes per query
# --------------------------------------------------------------------------- #
async def run_all_modes(
    engine: AsyncEngine,
    dataset_name: str,
    id_to_url: dict[int, str],
    k1: float,
    b: float,
    top_k: int,
    rrf_k: int,
    concurrency: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {"bm25": run, "vector": run, "rrf": run} where each run is the
    ranx-style {query_id: {doc_url: score}}."""
    dataset = ir_datasets.load(dataset_name)
    queries = []
    for query in dataset.queries_iter():
        text_attr = getattr(query, "text", None) or getattr(query, "title", None)
        if text_attr:
            queries.append((query.query_id, text_attr))

    bm25 = BM25Retriever(engine, BM25RetrieverConfig(k1=k1, b=b, top_k=top_k))
    vector = VectorRetriever(
        engine, VectorRetrieverConfig(embedding_model=EMBEDDING_MODEL, top_k=top_k)
    )

    runs: dict[str, dict[str, dict[str, float]]] = {
        "bm25": {},
        "vector": {},
        "rrf": {},
    }
    sem = asyncio.Semaphore(concurrency)

    def to_url_run(results) -> dict[str, float]:
        """Map a retriever's results to {url: score}, dropping any doc_id we
        can't resolve to a url (shouldn't happen if ingest matches the corpus)."""
        run: dict[str, float] = {}
        for r in results:
            url = id_to_url.get(r.doc_id)
            if url is not None:
                run[url] = float(r.score)
        return run

    async def score_one(qid: str, qtext: str) -> None:
        async with sem:
            # Both get the raw query; VectorRetriever applies the qwen3
            # `Instruct: ...\nQuery: ...` wrapper internally.
            bm25_results = await bm25.retrieve(query=qtext, top_k=top_k)
            vector_results = await vector.retrieve(query=qtext, top_k=top_k)

        # RRF only uses rank order, so feed the raw retriever lists (both best-
        # first by score desc) straight in.
        fused = reciprocal_rank_fusion([bm25_results, vector_results], k=rrf_k)

        runs["bm25"][qid] = to_url_run(bm25_results)
        # Vector score is already cosine similarity (higher better); use as-is.
        runs["vector"][qid] = to_url_run(vector_results)
        runs["rrf"][qid] = to_url_run(fused)

    await asyncio.gather(*(score_one(qid, qtext) for qid, qtext in queries))
    return runs


def load_qrels(dataset_name: str) -> dict[str, dict[str, int]]:
    dataset = ir_datasets.load(dataset_name)
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance <= 0:
            continue
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    return qrels


def score_run(
    run_dict: dict[str, dict[str, float]],
    qrels_dict: dict[str, dict[str, int]],
) -> tuple[dict[str, float], int]:
    """Filter to queries present in both qrels and run (and non-empty), then
    score with ranx. Returns (scores, num_scored_queries). Mirrors the
    run_dict/qrels_dict filtering in eval_bm25.py."""
    # ranx only scores queries present in both qrels and run.
    run_dict = {q: run_dict.get(q, {}) for q in qrels_dict}
    # ranx rejects empty dicts; drop queries with no retrieved docs.
    run_dict = {q: d for q, d in run_dict.items() if d}
    local_qrels = {q: qrels_dict[q] for q in run_dict}

    if not run_dict:
        return {m: 0.0 for m in METRICS}, 0

    scores = evaluate(Qrels(local_qrels), Run(run_dict), METRICS)
    # ranx returns a bare float when a single metric is passed; normalize.
    if not isinstance(scores, dict):
        scores = {METRICS[0]: scores}
    return scores, len(run_dict)


def print_comparison(
    args: argparse.Namespace,
    results: dict[str, tuple[dict[str, float], int]],
) -> None:
    order = ["bm25", "vector", "rrf"]
    labels = {"bm25": "BM25", "vector": "Vector", "rrf": "RRF"}

    print()
    print(f"dataset:        {args.dataset}")
    print(f"BM25 params:    k1={args.k1} b={args.b}")
    print(f"vector model:   {EMBEDDING_MODEL} (qwen3 Instruct/Query wrapper)")
    print(f"pool top_k:     {args.top_k}   RRF k: {args.rrf_k}")
    print(
        "scored queries: "
        + "  ".join(f"{labels[m]}={results[m][1]}" for m in order)
    )
    print("=" * 58)

    header = f"{'metric':>14} | " + " | ".join(f"{labels[m]:>8}" for m in order)
    print(header)
    print("-" * len(header))
    for metric in METRICS:
        cells = " | ".join(f"{results[m][0][metric]:>8.4f}" for m in order)
        # Mark the winning mode for each metric.
        best = max(order, key=lambda m: results[m][0][metric])
        print(f"{metric:>14} | {cells}   <- {labels[best]}")
    print("=" * 58)


async def async_main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
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

        if args.backfill:
            # Backfill opens its own engine; run it before we hold connections.
            run_backfill()

        if args.skip_eval:
            return

        # Vector/RRF are meaningless if the corpus isn't embedded -- warn loudly.
        null_emb = await count_null_embeddings(engine)
        if null_emb:
            LOG.warning(
                "%d documents have NULL embedding; vector + RRF numbers will be "
                "wrong. Run the backfill (or pass --backfill).",
                null_emb,
            )

        id_to_url = await load_id_to_url(engine)
        LOG.info("loaded _id->url map for %d documents", len(id_to_url))

        LOG.info("running BM25 + vector + RRF over %s", args.dataset)
        runs = await run_all_modes(
            engine,
            args.dataset,
            id_to_url,
            args.k1,
            args.b,
            args.top_k,
            args.rrf_k,
            args.concurrency,
        )

        qrels_dict = load_qrels(args.dataset)
        LOG.info("loaded qrels for %d queries", len(qrels_dict))

        results = {mode: score_run(run, qrels_dict) for mode, run in runs.items()}
        print_comparison(args, results)
    finally:
        await engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BM25 / vector / RRF retrieval on a BEIR dataset.",
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
        help="ingest the dataset corpus (BM25 index) before evaluating",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="run backfill_embeddings.py to fill NULL document.embedding rows",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap documents ingested (dry runs)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="ingest/backfill only; do not run queries or score",
    )
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"RRF rank constant k (default: {DEFAULT_RRF_K})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "candidate pool depth per retriever, also fed to RRF "
            f"(default: {DEFAULT_TOP_K}; metrics are @10/@100)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="queries to run in parallel (each does 1 embedding API call)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
