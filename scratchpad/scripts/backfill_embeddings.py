"""Backfill document.embedding with Fireworks qwen3-embedding-8b vectors.

One-off, resumable: only touches rows where embedding IS NULL, so it can be
re-run after an interruption. Embeds in batches against the Fireworks
embeddings endpoint (OpenAI-compatible) and writes back via UPDATE.

qwen3-embedding is asymmetric via INSTRUCTIONS, not prefixes:
  - DOCUMENTS (this script): embed the raw text with NO instruction/prefix.
  - QUERIES (the retriever): wrap with `Instruct: {task}\\nQuery: {query}`.
The native dim is 4096; we request `dimensions=768` (Matryoshka/MRL truncation)
to match the vector(768) column. Vectors are L2-normalized before writing
(qwen3 returns un-normalized vectors, norm ~60), so cosine/dot behave cleanly.

Run:
    uv run python scratchpad/scripts/backfill_embeddings.py
    uv run python scratchpad/scripts/backfill_embeddings.py --batch-size 128
"""

import argparse
import asyncio
import logging
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(SRC_DIR / ".env")

from sherloque.config import get_async_engine  # noqa: E402

LOG = logging.getLogger("backfill_embeddings")

MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
BASE_URL = "https://api.fireworks.ai/inference/v1"
EMBED_DIM = 768  # MRL truncation target; matches the vector(768) DB column.

SELECT_PENDING = text(
    "SELECT _id, full_text FROM document "
    "WHERE embedding IS NULL ORDER BY _id LIMIT :limit"
)
# get_async_engine() registers the pgvector type on every asyncpg connection,
# so we bind the embedding as a plain Python list[float] (no ::vector string
# cast -- a string would now fail asyncpg's vector codec).
UPDATE_EMBEDDING = text(
    "UPDATE document SET embedding = :vec WHERE _id = :id"
)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0.0 else [v / norm for v in vec]


async def embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    # DOCUMENTS get NO instruction/prefix (qwen3 asymmetric-via-instruction).
    # Retry with exponential backoff on Fireworks rate limits (429).
    delay = 2.0
    for attempt in range(8):
        try:
            return await _embed_once(client, texts)
        except RateLimitError:
            if attempt == 7:
                raise
            LOG.warning("rate limited; backing off %.1fs", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


async def _embed_once(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    resp = await client.embeddings.create(
        model=MODEL,
        input=texts,
        dimensions=EMBED_DIM,
    )
    # The API preserves input order, but sort by index to be safe.
    data = sorted(resp.data, key=lambda d: d.index)
    # L2-normalize each (truncated) vector before writing.
    return [_l2_normalize(d.embedding) for d in data]


async def async_main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        LOG.error("FIREWORKS_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    engine = get_async_engine()

    total_pending = 0
    started = time.perf_counter()
    try:
        async with engine.connect() as conn:
            remaining = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM document WHERE embedding IS NULL")
                )
            ).scalar_one()
        LOG.info("%d documents pending embedding", remaining)

        while True:
            async with engine.connect() as conn:
                rows = list(
                    await conn.execute(SELECT_PENDING, {"limit": args.batch_size})
                )
            if not rows:
                break

            ids = [r._id for r in rows]
            texts = [r.full_text or "" for r in rows]
            vectors = await embed_batch(client, texts)

            async with engine.connect() as conn:
                for doc_id, vec in zip(ids, vectors):
                    await conn.execute(
                        UPDATE_EMBEDDING,
                        {"id": doc_id, "vec": vec},
                    )
                await conn.commit()

            total_pending += len(rows)
            elapsed = time.perf_counter() - started
            LOG.info(
                "embedded %d/%d (%.1f docs/s)",
                total_pending,
                remaining,
                total_pending / max(elapsed, 1e-6),
            )
    finally:
        await client.close()
        await engine.dispose()

    LOG.info("done; embedded %d documents", total_pending)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
