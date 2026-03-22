from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio.engine import AsyncEngine

# load_dotenv("../../src/sherloque/.env")
print(load_dotenv(Path(__file__).parent.parent.parent / "src" / "sherloque" / ".env"))


async def create_index_tables(engine: AsyncEngine):
    async with engine.connect() as conn:
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS document
                 (
                     _id       BIGSERIAL PRIMARY KEY,
                     full_text TEXT    NOT NULL,
                     url       TEXT,
                     title     TEXT,
                     len       INTEGER NOT NULL CHECK (len >= 0),
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

        await conn.commit()

if __name__ == "__main__":
    import asyncio
    from config import get_async_engine
    asyncio.run(create_index_tables(get_async_engine()))
