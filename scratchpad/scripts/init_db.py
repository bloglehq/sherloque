import asyncio
import os

import psycopg
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio.engine import AsyncEngine

load_dotenv("../src/.env")


async def create_index_tables(engine: AsyncEngine):
    async with engine.connect() as conn:
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS url_list
                 (
                     _id BIGSERIAL PRIMARY KEY,
                     url TEXT UNIQUE NOT NULL
                 )
                 """),
        )
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS token_list
                 (
                     _id   BIGSERIAL PRIMARY KEY,
                     token TEXT UNIQUE NOT NULL
                 )
                 """),
        )
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS token_location
                 (
                     _id      BIGSERIAL PRIMARY KEY,
                     url_id   BIGINT  NOT NULL REFERENCES url_list (_id) ON DELETE CASCADE,
                     token_id BIGINT  NOT NULL REFERENCES token_list (_id) ON DELETE CASCADE,
                     location INTEGER NOT NULL,
                     UNIQUE (url_id, token_id, location)
                 )
                 """),
        )
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS link
                 (
                     _id     BIGSERIAL PRIMARY KEY,
                     from_id BIGINT NOT NULL REFERENCES url_list (_id) ON DELETE CASCADE,
                     to_id   BIGINT NOT NULL REFERENCES url_list (_id) ON DELETE CASCADE
                 )
                 """),
        )
        await conn.execute(
            text("""
                 CREATE TABLE IF NOT EXISTS link_tokens
                 (
                     _id      BIGSERIAL PRIMARY KEY,
                     token_id BIGINT NOT NULL REFERENCES token_list (_id) ON DELETE CASCADE,
                     link_id  BIGINT NOT NULL REFERENCES link (_id) ON DELETE CASCADE,
                     UNIQUE (token_id, link_id)
                 )
                 """),
        )

        await conn.execute(text("CREATE INDEX IF NOT EXISTS token_list_token_idx ON token_list(token)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS url_list_url_idx ON url_list(url)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS token_location_token_id_idx ON token_location(token_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS token_location_url_id_idx ON token_location(url_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS link_to_id_idx ON link(to_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS link_from_id_idx ON link(from_id)"))

        await conn.commit()
