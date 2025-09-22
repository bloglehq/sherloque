import logging

import psycopg
from nltk import word_tokenize, WordNetLemmatizer

import log_config
from config import get_settings
from config import manage_db_cursor

log_config.setup()
settings = get_settings()

LOG = logging.getLogger(__name__)


class QueryEngine:

    async def preprocess_text(self, text: str) -> list[str]:
        """Tokenize and stem the text using NLTK"""
        toks = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(tok) for tok in toks]

    @manage_db_cursor()
    async def get_match_rows(self, cursor: psycopg.AsyncCursor, query: str):
        field_list = "t0.url_id"
        table_list = ""
        clause_list = ""
        token_ids = []

        tokens = await self.preprocess_text(query)
        LOG.debug(f"Tokens: {tokens}")
        table_num = 0

        for token in tokens:
            cur = await cursor.execute(
                "SELECT _id FROM token_list WHERE token = %s",
                (token,)
            )
            token_record = await cur.fetchone()
            if token_record is None:
                continue
            token_id = token_record[0]
            token_ids.append(token_id)
            if table_num > 0:
                table_list += ", "
                clause_list += " AND "
                clause_list += f"t{table_num - 1}.url_id = t{table_num}.url_id AND "
            field_list += f", t{table_num}.location"
            table_list += f"token_location t{table_num}"
            clause_list += f"t{table_num}.token_id = {token_id}"
            table_num += 1

        full_query = f"SELECT {field_list} FROM {table_list} WHERE {clause_list}" \
            if all((field_list, table_list, clause_list)) else "SELECT"
        LOG.debug(f"Constructed SQL query: {full_query}")
        LOG.debug(f"Token IDs: {token_ids}")
        cur = await cursor.execute(full_query)
        rows = [row async for row in cur]
        return rows

    @manage_db_cursor()
    async def _get_match_rows(self, cursor: psycopg.AsyncCursor, query: str):
        tokens = await self.preprocess_text(query)
        if not tokens:
            return []
        token_placeholders = ", ".join(["%s"] * len(tokens))
        cur = await cursor.execute(
            f"SELECT _id FROM token_list WHERE token IN ({token_placeholders})",  # noqa
            tokens
        )
        token_ids = [record[0] for record in await cur.fetchall()]
        if not token_ids:
            LOG.debug("None of the tokens were found in the database.")

        num_tokens = len(tokens)
        token_id_placeholders = ", ".join(["%s"] * num_tokens)
        cur = await cursor.execute(
            f""" 
            SELECT url_id FROM token_location
            WHERE token_id IN ({token_id_placeholders})
            GROUP BY url_id
            HAVING COUNT(DISTINCT token_id) = %s
            """,
            (token_ids + [num_tokens])
        )
        return [row async for row in cur]


__all__ = [
    "QueryEngine",
]
