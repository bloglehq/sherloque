import psycopg
from nltk import word_tokenize, WordNetLemmatizer

from config import manage_db_cursor


@manage_db_cursor(commit=True)
async def create_index_tables(cursor: psycopg.AsyncCursor):
    # Core tables with explicit IDs and constraints suitable for PostgreSQL
    await cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS url_list
        (
            _id BIGSERIAL PRIMARY KEY,
            url TEXT UNIQUE NOT NULL
        )
        """
    )
    await cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS token_list
        (
            _id   BIGSERIAL PRIMARY KEY,
            token TEXT UNIQUE NOT NULL
        )
        """
    )
    await cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS token_location
        (
            _id      BIGSERIAL PRIMARY KEY,
            url_id   BIGINT  NOT NULL REFERENCES url_list (_id) ON DELETE CASCADE,
            token_id BIGINT  NOT NULL REFERENCES token_list (_id) ON DELETE CASCADE,
            location INTEGER NOT NULL,
            UNIQUE (url_id, token_id, location)
        )
        """
    )
    await cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS link
        (
            _id     BIGSERIAL PRIMARY KEY,
            from_id BIGINT NOT NULL REFERENCES url_list (_id) ON DELETE CASCADE,
            to_id   BIGINT NOT NULL REFERENCES url_list (_id) ON DELETE CASCADE
        )
        """
    )
    await cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS link_tokens
        (
            _id      BIGSERIAL PRIMARY KEY,
            token_id BIGINT NOT NULL REFERENCES token_list (_id) ON DELETE CASCADE,
            link_id  BIGINT NOT NULL REFERENCES link (_id) ON DELETE CASCADE,
            UNIQUE (token_id, link_id)
        )
        """
    )

    await cursor.execute("CREATE INDEX IF NOT EXISTS token_idx ON token_list(token)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS url_idx ON url_list(url)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS token_url_idx ON token_location(token_id)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS url_to_idx ON link(to_id)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS url_from_idx ON link(from_id)")


ALLOWED_TABLE_FIELDS = {
    "url_list": ["url"],
    "token_list": ["token"],
    "token_location": ["url_id", "token_id", "location"],
    "link": ["from_id", "to_id"],
    "link_tokens": ["token_id", "link_id"],
}


async def preprocess_text(text: str) -> list[str]:
    """Tokenize and stem the text using NLTK"""
    toks = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok) for tok in toks]
