import aiosqlite
from .config import manage_db_cursor

@manage_db_cursor(commit=True)
async def create_index_tables(cursor: aiosqlite.Cursor, some_arg=2):
    await cursor.execute("CREATE TABLE IF NOT EXISTS url_list(url)")
    await cursor.execute("CREATE TABLE IF NOT EXISTS token_list(token)")
    await cursor.execute("CREATE TABLE IF NOT EXISTS token_location(url_id, token_id, location)")
    await cursor.execute("CREATE TABLE IF NOT EXISTS link(from_id INTEGER, to_id INTEGER)")
    await cursor.execute("CREATE TABLE IF NOT EXISTS link_tokens(token_id, link_id)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS token_idx on token_list(token)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS url_idx on url_list(url)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS token_url_idx on token_location(token_id)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS url_to_idx on link(to_id)")
    await cursor.execute("CREATE INDEX IF NOT EXISTS url_from_idx on link(from_id)")

ALLOWED_TABLE_FIELDS = {
    "url_list": ["url"],
    "token_list": ["token"],
    "token_location": ["url_id", "token_id", "location"],
    "link": ["from_id", "to_id"],
    "link_tokens": ["token_id", "link_id"],
}