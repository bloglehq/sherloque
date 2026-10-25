from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Connection,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Row,
    Table,
    Text,
    UniqueConstraint,
    bindparam,
    cast,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.ext.asyncio.engine import AsyncEngine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateSchema


class _SchemaVector(Vector):
    """pgvector's usual value conversion with a schema-qualified SQL type."""

    cache_ok = True

    def __init__(self, dim: int, schema: str):
        super().__init__(dim)
        self.schema = schema


@compiles(_SchemaVector, "postgresql")
def _compile_schema_vector(vector_type, compiler, **kwargs):
    schema = compiler.dialect.identifier_preparer.quote_identifier(vector_type.schema)
    return f"{schema}.vector({vector_type.dim})"


class PostgresIndex:
    """A Postgres search index using an application-owned async engine.

    Initialization creates missing objects; it does not migrate existing tables.
    The caller remains responsible for disposing the engine.
    """

    EMBEDDING_DIMENSIONS = 768

    BM25_SQL = (
        # @formatter:off
        """
        WITH q_tokens AS (
            SELECT t._id AS token_id, t.doc_freq
            FROM {token} t
            WHERE t.token = ANY(:tokens)
        ), stats AS (
            SELECT COUNT(*)::float AS total_docs, AVG(len) ::float AS avgdl
            FROM {document}
        )
        SELECT d.title AS doc_title, d._id as doc_id, SUM(
            LN(
                (stats.total_docs - q.doc_freq + 0.5) / (q.doc_freq + 0.5) + 1
            ) * (tds.tf * (:k1 + 1)) / (
                tds.tf + :k1 * (1 - :b + :b * d.len / stats.avgdl)
           )
        ) ::float AS score
        FROM {term_doc_stats} tds
        JOIN q_tokens q ON tds.token_id = q.token_id
        JOIN {document} d ON tds.doc_id = d._id
        CROSS JOIN stats
        GROUP BY d._id
        ORDER BY score DESC
        LIMIT :top_k
        """
        # @formatter:on
    )

    def __init__(self, engine: AsyncEngine, schema: str):
        if not schema or "\x00" in schema or len(schema.encode("utf-8")) > 63:
            raise ValueError("schema must contain 1–63 UTF-8 bytes and no null bytes")
        self.engine = engine
        self.schema = schema
        self._metadata = MetaData(schema=schema)

        self._document = Table(
            "document",
            self._metadata,
            Column("_id", BigInteger, primary_key=True),
            Column("full_text", Text, nullable=False),
            Column("url", Text),
            Column("title", Text),
            Column("len", Integer, nullable=False),
            Column("embedding", Vector(self.EMBEDDING_DIMENSIONS)),
            CheckConstraint("len >= 0"),
            CheckConstraint("url IS NOT NULL OR title IS NOT NULL"),
        )
        self._token = Table(
            "token",
            self._metadata,
            Column("_id", BigInteger, primary_key=True),
            Column("token", Text, nullable=False, unique=True),
            Column("doc_freq", BigInteger, nullable=False, server_default="0"),
            CheckConstraint("doc_freq >= 0"),
        )
        self._token_position = Table(
            "token_position",
            self._metadata,
            Column("_id", BigInteger, primary_key=True),
            Column(
                "doc_id",
                BigInteger,
                ForeignKey(self._document.c._id, ondelete="CASCADE"),
                nullable=False,
            ),
            Column(
                "token_id",
                BigInteger,
                ForeignKey(self._token.c._id, ondelete="CASCADE"),
                nullable=False,
            ),
            Column("position", Integer, nullable=False),
            CheckConstraint("position >= 0"),
            UniqueConstraint("doc_id", "token_id", "position"),
        )
        self._term_doc_stats = Table(
            "term_doc_stats",
            self._metadata,
            Column("_id", BigInteger, primary_key=True),
            Column(
                "token_id",
                BigInteger,
                ForeignKey(self._token.c._id, ondelete="CASCADE"),
                nullable=False,
            ),
            Column(
                "doc_id",
                BigInteger,
                ForeignKey(self._document.c._id, ondelete="CASCADE"),
                nullable=False,
            ),
            Column("tf", BigInteger, nullable=False),
            CheckConstraint("tf >= 0"),
            UniqueConstraint("token_id", "doc_id"),
        )
        self._link = Table(
            "link",
            self._metadata,
            Column("_id", BigInteger, primary_key=True),
            Column(
                "from_doc_id",
                BigInteger,
                ForeignKey(self._document.c._id, ondelete="CASCADE"),
                nullable=False,
            ),
            Column(
                "to_doc_id",
                BigInteger,
                ForeignKey(self._document.c._id, ondelete="CASCADE"),
                nullable=False,
            ),
            UniqueConstraint("from_doc_id", "to_doc_id"),
        )

        for table, columns in (
            (self._document, ("url", "title")),
            (self._token, ("token",)),
            (self._token_position, ("token_id", "doc_id")),
            (self._term_doc_stats, ("token_id", "doc_id")),
            (self._link, ("to_doc_id", "from_doc_id")),
        ):
            for column in columns:
                Index(f"{table.name}_{column}_idx", table.c[column])

        self._embedding_index = Index(
            "document_embedding_hnsw",
            self._document.c.embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )
        format_table = self.engine.dialect.identifier_preparer.format_table
        self._bm25_sql = text(
            self.BM25_SQL.format(**{
                # Escape colons for the text-bind parser after quoting identifiers.
                table.name: format_table(table).replace(":", r"\:")
                for table in (self._document, self._token, self._term_doc_stats)
            })
        )

    async def initialize(self) -> None:
        """Create the schema, pgvector extension, tables, and indexes atomically.

        The database role needs permission to create the schema and, if pgvector
        is not already installed, the extension. Existing pgvector installations
        are reused in their original schema.
        """
        async with self.engine.begin() as connection:
            await connection.execute(CreateSchema(self.schema, if_not_exists=True))
            await connection.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            )
            vector_schema = await self._get_vector_schema(connection)
            self._document.c.embedding.type = _SchemaVector(
                self.EMBEDDING_DIMENSIONS, vector_schema
            )
            quote = connection.dialect.identifier_preparer.quote_identifier
            self._embedding_index.dialect_options["postgresql"]["ops"] = {
                "embedding": f"{quote(vector_schema)}.vector_cosine_ops",
            }
            await connection.run_sync(self._create_tables_and_indexes)

    async def _get_vector_schema(self, connection: AsyncConnection) -> str:
        return (
            await connection.execute(
                text("""
                    SELECT n.nspname
                    FROM pg_catalog.pg_extension e
                    JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
                    WHERE e.extname = 'vector'
                """)
            )
        ).scalar_one()

    def _create_tables_and_indexes(self, connection: Connection) -> None:
        self._metadata.create_all(connection)
        # create_all skips indexes on tables that already exist.
        for table in self._metadata.tables.values():
            for index in table.indexes:
                index.create(connection, checkfirst=True)

    async def retrieve_bm25(self, tokens: list[str], k1: float, b: float, top_k: int) -> list[Row]:
        async with self.engine.connect() as conn:
            cur = await conn.execute(
                self._bm25_sql,
                {"tokens": tokens, "k1": k1, "b": b, "top_k": top_k},
            )
            rows = [record for record in cur]
        return rows

    async def retrieve_vector(self, query_embedding: list[float], top_k: int) -> list[Row]:
        """Return indexed documents with cosine distance, lowest first."""
        # Bind a standard Postgres array, then cast on the server. This works
        # whether or not the borrowed engine has custom pgvector codecs.
        async with self.engine.connect() as connection:
            vector_schema = await self._get_vector_schema(connection)
            query_vector = cast(
                bindparam("query_embedding", type_=ARRAY(Float)),
                _SchemaVector(self.EMBEDDING_DIMENSIONS, vector_schema),
            )
            quote = connection.dialect.identifier_preparer.quote_identifier
            distance = self._document.c.embedding.op(
                f"OPERATOR({quote(vector_schema)}.<=>)", return_type=Float
            )(query_vector)
            statement = (
                select(
                    self._document.c._id.label("doc_id"),
                    self._document.c.title.label("doc_title"),
                    distance.label("distance"),
                )
                .where(self._document.c.embedding.is_not(None))
                .order_by(distance)
                .limit(top_k)
            )
            result = await connection.execute(
                statement, {"query_embedding": query_embedding}
            )
            return list(result)


__all__ = [
    "PostgresIndex",
]
