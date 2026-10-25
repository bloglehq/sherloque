import math
import unittest
from uuid import uuid4

from sqlalchemy import event, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import DropSchema

from sherloque.config import get_settings
from sherloque.index import PostgresIndex


class PostgresIndexValidationTests(unittest.TestCase):
    def test_rejects_schema_names_postgres_cannot_preserve(self) -> None:
        for schema in ("", "bad\x00name", "a" * 64, "é" * 32):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                PostgresIndex(None, schema)


class PostgresIndexIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        try:
            database_url = get_settings().engine_async_str
        except ValueError:
            self.skipTest("test database is not configured")
        self.engine = create_async_engine(database_url)
        self.schemas = [f"sherloque_index_test_{uuid4().hex}"]
        self.index = PostgresIndex(self.engine, self.schemas[0])

    async def asyncTearDown(self) -> None:
        if not hasattr(self, "engine"):
            return
        try:
            async with self.engine.begin() as connection:
                for schema in self.schemas:
                    await connection.execute(
                        DropSchema(schema, cascade=True, if_exists=True)
                    )
        finally:
            await self.engine.dispose()

    async def test_creates_lexical_and_vector_structures(self) -> None:
        await self.index.initialize()

        async with self.engine.connect() as connection:
            tables = await connection.run_sync(
                lambda conn: inspect(conn).get_table_names(schema=self.index.schema)
            )
            self.assertEqual(
                set(tables),
                {"document", "token", "token_position", "term_doc_stats", "link"},
            )
            embedding = (
                await connection.execute(
                    text("""
                        SELECT t.typname, a.atttypmod AS dimensions
                        FROM pg_attribute a
                        JOIN pg_class c ON c.oid = a.attrelid
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        JOIN pg_type t ON t.oid = a.atttypid
                        WHERE n.nspname = :schema
                          AND c.relname = 'document'
                          AND a.attname = 'embedding'
                    """),
                    {"schema": self.index.schema},
                )
            ).one()
            self.assertEqual(embedding.typname, "vector")
            self.assertEqual(embedding.dimensions, 768)
            indexes = (
                await connection.execute(
                    text("""
                        SELECT indexname, indexdef
                        FROM pg_indexes
                        WHERE schemaname = :schema
                    """),
                    {"schema": self.index.schema},
                )
            ).all()
            definitions = {row.indexname: row.indexdef for row in indexes}
            self.assertIn("USING hnsw", definitions["document_embedding_hnsw"])
            self.assertIn("vector_cosine_ops", definitions["document_embedding_hnsw"])
            self.assertIn("term_doc_stats_token_id_idx", definitions)
            self.assertIn("token_position_doc_id_idx", definitions)

    async def test_bm25_uses_the_index_schema(self) -> None:
        await self.index.initialize()

        # The engine's search_path does not contain this index's schema.
        self.assertEqual(await self.index.retrieve_bm25(["word"], 1.5, 0.75, 10), [])

    async def test_retrieval_is_isolated_from_the_connection_search_path(self) -> None:
        # Quotes and a bind-like colon must remain literal identifier characters.
        schema = f':literal "{uuid4().hex}'
        self.schemas.append(schema)
        other = PostgresIndex(self.engine, schema)
        query_vector = [1.0] + [0.0] * (self.index.EMBEDDING_DIMENSIONS - 1)

        for index, title in ((self.index, "First index"), (other, "Second index")):
            await index.initialize()
            async with self.engine.begin() as connection:
                doc_id = (
                    await connection.execute(
                        index._document.insert().values(
                            full_text="word", title=title, len=1, embedding=query_vector
                        ).returning(index._document.c._id)
                    )
                ).scalar_one()
                token_id = (
                    await connection.execute(
                        index._token.insert().values(token="word", doc_freq=1)
                        .returning(index._token.c._id)
                    )
                ).scalar_one()
                await connection.execute(
                    index._term_doc_stats.insert().values(
                        doc_id=doc_id, token_id=token_id, tf=1
                    )
                )

        async with self.engine.begin() as connection:
            quote = connection.dialect.identifier_preparer.quote_identifier
            search_path = f"{quote(other.schema)}, public"
            await connection.execute(
                text("SELECT set_config('search_path', :path, false)"),
                {"path": search_path},
            )

        for index, title in ((self.index, "First index"), (other, "Second index")):
            lexical = await index.retrieve_bm25(["word"], 1.5, 0.75, 10)
            dense = await index.retrieve_vector(query_vector, 10)
            self.assertEqual([row.doc_title for row in lexical], [title])
            self.assertAlmostEqual(lexical[0].score, math.log(4 / 3))
            self.assertEqual([row.doc_title for row in dense], [title])
            self.assertAlmostEqual(dense[0].distance, 0.0)
            self.assertEqual(await index.retrieve_bm25([], 1.5, 0.75, 10), [])
            self.assertEqual(await index.retrieve_vector(query_vector, 0), [])

        async with self.engine.connect() as connection:
            self.assertEqual(
                (await connection.execute(text("SHOW search_path"))).scalar_one(),
                search_path,
            )

    async def test_reinitializing_preserves_data_and_restores_missing_indexes(self) -> None:
        await self.index.initialize()
        async with self.engine.begin() as connection:
            await connection.execute(
                self.index._document.insert().values(
                    full_text="Existing document", title="Existing", len=2
                )
            )
            await connection.run_sync(
                lambda conn: next(
                    index for index in self.index._document.indexes
                    if index.name == "document_title_idx"
                ).drop(conn)
            )

        await self.index.initialize()

        async with self.engine.connect() as connection:
            self.assertEqual(
                (await connection.execute(self.index._document.select())).one().full_text,
                "Existing document",
            )
            indexes = await connection.run_sync(
                lambda conn: inspect(conn).get_indexes(
                    "document", schema=self.index.schema
                )
            )
            self.assertIn("document_title_idx", {index["name"] for index in indexes})

    async def test_foreign_keys_and_constraints_are_enforced(self) -> None:
        await self.index.initialize()
        async with self.engine.begin() as connection:
            doc_id = (
                await connection.execute(
                    self.index._document.insert().values(
                        full_text="Word", title="Title", len=1
                    ).returning(self.index._document.c._id)
                )
            ).scalar_one()
            token_id = (
                await connection.execute(
                    self.index._token.insert().values(token="word", doc_freq=1)
                    .returning(self.index._token.c._id)
                )
            ).scalar_one()
            await connection.execute(
                self.index._term_doc_stats.insert().values(
                    doc_id=doc_id, token_id=token_id, tf=1
                )
            )
            await connection.execute(
                self.index._token_position.insert().values(
                    doc_id=doc_id, token_id=token_id, position=0
                )
            )
            await connection.execute(
                self.index._document.delete().where(self.index._document.c._id == doc_id)
            )
            self.assertEqual(
                (await connection.execute(self.index._term_doc_stats.select())).all(), []
            )
            self.assertEqual(
                (await connection.execute(self.index._token_position.select())).all(), []
            )

        with self.assertRaises(IntegrityError):
            async with self.engine.begin() as connection:
                await connection.execute(
                    self.index._token.insert().values(token="invalid", doc_freq=-1)
                )

    async def test_indexes_in_different_schemas_are_isolated(self) -> None:
        self.schemas.append(f"sherloque_index_test_{uuid4().hex}")
        other = PostgresIndex(self.engine, self.schemas[-1])
        await self.index.initialize()
        await other.initialize()

        async with self.engine.begin() as connection:
            await connection.execute(
                self.index._document.insert().values(full_text="Only here", title="One", len=2)
            )
            self.assertEqual((await connection.execute(other._document.select())).all(), [])

    async def test_quoted_schema_names_are_literal_identifiers(self) -> None:
        schema = f'index"; -- {uuid4().hex}'
        self.schemas.append(schema)
        index = PostgresIndex(self.engine, schema)

        await index.initialize()

        async with self.engine.connect() as connection:
            tables = await connection.run_sync(
                lambda conn: inspect(conn).get_table_names(schema=schema)
            )
        self.assertIn("document", tables)

    async def test_search_path_and_borrowed_engine_are_preserved(self) -> None:
        async with self.engine.connect() as connection:
            before = (await connection.execute(text("SHOW search_path"))).scalar_one()
        original_pool = self.engine.pool

        await self.index.initialize()

        async with self.engine.connect() as connection:
            after = (await connection.execute(text("SHOW search_path"))).scalar_one()
            self.assertEqual((await connection.execute(text("SELECT 1"))).scalar_one(), 1)
        self.assertEqual(before, after)
        self.assertIs(self.engine.pool, original_pool)

    async def test_vector_operations_never_set_search_path(self) -> None:
        statements = []

        def capture(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement.lower())

        event.listen(self.engine.sync_engine, "before_cursor_execute", capture)
        try:
            await self.index.initialize()
            vector = [1.0] + [0.0] * (self.index.EMBEDDING_DIMENSIONS - 1)
            self.assertEqual(await self.index.retrieve_vector(vector, 10), [])
        finally:
            event.remove(self.engine.sync_engine, "before_cursor_execute", capture)

        for statement in statements:
            self.assertNotIn("set_config", statement)
            self.assertNotRegex(statement, r"\bset\s+(?:(?:local|session)\s+)?search_path\b")

    async def test_qualified_vector_operator_can_use_hnsw_index(self) -> None:
        await self.index.initialize()
        queries = []

        def capture(conn, cursor, statement, parameters, context, executemany):
            if "OPERATOR(" in statement:
                queries.append((statement, parameters))

        event.listen(self.engine.sync_engine, "before_cursor_execute", capture)
        try:
            vector = [1.0] + [0.0] * (self.index.EMBEDDING_DIMENSIONS - 1)
            await self.index.retrieve_vector(vector, 10)
        finally:
            event.remove(self.engine.sync_engine, "before_cursor_execute", capture)

        statement, parameters = queries[0]
        async with self.engine.begin() as connection:
            await connection.execute(text("SET LOCAL enable_seqscan = off"))
            plan = (
                await connection.exec_driver_sql(
                    "EXPLAIN (FORMAT JSON) " + statement, parameters
                )
            ).scalar_one()[0]["Plan"]

        nodes = [plan]
        while nodes:
            node = nodes.pop()
            if node.get("Index Name") == "document_embedding_hnsw":
                break
            nodes.extend(node.get("Plans", []))
        else:
            self.fail(f"Qualified vector query could not use the HNSW index: {plan}")

    async def test_failed_initialization_rolls_back_schema_and_tables(self) -> None:
        def fail_after_document(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().startswith("CREATE TABLE") and ".token (" in statement:
                raise RuntimeError("Simulated DDL failure")

        event.listen(self.engine.sync_engine, "before_cursor_execute", fail_after_document)
        try:
            with self.assertRaisesRegex(RuntimeError, "Simulated DDL failure"):
                await self.index.initialize()
        finally:
            event.remove(self.engine.sync_engine, "before_cursor_execute", fail_after_document)

        async with self.engine.connect() as connection:
            schemas = await connection.run_sync(lambda conn: inspect(conn).get_schema_names())
        self.assertNotIn(self.index.schema, schemas)
        await self.index.initialize()
