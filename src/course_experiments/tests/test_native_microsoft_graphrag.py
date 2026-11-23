from __future__ import annotations

import json
import unittest
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from course_experiments.evaluation.native_microsoft_graphrag import (
    AuditedGraphChat, AuditedGraphEmbedding, validate_pipeline,
    STANDARD_WORKFLOWS, bounded_sources, validate_source_provenance, NativeMicrosoftGraphRAG,
    NativeFailureCallbacks, NATIVE_SETTINGS, GraphSourceDocument,
)


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.content = "synthetic answer"

    async def achat(self, messages, **kwargs):
        return self.chat(messages, **kwargs)

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.content

    async def aembed(self, texts, **kwargs):
        return self.embed(texts, **kwargs)

    def embed(self, texts, **kwargs):
        return [[1.0] * 1024 for _ in texts]


class GraphRAGAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.errors = []
        self.config = SimpleNamespace(max_tokens=8192)
        self.chat = AuditedGraphChat(self.provider, self.config, "synthetic-stage", self.errors)

    async def test_complete_history_and_model_limits_forwarded(self):
        history = [{"role": "system", "content": "native prompt"}]
        first = await self.chat.achat("question", history=history, model_parameters={"max_tokens": 500})
        self.assertEqual(first.output.content, "synthetic answer")
        second = await self.chat.achat("gleaning", history=first.history)
        self.assertEqual(len(second.history), 5)
        self.assertEqual(self.provider.calls[0][1]["max_tokens"], 500)
        self.assertEqual(self.provider.calls[0][0][0], history[0])
        self.assertEqual(len(history), 1)

    async def test_local_and_global_freeze_supported_single_paragraph_type(self):
        seen = []
        async def query(**kwargs):
            seen.append(kwargs["response_type"])
            return "synthetic answer", {}
        adapter = NativeMicrosoftGraphRAG()
        adapter._check = lambda book: None
        adapter._tables = lambda: {key: None for key in ("entities", "communities", "community_reports", "text_units", "relationships")}
        adapter._models = nullcontext
        adapter.config = SimpleNamespace()
        adapter.community_level = 0
        adapter.api = SimpleNamespace(local_search=query, global_search=query)
        await adapter.query_native("synthetic", "synthetic question", method="local")
        await adapter.query_native("synthetic", "synthetic question", method="global")
        self.assertEqual(seen, ["Single Paragraph", "Single Paragraph"])

    async def test_json_schema_response_and_streaming_use_audited_provider(self):
        class Model:
            @classmethod
            def model_validate_json(cls, content):
                return json.loads(content)
        self.provider.content = '{"title":"synthetic"}'
        response = await self.chat.achat("report", json=True, json_model=Model)
        self.assertEqual(response.parsed_response["title"], "synthetic")
        self.assertEqual(self.provider.calls[0][1]["response_format"], {"type": "json_object"})
        chunks = [s async for s in self.chat.achat_stream("query")]
        self.assertEqual(chunks, [self.provider.content])

    async def test_budget_failure_survives_native_swallowing(self):
        async def failure(*args, **kwargs):
            raise RuntimeError("budget exhausted")
        self.provider.achat = failure
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            await self.chat.achat("synthetic")
        self.assertEqual(len(self.errors), 1)

    async def test_sticky_extraction_error_blocks_summary_and_embeddings_before_provider(self):
        self.errors.append(ValueError("synthetic truncated extraction"))
        model = AuditedGraphChat(self.provider, self.config, "summary", self.errors)
        with self.assertRaisesRegex(RuntimeError, "prior"):
            await model.achat("synthetic report")
        with self.assertRaisesRegex(RuntimeError, "prior"):
            model.chat("synthetic report")
        self.assertEqual(self.provider.calls, [])
        embedding = AuditedGraphEmbedding(self.provider, self.config, "embedding", self.errors)
        self.provider.embed = lambda *args, **kwargs: self.fail("provider must not be called after extraction error")
        with self.assertRaisesRegex(RuntimeError, "prior"):
            await embedding.aembed("synthetic")
        with self.assertRaisesRegex(RuntimeError, "prior"):
            embedding.embed("synthetic")

    async def test_workflow_callback_stops_swallowed_extract_before_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = NativeMicrosoftGraphRAG()
            adapter.target_book_id, adapter.community_level = "book", 0
            adapter.working_dir = Path(folder)
            adapter.config = SimpleNamespace(model_dump=lambda **kwargs: {})
            adapter.check_fatal, adapter.errors, adapter.truncation_recoveries = None, [], []
            adapter._models = nullcontext
            seen = []
            async def build(**kwargs):
                callbacks = kwargs["callbacks"]
                self.assertIsInstance(callbacks[0], NativeFailureCallbacks)
                for name in STANDARD_WORKFLOWS:
                    for callback in callbacks:callback.workflow_start(name, None)
                    seen.append(name)
                    if name == "extract_graph":adapter.errors.append(ValueError("synthetic native-swallowed extraction failure"))
                    for callback in callbacks:callback.workflow_end(name, None)
            adapter.api = SimpleNamespace(build_index=build)
            with patch.dict("sys.modules", {"graphrag.config.enums": SimpleNamespace(IndexingMethod=SimpleNamespace(Standard="standard"))}):
                with self.assertRaisesRegex(RuntimeError, "swallowed"):
                    await adapter.index_book("book", [GraphSourceDocument("document", "book", "complete synthetic original source")])
            self.assertEqual(seen, list(STANDARD_WORKFLOWS[:3]))
            self.assertFalse((Path(folder) / "native_index_complete.json").exists())

    def test_explicit_600_token_native_chunk_revision_preserves_other_controls(self):
        self.assertEqual(NATIVE_SETTINGS["chunks"]["size"], 600)
        self.assertEqual(NATIVE_SETTINGS["chunks"]["overlap"], 100)
        self.assertEqual(NATIVE_SETTINGS["chunks"]["group_by_columns"], ["id"])
        self.assertFalse(NATIVE_SETTINGS["chunks"]["prepend_metadata"])
        self.assertEqual(NATIVE_SETTINGS["extraction_truncation_recovery"]["retry_max_tokens"], 8192)

    async def test_only_truncated_extraction_retries_same_messages_at_8192(self):
        calls = []
        def raw(messages, **kwargs):
            calls.append((messages, kwargs))
            return {"choices": [{"finish_reason": "length" if len(calls) == 1 else "stop",
                                  "message": {"content": "partial" if len(calls) == 1 else "complete"}}],
                    "_native_cache_hit": len(calls) == 1}
        self.provider.chat_response = raw
        chat = AuditedGraphChat(self.provider, SimpleNamespace(max_tokens=4096), "microsoft_graphrag.synthetic.extract_graph", self.errors)
        result = await chat.achat("official continuation", history=[{"role": "user", "content": "synthetic extraction"}], name="extract-continuation-0")
        self.assertEqual(result.output.content, "complete")
        self.assertEqual([kwargs["max_tokens"] for _, kwargs in calls], [4096, 8192])
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertNotIn("partial", str(result.history))
        self.assertEqual(self.errors, [])
        self.assertEqual(chat.truncation_recoveries[0]["retry_finish_reason"], "stop")

    def test_sync_extraction_retry_still_rejects_second_length(self):
        calls = []
        def raw(messages, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"finish_reason": "length", "message": {"content": "truncated"}}]}
        self.provider.chat_response = raw
        chat = AuditedGraphChat(self.provider, SimpleNamespace(max_tokens=4096), "microsoft_graphrag.synthetic.extract_graph", self.errors)
        with self.assertRaisesRegex(ValueError, "truncated"):
            chat.chat("synthetic")
        self.assertEqual([row["max_tokens"] for row in calls], [4096, 8192])
        self.assertEqual(len(self.errors), 1)

    async def test_qa_and_summary_truncation_do_not_expand_limit(self):
        calls = []
        def raw(messages, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"finish_reason": "length", "message": {"content": "truncated"}}]}
        self.provider.chat_response = raw
        for stage in ("local_search_chat", "global_search", "summarize_descriptions", "community_reporting"):
            chat = AuditedGraphChat(self.provider, SimpleNamespace(max_tokens=4096), "microsoft_graphrag.synthetic." + stage, [])
            with self.assertRaises(ValueError):
                await chat.achat("synthetic")
        self.assertEqual([row["max_tokens"] for row in calls], [4096] * 4)

    async def test_non_length_failure_is_never_retried(self):
        calls = []
        def raw(messages, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"finish_reason": "content_filter", "message": {"content": "refused"}}]}
        self.provider.chat_response = raw
        chat = AuditedGraphChat(self.provider, SimpleNamespace(max_tokens=4096), "microsoft_graphrag.synthetic.extract_graph", self.errors)
        with self.assertRaises(ValueError):
            await chat.achat("synthetic")
        self.assertEqual(len(calls), 1)

    async def test_embedding_shape_enforced(self):
        model = AuditedGraphEmbedding(self.provider, self.config, "synthetic", self.errors)
        self.assertEqual(len(await model.aembed("synthetic")), 1024)
        self.provider.embed = lambda texts, **kwargs: [[1.0]]
        with self.assertRaises(ValueError):
            model.embed("synthetic")

    def test_extract_only_not_accepted_as_complete(self):
        partial = [SimpleNamespace(workflow="extract_graph", errors=None)]
        with self.assertRaises(RuntimeError):
            validate_pipeline(partial)
        good = [SimpleNamespace(workflow=name, errors=None) for name in STANDARD_WORKFLOWS]
        self.assertEqual(validate_pipeline(good), list(STANDARD_WORKFLOWS))
        good[-1].errors = [RuntimeError("failed")]
        with self.assertRaises(RuntimeError):
            validate_pipeline(good)

    def test_budget_source_selection_keeps_native_order_and_utf8(self):
        selected, context = bounded_sources(
            [{"id": str(i), "text": "甲😀乙" * 600} for i in range(8)], "book-a")
        self.assertEqual(selected[0]["native_text_unit_short_id"], "0")
        self.assertLessEqual(len(selected), 5)
        self.assertLessEqual(len(context.encode("utf-8")), 3000)
        self.assertEqual(selected[0]["target_book_id"], "book-a")

    def test_source_ids_follow_native_indexer_not_human_readable_ids(self):
        native_units = [SimpleNamespace(short_id="0", id="native-unit", text="synthetic", document_ids=["doc"])]
        rows = [{"id": "0", "text": "synthetic"}]
        allowed = validate_source_provenance(rows, native_units, {"doc"})
        self.assertEqual(allowed["0"].id, "native-unit")
        with self.assertRaises(ValueError):
            validate_source_provenance(rows, native_units, {"different-book-doc"})


if __name__ == "__main__":
    unittest.main()
