from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

from course_experiments.evaluation.native_lightrag import (
    NativeLightRAG, SourceDocument, bounded_evidence, verify_checkout, NATIVE_SETTINGS, DISCLOSURE,
)


class FakeCore:
    def __init__(self):
        self.calls = []
        self.chunks = []
        self.failure = False
        self.doc_status = self

    async def get_by_ids(self, ids):
        return [{"status": "processed"} for _ in ids]

    async def ainsert(self, texts, **kwargs):
        self.calls.append(("insert", texts, kwargs))
        return "track-test"

    async def aget_docs_by_track_id(self, track):
        return {"d": SimpleNamespace(status=SimpleNamespace(value="processed"))}

    async def aquery_data(self, query, param):
        self.calls.append(("retrieve", query, param))
        return {"status": "success", "data": {"chunks": self.chunks}}

    async def aquery_llm(self, query, param):
        self.calls.append(("query", query, param))
        return {"status": "failure" if self.failure else "success",
                "data": {"chunks": self.chunks}, "llm_response": {"content": "native answer"}}


class NativeLightRAGTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = FakeCore()
        self.adapter = NativeLightRAG(self.core, SimpleNamespace, Path(self.tmp.name), "book-a")
        self.addCleanup(self.tmp.cleanup)

    async def index(self):
        return await self.adapter.index_book("book-a", [SourceDocument("d1", "book-a", "合成说明文本")])

    async def test_official_timeout_configuration_covers_provider_queue_wait(self):
        import numpy as np
        captured = {}
        class Core:
            def __init__(self, **kwargs):
                captured.update(kwargs)
            async def initialize_storages(self):
                pass
        fake = SimpleNamespace(__version__="1.5.8", LightRAG=Core, QueryParam=SimpleNamespace)
        utils = SimpleNamespace(EmbeddingFunc=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch("course_experiments.evaluation.native_lightrag.verify_checkout", return_value=Path(self.tmp.name)), \
             patch("course_experiments.evaluation.native_lightrag.importlib.import_module", return_value=fake), \
             patch.dict(sys.modules, {"lightrag.utils": utils}):
            async def llm(*args, **kwargs):
                return "synthetic"
            async def embed(texts, **kwargs):
                return np.ones((len(texts), 1024))
            await NativeLightRAG.create(checkout=Path(self.tmp.name), index_root=Path(self.tmp.name),
                target_book_id="book-a", llm_callback=llm, embedding_callback=embed)
        self.assertEqual(captured["default_embedding_timeout"], 2700)
        self.assertEqual(captured["default_llm_timeout"], 2700)
        self.assertTrue(all(row["timeout"] == 2700 for row in captured["role_llm_configs"].values()))
        self.assertEqual(NATIVE_SETTINGS["callback_worker_timeout_seconds"], 5400)
        self.assertIn("queue", DISCLOSURE["callback_timeout"])
        self.assertEqual(captured["chunk_token_size"], 1200)
        self.assertEqual(captured["embedding_batch_num"], 8)

    async def test_failed_resume_requires_exact_authorization_and_preserves_native_insert(self):
        from course_experiments.evaluation.native_lightrag import _digest
        await self.index()
        manifest = json.loads((self.adapter.working_dir / "native_adapter_manifest.json").read_text())
        doc_ids = ["doc-" + _digest(["book-a", "d1"])]
        auth = {"kind": "official_manual_retry_after_callback_timeout", "target_book_id": "book-a",
                "manifest_digest": _digest(manifest), "document_ids": doc_ids}
        path = self.adapter.working_dir / "native_resume_authorization.json"
        path.write_text(json.dumps(auth))
        retried = []
        async def resume():
            retried.append(True)
        self.adapter._retry_failed_documents = resume
        await self.index()
        self.assertEqual(retried, [True])
        self.assertEqual(len(self.core.calls), 2)
        auth["document_ids"] = ["foreign"]
        path.write_text(json.dumps(auth))
        with self.assertRaisesRegex(ValueError, "resume authorization"):
            await self.index()
        self.assertEqual(len(self.core.calls), 2)

    async def test_index_native_and_book_isolation(self):
        report = await self.index()
        self.assertEqual(self.core.calls[0][0], "insert")
        self.assertEqual(report["target_book_id"], "book-a")
        self.assertTrue(report["extra_native_index_cost"])
        with self.assertRaises(ValueError):
            await self.adapter.retrieve("book-b", "合成问题")
        with self.assertRaises(ValueError):
            await self.adapter.index_book("book-a", [SourceDocument("x", "book-b", "跨书")])

    async def test_retrieval_native_hybrid_and_bounded_track(self):
        await self.index()
        file_path = self.core.calls[0][2]["file_paths"][0]
        self.core.chunks = [{"chunk_id": str(i), "file_path": file_path, "content": "证据" * 400} for i in range(10)]
        result = await self.adapter.retrieve("book-a", "合成问题")
        self.assertEqual(self.core.calls[-1][2].mode, "hybrid")
        self.assertEqual(self.core.calls[-1][2].response_type, "Single Paragraph")
        self.assertLessEqual(len(result["evidence"]), 5)
        self.assertLessEqual(len(result["context"].encode("utf-8")), 3000)
        self.assertEqual(result["track"], "native_retrieval_budgeted_generation")
        self.assertNotIn("entities", result["context"])
        native = await self.adapter.query_native("book-a", "合成问题")
        self.assertEqual(native["track"], "native_end_to_end")
        self.assertEqual(native["native_result"]["llm_response"]["content"], "native answer")

    async def test_cross_book_native_evidence_rejected(self):
        await self.index()
        self.core.chunks = [{"chunk_id": "bad", "file_path": "another-book/d", "content": "bad"}]
        with self.assertRaises(ValueError):
            await self.adapter.retrieve("book-a", "合成问题")

    async def test_changed_corpus_rejected_before_native_insert(self):
        await self.index()
        with self.assertRaises(ValueError):
            await self.adapter.index_book("book-a", [SourceDocument("d1", "book-a", "改变的文本")])
        self.assertEqual(len(self.core.calls), 1)

    async def test_failure_and_sticky_budget_error_visible(self):
        await self.index()
        self.core.failure = True
        with self.assertRaises(RuntimeError):
            await self.adapter.query_native("book-a", "合成问题")
        def fatal():
            raise RuntimeError("budget exhausted")
        self.adapter.check_fatal = fatal
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            await self.adapter.retrieve("book-a", "合成问题")

    async def test_unfinished_index_cannot_be_queried(self):
        async def failed(ids):
            return [{"status": "failed"} for _ in ids]
        self.core.get_by_ids = failed
        with self.assertRaises(RuntimeError):
            await self.index()
        with self.assertRaises(ValueError):
            await self.adapter.retrieve("book-a", "合成问题")

    def test_bounded_utf8_unique_chunks_and_no_reference_leakage(self):
        chunks = [{"chunk_id": "1", "file_path": "f", "content": "甲😀乙" * 1000}] * 2
        selected, context = bounded_evidence(chunks, {"f": "doc"}, "book-a")
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(len(context.encode("utf-8")), 3000)
        self.assertEqual(selected[0]["text"], context.split("\n", 1)[1])

    def test_non_checkout_fails_closed(self):
        with self.assertRaises(ValueError):
            verify_checkout(Path(self.tmp.name))


if __name__ == "__main__":
    unittest.main()
