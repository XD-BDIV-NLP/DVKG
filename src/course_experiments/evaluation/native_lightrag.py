from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence


OFFICIAL_REPOSITORY = "https://github.com/HKUDS/LightRAG"
OFFICIAL_COMMIT = "aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a"
OFFICIAL_VERSION = "1.5.8"
BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
NATIVE_SETTINGS = {"chunk_token_size": 1200, "chunk_overlap_token_size": 100,
                   "entity_extract_max_gleaning": 1, "language": "Chinese",
                   "top_k": 40, "chunk_top_k": 20, "max_entity_tokens": 6000,
                   "max_relation_tokens": 8000, "max_total_tokens": 30000,
                   "embedding_max_tokens": 8192, "embedding_batch_num": 8,
                   "reranker": False, "tiktoken_model_name": "gpt-4o-mini",
                   "response_type": "Single Paragraph",
                   "default_embedding_timeout": 2700, "default_llm_timeout": 2700,
                   "callback_worker_timeout_seconds": 5400}

# 原生轨保留图摘要与官方回答提示；受控轨只取原生排序后的原文块。
DISCLOSURE = {
    "callback_timeout": "official configurable embedding and all LLM role timeout=2700s; fixed commit worker limit=2x=5400s covers exact-request singleflight and global-slot queue waiting plus frozen 7-attempt HTTP retry and storage margin; bounded to 90 minutes within the existing 48h formal deadline; provider HTTP timeout, 4 slots, receipts, request cap and deadline remain authoritative; prior 60s embedding timeout failure retained",
    "response_type": "official supported Single Paragraph task-output configuration, frozen before full evaluation; not tuned on test scores",
    "native_entrypoints": ["LightRAG.ainsert", "LightRAG.aquery_data", "LightRAG.aquery_llm"],
    "retrieval_mode": "hybrid",
    "index_scope": "one independent index per target_book_id",
    "native_end_to_end": "official generation prompt and graph context; no 5-evidence/3000-byte claim",
    "budgeted_generation": "native hybrid retrieval; retain only source chunks in native order; UTF-8 truncate to at most 5 chunks and 3000 bytes including evidence labels; common generator must consume context only",
    "reranker": "disabled; no extra reranker model or external provider",
    "model_substitution": "frozen SiliconFlow DeepSeek-V4-Flash and BGE-M3 callbacks",
    "tokenization": "official tiktoken gpt-4o-mini token accounting; not the DeepSeek tokenizer",
    "index_cost": "native extraction, summaries, keyword extraction and embeddings are additional charged work and must be reported independently",
}


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    book_id: str
    text: str


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_checkout(checkout: Path) -> Path:
    checkout = checkout.resolve()
    if not (checkout / ".git").exists():
        raise ValueError("A pinned official LightRAG Git checkout is required")
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != OFFICIAL_COMMIT:
        raise ValueError(f"LightRAG commit differs: {head}; required {OFFICIAL_COMMIT}")
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if dirty:
        raise ValueError("Pinned official LightRAG sources have tracked modifications")
    return checkout


def bounded_evidence(
    chunks: Sequence[dict[str, Any]], source_map: dict[str, str], target_book_id: str,
) -> tuple[list[dict[str, Any]], str]:
    selected: list[dict[str, Any]] = []
    blocks: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        file_path = str(chunk.get("file_path", ""))
        if file_path not in source_map:
            raise ValueError(f"Native chunk provenance not in target-book index: {file_path}")
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        if len(selected) == 5:
            continue
        prefix = ("\n\n" if blocks else "") + f"[E{len(selected) + 1}]\n"
        remaining = 3000 - len("".join(blocks).encode("utf-8")) - len(prefix.encode("utf-8"))
        if remaining <= 0:
            continue
        text = str(chunk["content"]).encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
        if not text.strip():
            continue
        selected.append({"evidence_id": f"E{len(selected) + 1}", "text": text,
                         "native_chunk_id": chunk_id, "source_document_id": source_map[file_path],
                         "target_book_id": target_book_id,
                         "truncated": text != chunk["content"]})
        blocks.append(prefix + text)
    return selected, "".join(blocks)


class NativeLightRAG:
    """固定官方实现的薄适配；callback 负责收据、缓存、预算和唯一允许的 API。

    llm_callback: async (prompt, *, system_prompt=None, history_messages=None,
        purpose, **native_kwargs) -> str
    embedding_callback: async (texts: list[str], *, purpose) -> array(N,1024)
    check_fatal: () -> None，若 provider 已记录预算/鉴权故障则抛出。
    callback 必须拒绝任何模型或站点覆盖，并强制 enable_thinking=False。
    """

    def __init__(self, core: Any, query_param: Callable[..., Any], working_dir: Path,
                 target_book_id: str, check_fatal: Callable[[], None] | None = None):
        self.core = core
        self.query_param = query_param
        self.working_dir = working_dir
        self.target_book_id = target_book_id
        self.check_fatal = check_fatal

    @classmethod
    async def create(
        cls, *, checkout: Path, index_root: Path, target_book_id: str,
        llm_callback: Callable[..., Awaitable[str]],
        embedding_callback: Callable[..., Awaitable[Any]],
        check_fatal: Callable[[], None] | None = None,
    ) -> NativeLightRAG:
        checkout = verify_checkout(checkout)
        existing = sys.modules.get("lightrag")
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(checkout):
            raise RuntimeError("Another LightRAG build is already imported; use a clean Python process")
        sys.path.insert(0, str(checkout))
        native = importlib.import_module("lightrag")
        if native.__version__ != OFFICIAL_VERSION:
            raise RuntimeError("Unexpected official LightRAG version")
        from lightrag.utils import EmbeddingFunc
        import numpy as np

        if not target_book_id:
            raise ValueError("target_book_id is required")
        book_key = hashlib.sha256(target_book_id.encode("utf-8")).hexdigest()[:24]
        working_dir = index_root.resolve() / book_key
        working_dir.mkdir(parents=True, exist_ok=True)

        def binding(role: str):
            async def call(prompt, system_prompt=None, history_messages=None, **kwargs):
                # 这些是官方队列/缓存内部对象，不能进入 HTTP 请求或缓存键。
                kwargs.pop("hashing_kv", None)
                kwargs.pop("_priority", None)
                kwargs.pop("stream", None)
                result = await llm_callback(
                    prompt, system_prompt=system_prompt, history_messages=history_messages or [],
                    purpose=f"lightrag.{target_book_id}.{role}", **kwargs,
                )
                if not isinstance(result, str):
                    raise TypeError("LightRAG callback must return non-streaming text")
                return result
            return call

        async def embed(texts: list[str]):
            result = np.asarray(await embedding_callback(texts, purpose=f"lightrag.{target_book_id}.embedding"), dtype=np.float32)
            if result.shape != (len(texts), EMBEDDING_DIM) or not np.isfinite(result).all():
                raise ValueError("BGE-M3 callback must return finite (N,1024) vectors")
            return result

        core = native.LightRAG(
            working_dir=str(working_dir), workspace=book_key,
            llm_model_name=LLM_MODEL, llm_model_func=binding("query"),
            role_llm_configs={role: {"func": binding(role), "max_async": 4,
                                    "timeout": NATIVE_SETTINGS["default_llm_timeout"]}
                              for role in ("extract", "keyword", "query", "vlm")},
            llm_model_max_async=4, embedding_func_max_async=4,
            # 回调包含统一槽位等待，官方工作队列不能提前取消仍在排队的请求。
            default_embedding_timeout=NATIVE_SETTINGS["default_embedding_timeout"],
            default_llm_timeout=NATIVE_SETTINGS["default_llm_timeout"],
            embedding_func=EmbeddingFunc(embedding_dim=EMBEDDING_DIM, max_token_size=8192,
                                         model_name=EMBEDDING_MODEL, func=embed),
            chunk_token_size=1200, chunk_overlap_token_size=100,
            embedding_batch_num=8, max_parallel_insert=2,
            entity_extract_max_gleaning=1,
            kv_storage="JsonKVStorage", graph_storage="NetworkXStorage",
            vector_storage="NanoVectorDBStorage", doc_status_storage="JsonDocStatusStorage",
            rerank_model_func=None, tiktoken_model_name="gpt-4o-mini",
            addon_params={"language": "Chinese"}, user_prompt_prefix="",
        )
        await core.initialize_storages()
        adapter = cls(core, native.QueryParam, working_dir, target_book_id, check_fatal)
        adapter._fatal()
        return adapter

    def _fatal(self) -> None:
        if self.check_fatal is not None:
            self.check_fatal()

    def _check_book(self, target_book_id: str) -> None:
        if not target_book_id or target_book_id != self.target_book_id:
            raise ValueError("Query/index target differs from this independent book index")

    def _manifest(self) -> dict[str, Any]:
        path = self.working_dir / "native_adapter_manifest.json"
        if not path.exists():
            raise ValueError("Book corpus is not bound to this native index")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self._check_book(manifest["target_book_id"])
        if manifest["official_commit"] != OFFICIAL_COMMIT:
            raise ValueError("Index was produced by a different LightRAG commit")
        complete = self.working_dir / "native_index_complete.json"
        if not complete.exists() or json.loads(complete.read_text(encoding="utf-8")).get("manifest_digest") != _digest(manifest):
            raise ValueError("Native index has not completed this exact corpus/config")
        return manifest

    async def index_book(self, target_book_id: str, documents: Sequence[SourceDocument]) -> dict[str, Any]:
        self._check_book(target_book_id)
        self._fatal()
        documents = list(documents)
        if not documents or any(doc.book_id != target_book_id or not doc.text.strip() or not doc.document_id for doc in documents):
            raise ValueError("All documents must be nonempty and belong to target_book_id")
        if len({doc.document_id for doc in documents}) != len(documents):
            raise ValueError("Duplicate source document IDs")
        records = [{"document_id": doc.document_id, "text_sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
                    "text_utf8_bytes": len(doc.text.encode("utf-8"))} for doc in documents]
        source_map = {"lr-source-" + _digest([target_book_id, doc.document_id]): doc.document_id for doc in documents}
        manifest = {"target_book_id": target_book_id, "official_repository": OFFICIAL_REPOSITORY,
                    "official_commit": OFFICIAL_COMMIT, "official_version": OFFICIAL_VERSION,
                    "base_url": BASE_URL, "llm_model": LLM_MODEL, "embedding_model": EMBEDDING_MODEL,
                    "embedding_dim": EMBEDDING_DIM, "enable_thinking": False,
                    "native_settings": NATIVE_SETTINGS,
                    "records": records, "corpus_digest": _digest(records), "source_map": source_map,
                    "extra_native_index_cost": True, "disclosure": DISCLOSURE}
        path = self.working_dir / "native_adapter_manifest.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing native index corpus/config changed; allocate a new index_root")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        doc_ids = ["doc-" + _digest([target_book_id, doc.document_id]) for doc in documents]
        authorization = self.working_dir / "native_resume_authorization.json"
        if authorization.exists():
            proof = json.loads(authorization.read_text(encoding="utf-8"))
            if (proof.get("kind") != "official_manual_retry_after_callback_timeout" or
                    proof.get("target_book_id") != target_book_id or
                    proof.get("manifest_digest") != _digest(manifest) or
                    proof.get("document_ids") != doc_ids):
                raise ValueError("Native resume authorization differs from exact corpus/config")
            track = await self._retry_failed_documents()
            self._fatal()
        else:
            track = await self.core.ainsert(
                [doc.text for doc in documents],
                ids=doc_ids,
                file_paths=list(source_map),
            )
        self._fatal()
        statuses = await self.core.doc_status.get_by_ids(doc_ids)
        failed = {doc_id: status.get("status") if status else "missing"
                  for doc_id, status in zip(doc_ids, statuses)
                  if not status or status.get("status") != "processed"}
        if len(statuses) != len(doc_ids):
            raise RuntimeError("Native indexing returned incomplete document statuses")
        if failed:
            raise RuntimeError(f"Native indexing incomplete: {failed}")
        complete = self.working_dir / "native_index_complete.json"
        complete.write_text(json.dumps({"manifest_digest": _digest(manifest), "track_id": track}), encoding="utf-8")
        return {**manifest, "track_id": track, "status": "processed",
                "native_status_count": len(statuses)}

    async def _retry_failed_documents(self) -> str:
        # 固定提交默认不自动重试FAILED；沿用官方手动重试的独占排空与清理流程。
        from lightrag.kg.shared_storage import get_pipeline_ingress, get_namespace_lock
        from lightrag.kg.pipeline_ingress import PipelineIngressMessage, ManualRetryPublishResult
        request_id = uuid.uuid4().hex
        ingress = await get_pipeline_ingress(self.core.workspace)
        async with get_namespace_lock("pipeline_status", workspace=self.core.workspace):
            accepted = ingress.request_manual_retry(request_id, PipelineIngressMessage(
                kind="rescan", retry_failed=True, request_id=request_id))
        if accepted is not ManualRetryPublishResult.ACCEPTED:
            raise RuntimeError("Official native manual retry request was refused")
        await self.core.apipeline_process_enqueue_documents()
        return request_id

    def _params(self):
        return self.query_param(mode="hybrid", response_type=NATIVE_SETTINGS["response_type"], stream=False, top_k=40, chunk_top_k=20,
                                max_entity_tokens=6000, max_relation_tokens=8000, max_total_tokens=30000,
                                enable_rerank=False, include_references=True, disable_user_prompt_prefix=True)

    async def retrieve(self, target_book_id: str, query: str) -> dict[str, Any]:
        self._check_book(target_book_id)
        manifest = self._manifest()
        self._fatal()
        native_result = await self.core.aquery_data(query, param=self._params())
        self._fatal()
        if native_result.get("status") != "success":
            raise RuntimeError(f"Native retrieval failed: {native_result.get('message', native_result)}")
        evidence, context = bounded_evidence(native_result.get("data", {}).get("chunks", []), manifest["source_map"], target_book_id)
        return {"track": "native_retrieval_budgeted_generation", "target_book_id": target_book_id,
                "query": query, "evidence": evidence, "context": context,
                "context_utf8_bytes": len(context.encode("utf-8")), "native_result": native_result,
                "official_commit": OFFICIAL_COMMIT, "disclosure": DISCLOSURE}

    async def query_native(self, target_book_id: str, query: str) -> dict[str, Any]:
        self._check_book(target_book_id)
        manifest = self._manifest()
        self._fatal()
        native_result = await self.core.aquery_llm(query, param=self._params())
        self._fatal()
        if native_result.get("status") != "success":
            raise RuntimeError(f"Native query failed: {native_result.get('message', native_result)}")
        bounded_evidence(native_result.get("data", {}).get("chunks", []), manifest["source_map"], target_book_id)
        return {"track": "native_end_to_end", "target_book_id": target_book_id, "query": query,
                "native_result": native_result, "official_commit": OFFICIAL_COMMIT,
                "disclosure": DISCLOSURE}

    async def close(self) -> None:
        await self.core.finalize_storages()
