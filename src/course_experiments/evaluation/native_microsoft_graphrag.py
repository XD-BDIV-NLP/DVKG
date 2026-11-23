from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_COMMIT = "56a865bff0cac9fbbc24f343cf52d0a1850abba6"
OFFICIAL_VERSION = "2.2.1"
OFFICIAL_REPOSITORY = "https://github.com/microsoft/graphrag"
LLM_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
EMBEDDING_MODEL = "BAAI/bge-m3"
BASE_URL = "https://api.siliconflow.cn/v1"
NATIVE_SETTINGS = {"response_type": "Single Paragraph",
                   "index_configuration_revision": "native_chunks600_fail_fast_v1",
                   "chunks": {"size": 600, "overlap": 100, "group_by_columns": ["id"],
                              "strategy": "tokens", "encoding_model": "cl100k_base",
                              "prepend_metadata": False, "chunk_size_includes_metadata": False},
                   "fail_fast": "sticky provider/format failure blocks further model requests and workflow boundaries",
                   "extraction_truncation_recovery": {"initial_max_tokens": 4096, "retry_max_tokens": 8192,
                                                      "max_attempts": 2, "trigger": "raw finish_reason=length",
                                                      "scope": "registered extract_graph model only",
                                                      "same_prompt_and_history": True, "accept_only_stop": True}}
STANDARD_WORKFLOWS = (
    "create_base_text_units", "create_final_documents", "extract_graph", "finalize_graph",
    "create_communities", "create_final_text_units", "create_community_reports", "generate_text_embeddings",
)
TABLES = ("documents", "text_units", "entities", "relationships", "communities", "community_reports")
_MODEL_LOCK = asyncio.Lock()
DISCLOSURE = {
    "chunk_revision": "official chunks.size explicitly frozen at 600 tokens, overlap 100, per-document grouping; prior upstream-default 1200-token gate failed after both 4096 and 8192 extraction limits and remains preserved; full original source documents remain unchanged; official re-chunking creates additional index work and is not the same index configuration",
    "fail_fast": "official workflow_start/workflow_end callbacks and shared sticky model errors stop downstream work after native-swallowed failures; partial tables remain diagnostic and cannot count as complete",
    "response_type": "official supported Single Paragraph task-output configuration for local and global, frozen before full evaluation; not tuned on test scores",
    "index": "official Standard pipeline, including graph aggregation, Leiden communities, reports and required embeddings; claims disabled as official default",
    "injection": "public ModelFactory ChatModel/EmbeddingModel protocols; all model traffic uses shared AuditedNativeProvider; native FNLLM HTTP/retry/cache layer is replaced by audited provider",
    "truncation_recovery": "only extract_graph/gleaning: raw length at 4096 retries the identical messages once with 8192 via provider.chat_response; both preserved receipts and HTTP attempts share the existing budget; 8192 length fails; QA/summary/report limits stay frozen",
    "stream": "official query streaming protocol receives one completed audited non-streaming response chunk",
    "native_track": "official local/global search prompts and contexts; no five-source/3000-byte equivalence claim",
    "controlled_track": "official local-search context builder, then native Sources table only, at most five chunks and 3000 UTF-8 bytes including labels; common answer generator is external",
    "tokenization": "official cl100k_base token accounting, not DeepSeek tokenizer",
    "partial_assets": "legacy graphrag_extract outputs are extraction-only; never accepted as a completed index; existing audited account/budget must not be reset",
    "index_scope": "independent per-book Standard index; all index and community-report costs are additional",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphSourceDocument:
    document_id: str
    book_id: str
    text: str


@dataclass
class ChatOutput:
    content: str


@dataclass
class ChatResponse:
    output: ChatOutput
    history: list
    parsed_response: Any = None
    cache_hit: bool = False
    metrics: Any = None


class AuditedGraphChat:
    def __init__(self, provider, config, stage: str, errors: list, truncation_recoveries=None):
        self.provider, self.config, self.stage, self.errors = provider, config, stage, errors
        self.truncation_recoveries = truncation_recoveries if truncation_recoveries is not None else []

    def _request(self, prompt, history, kwargs):
        if self.errors:
            raise RuntimeError("Native GraphRAG prior model failure blocks further requests") from self.errors[0]
        params = dict(kwargs.get("model_parameters") or {})
        for key, default in {"temperature": 0, "top_p": 1, "n": 1, "frequency_penalty": 0, "presence_penalty": 0}.items():
            if params.get(key, default) not in (None, default):
                raise ValueError(f"Native model parameter differs from frozen provider: {key}")
        if any(key in params for key in ("model", "api_base", "base_url")):
            raise ValueError("Native model/provider overrides are forbidden")
        messages = [dict(item) for item in history or []] + [{"role": "user", "content": prompt}]
        max_tokens = params.get("max_tokens") or params.get("max_completion_tokens") or self.config.max_tokens
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("Explicit frozen generation max_tokens is required")
        json_mode = kwargs.get("json", False) or kwargs.get("json_model") is not None
        return messages, {"max_tokens": max_tokens,
                          "response_format": {"type": "json_object"} if json_mode else None,
                          "stage": self.stage + "." + str(kwargs.get("name", "chat"))}

    def _response(self, content, messages, kwargs):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Audited provider returned empty GraphRAG content")
        model = kwargs.get("json_model")
        parsed = model.model_validate_json(content) if model is not None else None
        return ChatResponse(ChatOutput(content), messages + [{"role": "assistant", "content": content}], parsed)

    def _raw_content(self, messages, options):
        raw = self.provider.chat_response(messages, **options)
        finish = raw["choices"][0].get("finish_reason")
        policy = NATIVE_SETTINGS["extraction_truncation_recovery"]
        if finish == "length" and self.stage.rsplit(".", 1)[-1] == "extract_graph" and options["max_tokens"] == policy["initial_max_tokens"]:
            # 重试不携带被截断回答，且不改变原生提示、输入正文或历史。
            event = {"stage": options["stage"], "messages_digest": _digest(messages),
                     "initial_max_tokens": options["max_tokens"], "initial_finish_reason": finish,
                     "initial_cache_hit": raw.get("_native_cache_hit"),
                     "retry_max_tokens": policy["retry_max_tokens"]}
            self.truncation_recoveries.append(event)
            retry_options = {**options, "max_tokens": policy["retry_max_tokens"],
                             "stage": options["stage"] + ".length_retry_8192"}
            raw = self.provider.chat_response(messages, **retry_options)
            finish = raw["choices"][0].get("finish_reason")
            event.update(retry_finish_reason=finish, retry_cache_hit=raw.get("_native_cache_hit"))
        if finish != "stop":
            raise ValueError("Native completion failed or was truncated; inspect preserved provider receipts")
        return raw["choices"][0]["message"]["content"]

    async def achat(self, prompt, history=None, **kwargs):
        try:
            messages, options = self._request(prompt, history, kwargs)
            if callable(getattr(self.provider, "chat_response", None)):
                content = await asyncio.to_thread(self._raw_content, messages, options)
            else:
                content = await self.provider.achat(messages, **options)
            return self._response(content, messages, kwargs)
        except Exception as exc:
            self.errors.append(exc)
            raise

    def chat(self, prompt, history=None, **kwargs):
        try:
            messages, options = self._request(prompt, history, kwargs)
            content = self._raw_content(messages, options) if callable(getattr(self.provider, "chat_response", None)) else self.provider.chat(messages, **options)
            return self._response(content, messages, kwargs)
        except Exception as exc:
            self.errors.append(exc)
            raise

    async def achat_stream(self, prompt, history=None, **kwargs):
        yield (await self.achat(prompt, history, **kwargs)).output.content

    def chat_stream(self, prompt, history=None, **kwargs):
        yield self.chat(prompt, history, **kwargs).output.content


class AuditedGraphEmbedding:
    def __init__(self, provider, config, stage: str, errors: list):
        self.provider, self.config, self.stage, self.errors = provider, config, stage, errors

    @staticmethod
    def _validated(values, count):
        rows = values.tolist() if hasattr(values, "tolist") else values
        if len(rows) != count or any(len(row) != 1024 or not all(math.isfinite(x) for x in row) for row in rows):
            raise ValueError("BGE-M3 must return finite (N,1024) vectors")
        return rows

    async def aembed_batch(self, text_list, **kwargs):
        try:
            if self.errors:
                raise RuntimeError("Native GraphRAG prior model failure blocks further requests") from self.errors[0]
            return self._validated(await self.provider.aembed(text_list, stage=self.stage), len(text_list))
        except Exception as exc:
            self.errors.append(exc)
            raise

    def embed_batch(self, text_list, **kwargs):
        try:
            if self.errors:
                raise RuntimeError("Native GraphRAG prior model failure blocks further requests") from self.errors[0]
            return self._validated(self.provider.embed(text_list, stage=self.stage), len(text_list))
        except Exception as exc:
            self.errors.append(exc)
            raise

    async def aembed(self, text, **kwargs):
        return (await self.aembed_batch([text], **kwargs))[0]

    def embed(self, text, **kwargs):
        return self.embed_batch([text], **kwargs)[0]


class NativeFailureCallbacks:
    def __init__(self, check):
        self.check = check

    def workflow_start(self, name, instance):
        self.check()

    def workflow_end(self, name, instance):
        self.check()


def validate_pipeline(outputs) -> list[str]:
    names = [item.workflow for item in outputs]
    if names != list(STANDARD_WORKFLOWS) or any(item.errors for item in outputs):
        raise RuntimeError(f"Official Standard index incomplete or failed: {names}")
    return names


def bounded_sources(rows: list[dict], target_book_id: str):
    selected, blocks, seen = [], [], set()
    for row in rows:
        source_id = str(row["id"])
        if source_id in seen or len(selected) >= 5:
            continue
        seen.add(source_id)
        prefix = ("\n\n" if blocks else "") + f"[E{len(selected) + 1}]\n"
        remaining = 3000 - len("".join(blocks).encode("utf-8")) - len(prefix.encode("utf-8"))
        text = str(row["text"]).encode("utf-8")[:max(0, remaining)].decode("utf-8", errors="ignore")
        if not text.strip():
            continue
        selected.append({"evidence_id": f"E{len(selected) + 1}", "text": text,
                         "native_text_unit_short_id": source_id, "target_book_id": target_book_id,
                         "truncated": text != row["text"]})
        blocks.append(prefix + text)
    return selected, "".join(blocks)


def validate_source_provenance(rows, native_units, document_ids):
    # 官方2.2.1将DataFrame索引作为Sources短ID，不能使用human_readable_id。
    allowed = {str(unit.short_id): unit for unit in native_units}
    for row in rows:
        source = allowed.get(str(row["id"]))
        if source is None or source.text != row["text"] or not set(source.document_ids or []).issubset(document_ids):
            raise ValueError("Native context source provenance differs from target-book index")
    return allowed


class NativeMicrosoftGraphRAG:
    @classmethod
    def create(cls, *, checkout: Path, index_root: Path, target_book_id: str,
               provider, max_tokens: int, community_level: int = 0, check_fatal=None):
        checkout = checkout.resolve()
        if not (3, 10) <= sys.version_info[:2] < (3, 13):
            raise RuntimeError("Official GraphRAG 2.2.1 requires Python >=3.10,<3.13")
        head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if head != OFFICIAL_COMMIT or dirty:
            raise ValueError("Unmodified fixed official GraphRAG checkout required")
        existing = sys.modules.get("graphrag")
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(checkout):
            raise RuntimeError("A foreign GraphRAG source is already imported")
        sys.path.insert(0, str(checkout))
        import graphrag
        if not Path(graphrag.__file__).resolve().is_relative_to(checkout):
            raise RuntimeError("GraphRAG import outside fixed checkout")
        from graphrag.language_model.factory import ModelFactory
        from graphrag.config.create_graphrag_config import create_graphrag_config
        if not target_book_id or max_tokens <= 0 or community_level < 0:
            raise ValueError("Book, positive max_tokens and nonnegative community level required")
        obj = cls()
        obj.provider, obj.check_fatal, obj.errors = provider, check_fatal, []
        obj.truncation_recoveries = []
        obj.target_book_id, obj.community_level = target_book_id, community_level
        obj.working_dir = index_root.resolve() / _digest(target_book_id)[:24]
        obj.working_dir.mkdir(parents=True, exist_ok=True)
        obj.stage = f"microsoft_graphrag.{target_book_id}"
        chat_type, embedding_type = "audited_graph_chat_" + _digest(str(obj.working_dir))[:16], "audited_graph_embedding_" + _digest(str(obj.working_dir))[:16]
        ModelFactory.register_chat(chat_type, lambda **kw: AuditedGraphChat(provider, kw["config"], obj.stage + "." + kw["name"], obj.errors, obj.truncation_recoveries))
        ModelFactory.register_embedding(embedding_type, lambda **kw: AuditedGraphEmbedding(provider, kw["config"], obj.stage + "." + kw["name"], obj.errors))
        obj.config = create_graphrag_config({
            "chunks": dict(NATIVE_SETTINGS["chunks"]),
            "models": {
                "default_chat_model": {"type": chat_type, "model": LLM_MODEL, "api_key": "callback-only-unused",
                                       "api_base": BASE_URL, "encoding_model": "cl100k_base", "max_tokens": max_tokens,
                                       "concurrent_requests": 4, "max_retries": 0, "async_mode": "asyncio"},
                "default_embedding_model": {"type": embedding_type, "model": EMBEDDING_MODEL, "api_key": "callback-only-unused",
                                            "api_base": BASE_URL, "encoding_model": "cl100k_base", "concurrent_requests": 4,
                                            "max_retries": 0, "async_mode": "asyncio"}},
            "input": {"type": "file", "file_type": "text", "base_dir": "input", "file_pattern": ".*\\.txt$"},
            "output": {"type": "file", "base_dir": "output"},
            "cache": {"type": "file", "base_dir": "cache"},
            "reporting": {"type": "file", "base_dir": "logs"},
            "vector_store": {"default_vector_store": {"type": "lancedb", "db_uri": str(obj.working_dir / "lancedb")}},
            "cluster_graph": {"seed": 42}, "extract_claims": {"enabled": False},
        }, root_dir=str(obj.working_dir))
        obj.api = importlib.import_module("graphrag.api")
        return obj

    def _check(self, book_id):
        if book_id != self.target_book_id:
            raise ValueError("Target book differs from this independent native index")
        if self.check_fatal is not None:
            self.check_fatal()
        if self.errors:
            raise RuntimeError("Native GraphRAG swallowed a provider/format error") from self.errors[0]

    @contextmanager
    def _models(self):
        from graphrag.language_model.manager import ModelManager
        manager = ModelManager()
        old_chat, old_embed = manager.chat_models, manager.embedding_models
        manager.chat_models, manager.embedding_models = {}, {}
        try:
            yield
        finally:
            manager.chat_models, manager.embedding_models = old_chat, old_embed

    def _completed(self):
        manifest_path = self.working_dir / "native_adapter_manifest.json"
        complete_path = self.working_dir / "native_index_complete.json"
        if not manifest_path.exists() or not complete_path.exists():
            raise ValueError("A completed official Standard index is required; partial extraction is insufficient")
        manifest, complete = (json.loads(p.read_text(encoding="utf-8")) for p in (manifest_path, complete_path))
        if complete["manifest_digest"] != _digest(manifest) or manifest["target_book_id"] != self.target_book_id:
            raise ValueError("Book/index manifest binding differs")
        for name, record in complete["tables"].items():
            if hashlib.sha256((self.working_dir / "output" / (name + ".parquet")).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Completed GraphRAG table changed: " + name)
        return complete

    async def index_book(self, target_book_id: str, documents: list[GraphSourceDocument]):
        self._check(target_book_id)
        if not documents or any(d.book_id != target_book_id or not d.document_id or not d.text.strip() for d in documents):
            raise ValueError("Nonempty single-book source documents required")
        if len({d.document_id for d in documents}) != len(documents):
            raise ValueError("Duplicate source document IDs")
        manifest = {"target_book_id": target_book_id, "official_commit": OFFICIAL_COMMIT,
                    "config": self.config.model_dump(mode="json"), "community_level": self.community_level,
                    "native_settings": NATIVE_SETTINGS,
                    "sources": [{"id": d.document_id, "file": _digest(d.document_id) + ".txt",
                                 "sha256": hashlib.sha256(d.text.encode("utf-8")).hexdigest()} for d in documents],
                    "disclosure": DISCLOSURE}
        path = self.working_dir / "native_adapter_manifest.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Corpus/native config differs; use a new index_root")
        if (self.working_dir / "native_index_complete.json").exists():
            return self._completed()
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        inputs = self.working_dir / "input"
        inputs.mkdir(exist_ok=True)
        for doc, source in zip(documents, manifest["sources"]):
            (inputs / source["file"]).write_text(doc.text, encoding="utf-8")
        from graphrag.config.enums import IndexingMethod
        async with _MODEL_LOCK:
            with self._models():
                outputs = await self.api.build_index(config=self.config, method=IndexingMethod.Standard,
                    callbacks=[NativeFailureCallbacks(lambda: self._check(target_book_id))])
        self._check(target_book_id)
        workflows = validate_pipeline(outputs)
        tables = self._tables(require_complete=False)
        if any(tables[name].empty for name in TABLES):
            raise RuntimeError("Standard index lacks nonempty graph/community/report artifacts")
        summary = {name: {"rows": len(table), "sha256": hashlib.sha256((self.working_dir / "output" / (name + ".parquet")).read_bytes()).hexdigest()}
                   for name, table in tables.items()}
        complete = {"status": "complete_standard_native_index", "target_book_id": target_book_id,
                    "manifest_digest": _digest(manifest), "official_commit": OFFICIAL_COMMIT,
                    "workflows": workflows, "tables": summary, "extra_native_index_cost": True,
                    "truncation_recoveries": self.truncation_recoveries,
                    "native_settings": NATIVE_SETTINGS}
        (self.working_dir / "native_index_complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
        return complete

    def _tables(self, require_complete=True):
        if require_complete:
            self._completed()
        import pandas as pd
        return {name: pd.read_parquet(self.working_dir / "output" / (name + ".parquet")) for name in TABLES}

    @staticmethod
    def _context(context):
        return {key: json.loads(value.to_json(orient="records", force_ascii=False)) if hasattr(value, "to_json") else value
                for key, value in context.items()}

    async def query_native(self, target_book_id: str, query: str, *, method: str = "local"):
        self._check(target_book_id)
        tables = self._tables()
        if method not in ("local", "global"):
            raise ValueError("Native Microsoft query method must be local or global")
        params = {key: tables[key] for key in ("entities", "communities", "community_reports")}
        params.update(config=self.config, community_level=self.community_level, response_type=NATIVE_SETTINGS["response_type"], query=query)
        if method == "local":
            params.update(text_units=tables["text_units"], relationships=tables["relationships"], covariates=None)
        else:
            params["dynamic_community_selection"] = False
        async with _MODEL_LOCK:
            with self._models():
                answer, context = await getattr(self.api, method + "_search")(**params)
        self._check(target_book_id)
        if not answer:
            raise RuntimeError("Native GraphRAG returned an empty answer")
        return {"track": "native_end_to_end", "method": method, "target_book_id": target_book_id,
                "query": query, "answer": answer, "native_context": self._context(context),
                "official_commit": OFFICIAL_COMMIT, "disclosure": DISCLOSURE}

    async def retrieve(self, target_book_id: str, query: str):
        self._check(target_book_id)
        tables = self._tables()
        from graphrag.config.embeddings import entity_description_embedding
        from graphrag.query.factory import get_local_search_engine
        from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports, read_indexer_text_units, read_indexer_relationships
        from graphrag.utils.api import get_embedding_store
        def build():
            store = get_embedding_store(config_args={key: item.model_dump() for key, item in self.config.vector_store.items()}, embedding_name=entity_description_embedding)
            engine = get_local_search_engine(
                self.config, reports=read_indexer_reports(tables["community_reports"], tables["communities"], self.community_level),
                text_units=read_indexer_text_units(tables["text_units"]),
                entities=read_indexer_entities(tables["entities"], tables["communities"], self.community_level),
                relationships=read_indexer_relationships(tables["relationships"]), covariates={},
                response_type=NATIVE_SETTINGS["response_type"], description_embedding_store=store)
            return engine.context_builder.build_context(query=query, **engine.context_builder_params)
        async with _MODEL_LOCK:
            with self._models():
                context = await asyncio.to_thread(build)
        self._check(target_book_id)
        context_data = self._context(context.context_records)
        rows = context_data.get("sources", [])
        doc_ids = set(tables["documents"]["id"].astype(str))
        allowed = validate_source_provenance(rows, read_indexer_text_units(tables["text_units"]), doc_ids)
        evidence, text = bounded_sources(rows, target_book_id)
        for item in evidence:
            source = allowed[item["native_text_unit_short_id"]]
            item.update(native_text_unit_id=source.id, native_document_ids=source.document_ids)
        return {"track": "native_local_retrieval_budgeted_generation", "target_book_id": target_book_id,
                "query": query, "evidence": evidence, "context": text, "context_utf8_bytes": len(text.encode("utf-8")),
                "native_context": context_data, "official_commit": OFFICIAL_COMMIT, "disclosure": DISCLOSURE}
