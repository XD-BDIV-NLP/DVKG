from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contributions import write_immutable
from ..manifest import digest
from .native_microsoft_graphrag import GraphSourceDocument, NativeMicrosoftGraphRAG, OFFICIAL_COMMIT


def source_document_id(book_id: str, chunk_id: str) -> str:
    return digest([book_id, chunk_id])


@dataclass
class MicrosoftSession:
    adapter: Any
    provider: Any
    book_id: str
    index_receipt: dict
    binding_digest: str
    native_units: dict
    native_documents: dict
    passages: dict
    structure: dict

    async def close(self):
        # 官方API每次操作已在finally恢复ModelManager，无常驻模型或向量库会话。
        close = getattr(self.adapter, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
        if isinstance(self.adapter, NativeMicrosoftGraphRAG):
            from graphrag.language_model.factory import ModelFactory
            for config in self.adapter.config.models.values():
                if config.type.startswith("audited_graph_chat_"):
                    ModelFactory._chat_registry.pop(config.type, None)
                if config.type.startswith("audited_graph_embedding_"):
                    ModelFactory._embedding_registry.pop(config.type, None)
        self.native_units.clear()
        self.native_documents.clear()


async def open_microsoft(plan, index_root, provider, adapter_factory=None):
    book, rows, structure = plan["book"], plan["passages"], plan["structure"]
    if structure.get("book_id") != book or not rows:
        raise ValueError("Microsoft full helper requires the complete target-book corpus")
    passages = {}
    sources = []
    binding = []
    for row in rows:
        if row["book_id"] != book or not row["text"] or row["chunk_id"] in passages:
            raise ValueError("Foreign, empty or duplicate original source passage")
        for span in row["evidence_spans"]:
            evidence = structure["evidence"][span["evidence_id"]]
            if evidence["book_id"] != book or evidence["chunk_id"] != row["chunk_id"] or row["text"][span["start"]:span["end"]] != evidence["text"]:
                raise ValueError("Original passage evidence-span binding differs")
        passages[row["chunk_id"]] = row
        identifier = source_document_id(book, row["chunk_id"])
        sources.append(GraphSourceDocument(identifier, book, row["text"]))
        binding.append({"chunk_id": row["chunk_id"], "document_id": identifier,
                        "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                        "evidence_spans": row["evidence_spans"]})
    max_tokens = plan.get("microsoft_max_tokens", 4096)
    level = plan.get("microsoft_community_level", 0)
    factory = adapter_factory or NativeMicrosoftGraphRAG.create
    adapter = factory(checkout=Path(plan["source"]), index_root=Path(index_root), target_book_id=book,
                      provider=provider, max_tokens=max_tokens, community_level=level,
                      check_fatal=provider.raise_if_fatal)
    before = provider.statistics()
    identity = await adapter.index_book(book, sources)
    provider.raise_if_fatal()
    if identity.get("status") != "complete_standard_native_index":
        raise RuntimeError("Full Microsoft helper refuses incomplete/legacy extraction-only index")
    tables = adapter._tables()
    filenames = {digest(doc.document_id) + ".txt": row for doc, row in zip(sources, rows)}
    by_text = {}
    for row in rows:
        by_text.setdefault(row["text"], []).append(row)
    document_map = {}
    for doc in tables["documents"].to_dict(orient="records"):
        original = filenames.get(doc["title"])
        if original is None or original["text"] != doc["text"]:
            raise ValueError("Native documents do not match complete original source passages")
        # 官方文档ID按正文计算；重复正文保留所有原始来源，不任意挑选金标准位置。
        document_map.setdefault(str(doc["id"]), {})
        document_map[str(doc["id"])].update({r["chunk_id"]: r for r in by_text[doc["text"]]})
    unit_map = {str(index): row for index, row in zip(tables["text_units"].index, tables["text_units"].to_dict(orient="records"))}
    receipt = {"identity": identity, "before": before, "after": provider.statistics(),
               "target_book_id": book, "source_passage_count": len(sources),
               "source_characters": sum(len(doc.text) for doc in sources),
               "source_utf8_bytes": sum(len(doc.text.encode("utf-8")) for doc in sources),
               "source_binding_digest": digest(binding), "max_tokens": max_tokens, "community_level": level,
               "source_policy": "complete_original_native_corpus_passages_no_body_removed",
               "extra_native_index_cost": True}
    return MicrosoftSession(adapter, provider, book, receipt, digest(binding), unit_map,
                            document_map, passages, structure)


def _positions(text, fragment):
    offset = 0
    while fragment:
        found = text.find(fragment, offset)
        if found < 0:
            return
        yield found, found + len(fragment)
        offset = found + 1


def _locate(session, native_sources):
    context, locations, seen = [], [], {}
    for source in native_sources:
        short_id = str(source["id"])
        unit = session.native_units.get(short_id)
        if unit is None or unit["text"] != source["text"]:
            raise ValueError("Native local Sources differs from actual indexed text_units")
        matches, mapped_ids = [], set()
        for doc_id in unit["document_ids"]:
            candidates = session.native_documents.get(str(doc_id))
            if not candidates:
                raise ValueError("Native source document is outside target-book index")
            for passage in candidates.values():
                for start, end in _positions(passage["text"], unit["text"]):
                    matched = []
                    for span in passage["evidence_spans"]:
                        left, right = max(start, span["start"]), min(end, span["end"])
                        if left >= right:
                            continue
                        eid = span["evidence_id"]
                        evidence = session.structure["evidence"][eid]
                        fragment = passage["text"][left:right]
                        key = (eid, left - span["start"], right - span["start"], fragment)
                        if key in seen:
                            record = context[seen[key]]
                            if short_id not in record["native_source_ids"]:
                                record["native_source_ids"].append(short_id)
                        else:
                            seen[key] = len(context)
                            context.append({"book_id": session.book_id, "evidence_id": eid,
                                            "text": fragment, "chunk_id": passage["chunk_id"],
                                            "chapter_id": evidence.get("chapter_id"),
                                            "evidence_start": left - span["start"], "evidence_end": right - span["start"],
                                            "native_source_ids": [short_id], "source_kind": "original_evidence_fragment"})
                        matched.append(eid)
                        mapped_ids.add(eid)
                    matches.append({"chunk_id": passage["chunk_id"], "passage_start": start, "passage_end": end,
                                    "evidence_ids": matched})
        if not matches:
            # 不能精确定位的分词边界仍保留实际上下文，但不给原文引用命中信用。
            context.append({"book_id": session.book_id, "evidence_id": None, "text": unit["text"],
                            "native_source_ids": [short_id], "source_kind": "unresolved_native_text_unit"})
        locations.append({"native_source_short_id": short_id, "native_text_unit_id": unit["id"],
                          "native_document_ids": list(unit["document_ids"]), "matches": matches,
                          "mapped_evidence_ids": sorted(mapped_ids), "resolved": bool(matches),
                          "ambiguous_location": len(matches) > 1})
    return context, locations


def _citation_mapping(answer, locations):
    lookup = {r["native_source_short_id"]: r for r in locations}
    result, seen = [], set()
    for block in re.findall(r"\[\s*Data:\s*([^\]]+)\]", answer, flags=re.IGNORECASE):
        for group in re.findall(r"Sources\s*\(([^)]+)\)", block, flags=re.IGNORECASE):
            for raw in group.split(","):
                identifier = raw.strip()
                if not identifier.isdigit() or identifier in seen:
                    continue
                seen.add(identifier)
                location = lookup.get(identifier)
                result.append({"native_source_short_id": identifier, "source_kind": "original_text_unit",
                               "resolved": bool(location and location["resolved"]),
                               "mapped_evidence_ids": location["mapped_evidence_ids"] if location else []})
    return result


async def query_microsoft(session, plan, question, *, method="local", output_dir=None):
    book = question["target_book_id"]
    if book != session.book_id or plan["book"] != book or method not in ("local", "global"):
        raise ValueError("Microsoft full query target book/method differs")
    session.provider.raise_if_fatal()
    identity = digest({"book": book, "question_id": question.get("question_id"), "query": question["question"],
                       "method": method, "source_binding": session.binding_digest,
                       "official_commit": OFFICIAL_COMMIT,
                       "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    path = Path(output_dir) / f"microsoft_{method}_native_qa.json" if output_dir is not None else None
    if path is not None and path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["identity"] != identity:
            raise ValueError("Microsoft native QA checkpoint differs from question/source binding")
        return stored
    # 只传问题文本，参考答案、支持证据和任务类型均不进入原生检索/QA。
    native = await session.adapter.query_native(book, question["question"], method=method)
    session.provider.raise_if_fatal()
    native_context = native["native_context"]
    context, locations = _locate(session, native_context.get("sources", [])) if method == "local" else ([], [])
    result = {"identity": identity, "question_id": question.get("question_id"), "target_book_id": book,
              "query": question["question"], "answer": native["answer"], "method": method,
              "comparison_role": "primary" if method == "local" else "auxiliary",
              "provided_context": context, "source_locations": locations,
              "native_citation_mapping": _citation_mapping(native["answer"], locations) if method == "local" else [],
              "native_context": native_context, "native_result": native,
              "context_kind": "native_local_original_sources" if method == "local" else "community_reports",
              "source_grounding_evaluable": method == "local" and bool(context),
              "grounding_evaluable": method == "local" and bool(context),
              "original_source_citation_evaluable": method == "local" and bool(locations) and all(row["resolved"] for row in locations),
              "finish_metadata": {"validation": "GraphRAG audited binding accepts only finish_reason=stop; raw receipts retained by shared provider",
                                  "raw_finish_reasons_returned_by_native_api": False},
              "citation_policy": "only exact original source/evidence intersections; reports/entities/relationships never treated as original quotes",
              "mapping_status": "resolved" if all(row["resolved"] for row in locations) else "partial",
              "official_commit": OFFICIAL_COMMIT}
    if path is not None:
        write_immutable(path, result)
    return result
