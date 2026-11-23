"""查询后零API来源定位派生；不改变原GraphRAG索引、模型输入或原QA检查点。"""
from __future__ import annotations

import copy
import hashlib

from ..manifest import digest, file_record, read_document
from . import native_microsoft_spans
from .native_microsoft_full import _citation_mapping

POLICY = dict(version="microsoft_query_token_byte_source_mapping_v1",
              index_reuse="byte_exact_original_index_contract_and_complete_standard_receipt",
              quote="only_complete_original_S_characters_inside_official_native_window_bytes",
              native_input_changed=False, new_model_calls=0)


def protocol_record():
    return dict(policy=POLICY, mapping_code=file_record(__file__),
                token_mapping_code=file_record(native_microsoft_spans.__file__))


def bind_query_revision(plan, original_index_contract):
    stored = read_document(original_index_contract["path"])
    if file_record(original_index_contract["path"]) != original_index_contract:
        raise ValueError("Original index contract bytes changed")
    index_identity = stored.pop("index_identity")
    if digest(stored) != index_identity or stored != plan["index_contract"]:
        raise ValueError("Query revision requires byte-identical original index configuration/code")
    result = copy.deepcopy(plan)
    result["contract"].update(query_mapping_revision=protocol_record(),
                               reused_original_index_contract=original_index_contract)
    return result


async def cached_index_only(adapter, target_book_id, documents):
    """没有完整索引就失败；绝不回退到付费index_book。"""
    from .native_microsoft_graphrag import NATIVE_SETTINGS, OFFICIAL_COMMIT, _digest
    manifest = read_document(adapter.working_dir / "native_adapter_manifest.json")
    sources = [{"id": doc.document_id, "file": _digest(doc.document_id) + ".txt",
                "sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest()}
               for doc in documents]
    if (target_book_id != adapter.target_book_id or
            any(doc.book_id != target_book_id for doc in documents) or
            manifest["sources"] != sources or manifest["target_book_id"] != target_book_id or
            manifest["official_commit"] != OFFICIAL_COMMIT or
            manifest["config"] != adapter.config.model_dump(mode="json") or
            manifest["native_settings"] != NATIVE_SETTINGS):
        raise ValueError("Completed original index/source/config differs from query-only revision")
    return adapter._completed()


def cached_index_factory(**kwargs):
    from .native_microsoft_graphrag import NativeMicrosoftGraphRAG
    adapter = NativeMicrosoftGraphRAG.create(**kwargs)
    async def existing(target_book_id, documents):
        return await cached_index_only(adapter, target_book_id, documents)
    adapter.index_book = existing
    return adapter


def remap_native_result(session, native, *, encoding=None, size=600, overlap=100, base_record=None):
    if base_record is not None and file_record(base_record["path"]) != base_record:
        raise ValueError("Original native QA checkpoint bytes changed")
    result = copy.deepcopy(native)
    protocol = protocol_record()
    derivation = dict(protocol=protocol, original_identity=native["identity"],
                      original_QA=base_record, original_output_digest=digest(native),
                      new_model_calls=0, original_native_answer_and_context_unchanged=True)
    if native["method"] == "global":
        derivation["status"] = "not_applicable_global_report_context"
        result["mapping_derivation"] = derivation
        return result
    if native["method"] != "local":
        raise ValueError("Explicit original Microsoft local/global query result required")
    context, locations, seen = [], [], {}
    for source in native["native_context"].get("sources", []):
        short_id = str(source["id"])
        unit = session.native_units.get(short_id)
        if unit is None or unit["text"] != source["text"]:
            raise ValueError("Native query source differs from actual indexed text unit")
        if encoding is None:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
        matches, mapped = [], set()
        for doc_id in unit["document_ids"]:
            candidates = session.native_documents.get(str(doc_id))
            if not candidates:
                raise ValueError("Native query document is outside the target-book index")
            for passage in candidates.values():
                for window in native_microsoft_spans.locate_native_text(
                        passage["text"], unit["text"], encoding, size=size, overlap=overlap):
                    quotes = native_microsoft_spans.evidence_intersections(passage, window)
                    evidence_ids = []
                    for quote in quotes:
                        eid = quote["evidence_id"]
                        evidence = session.structure["evidence"][eid]
                        if (evidence["book_id"] != session.book_id or
                                evidence["chunk_id"] != passage["chunk_id"] or
                                evidence["text"][quote["evidence_start"]:quote["evidence_end"]] != quote["text"]):
                            raise ValueError("Token-byte quote does not equal the frozen original S")
                        key = (eid, quote["evidence_start"], quote["evidence_end"], quote["text"])
                        if key in seen:
                            old = context[seen[key]]
                            if short_id not in old["native_source_ids"]:
                                old["native_source_ids"].append(short_id)
                        else:
                            seen[key] = len(context)
                            context.append(dict(book_id=session.book_id, evidence_id=eid, evidence_ids=[eid],
                                                text=quote["text"], chunk_id=passage["chunk_id"],
                                                chapter_id=evidence.get("chapter_id"),
                                                evidence_start=quote["evidence_start"], evidence_end=quote["evidence_end"],
                                                native_source_ids=[short_id], source_kind="original_evidence_fragment",
                                                native_token_window=quote["native_token_window"]))
                        evidence_ids.append(eid)
                        mapped.add(eid)
                    matches.append(dict(chunk_id=passage["chunk_id"], **{k: v for k, v in window.items() if k != "native_text"},
                                        evidence_ids=evidence_ids))
        if not matches:
            raise ValueError("Indexed native source has no exact original token-byte window")
        locations.append(dict(native_source_short_id=short_id, native_text_unit_id=unit["id"],
                              native_document_ids=list(unit["document_ids"]), matches=matches,
                              mapped_evidence_ids=sorted(mapped), resolved=bool(mapped),
                              token_byte_position_resolved=True, ambiguous_location=len(matches) > 1))
    derivation["status"] = "zero_API_original_token_bytes_to_S_derived"
    result.update(identity=digest(dict(original_identity=native["identity"], query_mapping_revision=protocol)),
                  mapping_derivation=derivation, provided_context=context, source_locations=locations,
                  native_citation_mapping=_citation_mapping(native["answer"], locations),
                  source_grounding_evaluable=bool(context), grounding_evaluable=bool(context),
                  original_source_citation_evaluable=bool(locations) and all(row["resolved"] for row in locations),
                  mapping_status="resolved" if all(row["resolved"] for row in locations) else "partial",
                  citation_policy=POLICY["quote"])
    return result
