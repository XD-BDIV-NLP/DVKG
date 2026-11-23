"""独立保留题生成；原文、题答和原始收据仅存服务器，不读取方法结果。"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
from itertools import islice
import json
from pathlib import Path
import sqlite3
import time
import traceback

from filelock import FileLock
import numpy as np

from ..contributions import write_immutable
from ..manifest import digest, file_record, read_document
from . import dual_view_tasks as task
from .dual_view_pipeline import candidate_bundles, store_vectors
from .native_provider import AuditedNativeProvider
from .protocol import check_runtime, runtime_for
from .run_repaired_full import atomic, credentials


SOURCE_AUDIT_ID = "00ea5985771806d104f5e35eb8d17af2a19346eec4082e91378a96ea87531637"
CANDIDATE_DIGEST = "eebfdf01773980189e0e71a9eaf041e3e2f710a3eb848456609067f363f19ef6"
LIMITS = dict(max_http_attempts=30000, max_seconds=172800, max_concurrency=4)
POLICY = dict(version="independent_holdout_128_v1", seed=42, books=8, per_book=16, per_type=4,
    target_count=128, max_candidates_per_type=20, max_local_http_attempts=1800,
    max_output_tokens=1536, duplicate_char_jaccard=task.TASK_POLICY["duplicate_char_jaccard"],
    duplicate_embedding_cosine=task.TASK_POLICY["duplicate_embedding_cosine"],
    source_policy="whole_observed_and_previous_question_chunks_excluded;development_whole_chapters_excluded",
    audit="unchanged_GENERATE_SUPPORT_NECESSITY;verbatim_quotes_and_all_cited_units_supported;leave_one_chapter_out",
    format_failure="reject_candidate_with_raw_response_preserved;no_same_body_format_retry",
    shortages="seal_actual_accepted_count_and_explicit_shortage;never_relax_or_expand_candidates",
    preregistration="docs/plans/2026-10-22-complete-course-benchmark.md#新确认集数量预注册与来源可用性",
    release="sealed_common_set_for_all_methods;quality_use_requires_separate_DVKG_development_selection_freeze",
    human_audited=False)


class HoldoutBudgetExceeded(RuntimeError):
    pass


class LocalHTTPBudget:
    def __init__(self, provider, root, identity, cap=1800):
        self.provider, self.root, self.cap = provider, Path(root), cap
        self.prefix = "independent_holdout/" + identity + "/"
        self.lock = FileLock(str(provider.root / ("holdout_scope_" + digest(identity) + ".lock")), timeout=30)

    def stage(self, name):
        return self.prefix + name

    def count(self):
        with closing(sqlite3.connect(self.provider.database, timeout=30)) as db:
            return db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,?)=?",
                              (len(self.prefix), self.prefix)).fetchone()[0]

    def __enter__(self):
        write_immutable(self.root / "http_scope.json", dict(prefix=self.prefix, cap=self.cap,
            accounting="shared_formal_provider_request_started_events;cache_hits_free;all_transport_retries_counted"))
        self.original = self.provider.runtime._record
        def bounded(key, stage, status, value):
            if status != "request_started":
                return self.original(key, stage, status, value)
            if not stage.startswith(self.prefix):
                raise ValueError("Holdout request has no frozen accounting scope")
            # 关闭只读连接后才进入原provider事务，避免跨账本自锁。
            with self.lock:
                if self.count() >= self.cap:
                    raise HoldoutBudgetExceeded("Independent holdout cumulative 1800 HTTP scope exhausted")
                return self.original(key, stage, status, value)
        self.provider.runtime._record = bounded
        return self

    def __exit__(self, *args):
        self.provider.runtime._record = self.original


def checked(record):
    if file_record(record["path"]) != record:
        raise ValueError("Frozen holdout source file changed")
    return read_document(record["path"])


def load_plan(candidate_path, *, expected_identity=SOURCE_AUDIT_ID, expected_candidates=CANDIDATE_DIGEST):
    candidate_path = Path(candidate_path)
    source = read_document(candidate_path)
    identity, candidates = source["identity"], source["candidates"]
    if digest(identity) != expected_identity or candidate_path.parent.name != expected_identity:
        raise ValueError("Frozen source audit identity differs")
    if digest(candidates) != expected_candidates or identity["candidate_digest"] != expected_candidates:
        raise ValueError("Frozen candidate order digest differs")
    if identity["seed"] != 42:
        raise ValueError("Frozen candidate seed must be 42")
    assets, datasets, observed = (checked(identity[k]) for k in ("assets", "datasets", "observed"))
    books = sorted(candidates)
    if len(books) != 8 or set(books) != set(datasets["books"]) or set(books) != set(assets["unit_assets"]):
        raise ValueError("Exactly eight authorized holdout books required")
    structures, bundles, known, source_records, construction, configuration, counts = {}, {}, list(observed["questions"]), {}, None, None, {}
    for book in books:
        structure, units = checked(assets["structures"][book]), checked(assets["unit_assets"][book])
        evidence, audit = structure["evidence"], candidates[book]
        if structure["book_id"] != book or any(r["book_id"] != book for r in evidence.values()):
            raise ValueError("Foreign-book source structure")
        old = [checked(datasets["books"][book][split])["questions"] for split in ("development", "confirmation")]
        if len(old[0]) != 4 or not 37 <= len(old[1]) <= 40:
            raise ValueError("Frozen historical question counts differ")
        counts[book] = [len(v) for v in old]
        previous = [q for group in old for q in group]
        if any(q["target_book_id"] != book for q in previous):
            raise ValueError("Historical question belongs to another book")
        if [hashlib.sha256(q["question"].encode()).hexdigest() for q in previous] != audit["old_question_hashes"]:
            raise ValueError("Historical questions differ from source audit")
        known.extend(q["question"] for q in previous)
        excluded = set(observed["excluded"][book]) | {e for q in previous for e in q["evidence_ids"]}
        excluded_chunks = sorted({evidence[e]["chunk_id"] for e in excluded if e in evidence})
        dev_chapters = sorted({c for q in old[0] for c in q["chapter_ids"]})
        eligible = sorted((e for e, r in evidence.items() if r["chunk_id"] not in excluded_chunks and
            r["chapter_id"] and r["chapter_id"] not in dev_chapters), key=lambda e: digest([42, book, e]))
        if excluded_chunks != audit["excluded_source_chunks"] or dev_chapters != audit["development_chapters"] or eligible != audit["candidate_evidence_ids"]:
            raise ValueError("Candidate order or whole-source exclusion differs")
        row_ids = [r["evidence_ids"][0] for r in units["rows"]]
        if len(set(row_ids)) != len(row_ids) or set(row_ids) != set(evidence):
            raise ValueError("Full original evidence units required")
        for row in units["rows"]:
            if len(row["evidence_ids"]) != 1 or row["book_id"] != book or row["text"] != evidence[row["evidence_ids"][0]]["text"]:
                raise ValueError("Source units differ from original textbook evidence")
        if file_record(units["vectors"]["path"]) != units["vectors"]:
            raise ValueError("Source selection vectors changed")
        structures[book] = structure
        bundles[book] = {kind: list(islice(candidate_bundles(structure, units, eligible, kind), 20)) for kind in task.TYPES}
        source_records[book] = dict(structure=assets["structures"][book], units=assets["unit_assets"][book], vectors=units["vectors"])
        cohort = next(v for v in assets["cohorts"].values() if book in v["active_books"])
        current = checked(cohort["construction"])
        config = checked(current["files"]["assets"]["runtime_config"])
        check_runtime(config)
        if configuration is not None and configuration != config:
            raise ValueError("All books must use the same frozen model runtime")
        construction, configuration = current, config
    if sum(v[1] for v in counts.values()) != 313:
        raise ValueError("All 32 development and 313 historical confirmation questions required")
    known = sorted(set(known))
    files = {name: file_record(Path(__file__).parent / name) for name in
        ("run_independent_holdout.py", "dual_view_tasks.py", "dual_view_pipeline.py", "native_provider.py", "protocol.py", "run_repaired_full.py")}
    from domain_ckg_workspace.ckg_incremental import runtime
    files["model_runtime.py"] = file_record(runtime.__file__)
    for name in ("manifest.py", "contributions.py"):
        files[name] = file_record(Path(__file__).parents[1] / name)
    contract = dict(policy=POLICY, source_audit=file_record(candidate_path), source_identity=identity,
        source_records=source_records, candidate_bundles_digest=digest(bundles), known_questions_digest=digest(known),
        known_questions_count=len(known), historical_counts=counts, runtime_config=configuration, code=files,
        prompts=dict(generate=task.GENERATE, support=task.SUPPORT, necessity=task.NECESSITY), global_limits=LIMITS,
        method_scores_read=False, retrieval_results_read=False)
    return dict(identity=digest(contract), contract=contract, books=books, structures=structures, bundles=bundles,
                known_questions=known, construction=construction, source_scope=expected_identity)


def validate_audit_quotes(audit, evidence, book, *, required=()):
    quotes = audit.get("quotes")
    if not isinstance(quotes, list):
        raise ValueError("Independent audit quote list missing")
    found = set()
    for quote in quotes:
        if not isinstance(quote, dict):
            raise ValueError("Invalid independent audit quote")
        eid = quote.get("evidence_id")
        row = evidence.get(eid)
        text = quote.get("quote")
        if row is None or row["book_id"] != book or not row.get("chapter_id"):
            raise ValueError("Independent audit cites unavailable book or chapter")
        if not isinstance(text, str) or len(text.strip()) < 6 or text not in row["text"]:
            raise ValueError("Independent audit quote is not verbatim original evidence")
        found.add(eid)
    if not set(required) <= found:
        raise ValueError("Independent audit does not support every cited source unit")


def vector(provider, texts, root, stage):
    root = Path(root)
    path = root / (digest(texts) + ".npy")
    write_immutable(path.with_suffix(".input.json"), dict(texts_digest=digest(texts), count=len(texts)))
    meta = path.with_suffix(".npy.meta.json")
    if meta.exists():
        record = read_document(meta)["vectors"]
        if file_record(path) != record:
            raise ValueError("Holdout question vectors changed")
        checked(read_document(path.with_suffix(".receipts.json"))["provider_receipt"])
    else:
        values = np.asarray(provider.embed(texts, stage=stage), dtype=np.float32)
        if values.shape != (len(texts), 1024) or not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) <= 0).any():
            raise ValueError("Invalid frozen holdout BGE-M3 vectors")
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        body = dict(model=provider.runtime.config["embedding"]["model"], input=texts, encoding_format="float")
        key = digest(dict(endpoint="/embeddings", body=body, config=provider.runtime.config))
        write_immutable(path.with_suffix(".receipts.json"), dict(request_id=key,
            provider_receipt=file_record(provider.root / "receipts" / (key + ".json"))))
        record = store_vectors(path, values)
    values = np.load(record["path"], allow_pickle=False)
    if values.shape != (len(texts), 1024) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
        raise ValueError("Invalid saved holdout question vectors")
    return values, record


def json_call(provider, prompt, payload, folder, stage):
    path = Path(folder) / "response.json"
    identity = digest(dict(prompt=prompt, payload=payload, max_tokens=1536, response_format={"type": "json_object"}))
    if path.exists():
        saved = read_document(path)
        if saved["input_identity"] != identity:
            raise ValueError("Holdout model checkpoint input differs")
        checked(saved["provider_receipt"])
    else:
        messages = [dict(role="system", content=prompt), dict(role="user", content=json.dumps(payload, ensure_ascii=False))]
        raw = provider.chat_response(messages,
            max_tokens=1536, response_format={"type": "json_object"}, stage=stage)
        body = dict(model=provider.runtime.config["llm"]["model"], messages=messages, temperature=0,
            seed=42, enable_thinking=False, max_tokens=1536, response_format={"type": "json_object"})
        key = digest(dict(endpoint="/chat/completions", body=body, config=provider.runtime.config))
        saved = dict(input_identity=identity, response=raw, request_id=key,
            provider_receipt=file_record(provider.root / "receipts" / (key + ".json")))
        write_immutable(path, saved)
    choices = saved["response"].get("choices")
    if not isinstance(choices, list) or not choices or choices[0].get("finish_reason") != "stop":
        raise ValueError("Holdout completion truncated or incomplete; original response preserved")
    value = json.loads(choices[0]["message"]["content"])
    if not isinstance(value, dict):
        raise ValueError("Holdout completion must be a JSON object")
    return value


def generate(plan, root, provider):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(root / "operation.lock"), timeout=0):
            write_immutable(root / "contract.json", dict(identity=plan["identity"], **plan["contract"]))
            if (root / "sealed.json").exists():
                sealed = read_document(root / "sealed.json")
                index = checked(sealed["dataset_index"])
                for record in index["books"].values():
                    checked(record)
                return sealed
            with LocalHTTPBudget(provider, root, plan.get("source_scope", SOURCE_AUDIT_ID)) as budget:
                known = list(plan["known_questions"])
                matrices = [vector(provider, known[first:first+16], root / "vectors", budget.stage("history_embeddings"))[0]
                    for first in range(0, len(known), 16)]
                vectors = list(np.concatenate(matrices)) if matrices else []
                references, total, rejected = {}, 0, 0
                for book in plan["books"]:
                    questions, counts, decisions = [], {}, []
                    for kind in task.TYPES:
                        good, attempts = [], 0
                        for number, bundle in enumerate(plan["bundles"][book][kind][:20], 1):
                            if len(good) >= 4:
                                break
                            provider.raise_if_fatal()
                            attempts = number
                            folder = root / "candidates" / book / kind / (str(number).zfill(3) + "_" + digest(bundle)[:16])
                            path = folder / "decision.json"
                            decision_identity = digest([plan["identity"], book, kind, number, bundle])
                            evidence = {e: plan["structures"][book]["evidence"][e] for e in bundle}
                            sources = [dict(evidence_id=e, chapter_id=r["chapter_id"], text=r["text"]) for e, r in evidence.items()]
                            if path.exists():
                                decision = read_document(path)
                                if decision["input_identity"] != decision_identity or decision["decision_digest"] != digest({k:v for k,v in decision.items() if k != "decision_digest"}):
                                    raise ValueError("Holdout candidate decision changed")
                            else:
                                try:
                                    value = json_call(provider, task.GENERATE, dict(target_book_id=book, type=kind, sources=sources), folder / "generate", budget.stage("generate"))
                                    if value.get("skip") is True:
                                        raise ValueError("Generator declined unsupported candidate")
                                    q = task.validate_task(value, evidence, book, kind)
                                    if task.near_duplicate(q["question"], known):
                                        raise ValueError("Character near duplicate of historical or accepted question")
                                    matrix, record = vector(provider, [q["question"]], root / "vectors", budget.stage("question_embedding"))
                                    if vectors and float(np.max(np.asarray(vectors) @ matrix[0])) >= task.TASK_POLICY["duplicate_embedding_cosine"]:
                                        raise ValueError("Semantic near duplicate of historical or accepted question")
                                    audit = json_call(provider, task.SUPPORT, dict(target_book_id=book, type=kind, task=q, sources=sources), folder / "support", budget.stage("support"))
                                    if audit.get("supported") is not True or audit.get("coherent") is not True or kind == "conditions" and audit.get("explicit_conditions") is not True:
                                        raise ValueError("Independent source support audit rejected")
                                    validate_audit_quotes(audit, evidence, book, required=q["evidence_ids"])
                                    necessary = {}
                                    if kind == "cross_chapter":
                                        for chapter in q["chapter_ids"]:
                                            partial = {e:r for e,r in evidence.items() if r["chapter_id"] != chapter}
                                            necessary[chapter] = json_call(provider, task.NECESSITY,
                                                dict(question=q["question"], sources=[s for s in sources if s["chapter_id"] != chapter]),
                                                folder / "leave_out" / digest(chapter), budget.stage("necessity"))
                                            validate_audit_quotes(necessary[chapter], partial, book)
                                            if necessary[chapter].get("answerable") is not False:
                                                raise ValueError("Cross-chapter evidence not independently necessary")
                                    decision = dict(accepted=True, question=q, vector=record, source_audit=audit, necessity=necessary)
                                except (ValueError, KeyError, TypeError) as exc:
                                    decision = dict(accepted=False, error_type=type(exc).__name__, reason=str(exc))
                                decision.update(input_identity=decision_identity, source_bundle=bundle,
                                    model_checkpoints=[file_record(p) for p in sorted(folder.rglob("response.json"))])
                                decision["decision_digest"] = digest(decision)
                                write_immutable(path, decision)
                            for record in decision["model_checkpoints"]:
                                checked(record)
                            decisions.append(file_record(path))
                            if decision["accepted"]:
                                q = task.validate_task(decision["question"], evidence, book, kind)
                                if q != decision["question"] or file_record(decision["vector"]["path"]) != decision["vector"]:
                                    raise ValueError("Accepted holdout task or vector changed")
                                good.append(q)
                                known.append(q["question"])
                                vectors.append(np.load(decision["vector"]["path"], allow_pickle=False)[0])
                            else:
                                rejected += 1
                            atomic(root / "state.json", dict(status="generating", book_id=book, type=kind,
                                candidates=number, accepted_current_type=len(good), target_current_type=4,
                                local_http_attempts=budget.count(), raw_text_exported=False))
                        counts[kind] = dict(target=4, accepted=len(good), candidates=attempts, shortage=4-len(good))
                        questions.extend(good)
                    frozen = dict(target_book_id=book, split="independent_confirmation", questions=questions,
                        by_type=counts, target_count=16, accepted_count=len(questions), policy=POLICY,
                        decisions=decisions, source_contract=file_record(root / "contract.json"), human_audited=False)
                    frozen["dataset_id"] = digest(frozen)
                    path = root / "datasets" / (book + ".json")
                    write_immutable(path, frozen)
                    references[book] = file_record(path)
                    total += len(questions)
                index = root / "datasets/index.json"
                write_immutable(index, dict(books=references, target_count=128, accepted_count=total, policy=POLICY))
                result = dict(status="sealed_complete" if total == 128 else "sealed_insufficient", target_count=128,
                    accepted_count=total, shortage=128-total, rejected_candidates=rejected, dataset_index=file_record(index),
                    contract=file_record(root / "contract.json"), local_http_attempts=budget.count(),
                    released_for_quality_evaluation=False, release_policy=POLICY["release"], raw_text_exported=False,
                    method_scores_read=False, human_audited=False)
                write_immutable(root / "sealed.json", result)
                atomic(root / "state.json", result)
                return result
    except Exception as exc:
        error = root / ("interruption_" + str(time.time_ns()) + ".log")
        error.write_text(traceback.format_exc(), encoding="utf-8")
        atomic(root / "state.json", dict(status="interrupted_preserved_checkpoints", error_type=type(exc).__name__,
            error_record=file_record(error), raw_text_exported=False))
        raise
    finally:
        provider.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    candidate = args.run_root / "independent_holdout_source_audit" / SOURCE_AUDIT_ID / "candidate_order.json"
    plan = load_plan(candidate)
    if read_document(args.run_root / "formal_provider/limits.json")["limits"] != LIMITS:
        raise ValueError("Existing 30000/48h/4-slot formal budget must be reused without reset")
    if args.check_only:
        print(json.dumps(dict(status="checked_no_api", target_count=128, raw_text_exported=False, model_calls=0,
            candidate_digest=CANDIDATE_DIGEST, identity=plan["identity"], known_question_count=len(plan["known_questions"]),
            candidates={b:{k:len(v) for k,v in plan["bundles"][b].items()} for b in plan["books"]})))
        return 0
    root = args.run_root / "independent_holdout_128"
    credentials()
    provider = AuditedNativeProvider(runtime_for(plan["construction"], root / "transport"), args.run_root / "formal_provider", **LIMITS)
    try:
        result = generate(plan, root, provider)
    except Exception as exc:
        print(json.dumps(dict(status="interrupted_preserved_checkpoints", error_type=type(exc).__name__, raw_text_exported=False)))
        return 2
    print(json.dumps(result))
    return 0 if result["status"] == "sealed_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
