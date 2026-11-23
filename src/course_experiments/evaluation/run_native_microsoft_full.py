"""GraphRAG单书完整索引及local主轨/global辅助轨；仅由总控启动真实调用。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from filelock import FileLock

from ..contributions import write_immutable
from ..manifest import digest, file_record, read_document
from . import run_native_full as shared
from .native_corpus import native_passages
from .native_microsoft_full import open_microsoft, query_microsoft
from .native_microsoft_graphrag import OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, NATIVE_SETTINGS, DISCLOSURE
from .native_provider import AuditedNativeProvider, NativeAccountUnavailable, NativeBudgetExceeded
from .protocol import check_runtime, runtime_for
from .rag import evidence_recall, score_answer
from .run_native_rag import PARENT, checked
from .run_repaired_full import atomic, credentials


TRACKS = ("local", "global")
POLICY = dict(shared.POLICY, version="microsoft_native_full_book_v1",
              microsoft_local="primary_official_local_query_single_paragraph",
              microsoft_global="auxiliary_official_global_map_reduce_single_paragraph",
              source_grounding="global_reports_only_not_applicable_no_zero_penalty;local_exact_original_sources",
              index_count="one_Standard_index_per_target_book_shared_by_local_global_and_both_splits",
              native_settings=NATIVE_SETTINGS, native_disclosure=DISCLOSURE)


def summarize(questions, records, track):
    expected = {q["question_id"] for q in questions}
    if len(expected) != len(questions) or set(records) - expected:
        raise ValueError("GraphRAG question identifiers differ from frozen denominator")
    completed = [r for r in records.values() if r["status"] == "completed"]
    n = len(questions)
    metrics = {key: sum(r["scores"][key] for r in completed) / n if n else None for key in ("f1", "em")}
    metrics["machine_judge_mean"] = sum(r["machine_judge"]["score"] for r in completed) / n if n else None
    evaluable = track == "local" and all(r.get("source_grounding_evaluable", True) for r in completed)
    metrics["machine_grounding_mean"] = sum(r["machine_judge"]["grounding"] for r in completed) / n if evaluable and n else None
    metrics.update(citation_correctness=None, citation_count=None)
    return dict(status="completed" if n and len(completed) == n else "partial", question_count=n,
                completed_count=len(completed), failed_count=sum(r["status"] == "failed" for r in records.values()),
                pending_count=n-len(records), denominator="all_frozen_questions_incomplete_zero_no_silent_dropping",
                metrics=metrics, source_grounding_evaluable=evaluable,
                grounding_denominator="all_frozen_questions" if evaluable else "not_applicable_source_context_unavailable",
                citation_policy=shared.POLICY["citations"], human_audited=False)


def summary_for(plan, records, root, identity, statistics):
    tracks = {}
    for track in TRACKS:
        splits = {}
        for split, dataset in plan["datasets"].items():
            splits[split] = summarize(dataset["questions"], records[track][split], track)
            splits[split]["by_type"] = {kind: summarize([q for q in dataset["questions"] if q["type"] == kind],
                {key: row for key, row in records[track][split].items() if row["question_type"] == kind}, track)
                for kind in sorted({q["type"] for q in dataset["questions"]})}
        tracks[track] = dict(status="completed" if all(v["status"] == "completed" for v in splits.values()) else "partial",
                             comparison_role="primary" if track == "local" else "auxiliary", splits=splits,
                             context_kind="native_local_original_sources" if track == "local" else "community_reports")
    return dict(status="completed" if all(v["status"] == "completed" for v in tracks.values()) else "partial",
                method="graphrag", book_id=plan["book"], root=str(root), job_identity=identity, tracks=tracks,
                cohort_memberships=plan["cohort_memberships"], statistics=statistics,
                statistics_scope="shared_formal_provider_global_not_per_method", raw_text_exported=False,
                request_attribution="one_index_and_per_track_question_requests_jsonl_receipt_ids",
                judge_version=shared.POLICY["judge_version"], full_eight_book_benchmark_completed=False)


def _normalize_context(native):
    return [{**row, "evidence_ids": [row["evidence_id"]] if row.get("evidence_id") is not None else []}
            for row in native["provided_context"]]


async def run_book(plan, run_root, provider, *, opener=None, query_fn=None, judge_fn=None):
    opener, query_fn, judge_fn = opener or open_microsoft, query_fn or query_microsoft, judge_fn or shared.blind_judge
    identity, index_identity = digest(plan["contract"]), digest(plan["index_contract"])
    root = Path(run_root) / "native_full/graphrag" / plan["book"] / identity[:16]
    index_root = Path(run_root) / "native_indices/graphrag" / plan["book"] / index_identity[:16]
    records = {track: {split: {} for split in plan["datasets"]} for track in TRACKS}
    session, tracker, result = None, None, None
    def state(phase, status="running", **extra):
        atomic(root / "state.json", dict(status=status, phase=phase, pid=os.getpid(), updated_unix=time.time(),
            job_identity=identity, summary=summary_for(plan, records, root, identity, provider.statistics()), **extra))
    try:
        root.mkdir(parents=True, exist_ok=True)
        write_immutable(root / "contract.json", dict(job_identity=identity, **plan["contract"]))
        for track in TRACKS:
            for split, dataset in plan["datasets"].items():
                for q in dataset["questions"]:
                    path = root / track / split / digest(q["question_id"]) / "record.json"
                    if path.exists():
                        row = shared._read_checkpoint(path, identity)
                        if row["question_id"] != q["question_id"] or row["question_type"] != q["type"] or row["track"] != track:
                            raise ValueError("GraphRAG question checkpoint differs from frozen task")
                        records[track][split][q["question_id"]] = row
        provider.raise_if_fatal()
        current = summary_for(plan, records, root, identity, provider.statistics())
        if current["status"] == "completed":
            result = current
        else:
            if hasattr(provider, "request_executor"):
                tracker = shared.ReceiptTracker(provider)
            index_root.mkdir(parents=True, exist_ok=True)
            write_immutable(index_root / "contract.json", dict(index_identity=index_identity, **plan["index_contract"]))
            with FileLock(str(index_root / "operation.lock"), timeout=172800):
                state("native_index")
                with tracker.scope(index_root / "requests.jsonl") if tracker else shared._no_scope():
                    session = await opener(plan, index_root / "storage", provider)
                provider.raise_if_fatal()
                receipt_path = index_root / "index_receipt.json"
                if receipt_path.exists():
                    shared._read_checkpoint(receipt_path, index_identity)
                else:
                    shared._checkpoint(receipt_path, dict(native_index=session.index_receipt,
                        requests_path=str(index_root / "requests.jsonl"), completed_unix=time.time()), index_identity)
                for track in TRACKS:
                    for split, dataset in plan["datasets"].items():
                        for question in dataset["questions"]:
                            qid = question["question_id"]
                            if records[track][split].get(qid, {}).get("status") == "completed":
                                continue
                            provider.raise_if_fatal()
                            qroot = root / track / split / digest(qid)
                            state("native_question", track=track, split=split, question_id=qid)
                            try:
                                with tracker.scope(qroot / "requests.jsonl") if tracker else shared._no_scope():
                                    qa_path = qroot / "native_qa.json"
                                    if qa_path.exists():
                                        native = shared._read_checkpoint(qa_path, identity)
                                    else:
                                        native = await query_fn(session, plan, question, method=track, output_dir=qroot)
                                        native["provided_context"] = _normalize_context(native)
                                        native = shared._checkpoint(qa_path, native, identity)
                                    if not isinstance(native["answer"], str) or not native["answer"].strip():
                                        raise ValueError("Native Microsoft answer is empty")
                                    judgment = dict(judge_fn(provider, question, native, qroot, identity))
                                provider.raise_if_fatal()
                                evaluable = bool(native.get("source_grounding_evaluable"))
                                # 同一盲评保留correctness；无原文能力的报告轨不使用raw grounding数值。
                                if not evaluable:
                                    judgment["grounding"] = None
                                    judgment["grounding_status"] = "not_applicable_source_context_unavailable"
                                judgment["source_grounding_evaluable"] = evaluable
                                scores = dict(score_answer(native["answer"], question["answer"]),
                                    evidence_recall=evidence_recall(question, native["provided_context"]) if evaluable else None,
                                    citation_count=None, citation_correctness=None)
                                row = dict(status="completed", question_id=qid, question_type=question["type"], split=split,
                                    track=track, comparison_role="primary" if track == "local" else "auxiliary",
                                    target_book_id=plan["book"], answer=native["answer"], provided_context=native["provided_context"],
                                    native_result=native["native_result"], native_context=native["native_context"],
                                    source_locations=native["source_locations"], native_citation_mapping=native["native_citation_mapping"],
                                    context_kind=native["context_kind"], finish_metadata=native["finish_metadata"],
                                    source_grounding_evaluable=evaluable, grounding_evaluable=evaluable,
                                    machine_judge=judgment, scores=scores, human_audited=False,
                                    receipts_path=str(qroot / "requests.jsonl"), completed_unix=time.time())
                            except Exception as exc:
                                row = dict(status="failed", question_id=qid, question_type=question["type"], split=split,
                                    track=track, target_book_id=plan["book"], **shared._error(qroot, exc), human_audited=False)
                                records[track][split][qid] = shared._checkpoint(qroot / "record.json", row, identity, immutable=False)
                                provider.raise_if_fatal()
                                if isinstance(exc, (NativeAccountUnavailable, NativeBudgetExceeded)):
                                    raise
                                continue
                            records[track][split][qid] = shared._checkpoint(qroot / "record.json", row, identity, immutable=False)
                            state("question_checkpointed", track=track, split=split, question_id=qid)
                result = summary_for(plan, records, root, identity, provider.statistics())
        provider.raise_if_fatal()
    except Exception as exc:
        state("halted" if isinstance(exc, (NativeAccountUnavailable, NativeBudgetExceeded)) else "failed",
              status="halted" if isinstance(exc, (NativeAccountUnavailable, NativeBudgetExceeded)) else "failed", **shared._error(root, exc))
        raise
    finally:
        try:
            if session is not None:
                await session.close()
            if result is not None:
                provider.raise_if_fatal()
                complete_path = root / "complete.json"
                if complete_path.exists():
                    previous = read_document(complete_path)
                    if {k:v for k,v in previous.items() if k != "statistics"} != {k:v for k,v in result.items() if k != "statistics"}:
                        raise ValueError("Completed Microsoft summary differs from question checkpoints")
                    result = previous
                atomic(root / "result.json", result)
                if result["status"] == "completed":
                    write_immutable(complete_path, result)
                state("finished", status=result["status"])
        except Exception as exc:
            state("finalization_failed", status="failed", **shared._error(root, exc))
            raise
        finally:
            if tracker is not None:
                tracker.close()
            provider.close()
    return result


def prepare_plan(run_root, book, *, parent=PARENT):
    parent, run_root = Path(parent), Path(run_root)
    asset_record, dataset_record = file_record(parent / "assets/result.json"), file_record(parent / "datasets/index.json")
    assets, index = checked(asset_record), checked(dataset_record)
    books = sorted(index["books"])
    if len(books) != 8 or set(books) != set(assets["unit_assets"]) or book not in books:
        raise ValueError("Exact eight authorized books required")
    counts, selected = {}, None
    for candidate in books:
        datasets = {split: checked(index["books"][candidate][split]) for split in ("development", "confirmation")}
        counts[candidate] = {split: len(value["questions"]) for split, value in datasets.items()}
        ids = [q["question_id"] for dataset in datasets.values() for q in dataset["questions"]]
        if counts[candidate]["development"] != 4 or not 37 <= counts[candidate]["confirmation"] <= 40 or len(set(ids)) != len(ids):
            raise ValueError("Frozen Microsoft full question counts or identifiers differ")
        if any(q["target_book_id"] != candidate for dataset in datasets.values() for q in dataset["questions"]):
            raise ValueError("Foreign-book question in frozen dataset")
        if candidate == book:
            selected = datasets
    if sum(c["confirmation"] for c in counts.values()) != 313:
        raise ValueError("All 313 confirmation and 32 development questions required per track")
    units, structure = checked(assets["unit_assets"][book]), checked(assets["structures"][book])
    if {eid for row in units["rows"] for eid in row["evidence_ids"]} != set(structure["evidence"]):
        raise ValueError("Full Microsoft corpus lacks original evidence")
    passages, audit = native_passages(units["rows"], structure, book)
    if any(not set(q["evidence_ids"]) <= set(structure["evidence"]) for dataset in selected.values() for q in dataset["questions"]):
        raise ValueError("Question support outside target-book corpus")
    memberships = sorted(c for c, value in assets["cohorts"].items() if book in value["active_books"])
    if not memberships:
        raise ValueError("Frozen construction provenance missing")
    construction_record = assets["cohorts"][memberships[0]]["construction"]
    construction = checked(construction_record)
    config_record = construction["files"]["assets"]["runtime_config"]
    config = checked(config_record)
    check_runtime(config)
    source = (run_root / "sources/MicrosoftGraphRAG").resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != OFFICIAL_COMMIT or dirty:
        raise ValueError("Fixed unmodified official Microsoft source required")
    source_record = dict(path=str(source), commit=OFFICIAL_COMMIT, repository=OFFICIAL_REPOSITORY)
    names = ["run_native_microsoft_full.py", "native_microsoft_full.py", "native_microsoft_graphrag.py",
             "run_native_full.py", "native_provider.py", "native_corpus.py", "run_native_rag.py", "protocol.py", "run_repaired_full.py", "rag.py"]
    code = {name: file_record(Path(__file__).parent / name) for name in names}
    from domain_ckg_workspace.ckg_incremental import runtime as original_runtime
    code["model_runtime.py"] = file_record(original_runtime.__file__)
    code["manifest.py"] = file_record(Path(__file__).parents[1] / "manifest.py")
    code["contributions.py"] = file_record(Path(__file__).parents[1] / "contributions.py")
    index_contract = dict(method="graphrag", book=book, source=source_record,
        source_units_sha256=assets["unit_assets"][book]["sha256"], structure_sha256=assets["structures"][book]["sha256"],
        corpus_audit=audit, runtime_config=config, native_settings=NATIVE_SETTINGS, max_tokens=4096, community_level=0,
        implementation={key: value["sha256"] for key, value in code.items()})
    contract = dict(format_version=1, method="graphrag", book=book, asset_manifest=asset_record,
        dataset_index=dataset_record, datasets=index["books"][book], global_question_counts=counts,
        source_units=assets["unit_assets"][book], structure=assets["structures"][book],
        construction=construction_record, runtime_config_record=config_record, runtime_config=config,
        source=source_record, code=code, policy=POLICY, judge_prompt=shared.JUDGE, provider_limits=shared.LIMITS,
        corpus_audit=audit, cohort_memberships=memberships, index_identity=digest(index_contract))
    return dict(method="graphrag", book=book, passages=passages, passage_audit=audit, structure=structure,
                datasets=selected, source=source, contract=contract, index_contract=index_contract,
                cohort_memberships=memberships, allowed_books=books, construction=construction,
                microsoft_max_tokens=4096, microsoft_community_level=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--book", required=True)
    args = parser.parse_args()
    plan = prepare_plan(args.run_root, args.book)
    if read_document(args.run_root / "formal_provider/limits.json")["limits"] != shared.LIMITS:
        raise ValueError("Reuse frozen formal_provider limits/account; budgets must never be reset")
    identity = digest(plan["contract"])
    transport = args.run_root / "native_full/graphrag" / args.book / identity[:16] / "transport"
    credentials()
    provider = AuditedNativeProvider(runtime_for(plan["construction"], transport), args.run_root / "formal_provider", **shared.LIMITS)
    result = asyncio.run(run_book(plan, args.run_root, provider))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
