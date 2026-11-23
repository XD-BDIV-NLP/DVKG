"""计组两本新增 200 题；历史题集和原执行器保持冻结。"""
from __future__ import annotations

from itertools import islice
from pathlib import Path
import time
import traceback
import numpy as np
from filelock import FileLock

from ..manifest import digest, file_record, read_document
from ..contributions import write_immutable
from . import dual_view_tasks as task
from .dual_view_pipeline import candidate_bundles
from .run_repaired_full import atomic
from .run_independent_holdout import (
    LocalHTTPBudget as OriginalBudget, checked, vector, json_call,
    validate_audit_quotes, SOURCE_AUDIT_ID, CANDIDATE_DIGEST,
)

BOOKS=('book_25f718562685','book_49bb8d31ddb9')
TYPES=task.TYPES
DEADLINE=1792950136.907112
POLICY=dict(version='paper_AB200_independent_v1',seed=42,books=2,per_book=100,
    per_type=25,target_count=200,max_candidates_per_type=125,max_local_http_attempts=6000,
    max_output_tokens=1536,duplicate_char_jaccard=.85,duplicate_embedding_cosine=.97,
    source_policy='whole_observed_and_all_previous_accepted_question_chunks_excluded;development_whole_chapters_excluded',
    audit='unchanged_GENERATE_SUPPORT_NECESSITY;verbatim_quotes;all_cited_units_supported;leave_one_chapter_out',
    shortages='explicit_shortage_no_relaxed_evidence_or_supplementing_with_old80',
    release='same_sealed_200_for_all_methods;require_exact_book_type_quotas',
    old80_role='observed_regression_only',human_audited=False,global_slots=4,
    original_deadline=DEADLINE,within_existing_native45000_and_global200000=True)

class LocalHTTPBudget(OriginalBudget):
    def __init__(self,provider,root,identity):
        super().__init__(provider,provider.root.parent/'paper_AB200_budget_20261025','paper_AB200_20261025',cap=6000)
        self.prefix='paper_AB200_20261025/generation/'

def eligible_ids(book,evidence,excluded,development_chapters):
    if any(r['book_id']!=book for r in evidence.values()):
        raise ValueError('Foreign textbook evidence')
    if not set(excluded)<=set(evidence):
        raise ValueError('Historical evidence missing from original textbook')
    chunks={evidence[e]['chunk_id'] for e in excluded}
    return sorted((e for e,r in evidence.items() if r['chunk_id'] not in chunks
        and r['chapter_id'] and r['chapter_id'] not in development_chapters),key=lambda e:digest([42,book,e]))

def sealed_counts(counts):
    if set(counts)!=set(BOOKS) or any(set(row)!=set(TYPES) for row in counts.values()):
        raise ValueError('Exact two textbooks and four question types required')
    if any(type(n)is not int or not 0<=n<=25 for row in counts.values() for n in row.values()):
        raise ValueError('Question type quota exceeded')
    total=sum(sum(row.values()) for row in counts.values())
    return dict(status='sealed_complete' if total==200 else 'sealed_insufficient',accepted_count=total,shortage=200-total)

def load_plan(run_root,registry):
    run_root=Path(run_root)
    audit_path=run_root/'independent_holdout_source_audit'/SOURCE_AUDIT_ID/'candidate_order.json'
    source=read_document(audit_path)
    if digest(source['identity'])!=SOURCE_AUDIT_ID or digest(source['candidates'])!=CANDIDATE_DIGEST:
        raise ValueError('Original source audit changed')
    source_identity=source['identity']
    assets,datasets,observed=(checked(source_identity[k]) for k in ('assets','datasets','observed'))
    known=list(observed['questions']);previous={b:[] for b in BOOKS};history=[]
    for b,row in datasets['books'].items():
        for split in ('development','confirmation'):
            data=checked(row[split]);history.append(row[split]);known.extend(q['question'] for q in data['questions'])
            if b in previous:previous[b].extend(data['questions'])
    for name in ('independent_holdout_128','blind64_20261022_v1'):
        path=run_root/name/'sealed.json'
        if not path.exists():continue
        sealed=read_document(path);history.append(file_record(path));index=checked(sealed['dataset_index'])
        for b,ref in index['books'].items():
            data=checked(ref);history.append(ref);known.extend(q['question'] for q in data['questions'])
            if b in previous:previous[b].extend(data['questions'])
    structures={};bundles={};records={};exclusions={};construction=None
    for book in BOOKS:
        structure=checked(assets['structures'][book]);units=checked(assets['unit_assets'][book]);evidence=structure['evidence']
        if structure['book_id']!=book:raise ValueError('Original book identity changed')
        rows=units['rows'];ids=[r['evidence_ids'][0] for r in rows]
        if len(set(ids))!=len(rows) or set(ids)!=set(evidence):raise ValueError('Incomplete original paragraph units')
        if any(r['book_id']!=book or len(r['evidence_ids'])!=1 or r['text']!=evidence[r['evidence_ids'][0]]['text'] for r in rows):
            raise ValueError('Paragraph units differ from textbook')
        excluded=set(observed['excluded'][book])|{e for q in previous[book] for e in q['evidence_ids']}
        dev=checked(datasets['books'][book]['development'])['questions'];chapters={c for q in dev for c in q['chapter_ids']}
        eligible=eligible_ids(book,evidence,excluded,chapters)
        structures[book]=structure
        bundles[book]={kind:list(islice(candidate_bundles(structure,units,eligible,kind),125)) for kind in TYPES}
        exclusions[book]=dict(previous_questions=len(previous[book]),excluded_source_chunks=sorted({evidence[e]['chunk_id'] for e in excluded}),development_chapters=sorted(chapters),eligible_evidence_ids=eligible)
        records[book]=dict(structure=assets['structures'][book],units=assets['unit_assets'][book],vectors=units['vectors'])
        cohort=next(v for v in assets['cohorts'].values() if book in v['active_books'])
        current=checked(cohort['construction'])
        if construction is not None and checked(construction['files']['assets']['runtime_config'])!=checked(current['files']['assets']['runtime_config']):
            raise ValueError('Frozen model configurations differ')
        construction=current
    contract=dict(policy=POLICY,source_audit=file_record(audit_path),source_records=records,
        prior_datasets=history,exclusions=exclusions,candidate_bundles_digest=digest(bundles),
        known_questions_digest=digest(sorted(set(known))),known_questions_count=len(set(known)),
        method_registry=registry,code=file_record(__file__),
        generation_code=file_record(Path(task.__file__).parent/'run_independent_holdout.py'),
        prompts=dict(generate=task.GENERATE,support=task.SUPPORT,necessity=task.NECESSITY),
        method_scores_read=False,retrieval_results_read=False)
    return dict(identity=digest(contract),contract=contract,books=list(BOOKS),structures=structures,
        bundles=bundles,known_questions=sorted(set(known)),construction=construction,source_scope='paper_AB200_20261025')


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
                        for number, bundle in enumerate(plan["bundles"][book][kind][:125], 1):
                            if len(good) >= 25:
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
                                candidates=number, accepted_current_type=len(good), target_current_type=25,
                                local_http_attempts=budget.count(), accepted_total=total+len(questions)+len(good), raw_text_exported=False))
                        counts[kind] = dict(target=25, accepted=len(good), candidates=attempts, shortage=25-len(good))
                        questions.extend(good)
                    frozen = dict(target_book_id=book, split="independent_confirmation", questions=questions,
                        by_type=counts, target_count=100, accepted_count=len(questions), policy=POLICY,
                        decisions=decisions, source_contract=file_record(root / "contract.json"), human_audited=False)
                    frozen["dataset_id"] = digest(frozen)
                    path = root / "datasets" / (book + ".json")
                    write_immutable(path, frozen)
                    references[book] = file_record(path)
                    total += len(questions)
                quota = sealed_counts({b:{k:checked(ref)["by_type"][k]["accepted"] for k in TYPES} for b,ref in references.items()})
                if quota["accepted_count"] != total:raise ValueError("Sealed question count differs")
                index = root / "datasets/index.json"
                write_immutable(index, dict(books=references, target_count=200, accepted_count=total, policy=POLICY))
                result = dict(status="sealed_complete" if total == 200 else "sealed_insufficient", target_count=200,
                    accepted_count=total, shortage=200-total, rejected_candidates=rejected, dataset_index=file_record(index),
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

