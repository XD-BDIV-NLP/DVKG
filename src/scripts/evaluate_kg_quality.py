"""Evaluate extracted concept KG quality.

The evaluator has three layers:
- rule checks for traceability, evidence, schema, and endpoint consistency;
- graph-theoretic metrics for structural health;
- optional LLM/PyKEEN evaluators for factual judgement and link prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, extract_json_payload, load_json, load_yaml, log, normalize_key, normalize_whitespace, set_log_file, write_json


REPORT_FILE = "kg_quality_report.json"
ISSUES_FILE = "kg_quality_issues.csv"
RISK_RELATIONS_FILE = "kg_quality_risk_relations.csv"
LOG_FILE = "kg_quality_eval.log"
LLM_JUDGEMENTS_FILE = "kg_quality_llm_judgements.jsonl"
PYKEEN_TRIPLES_FILE = "pykeen_triples.tsv"
PYKEEN_METRICS_FILE = "pykeen_metrics.json"
RAKG_ENTITY_FIDELITY_FILE = "rakg_entity_fidelity_judgements.jsonl"
OUTPUT_FILE_NAMES = {
    REPORT_FILE,
    ISSUES_FILE,
    RISK_RELATIONS_FILE,
    LOG_FILE,
    LLM_JUDGEMENTS_FILE,
    PYKEEN_TRIPLES_FILE,
    PYKEEN_METRICS_FILE,
    RAKG_ENTITY_FIDELITY_FILE,
}


@dataclass
class EvaluationInputs:
    output_dir: Path
    concept_kg: dict[str, Any]
    doc_nodes: dict[str, dict[str, Any]]
    schema: dict[str, Any]


class OpenAICompatibleJudgeClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model: str,
        timeout_seconds: int = 90,
        retries: int = 2,
        temperature: float = 0,
        response_format_json: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.temperature = temperature
        self.response_format_json = response_format_json

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable `{self.api_key_env}` is required for LLM judge requests.")
        if not self.model:
            raise RuntimeError("A judge model is required. Use --judge-model or configure llm.model in settings.yaml.")
        self.api_key = api_key

    def evaluate_relation(self, relation: dict[str, Any], source_context: str) -> dict[str, Any]:
        prompt = self._render_prompt(relation, source_context)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict knowledge graph quality evaluator. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        raw = self._post_chat_completion(payload)
        judged = extract_json_payload(raw)
        if not isinstance(judged, dict):
            raise RuntimeError("Judge response must be a JSON object.")
        return {
            "verdict": str(judged.get("verdict") or "").strip(),
            "is_accurate": bool(judged.get("is_accurate", False)),
            "score": _safe_float(judged.get("score"), default=0.0),
            "error_type": str(judged.get("error_type") or "").strip(),
            "reason": normalize_whitespace(str(judged.get("reason") or "")),
            "suggested_fix": judged.get("suggested_fix"),
        }

    def evaluate_entity(self, entity: dict[str, Any], source_context: str) -> dict[str, Any]:
        prompt = self._render_entity_prompt(entity, source_context)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict knowledge graph entity fidelity evaluator. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        raw = self._post_chat_completion(payload)
        judged = extract_json_payload(raw)
        if not isinstance(judged, dict):
            raise RuntimeError("Entity judge response must be a JSON object.")
        return {
            "verdict": str(judged.get("verdict") or "").strip(),
            "is_faithful": bool(judged.get("is_faithful", False)),
            "score": _safe_float(judged.get("score"), default=0.0),
            "error_type": str(judged.get("error_type") or "").strip(),
            "reason": normalize_whitespace(str(judged.get("reason") or "")),
            "suggested_fix": judged.get("suggested_fix"),
        }

    def _post_chat_completion(self, payload: dict[str, Any]) -> str:
        last_error: Exception | None = None
        request_payload = dict(payload)
        tried_without_response_format = "response_format" not in request_payload
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                url=f"{self.base_url}/chat/completions",
                data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                    "Connection": "close",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                    return response_payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                last_error = self._format_http_error(exc)
                if (
                    not tried_without_response_format
                    and "response_format" in request_payload
                    and getattr(exc, "code", None) in {400, 404, 422}
                ):
                    request_payload = dict(request_payload)
                    request_payload.pop("response_format", None)
                    tried_without_response_format = True
                    log("llm_judge", "Provider rejected JSON response_format; retrying once without it.")
                    continue
                if attempt >= self.retries:
                    break
                time.sleep(1.5 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM judge request failed: {last_error}")

    def _format_http_error(self, exc: urllib.error.HTTPError) -> RuntimeError:
        response_body = ""
        try:
            if exc.fp is not None:
                response_body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:  # noqa: BLE001
            response_body = ""
        suffix = f": {response_body}" if response_body else f": {exc.reason}"
        return RuntimeError(f"HTTP {getattr(exc, 'code', None)} from {self.base_url}{suffix}")

    def _render_prompt(self, relation: dict[str, Any], source_context: str) -> str:
        evidence = "\n".join(f"- {item}" for item in relation.get("evidences", []) if item)
        return f"""
请基于给定上下文，严格评测知识图谱三元组质量。

【上下文文本】
{source_context}

【待评测三元组】
头实体: {relation.get("head", "")}
关系: {relation.get("relation", "")}
尾实体: {relation.get("tail", "")}

【抽取证据】
{evidence or "(无)"}

请只依据上下文和证据判断，不使用外部知识。重点检查：
1. 三元组是否被上下文明确支持；
2. 关系类型是否合适；
3. head/tail 方向是否正确；
4. evidence 是否能支撑该三元组。

请输出 JSON，字段固定为：
{{
  "verdict": "accurate | partially_supported | unsupported | wrong_relation | wrong_direction | ambiguous",
  "is_accurate": true,
  "score": 0.0,
  "error_type": "",
  "reason": "简短中文理由",
  "suggested_fix": null
}}
score 范围为 0 到 1。
""".strip()


    def _render_entity_prompt(self, entity: dict[str, Any], source_context: str) -> str:
        definitions = "\n".join(f"- {item}" for item in entity.get("merged_definitions", []) if item)
        aliases = ", ".join(str(item) for item in entity.get("aliases", []) if item)
        return f"""
Please evaluate whether the extracted knowledge graph entity is faithful to the given source context.

Source context:
{source_context}

Extracted entity:
Name: {entity.get("canonical_name", "")}
Type: {entity.get("type", "")}
Aliases: {aliases or "(none)"}
Definitions:
{definitions or "(none)"}

Judge only from the source context. Do not use outside knowledge.
Return JSON with exactly these fields:
{{
  "verdict": "faithful | partially_supported | unsupported | wrong_type | hallucinated | ambiguous",
  "is_faithful": true,
  "score": 0.0,
  "error_type": "",
  "reason": "brief reason",
  "suggested_fix": null
}}
score must be between 0 and 1.
""".strip()


# CLI and orchestration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the quality of an extracted concept knowledge graph.")
    parser.add_argument("--kg-output-dir", required=True, type=Path, help="Directory containing concept_kg.json.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=PROJECT_ROOT / "config" / "schema.yaml",
        help="Schema YAML defining allowed entity and relation types.",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=PROJECT_ROOT / "config" / "settings.yaml",
        help="Settings YAML used for optional LLM judge defaults.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for evaluation outputs. Defaults to outputs/<kg-name>_quality_<timestamp>.",
    )
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Allow writing into an existing report directory that already contains evaluation outputs.",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Maximum relations to send to the optional LLM judge. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--run-llm-judge", action="store_true", help="Run source-grounded LLM-as-judge evaluation.")
    parser.add_argument("--judge-base-url", default="", help="Optional judge API base URL override.")
    parser.add_argument("--judge-api-key-env", default="", help="Optional judge API key environment variable override.")
    parser.add_argument("--judge-model", default="", help="Optional judge model override.")
    parser.add_argument("--judge-timeout-seconds", type=int, default=90, help="Timeout for each judge request.")
    parser.add_argument("--judge-retries", type=int, default=2, help="Retries for each judge request.")
    parser.add_argument(
        "--judge-response-format-json",
        dest="judge_response_format_json",
        action="store_true",
        default=None,
        help="Force OpenAI JSON response_format for LLM judge requests.",
    )
    parser.add_argument(
        "--no-judge-response-format-json",
        dest="judge_response_format_json",
        action="store_false",
        help="Do not send OpenAI JSON response_format for LLM judge requests.",
    )
    parser.add_argument("--context-chars", type=int, default=5000, help="Maximum source context characters per judged relation.")
    parser.add_argument(
        "--run-rakg-entity-fidelity",
        action="store_true",
        help="Run the RAKG-paper Entity Fidelity metric with an LLM judge over sampled entities.",
    )
    parser.add_argument(
        "--entity-sample-size",
        type=int,
        default=100,
        help="Maximum entities to send to the RAKG Entity Fidelity judge. Use 0 for all.",
    )
    parser.add_argument(
        "--rakg-standard-kg",
        type=Path,
        default=None,
        help="Optional standard/reference KG JSON for RAKG EC and RNS metrics.",
    )
    parser.add_argument(
        "--export-rakg-eval-inputs",
        action="store_true",
        help="Export RAKG-compatible input files alongside this quality report.",
    )
    parser.add_argument(
        "--rakg-output-dir",
        type=Path,
        default=None,
        help="Directory for RAKG-compatible files. Defaults to <report-dir>/rakg_eval_inputs.",
    )
    parser.add_argument("--run-pykeen", action="store_true", help="Run optional PyKEEN link-prediction evaluation if pykeen is installed.")
    parser.add_argument("--pykeen-model", default="TransE", help="PyKEEN model name.")
    parser.add_argument("--pykeen-epochs", type=int, default=50, help="PyKEEN training epochs.")
    parser.add_argument("--pykeen-test-ratio", type=float, default=0.2, help="Held-out ratio for PyKEEN evaluation.")
    parser.add_argument(
        "--pykeen-device",
        default="cpu",
        help="Device for PyKEEN training/evaluation. Use cpu, cuda, cuda:0, or auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = resolve_report_dir(args)
    set_log_file(report_dir / LOG_FILE)

    inputs = load_inputs(args)
    log("quality", f"Loaded KG with {len(inputs.concept_kg.get('entities', []))} entities and {len(inputs.concept_kg.get('relations', []))} relations")

    entity_metrics, entity_issues = evaluate_entities(inputs)
    relation_metrics, relation_issues, risk_rows = evaluate_relations(inputs)
    graph_metrics = evaluate_graph(inputs.concept_kg)

    report: dict[str, Any] = {
        "metadata": {
            "kg_output_dir": str(args.kg_output_dir),
            "entity_count": len(inputs.concept_kg.get("entities", [])),
            "relation_count": len(inputs.concept_kg.get("relations", [])),
            "doc_node_count": len(inputs.doc_nodes),
        },
        "scores": build_scores(entity_metrics, relation_metrics, graph_metrics),
        "entity_metrics": entity_metrics,
        "relation_metrics": relation_metrics,
        "graph_metrics": graph_metrics,
        "issue_summary": summarize_issues(entity_issues + relation_issues),
        "top_risk_relations": risk_rows[:30],
        "optional_evaluations": {},
    }

    llm_judge_result: dict[str, Any] | None = None
    entity_fidelity_result: dict[str, Any] | None = None

    if args.run_llm_judge:
        llm_judge_result = run_llm_judge(inputs, args, report_dir, risk_rows)
        report["optional_evaluations"]["llm_judge"] = llm_judge_result

    if args.run_rakg_entity_fidelity:
        entity_fidelity_result = run_rakg_entity_fidelity(inputs, args, report_dir)
        report["optional_evaluations"]["rakg_entity_fidelity"] = entity_fidelity_result

    if args.run_pykeen:
        report["optional_evaluations"]["pykeen"] = run_pykeen(inputs, args, report_dir)

    if args.export_rakg_eval_inputs:
        report["optional_evaluations"]["rakg_eval_inputs"] = export_rakg_eval_inputs(inputs, args, report_dir)

    report["rakg_paper_metrics"] = build_rakg_paper_metrics(
        inputs=inputs,
        args=args,
        scores=report["scores"],
        llm_judge_result=llm_judge_result,
        entity_fidelity_result=entity_fidelity_result,
    )

    write_json(report_dir / REPORT_FILE, report)
    write_issues_csv(report_dir / ISSUES_FILE, entity_issues + relation_issues)
    write_risk_relations_csv(report_dir / RISK_RELATIONS_FILE, risk_rows)
    log("quality", f"Wrote quality report to {report_dir / REPORT_FILE}")


def resolve_report_dir(args: argparse.Namespace) -> Path:
    base_dir = args.report_dir or default_report_dir(args.kg_output_dir)
    if args.overwrite_report:
        return ensure_dir(base_dir)
    if not report_dir_has_existing_outputs(base_dir):
        return ensure_dir(base_dir)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for suffix in [stamp, *[f"{stamp}_{index}" for index in range(1, 100)]]:
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix}")
        if not report_dir_has_existing_outputs(candidate):
            log("quality", f"Report directory already has outputs; using {candidate} instead.")
            return ensure_dir(candidate)
    raise RuntimeError(f"Could not find a free report directory derived from {base_dir}")


def default_report_dir(kg_output_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    kg_name = kg_output_dir.name or "concept_kg"
    if kg_name == "concept_kg" and kg_output_dir.parent.name:
        kg_name = kg_output_dir.parent.name
    return PROJECT_ROOT / "outputs" / f"{safe_path_name(kg_name)}_quality_{stamp}"


def report_dir_has_existing_outputs(path: Path) -> bool:
    return path.exists() and any((path / file_name).exists() for file_name in OUTPUT_FILE_NAMES)


def safe_path_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return safe.strip("._-") or "concept_kg"


# Input loading


def load_inputs(args: argparse.Namespace) -> EvaluationInputs:
    output_dir = args.kg_output_dir
    concept_path = output_dir / "concept_kg.json"
    doc_nodes_path = output_dir / "extractable_doc_nodes.json"
    if not concept_path.exists():
        raise FileNotFoundError(f"Missing concept KG file: {concept_path}")
    if not doc_nodes_path.exists():
        raise FileNotFoundError(f"Missing document nodes file: {doc_nodes_path}")

    concept_kg = load_json(concept_path)
    doc_node_rows = load_json(doc_nodes_path)
    if not isinstance(concept_kg, dict):
        raise ValueError(f"Expected JSON object in {concept_path}")
    if not isinstance(doc_node_rows, list):
        raise ValueError(f"Expected JSON array in {doc_nodes_path}")

    doc_nodes = {
        str(item.get("source_node_id", "")): item
        for item in doc_node_rows
        if isinstance(item, dict) and item.get("source_node_id")
    }
    schema = load_yaml(args.schema)
    return EvaluationInputs(output_dir=output_dir, concept_kg=concept_kg, doc_nodes=doc_nodes, schema=schema)


# Rule-based quality checks


def evaluate_entities(inputs: EvaluationInputs) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entities = _as_list(inputs.concept_kg.get("entities"))
    allowed_types = set((inputs.schema.get("entity_types") or {}).keys())
    by_id: dict[str, dict[str, Any]] = {}
    name_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    no_definition = 0
    no_raw_ids = 0
    source_refs = 0
    missing_source_refs = 0

    for entity in entities:
        entity_id = str(entity.get("entity_id") or "")
        name = normalize_whitespace(entity.get("canonical_name"))
        entity_type = normalize_whitespace(entity.get("type"))
        sources = _as_list(entity.get("merged_source_node_ids"))
        definitions = [normalize_whitespace(item) for item in _as_list(entity.get("merged_definitions")) if normalize_whitespace(item)]
        raw_ids = _as_list(entity.get("merged_raw_entity_ids"))

        if not entity_id:
            issues.append(issue("entity", "", "missing_entity_id", "high", "Entity has no entity_id."))
        elif entity_id in by_id:
            issues.append(issue("entity", entity_id, "duplicate_entity_id", "high", "Duplicate entity_id."))
        else:
            by_id[entity_id] = entity

        if not name:
            issues.append(issue("entity", entity_id, "missing_name", "high", "Entity has empty canonical_name."))
        else:
            name_counts[normalize_key(name)] += 1

        if not entity_type or entity_type not in allowed_types:
            issues.append(issue("entity", entity_id, "unknown_entity_type", "medium", f"Unknown entity type: {entity_type or '(empty)'}."))
        type_counts[entity_type or "(empty)"] += 1

        if not definitions:
            no_definition += 1
        if not raw_ids:
            no_raw_ids += 1

        if not sources:
            issues.append(issue("entity", entity_id, "missing_source_node", "medium", "Entity has no source node."))
        for source_id in sources:
            source_refs += 1
            if source_id not in inputs.doc_nodes:
                missing_source_refs += 1
                issues.append(issue("entity", entity_id, "missing_source_context", "medium", f"Source node {source_id} is not in extractable_doc_nodes.json."))

    for name_key, count in name_counts.items():
        if count > 1:
            issues.append(issue("entity", name_key, "duplicate_canonical_name", "medium", f"{count} entities share the same normalized canonical name."))

    metrics = {
        "entity_count": len(entities),
        "entity_type_counts": dict(sorted(type_counts.items())),
        "duplicate_canonical_name_count": sum(1 for count in name_counts.values() if count > 1),
        "missing_definition_count": no_definition,
        "missing_definition_rate": ratio(no_definition, len(entities)),
        "synthetic_or_untraced_entity_count": no_raw_ids,
        "synthetic_or_untraced_entity_rate": ratio(no_raw_ids, len(entities)),
        "source_reference_count": source_refs,
        "missing_source_reference_count": missing_source_refs,
        "source_reference_validity_rate": 1 - ratio(missing_source_refs, source_refs),
    }
    return metrics, issues


def evaluate_relations(inputs: EvaluationInputs) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    entities = _as_list(inputs.concept_kg.get("entities"))
    relations = _as_list(inputs.concept_kg.get("relations"))
    entity_ids = {str(entity.get("entity_id") or "") for entity in entities if entity.get("entity_id")}
    entity_name_by_id = {str(entity.get("entity_id")): normalize_whitespace(entity.get("canonical_name")) for entity in entities}
    allowed_relation_types = set((inputs.schema.get("relation_types") or {}).keys())

    relation_type_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    missing_evidence = 0
    evidence_refs = 0
    evidence_found = 0
    source_refs = 0
    missing_source_refs = 0
    unresolved_endpoint_count = 0
    self_loop_count = 0
    name_mismatch_count = 0

    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        is_inferred = bool(relation.get("is_inferred", False))
        if is_inferred:
            continue
        relation_type = normalize_whitespace(relation.get("relation"))
        head_id = str(relation.get("head_entity_id") or "")
        tail_id = str(relation.get("tail_entity_id") or "")
        head = normalize_whitespace(relation.get("head"))
        tail = normalize_whitespace(relation.get("tail"))
        sources = [str(item) for item in _as_list(relation.get("source_node_ids")) if str(item)]
        evidences = [normalize_whitespace(item) for item in _as_list(relation.get("evidences")) if normalize_whitespace(item)]
        risk_reasons: list[str] = []
        risk_score = 0

        if not relation_id:
            issues.append(issue("relation", "", "missing_relation_id", "high", "Relation has no relation_id."))
            risk_reasons.append("missing_relation_id")
            risk_score += 3

        if not relation_type or relation_type not in allowed_relation_types:
            issues.append(issue("relation", relation_id, "unknown_relation_type", "high", f"Unknown relation type: {relation_type or '(empty)'}."))
            risk_reasons.append("unknown_relation_type")
            risk_score += 3
        relation_type_counts[relation_type or "(empty)"] += 1

        if head_id not in entity_ids or tail_id not in entity_ids:
            unresolved_endpoint_count += 1
            issues.append(issue("relation", relation_id, "unresolved_endpoint", "high", "Relation endpoint id is missing from entities."))
            risk_reasons.append("unresolved_endpoint")
            risk_score += 4

        if head_id and tail_id and head_id == tail_id:
            self_loop_count += 1
            issues.append(issue("relation", relation_id, "self_loop", "medium", "Relation points from an entity to itself."))
            risk_reasons.append("self_loop")
            risk_score += 2

        expected_head = entity_name_by_id.get(head_id, "")
        expected_tail = entity_name_by_id.get(tail_id, "")
        if expected_head and head and normalize_key(expected_head) != normalize_key(head):
            name_mismatch_count += 1
            issues.append(issue("relation", relation_id, "head_name_mismatch", "low", f"Head text `{head}` differs from entity name `{expected_head}`."))
            risk_reasons.append("head_name_mismatch")
            risk_score += 1
        if expected_tail and tail and normalize_key(expected_tail) != normalize_key(tail):
            name_mismatch_count += 1
            issues.append(issue("relation", relation_id, "tail_name_mismatch", "low", f"Tail text `{tail}` differs from entity name `{expected_tail}`."))
            risk_reasons.append("tail_name_mismatch")
            risk_score += 1

        if not sources:
            issues.append(issue("relation", relation_id, "missing_source_node", "high", "Relation has no source node."))
            risk_reasons.append("missing_source_node")
            risk_score += 3
        for source_id in sources:
            source_refs += 1
            if source_id not in inputs.doc_nodes:
                missing_source_refs += 1
                issues.append(issue("relation", relation_id, "missing_source_context", "high", f"Source node {source_id} is not in extractable_doc_nodes.json."))
                risk_reasons.append("missing_source_context")
                risk_score += 3

        if not evidences:
            missing_evidence += 1
            issues.append(issue("relation", relation_id, "missing_evidence", "high", "Relation has no evidence text."))
            risk_reasons.append("missing_evidence")
            risk_score += 4
        else:
            contexts = [str(inputs.doc_nodes.get(source_id, {}).get("content", "")) for source_id in sources]
            combined_context = "\n".join(contexts)
            for evidence in evidences:
                evidence_refs += 1
                if normalized_contains(combined_context, evidence):
                    evidence_found += 1
                else:
                    issues.append(issue("relation", relation_id, "evidence_not_found_in_source", "medium", f"Evidence is not an exact normalized substring: {evidence[:120]}"))
                    risk_reasons.append("evidence_not_found_in_source")
                    risk_score += 2

        if head and tail and normalize_key(head) == normalize_key(tail):
            risk_reasons.append("same_head_tail_text")
            risk_score += 2

        risk_rows.append(
            {
                "relation_id": relation_id,
                "head": head,
                "relation": relation_type,
                "tail": tail,
                "risk_score": risk_score,
                "risk_reasons": " | ".join(sorted(set(risk_reasons))),
                "source_node_ids": " | ".join(sources),
                "evidences": " | ".join(evidences),
            }
        )

    risk_rows.sort(key=lambda row: (-int(row["risk_score"]), row["head"], row["relation"], row["tail"]))
    metrics = {
        "relation_count": len(relations),
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "missing_evidence_count": missing_evidence,
        "evidence_coverage_rate": 1 - ratio(missing_evidence, len(relations)),
        "evidence_reference_count": evidence_refs,
        "evidence_found_in_source_count": evidence_found,
        "evidence_found_in_source_rate": ratio(evidence_found, evidence_refs),
        "source_reference_count": source_refs,
        "missing_source_reference_count": missing_source_refs,
        "source_reference_validity_rate": 1 - ratio(missing_source_refs, source_refs),
        "unresolved_endpoint_count": unresolved_endpoint_count,
        "self_loop_count": self_loop_count,
        "endpoint_name_mismatch_count": name_mismatch_count,
        "high_risk_relation_count": sum(1 for row in risk_rows if int(row["risk_score"]) >= 4),
        "high_risk_relation_rate": ratio(sum(1 for row in risk_rows if int(row["risk_score"]) >= 4), len(relations)),
    }
    return metrics, issues, risk_rows


# Graph-theoretic structural metrics


def evaluate_graph(concept_kg: dict[str, Any]) -> dict[str, Any]:
    entities = _as_list(concept_kg.get("entities"))
    relations = _as_list(concept_kg.get("relations"))
    entity_ids = [str(entity.get("entity_id")) for entity in entities if entity.get("entity_id")]
    entity_by_id = {
        str(entity.get("entity_id")): {
            "name": normalize_whitespace(entity.get("canonical_name")),
            "type": normalize_whitespace(entity.get("type")),
        }
        for entity in entities
        if entity.get("entity_id")
    }
    adjacency: dict[str, set[str]] = {entity_id: set() for entity_id in entity_ids}
    out_adjacency: dict[str, set[str]] = {entity_id: set() for entity_id in entity_ids}
    in_adjacency: dict[str, set[str]] = {entity_id: set() for entity_id in entity_ids}
    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    edge_keys: list[tuple[str, str, str]] = []
    directed_pairs: list[tuple[str, str]] = []
    type_relation_counter: Counter[str] = Counter()

    for relation in relations:
        head_id = str(relation.get("head_entity_id") or "")
        tail_id = str(relation.get("tail_entity_id") or "")
        relation_type = normalize_whitespace(relation.get("relation"))
        if head_id not in adjacency or tail_id not in adjacency:
            continue
        adjacency[head_id].add(tail_id)
        adjacency[tail_id].add(head_id)
        out_adjacency[head_id].add(tail_id)
        in_adjacency[tail_id].add(head_id)
        out_degree[head_id] += 1
        in_degree[tail_id] += 1
        edge_keys.append((head_id, relation_type, tail_id))
        directed_pairs.append((head_id, tail_id))
        head_type = entity_by_id.get(head_id, {}).get("type", "")
        tail_type = entity_by_id.get(tail_id, {}).get("type", "")
        type_relation_counter[f"{head_type or '(empty)'} --{relation_type or '(empty)'}--> {tail_type or '(empty)'}"] += 1

    isolated = [entity_id for entity_id in entity_ids if in_degree[entity_id] + out_degree[entity_id] == 0]
    components = connected_components(adjacency)
    component_sizes = sorted((len(component) for component in components), reverse=True)
    total_degree = [in_degree[entity_id] + out_degree[entity_id] for entity_id in entity_ids]
    sccs = strongly_connected_components(out_adjacency)
    scc_sizes = sorted((len(component) for component in sccs), reverse=True)
    pagerank = page_rank(out_adjacency)
    core_numbers = k_core_numbers(adjacency)
    max_core = max(core_numbers.values()) if core_numbers else 0
    pair_reciprocity = reciprocity_rate(directed_pairs)
    duplicated_edges = len(edge_keys) - len(set(edge_keys))

    return {
        "isolated_entity_count": len(isolated),
        "isolated_entity_rate": ratio(len(isolated), len(entity_ids)),
        "weakly_connected_component_count": len(components),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "largest_component_rate": ratio(component_sizes[0] if component_sizes else 0, len(entity_ids)),
        "average_total_degree": sum(total_degree) / len(total_degree) if total_degree else 0,
        "max_total_degree": max(total_degree) if total_degree else 0,
        "density": graph_density(len(entity_ids), len(relations)),
        "duplicate_typed_edge_count": duplicated_edges,
        "degree_metrics": {
            "in_degree": numeric_summary([in_degree[entity_id] for entity_id in entity_ids]),
            "out_degree": numeric_summary([out_degree[entity_id] for entity_id in entity_ids]),
            "total_degree": numeric_summary(total_degree),
            "total_degree_gini": gini(total_degree),
            "top_total_degree_entities": top_entities_by_score(
                {entity_id: in_degree[entity_id] + out_degree[entity_id] for entity_id in entity_ids},
                entity_by_id,
            ),
            "top_in_degree_entities": top_entities_by_score({entity_id: in_degree[entity_id] for entity_id in entity_ids}, entity_by_id),
            "top_out_degree_entities": top_entities_by_score({entity_id: out_degree[entity_id] for entity_id in entity_ids}, entity_by_id),
        },
        "directed_metrics": {
            "strongly_connected_component_count": len(sccs),
            "largest_strongly_connected_component_size": scc_sizes[0] if scc_sizes else 0,
            "largest_strongly_connected_component_rate": ratio(scc_sizes[0] if scc_sizes else 0, len(entity_ids)),
            "reciprocal_pair_rate": pair_reciprocity,
        },
        "centrality": {
            "top_pagerank_entities": top_entities_by_score(pagerank, entity_by_id, limit=20),
        },
        "k_core": {
            "max_core_number": max_core,
            "core_number_counts": dict(sorted(Counter(core_numbers.values()).items())),
            "top_core_entities": top_entities_by_score(core_numbers, entity_by_id, limit=20),
        },
        "relation_subgraphs": relation_subgraph_metrics(relations, entity_by_id),
        "top_type_relation_patterns": [
            {"pattern": pattern, "count": count}
            for pattern, count in type_relation_counter.most_common(30)
        ],
    }


# Optional evaluators


def run_llm_judge(
    inputs: EvaluationInputs,
    args: argparse.Namespace,
    report_dir: Path,
    risk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = load_yaml(args.settings)
    llm_settings = settings.get("llm", {})
    client = OpenAICompatibleJudgeClient(
        base_url=args.judge_base_url or str(llm_settings.get("base_url", "https://api.openai.com/v1")),
        api_key_env=args.judge_api_key_env or str(llm_settings.get("api_key_env", "OPENAI_API_KEY")),
        model=args.judge_model or str(llm_settings.get("model", "")),
        timeout_seconds=args.judge_timeout_seconds,
        retries=args.judge_retries,
        temperature=0,
        response_format_json=resolve_judge_response_format(args, llm_settings),
    )

    relations = {str(item.get("relation_id") or ""): item for item in _as_list(inputs.concept_kg.get("relations"))}
    sampled_ids = sample_relation_ids(risk_rows, args.sample_size, args.seed)
    output_path = report_dir / LLM_JUDGEMENTS_FILE
    verdict_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    score_sum = 0.0
    attempted_count = 0
    valid_count = 0
    accurate_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for index, relation_id in enumerate(sampled_ids, start=1):
            relation = relations.get(relation_id)
            if relation is None:
                continue
            log("llm_judge", f"Judging relation {index}/{len(sampled_ids)}: {relation_id}")
            try:
                judged = client.evaluate_relation(relation, build_relation_context(inputs, relation, args.context_chars))
            except Exception as exc:  # noqa: BLE001
                judged = {
                    "verdict": "judge_error",
                    "is_accurate": False,
                    "score": None,
                    "error_type": "judge_error",
                    "reason": str(exc),
                    "suggested_fix": None,
                }
                error_counts[short_error_reason(str(exc))] += 1
            row = {
                "relation_id": relation_id,
                "head": relation.get("head", ""),
                "relation": relation.get("relation", ""),
                "tail": relation.get("tail", ""),
                **judged,
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            attempted_count += 1
            verdict_counts[str(judged.get("verdict") or "(empty)")] += 1
            if judged.get("verdict") != "judge_error":
                valid_count += 1
                score_sum += _safe_float(judged.get("score"), default=0.0)
                if judged.get("is_accurate"):
                    accurate_count += 1

    return {
        "enabled": True,
        "status": judge_status(attempted_count, valid_count),
        "attempted_relation_count": attempted_count,
        "judged_relation_count": valid_count,
        "error_count": attempted_count - valid_count,
        "sample_size": args.sample_size,
        "sampled_relation_count": len(sampled_ids),
        "judgement_file": str(output_path),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "error_summary": counter_to_rows(error_counts),
        "accuracy_rate": ratio(accurate_count, valid_count),
        "average_score": score_sum / valid_count if valid_count else None,
    }


def run_rakg_entity_fidelity(inputs: EvaluationInputs, args: argparse.Namespace, report_dir: Path) -> dict[str, Any]:
    settings = load_yaml(args.settings)
    llm_settings = settings.get("llm", {})
    client = OpenAICompatibleJudgeClient(
        base_url=args.judge_base_url or str(llm_settings.get("base_url", "https://api.openai.com/v1")),
        api_key_env=args.judge_api_key_env or str(llm_settings.get("api_key_env", "OPENAI_API_KEY")),
        model=args.judge_model or str(llm_settings.get("model", "")),
        timeout_seconds=args.judge_timeout_seconds,
        retries=args.judge_retries,
        temperature=0,
        response_format_json=resolve_judge_response_format(args, llm_settings),
    )

    entities = [entity for entity in _as_list(inputs.concept_kg.get("entities")) if isinstance(entity, dict)]
    sampled_entities = sample_entities(entities, args.entity_sample_size, args.seed)
    output_path = report_dir / RAKG_ENTITY_FIDELITY_FILE
    verdict_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    score_sum = 0.0
    attempted_count = 0
    valid_count = 0
    faithful_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for index, entity in enumerate(sampled_entities, start=1):
            entity_id = str(entity.get("entity_id") or "")
            log("rakg_ef", f"Judging entity {index}/{len(sampled_entities)}: {entity_id}")
            try:
                judged = client.evaluate_entity(entity, build_entity_context(inputs, entity, args.context_chars))
            except Exception as exc:  # noqa: BLE001
                judged = {
                    "verdict": "judge_error",
                    "is_faithful": False,
                    "score": None,
                    "error_type": "judge_error",
                    "reason": str(exc),
                    "suggested_fix": None,
                }
                error_counts[short_error_reason(str(exc))] += 1
            row = {
                "entity_id": entity_id,
                "name": entity.get("canonical_name", ""),
                "type": entity.get("type", ""),
                **judged,
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            attempted_count += 1
            verdict_counts[str(judged.get("verdict") or "(empty)")] += 1
            if judged.get("verdict") != "judge_error":
                valid_count += 1
                score_sum += _safe_float(judged.get("score"), default=0.0)
                if judged.get("is_faithful"):
                    faithful_count += 1

    return {
        "enabled": True,
        "status": judge_status(attempted_count, valid_count),
        "attempted_entity_count": attempted_count,
        "judged_entity_count": valid_count,
        "error_count": attempted_count - valid_count,
        "sample_size": args.entity_sample_size,
        "sampled_entity_count": len(sampled_entities),
        "judgement_file": str(output_path),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "error_summary": counter_to_rows(error_counts),
        "faithful_rate": ratio(faithful_count, valid_count),
        "average_score": score_sum / valid_count if valid_count else None,
    }


def run_pykeen(inputs: EvaluationInputs, args: argparse.Namespace, report_dir: Path) -> dict[str, Any]:
    triples = [
        [
            normalize_whitespace(relation.get("head")),
            normalize_whitespace(relation.get("relation")),
            normalize_whitespace(relation.get("tail")),
        ]
        for relation in _as_list(inputs.concept_kg.get("relations"))
        if normalize_whitespace(relation.get("head")) and normalize_whitespace(relation.get("relation")) and normalize_whitespace(relation.get("tail"))
    ]
    triples_path = report_dir / PYKEEN_TRIPLES_FILE
    with triples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(triples)

    if len(triples) < 10:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "PyKEEN evaluation needs at least 10 triples for a meaningful train/test split.",
            "triple_file": str(triples_path),
        }

    try:
        import numpy as np
        import torch
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError as exc:
        return {
            "enabled": True,
            "status": "missing_dependency",
            "reason": f"{exc}. Install pykeen to run this optional evaluator.",
            "triple_file": str(triples_path),
        }

    device = resolve_pykeen_device(args.pykeen_device, torch)
    mapped_triples = np.array(triples, dtype=str)
    factory = TriplesFactory.from_labeled_triples(mapped_triples)
    training_factory, testing_factory = factory.split([1 - args.pykeen_test_ratio, args.pykeen_test_ratio], random_state=args.seed)
    result = pipeline(
        training=training_factory,
        testing=testing_factory,
        model=args.pykeen_model,
        training_kwargs={"num_epochs": args.pykeen_epochs},
        random_seed=args.seed,
        device=device,
    )
    metrics = result.metric_results.to_dict()
    write_json(report_dir / PYKEEN_METRICS_FILE, metrics)
    return {
        "enabled": True,
        "status": "completed",
        "model": args.pykeen_model,
        "epochs": args.pykeen_epochs,
        "device": device,
        "train_triples": int(training_factory.num_triples),
        "test_triples": int(testing_factory.num_triples),
        "triple_file": str(triples_path),
        "metrics_file": str(report_dir / PYKEEN_METRICS_FILE),
        "selected_metrics": select_pykeen_metrics(metrics),
    }


def export_rakg_eval_inputs(inputs: EvaluationInputs, args: argparse.Namespace, report_dir: Path) -> dict[str, Any]:
    from scripts.export_rakg_eval_inputs import (  # noqa: PLC0415
        COUNT_STATS_FILE,
        DEEPEVAL_REL_FILE,
        MANIFEST_FILE,
        RAKG_GRAPH_FILE,
        build_deepeval_relation_rows,
        build_rakg_graph,
        write_jsonl,
    )

    output_dir = ensure_dir(args.rakg_output_dir or (report_dir / "rakg_eval_inputs"))
    entity_by_id = {entity.get("entity_id"): entity for entity in _as_list(inputs.concept_kg.get("entities"))}
    entity_type_by_name = {
        entity.get("canonical_name"): entity.get("type", "")
        for entity in _as_list(inputs.concept_kg.get("entities"))
        if entity.get("canonical_name")
    }

    rakg_graph = build_rakg_graph(inputs.concept_kg)
    deepeval_rows = build_deepeval_relation_rows(
        concept_kg=inputs.concept_kg,
        doc_nodes=inputs.doc_nodes,
        entity_by_id=entity_by_id,
        entity_type_by_name=entity_type_by_name,
        context_chars=args.context_chars,
    )
    count_stats = {
        "file_name": str(inputs.output_dir / "concept_kg.json"),
        "entities": len(rakg_graph["entities"]),
        "attributes": 0,
        "relations": len(rakg_graph["relations"]),
    }

    graph_path = output_dir / RAKG_GRAPH_FILE
    deepeval_path = output_dir / DEEPEVAL_REL_FILE
    count_path = output_dir / COUNT_STATS_FILE
    manifest_path = output_dir / MANIFEST_FILE

    write_json(graph_path, rakg_graph)
    write_jsonl(deepeval_path, deepeval_rows)
    write_jsonl(count_path, [count_stats])
    write_json(
        manifest_path,
        {
            "source_kg_dir": str(inputs.output_dir),
            "files": {
                "rakg_graph": str(graph_path),
                "deepeval_relation_data": str(deepeval_path),
                "count_stats": str(count_path),
            },
            "counts": {
                "entities": len(rakg_graph["entities"]),
                "relations": len(rakg_graph["relations"]),
                "deepeval_relation_rows": len(deepeval_rows),
                "doc_nodes": len(inputs.doc_nodes),
            },
            "usage_notes": [
                "Use rakg_eval_kg.json with RAKG ideal_kg_eval when you also have a reference/ideal KG in the same format.",
                "Use deepeval_rel_data.jsonl as input for a patched RAKG llm_eval/deepeval_test_kg.py.",
                "Use rakg_count_stats.jsonl for count_eval-style statistics or visualization.",
                "MINE_eval needs a QA/retrieval benchmark, so this adapter only prepares the graph side.",
            ],
        },
    )
    log("rakg", f"Wrote RAKG-compatible evaluation inputs to {output_dir}")
    return {
        "enabled": True,
        "status": "completed",
        "output_dir": str(output_dir),
        "files": {
            "rakg_graph": str(graph_path),
            "deepeval_relation_data": str(deepeval_path),
            "count_stats": str(count_path),
            "manifest": str(manifest_path),
        },
        "counts": {
            "entities": len(rakg_graph["entities"]),
            "relations": len(rakg_graph["relations"]),
            "deepeval_relation_rows": len(deepeval_rows),
            "doc_nodes": len(inputs.doc_nodes),
        },
    }


def build_rakg_paper_metrics(
    inputs: EvaluationInputs,
    args: argparse.Namespace,
    scores: dict[str, Any],
    llm_judge_result: dict[str, Any] | None,
    entity_fidelity_result: dict[str, Any] | None,
) -> dict[str, Any]:
    entity_count = len(_as_list(inputs.concept_kg.get("entities")))
    relation_count = len(_as_list(inputs.concept_kg.get("relations")))
    metrics: dict[str, Any] = {
        "source": {
            "paper": "RAKG.pdf",
            "implemented_metrics": ["ED", "RR", "EF", "RF", "EC", "RNS", "MINE_accuracy"],
            "notes": [
                "ED and RR are computed directly from the extracted KG.",
                "EF follows the paper only when --run-rakg-entity-fidelity is enabled.",
                "RF is mapped to the existing source-grounded LLM relation judge when --run-llm-judge is enabled.",
                "EC and RNS require --rakg-standard-kg.",
                "MINE accuracy requires a separate fact/QA benchmark and is reported as not configured here.",
            ],
        },
        "entity_density": {
            "symbol": "ED",
            "formula": "ED = Ne",
            "value": entity_count,
        },
        "relationship_richness": {
            "symbol": "RR",
            "formula": "RR = Nr / Ne",
            "value": ratio(relation_count, entity_count),
            "entity_count": entity_count,
            "relation_count": relation_count,
        },
        "entity_fidelity": {
            "symbol": "EF",
            "formula": "EF = average(LLMJudge_entity(e_i, retrieved_text_i))",
            "status": "not_run",
            "reason": "Use --run-rakg-entity-fidelity to run the paper-style entity LLM judge.",
            "rule_proxy": scores.get("entity_traceability_score"),
        },
        "relationship_fidelity": {
            "symbol": "RF",
            "formula": "RF = average(LLMJudge_relation(r_i, retrieved_text_i, retrieved_kg_i))",
            "status": "not_run",
            "reason": "Use --run-llm-judge to run the source-grounded relation LLM judge.",
            "rule_proxy": scores.get("relation_evidence_score"),
        },
        "mine_accuracy": {
            "symbol": "MINE_accuracy",
            "status": "not_configured",
            "reason": "The paper computes MINE accuracy from manually verified facts/QA. Provide a fact benchmark before running this metric.",
        },
        "standard_kg_comparison": {
            "entity_coverage": {
                "symbol": "EC",
                "formula": "EC = |E intersect E*| / |E*|",
                "status": "requires_standard_kg",
            },
            "relation_network_similarity": {
                "symbol": "RNS",
                "formula": "RNS = sum(RelSim_i * EntityWeight_i), for e_i in E intersect E*",
                "status": "requires_standard_kg",
            },
        },
    }

    if entity_fidelity_result:
        metrics["entity_fidelity"].update(
            {
                "status": entity_fidelity_result.get("status", "completed"),
                "value": entity_fidelity_result.get("average_score"),
                "faithful_rate": entity_fidelity_result.get("faithful_rate", 0),
                "judged_entity_count": entity_fidelity_result.get("judged_entity_count", 0),
                "attempted_entity_count": entity_fidelity_result.get("attempted_entity_count", 0),
                "error_count": entity_fidelity_result.get("error_count", 0),
                "judgement_file": entity_fidelity_result.get("judgement_file", ""),
                "verdict_counts": entity_fidelity_result.get("verdict_counts", {}),
            }
        )
        metrics["entity_fidelity"].pop("reason", None)

    if llm_judge_result:
        metrics["relationship_fidelity"].update(
            {
                "status": llm_judge_result.get("status", "completed"),
                "value": llm_judge_result.get("average_score"),
                "accuracy_rate": llm_judge_result.get("accuracy_rate", 0),
                "judged_relation_count": llm_judge_result.get("judged_relation_count", 0),
                "attempted_relation_count": llm_judge_result.get("attempted_relation_count", 0),
                "error_count": llm_judge_result.get("error_count", 0),
                "judgement_file": llm_judge_result.get("judgement_file", ""),
                "verdict_counts": llm_judge_result.get("verdict_counts", {}),
            }
        )
        metrics["relationship_fidelity"].pop("reason", None)

    if args.rakg_standard_kg:
        standard_graph = load_kg_graph_for_rakg_metrics(args.rakg_standard_kg)
        eval_graph = current_kg_as_rakg_graph(inputs.concept_kg)
        metrics["standard_kg_comparison"] = compute_rakg_standard_metrics(eval_graph, standard_graph)

    return metrics


def current_kg_as_rakg_graph(concept_kg: dict[str, Any]) -> dict[str, Any]:
    from scripts.export_rakg_eval_inputs import build_rakg_graph  # noqa: PLC0415

    return build_rakg_graph(concept_kg)


def load_kg_graph_for_rakg_metrics(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Standard KG must be a JSON object: {path}")
    if has_concept_kg_shape(data):
        return current_kg_as_rakg_graph(data)
    if "entities" not in data or "relations" not in data:
        raise ValueError(f"Standard KG must contain entities and relations: {path}")
    return normalize_rakg_graph(data)


def has_concept_kg_shape(data: dict[str, Any]) -> bool:
    entities = _as_list(data.get("entities"))
    relations = _as_list(data.get("relations"))
    return bool(entities and isinstance(entities[0], dict) and "canonical_name" in entities[0]) or bool(
        relations and isinstance(relations[0], dict) and "head_entity_id" in relations[0]
    )


def normalize_rakg_graph(data: dict[str, Any]) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    for entity in _as_list(data.get("entities")):
        if not isinstance(entity, dict):
            continue
        name = normalize_whitespace(str(entity.get("name") or entity.get("canonical_name") or ""))
        if not name:
            continue
        entities.append(
            {
                "name": name,
                "type": normalize_whitespace(str(entity.get("type") or "")),
                "description": normalize_whitespace(str(entity.get("description") or "")),
            }
        )

    relations: list[list[str]] = []
    for relation in _as_list(data.get("relations")):
        if isinstance(relation, dict):
            head = relation.get("head")
            rel_name = relation.get("relation")
            tail = relation.get("tail")
            desc = " | ".join(str(item) for item in _as_list(relation.get("evidences")) if item)
        elif isinstance(relation, list):
            padded = relation + [""] * (4 - len(relation))
            head, rel_name, tail, desc = padded[:4]
        else:
            continue
        head_text = normalize_whitespace(str(head or ""))
        rel_text = normalize_whitespace(str(rel_name or ""))
        tail_text = normalize_whitespace(str(tail or ""))
        if head_text and rel_text and tail_text:
            relations.append([head_text, rel_text, tail_text, normalize_whitespace(str(desc or ""))])
    return {"entities": entities, "relations": relations}


def compute_rakg_standard_metrics(eval_graph: dict[str, Any], standard_graph: dict[str, Any]) -> dict[str, Any]:
    eval_graph = normalize_rakg_graph(eval_graph)
    standard_graph = normalize_rakg_graph(standard_graph)
    eval_entities = {rakg_entity_key(entity.get("name", "")): entity for entity in eval_graph["entities"]}
    standard_entities = {rakg_entity_key(entity.get("name", "")): entity for entity in standard_graph["entities"]}
    matched_keys = sorted(key for key in standard_entities if key in eval_entities)

    eval_networks = relation_networks_by_entity(eval_graph)
    standard_networks = relation_networks_by_entity(standard_graph)
    weights = {key: max(1, len(standard_networks.get(key, set()))) for key in matched_keys}
    total_weight = sum(weights.values())
    weighted_similarity = 0.0
    per_entity: list[dict[str, Any]] = []
    for key in matched_keys:
        std_edges = standard_networks.get(key, set())
        eval_edges = eval_networks.get(key, set())
        rel_sim = relation_network_f1(eval_edges, std_edges)
        weight = weights[key] / total_weight if total_weight else 0.0
        weighted_similarity += rel_sim * weight
        per_entity.append(
            {
                "entity": standard_entities[key].get("name", ""),
                "relation_similarity": rel_sim,
                "weight": weight,
                "standard_relation_count": len(std_edges),
                "evaluated_relation_count": len(eval_edges),
            }
        )
    per_entity.sort(key=lambda row: (float(row["relation_similarity"]), -float(row["weight"]), row["entity"]))

    return {
        "matching_mode": "normalized_exact_entity_name",
        "entity_coverage": {
            "symbol": "EC",
            "formula": "EC = |E intersect E*| / |E*|",
            "status": "completed",
            "value": ratio(len(matched_keys), len(standard_entities)),
            "matched_entity_count": len(matched_keys),
            "standard_entity_count": len(standard_entities),
            "evaluated_entity_count": len(eval_entities),
        },
        "relation_network_similarity": {
            "symbol": "RNS",
            "formula": "RNS = sum(RelSim_i * EntityWeight_i), for e_i in E intersect E*",
            "status": "completed",
            "value": max(0.0, min(1.0, weighted_similarity)),
            "matched_entity_count": len(matched_keys),
            "entity_weighting": "standard incident relation count, smoothed with minimum weight 1",
            "relation_similarity_function": "one-hop incident relation F1 over (direction, relation, neighbor)",
            "lowest_relation_similarity_entities": per_entity[:20],
        },
    }


def relation_networks_by_entity(graph: dict[str, Any]) -> dict[str, set[tuple[str, str, str]]]:
    networks: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for relation in _as_list(graph.get("relations")):
        if not isinstance(relation, list):
            continue
        padded = relation + [""] * (4 - len(relation))
        head, rel_name, tail = padded[:3]
        head_key = rakg_entity_key(head)
        tail_key = rakg_entity_key(tail)
        rel_key = normalize_whitespace(str(rel_name)).lower()
        if not head_key or not tail_key or not rel_key:
            continue
        networks[head_key].add(("out", rel_key, tail_key))
        networks[tail_key].add(("in", rel_key, head_key))
    return networks


def relation_network_f1(eval_edges: set[tuple[str, str, str]], standard_edges: set[tuple[str, str, str]]) -> float:
    if not eval_edges and not standard_edges:
        return 1.0
    if not eval_edges or not standard_edges:
        return 0.0
    intersection = len(eval_edges & standard_edges)
    return (2 * intersection) / (len(eval_edges) + len(standard_edges))


def rakg_entity_key(value: Any) -> str:
    return normalize_whitespace(str(value or "")).casefold()


# Scoring and context helpers


def build_scores(entity_metrics: dict[str, Any], relation_metrics: dict[str, Any], graph_metrics: dict[str, Any]) -> dict[str, Any]:
    entity_traceability = average(
        [
            1 - entity_metrics["missing_definition_rate"],
            1 - entity_metrics["synthetic_or_untraced_entity_rate"],
            entity_metrics["source_reference_validity_rate"],
        ]
    )
    relation_evidence = average(
        [
            relation_metrics["evidence_coverage_rate"],
            relation_metrics["evidence_found_in_source_rate"],
            relation_metrics["source_reference_validity_rate"],
        ]
    )
    structural_health = average(
        [
            1 - graph_metrics["isolated_entity_rate"],
            graph_metrics["largest_component_rate"],
            1 - relation_metrics["high_risk_relation_rate"],
        ]
    )
    overall = average([entity_traceability, relation_evidence, structural_health])
    return {
        "overall_score": round(overall, 4),
        "entity_traceability_score": round(entity_traceability, 4),
        "relation_evidence_score": round(relation_evidence, 4),
        "structural_health_score": round(structural_health, 4),
    }


def build_relation_context(inputs: EvaluationInputs, relation: dict[str, Any], limit: int) -> str:
    chunks: list[str] = []
    for source_id in _as_list(relation.get("source_node_ids")):
        node = inputs.doc_nodes.get(str(source_id))
        if not node:
            continue
        label = normalize_whitespace(node.get("label") or node.get("title") or source_id)
        content = normalize_whitespace(node.get("content"))
        if content:
            chunks.append(f"[{source_id}] {label}\n{content}")
    context = "\n\n".join(chunks)
    if len(context) <= limit:
        return context
    return context[:limit]


def build_entity_context(inputs: EvaluationInputs, entity: dict[str, Any], limit: int) -> str:
    chunks: list[str] = []
    for source_id in _as_list(entity.get("merged_source_node_ids")):
        node = inputs.doc_nodes.get(str(source_id))
        if not node:
            continue
        label = normalize_whitespace(node.get("label") or node.get("title") or source_id)
        content = normalize_whitespace(node.get("content"))
        if content:
            chunks.append(f"[{source_id}] {label}\n{content}")
    context = "\n\n".join(chunks)
    if len(context) <= limit:
        return context
    return context[:limit]


def resolve_judge_response_format(args: argparse.Namespace, llm_settings: dict[str, Any]) -> bool:
    if args.judge_response_format_json is not None:
        return bool(args.judge_response_format_json)
    return bool(llm_settings.get("response_format_json", True))


def judge_status(attempted_count: int, valid_count: int) -> str:
    if attempted_count == 0:
        return "skipped"
    if valid_count == 0:
        return "failed"
    if valid_count < attempted_count:
        return "completed_with_errors"
    return "completed"


def short_error_reason(reason: str, limit: int = 240) -> str:
    cleaned = normalize_whitespace(reason)
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def counter_to_rows(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [{"reason": reason, "count": count} for reason, count in counter.most_common(limit)]


def resolve_pykeen_device(device_arg: str, torch_module: Any) -> str:
    requested = (device_arg or "cpu").strip().lower()
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(
            f"PyKEEN device `{device_arg}` was requested, but torch.cuda.is_available() is false. "
            "Check the NVIDIA driver and the installed PyTorch CUDA build, or use --pykeen-device cpu."
        )
    return requested


def sample_relation_ids(risk_rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[str]:
    ids = [str(row.get("relation_id") or "") for row in risk_rows if row.get("relation_id")]
    if sample_size <= 0 or sample_size >= len(ids):
        return ids
    high_risk = [str(row["relation_id"]) for row in risk_rows if int(row.get("risk_score") or 0) > 0 and row.get("relation_id")]
    remaining = [item for item in ids if item not in set(high_risk)]
    selected = high_risk[: max(1, sample_size // 2)]
    random.Random(seed).shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    return selected[:sample_size]


def sample_entities(entities: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(entities):
        return entities
    ordered = list(entities)
    random.Random(seed).shuffle(ordered)
    return ordered[:sample_size]


# Pure graph algorithms


def connected_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for node in adjacency:
        if node in seen:
            continue
        component: set[str] = set()
        queue: deque[str] = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def strongly_connected_components(out_adjacency: dict[str, set[str]]) -> list[set[str]]:
    nodes = list(out_adjacency.keys())
    reverse: dict[str, set[str]] = {node: set() for node in nodes}
    for node, neighbors in out_adjacency.items():
        for neighbor in neighbors:
            if neighbor in reverse:
                reverse[neighbor].add(node)

    visited: set[str] = set()
    order: list[str] = []
    for node in nodes:
        if node in visited:
            continue
        stack: list[tuple[str, bool]] = [(node, False)]
        while stack:
            current, expanded = stack.pop()
            if expanded:
                order.append(current)
                continue
            if current in visited:
                continue
            visited.add(current)
            stack.append((current, True))
            for neighbor in out_adjacency[current]:
                if neighbor not in visited:
                    stack.append((neighbor, False))

    components: list[set[str]] = []
    visited.clear()
    for node in reversed(order):
        if node in visited:
            continue
        component: set[str] = set()
        stack = [(node, False)]
        visited.add(node)
        while stack:
            current, _ = stack.pop()
            component.add(current)
            for neighbor in reverse[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append((neighbor, False))
        components.append(component)
    return components


def page_rank(out_adjacency: dict[str, set[str]], damping: float = 0.85, iterations: int = 50) -> dict[str, float]:
    nodes = list(out_adjacency.keys())
    if not nodes:
        return {}
    n = len(nodes)
    rank = {node: 1.0 / n for node in nodes}
    base = (1.0 - damping) / n
    for _ in range(iterations):
        next_rank = {node: base for node in nodes}
        dangling_rank = sum(rank[node] for node in nodes if not out_adjacency[node])
        dangling_share = damping * dangling_rank / n
        for node in nodes:
            next_rank[node] += dangling_share
        for node, neighbors in out_adjacency.items():
            if not neighbors:
                continue
            share = damping * rank[node] / len(neighbors)
            for neighbor in neighbors:
                if neighbor in next_rank:
                    next_rank[neighbor] += share
        rank = next_rank
    return rank


def k_core_numbers(adjacency: dict[str, set[str]]) -> dict[str, int]:
    remaining = set(adjacency.keys())
    degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
    core: dict[str, int] = {}
    current_core = 0
    while remaining:
        node = min(remaining, key=lambda item: degree[item])
        current_core = max(current_core, degree[node])
        core[node] = current_core
        remaining.remove(node)
        for neighbor in adjacency[node]:
            if neighbor in remaining:
                degree[neighbor] = max(0, degree[neighbor] - 1)
    return core


def relation_subgraph_metrics(relations: list[Any], entity_by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    edges_by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    nodes_by_type: dict[str, set[str]] = defaultdict(set)
    source_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    target_type_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for relation in relations:
        if not isinstance(relation, dict):
            continue
        relation_type = normalize_whitespace(relation.get("relation")) or "(empty)"
        head_id = str(relation.get("head_entity_id") or "")
        tail_id = str(relation.get("tail_entity_id") or "")
        if head_id not in entity_by_id or tail_id not in entity_by_id:
            continue
        edges_by_type[relation_type].append((head_id, tail_id))
        nodes_by_type[relation_type].update([head_id, tail_id])
        source_type_counts[relation_type][entity_by_id[head_id].get("type") or "(empty)"] += 1
        target_type_counts[relation_type][entity_by_id[tail_id].get("type") or "(empty)"] += 1

    metrics: dict[str, Any] = {}
    for relation_type in sorted(edges_by_type):
        edges = edges_by_type[relation_type]
        nodes = nodes_by_type[relation_type]
        undirected = {node: set() for node in nodes}
        directed = {node: set() for node in nodes}
        out_counts: Counter[str] = Counter()
        in_counts: Counter[str] = Counter()
        for head_id, tail_id in edges:
            undirected[head_id].add(tail_id)
            undirected[tail_id].add(head_id)
            directed[head_id].add(tail_id)
            out_counts[head_id] += 1
            in_counts[tail_id] += 1
        wcc_sizes = sorted((len(component) for component in connected_components(undirected)), reverse=True)
        scc_sizes = sorted((len(component) for component in strongly_connected_components(directed)), reverse=True)
        cycle_sample = find_directed_cycle(directed)
        metrics[relation_type] = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "density": graph_density(len(nodes), len(edges)),
            "weakly_connected_component_count": len(wcc_sizes),
            "largest_weak_component_rate": ratio(wcc_sizes[0] if wcc_sizes else 0, len(nodes)),
            "strongly_connected_component_count": len(scc_sizes),
            "largest_strong_component_rate": ratio(scc_sizes[0] if scc_sizes else 0, len(nodes)),
            "reciprocal_pair_rate": reciprocity_rate(edges),
            "has_directed_cycle": bool(cycle_sample),
            "cycle_sample": format_entity_path(cycle_sample, entity_by_id),
            "out_degree": numeric_summary([out_counts[node] for node in nodes]),
            "in_degree": numeric_summary([in_counts[node] for node in nodes]),
            "top_source_entity_types": dict(source_type_counts[relation_type].most_common(8)),
            "top_target_entity_types": dict(target_type_counts[relation_type].most_common(8)),
        }
    return metrics


def find_directed_cycle(out_adjacency: dict[str, set[str]]) -> list[str]:
    color = {node: 0 for node in out_adjacency}
    parent: dict[str, str] = {}

    for start in out_adjacency:
        if color[start] != 0:
            continue
        color[start] = 1
        parent[start] = ""
        stack: list[tuple[str, Any]] = [(start, iter(out_adjacency[start]))]
        while stack:
            node, iterator = stack[-1]
            try:
                neighbor = next(iterator)
            except StopIteration:
                color[node] = 2
                stack.pop()
                continue
            if neighbor not in out_adjacency:
                continue
            if color[neighbor] == 0:
                color[neighbor] = 1
                parent[neighbor] = node
                stack.append((neighbor, iter(out_adjacency[neighbor])))
                continue
            if color[neighbor] == 1:
                cycle = [neighbor]
                current = node
                while current != neighbor and current:
                    cycle.append(current)
                    current = parent.get(current, "")
                cycle.append(neighbor)
                cycle.reverse()
                return cycle
    return []


def format_entity_path(path: list[str], entity_by_id: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "entity_id": entity_id,
            "name": entity_by_id.get(entity_id, {}).get("name", ""),
            "type": entity_by_id.get(entity_id, {}).get("type", ""),
        }
        for entity_id in path
    ]


def reciprocity_rate(directed_pairs: list[tuple[str, str]]) -> float:
    if not directed_pairs:
        return 0.0
    pair_set = set(directed_pairs)
    reciprocal_count = sum(1 for head_id, tail_id in directed_pairs if (tail_id, head_id) in pair_set)
    return reciprocal_count / len(directed_pairs)


def numeric_summary(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "max": 0}
    sorted_values = sorted(float(value) for value in values)
    return {
        "min": sorted_values[0],
        "mean": sum(sorted_values) / len(sorted_values),
        "median": percentile(sorted_values, 0.5),
        "p90": percentile(sorted_values, 0.9),
        "p95": percentile(sorted_values, 0.95),
        "max": sorted_values[-1],
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def gini(values: list[int | float]) -> float:
    clean_values = sorted(float(value) for value in values if value >= 0)
    if not clean_values:
        return 0.0
    total = sum(clean_values)
    if total == 0:
        return 0.0
    weighted_sum = sum((index + 1) * value for index, value in enumerate(clean_values))
    n = len(clean_values)
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def top_entities_by_score(
    scores: dict[str, int | float],
    entity_by_id: dict[str, dict[str, str]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    rows = []
    for entity_id, score in scores.items():
        if score <= 0:
            continue
        entity = entity_by_id.get(entity_id, {})
        rows.append(
            {
                "entity_id": entity_id,
                "name": entity.get("name", ""),
                "type": entity.get("type", ""),
                "score": round(float(score), 8),
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), row["name"], row["entity_id"]))
    return rows[:limit]


# Output and generic utilities


def select_pykeen_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    wanted_suffixes = (
        "both.realistic.mean_reciprocal_rank",
        "both.realistic.hits_at_1",
        "both.realistic.hits_at_3",
        "both.realistic.hits_at_10",
        "both.optimistic.mean_reciprocal_rank",
        "both.optimistic.hits_at_10",
    )

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), item)
            return
        for suffix in wanted_suffixes:
            if prefix.endswith(suffix):
                selected[prefix] = value

    walk("", metrics)
    return selected


def summarize_issues(issues: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    for item in issues:
        by_type[str(item.get("issue_type") or "(empty)")] += 1
        by_severity[str(item.get("severity") or "(empty)")] += 1
    return {
        "total_issue_count": len(issues),
        "by_type": dict(sorted(by_type.items())),
        "by_severity": dict(sorted(by_severity.items())),
    }


def write_issues_csv(path: Path, issues: list[dict[str, Any]]) -> None:
    fieldnames = ["item_kind", "item_id", "issue_type", "severity", "message"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in issues:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_risk_relations_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["relation_id", "head", "relation", "tail", "risk_score", "risk_reasons", "source_node_ids", "evidences"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def issue(item_kind: str, item_id: str, issue_type: str, severity: str, message: str) -> dict[str, Any]:
    return {
        "item_kind": item_kind,
        "item_id": item_id,
        "issue_type": issue_type,
        "severity": severity,
        "message": message,
    }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalized_contains(text: str, needle: str) -> bool:
    normalized_text = normalize_whitespace(text)
    normalized_needle = normalize_whitespace(needle)
    return bool(normalized_needle) and normalized_needle in normalized_text


def ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def average(values: list[float]) -> float:
    clean_values = [max(0.0, min(1.0, value)) for value in values if not math.isnan(value)]
    return sum(clean_values) / len(clean_values) if clean_values else 0.0


def graph_density(entity_count: int, relation_count: int) -> float:
    if entity_count <= 1:
        return 0.0
    return relation_count / (entity_count * (entity_count - 1))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
