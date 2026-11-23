"""Verify DVKG's text-free result archive using the Python standard library."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import mean, median, stdev


ROOT = Path(__file__).resolve().parent
HEX = re.compile(r"[0-9a-f]{64}\Z")
PARAGRAPH_ID = re.compile(
    r"book_[a-f0-9]{12}/paragraph/(?:[a-f0-9]{24,64}|"
    r"(?:physical|pdfpara):book_[a-f0-9]{12}:(?:\d+:\d+:\d+|merge:[a-f0-9]{16}))\Z"
)
COURSES = {"CO", "OS", "NET", "computer_organization", "operating_systems", "computer_networks"}
POOL_SCOPES = ("same_subject_pool", "mixed_six_book_pool")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def rows(name):
    with (ROOT / name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def require(condition, label):
    if not condition:
        raise ValueError(label)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), label)


def verify_checksums():
    require(not any(p.is_symlink() for p in ROOT.rglob("*")), "symlink in archive")
    targets = {}
    for line in (ROOT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and name not in targets, "invalid checksum path")
        require(HEX.fullmatch(expected), "invalid checksum hash")
        require(sha256(path.read_bytes()).hexdigest() == expected, f"checksum: {name}")
        targets[name] = expected
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    require(set(targets) == actual - {"checksums.sha256"}, "unlisted or missing archive files")
    return len(targets)


def verify_inventory():
    books = read("textbooks.json")["books"]
    require(len({b["book_id"] for b in books}) == len(books) == 8, "eight books")
    require(len({b["course_id"] for b in books}) == 3, "three courses")
    require(sum(b["source_units"] for b in books) == 2290, "source unit count")
    require(sum(b["original_paragraphs"] for b in books) == 36048, "paragraph count")
    require(sum(b["chapters"] for b in books) == 87, "chapter count")
    require(sum(b["sections"] for b in books) == 2083, "section count")
    require(sum(b["current_main_table_pool"] for b in books) == 6, "main textbook pool")
    topology = read("topology.json")["rows"]
    require(len(topology) == 25, "topology count")
    for row in topology:
        n, m, degree, density, clustering, components, nfi, length = row["values"]
        close(degree, 2 * m / n, "mean degree")
        close(density, 2000 * m / (n * (n - 1)), "scaled density")
        require(0 <= clustering <= 1 and 0 <= nfi <= 1 and 1 <= components <= n and length >= 0, "topology ranges")
    geometry = read("geometry.json")["rows"]
    require(len(geometry) == 21, "geometry count")
    for row in geometry:
        close(row["global_efficiency"], row["inverse_distance_sum"] / row["possible_ordered_pairs"], "global efficiency normalization")
        close(row["finite_pair_mean_distance"], row["finite_distance_sum"] / row["finite_ordered_pairs"], "finite distance mean")
        if row["simple_edges"]:
            close(row["mean_forman_edge_curvature"], row["forman_edge_curvature_sum"] / row["simple_edges"], "Forman edge mean")


def verify_kge():
    count = 0
    for path in (ROOT / "kge_splits").glob("*.json"):
        bundle = read(path.relative_to(ROOT))
        expected_id = bundle.pop("bundle_id")
        canonical = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        require(sha256(canonical.encode()).hexdigest() == expected_id, f"bundle ID: {path.name}")
        require(not bundle["raw_text_exported"], f"graph text privacy: {path.name}")
        fold_pairs = {}
        flattened = set()
        train = bundle["splits"]["train"]
        entities = {v for head, _, tail in train for v in (head, tail)}
        relations = {r for _, r, _ in train}
        for fold, triples in bundle["splits"].items():
            require(len(triples) == bundle["counts"][fold], f"split count: {path.name}/{fold}")
            for head, relation, tail in triples:
                triple = (head, relation, tail)
                require(all(HEX.fullmatch(v) for v in triple), "opaque triple ID")
                require(triple not in flattened, "repeated triple")
                flattened.add(triple)
                pair = tuple(sorted((head, tail)))
                require(fold_pairs.setdefault(pair, fold) == fold, "cross-fold pair leakage")
                require(head in entities and tail in entities and relation in relations, "non-transductive split")
        require(len(flattened) == bundle["counts"]["unique_triples"], "unique triple count")
        count += 1
    require(count == 4, "KGE bundle count")
    metrics = read("kge_metrics.json")
    require(len(metrics["runs"]) == 24 and len(metrics["summaries"]) == 8, "KGE run count")
    for row in metrics["summaries"]:
        runs = [r for r in metrics["runs"] if (r["name"], r["model"]) == (row["name"], row["model"])]
        require(sorted(r["seed"] for r in runs) == [42, 43, 44], "KGE seeds")
        require(all(r["actual_epochs"] == 1000 for r in runs), "KGE epoch count")
        for name, metric in row["metrics"].items():
            values = [r["metrics"][name] for r in runs]
            close(mean(values), metric["mean"], "KGE mean")
            close(stdev(values), metric["std"], "KGE sample standard deviation")
    return count


def macro(rows, field, book_field="book"):
    by_book = defaultdict(list)
    for row in rows:
        by_book[row[book_field]].append(row[field])
    return mean(mean(values) for values in by_book.values())


def verify_qa():
    qa = {}
    expected_counts = {"textbook_qa": 1380, "ablation": 690, "expanded_pools_archived": 3530}
    keys = set("course method book qid scope support_paragraph_ids retrieved_paragraph_ids expected_f1 expected_recall dataset_sha256 record_sha256".split())
    for name, expected in expected_counts.items():
        group = rows(f"qa/{name}.jsonl")
        require(len(group) == expected, f"QA row count: {name}")
        identities = set()
        for row in group:
            require(set(row) == keys, "unexpected QA field")
            require(row["course"] in COURSES, "unknown QA course")
            require(all(HEX.fullmatch(row[k]) for k in ("qid", "dataset_sha256", "record_sha256")), "non-opaque QA run ID")
            identity = (row["method"], row["scope"], row["book"], row["qid"])
            require(identity not in identities, "duplicate QA row")
            identities.add(identity)
            gold = set(row["support_paragraph_ids"])
            ranked = list(dict.fromkeys(row["retrieved_paragraph_ids"]))[:5]
            require(bool(gold), "empty annotated support set")
            require(
                all(PARAGRAPH_ID.fullmatch(v) for v in gold)
                and all(PARAGRAPH_ID.fullmatch(v) for v in row["retrieved_paragraph_ids"]),
                "non-opaque paragraph ID",
            )
            close(len(gold.intersection(ranked)) / len(gold), row["expected_recall"], "original Recall@5")
            require(0 <= row["expected_f1"] <= 1, "saved F1 range")
        qa[name] = group
    for name, expected in read("qa/aggregates.json").items():
        for e in expected:
            subset = [r for r in qa[name] if (r["method"], r["course"]) == (e["method"], e["course"])]
            for saved, key in (("expected_f1", "f1"), ("expected_recall", "paragraph_recall_at5")):
                value = mean(r[saved] for r in subset) if name == "ablation" else macro(subset, saved)
                close(value, e[key], f"{name} {key}")
    latest = read("qa/expanded_pools_current.json")["rows"]
    reranker = rows("qa/expanded_pools_cross_encoder_saved_f1.jsonl")
    require(Counter(r["scope"] for r in reranker) == {scope:353 for scope in POOL_SCOPES}, "reranker F1 row coverage")
    require(all(r["status"] == "completed" for r in reranker), "reranker completion")
    for e in latest:
        if e["method"] == "DVKG+cross-encoder":
            subset = [r for r in reranker if r["scope"] == e["scope"]]
            close(macro(subset, "f1", "book_id"), e["f1"], "reranker F1 aggregate")
        else:
            subset = [r for r in qa["expanded_pools_archived"] if (r["method"], r["scope"]) == (e["method"], e["scope"])]
            close(macro(subset, "expected_f1"), e["f1"], "expanded-pool baseline F1 aggregate")
            close(macro(subset, "expected_recall"), e["paragraph_recall_at5"], "expanded-pool baseline Recall aggregate")
        require(len(subset) == e["question_count"] == 353, "current expanded-pool group count")
    return sum(expected_counts.values()), len(reranker)


def verify_efficiency():
    cost = read("construction_cost.json")
    dvkg = next(r for r in cost["rows"] if r["method"] == "DVKG")
    stages = cost["dvkg_stages"].values()
    require(len(cost["rows"]) == 5, "cost row count")
    for direction in ("input", "output"):
        require(sum(s[f"LLM_{direction}_tokens"] for s in stages) == dvkg[f"llm_{direction}_tokens"], "DVKG token sum")
    close((sum(s["elapsed_seconds"] for s in stages) + cost["dvkg_unattributed_seconds"]) / 60, dvkg["active_minutes"], "DVKG active time sum")
    timing = read("query_latency.json")
    groups = defaultdict(list)
    for r in timing["samples"]:
        groups[(r["course"], r["method"], r["condition"])].append(r["seconds"])
    require(len(groups) == 12 and all(len(g) == 10 and min(g) > 0 for g in groups.values()), "timing sample groups")
    for (course, method, condition), seconds in sorted(groups.items()):
        p95 = sorted(seconds)[math.ceil(0.95 * len(seconds)) - 1]
        print(f"Latency {course:3} {method:11} {condition:20} P50 {median(seconds):6.2f} s, P95 {p95:6.2f} s")
    require(len(timing["network_native_aggregates"]) == 3, "aggregate-only network timing groups")
    return len(timing["samples"])


def main():
    checksums = verify_checksums()
    verify_inventory()
    splits = verify_kge()
    recall, f1 = verify_qa()
    timing = verify_efficiency()
    print(f"Verified {checksums} file hashes, {splits} KGE bundles, {recall} Recall@5 rows, {f1} reranker F1 scores for aggregation, and {timing} latency samples.")


if __name__ == "__main__":
    main()
