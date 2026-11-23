from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_kg_quality.py"


METRIC_KEYS = {
    "realistic_mrr": ("both.realistic.mean_reciprocal_rank", "both.realistic.inverse_harmonic_mean_rank"),
    "realistic_hits_at_1": ("both.realistic.hits_at_1",),
    "realistic_hits_at_3": ("both.realistic.hits_at_3",),
    "realistic_hits_at_10": ("both.realistic.hits_at_10",),
    "optimistic_mrr": ("both.optimistic.mean_reciprocal_rank", "both.optimistic.inverse_harmonic_mean_rank"),
    "optimistic_hits_at_10": ("both.optimistic.hits_at_10",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PyKEEN epoch sweep and plot link-prediction metrics.")
    parser.add_argument("--kg-output-dir", required=True, type=Path, help="Directory containing concept_kg.json.")
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "outputs" / "pykeen_epoch_sweep", help="Root directory for sweep outputs.")
    parser.add_argument("--epochs", nargs="+", type=int, required=True, help="Epoch values to run, e.g. --epochs 500 1000 2000.")
    parser.add_argument("--pykeen-model", default="TransE", help="PyKEEN model name.")
    parser.add_argument("--pykeen-device", default="auto", help="Device for PyKEEN. Use cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--pykeen-test-ratio", type=float, default=0.2, help="Held-out ratio for PyKEEN evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse run directories that already contain kg_quality_report.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.report_root / f"{args.pykeen_model}_sweep_{run_id}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for epoch in args.epochs:
        run_dir = sweep_dir / f"epochs_{epoch}"
        report_path = run_dir / "kg_quality_report.json"
        if not (args.skip_existing and report_path.exists()):
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(EVALUATOR),
                "--kg-output-dir",
                str(args.kg_output_dir),
                "--report-dir",
                str(run_dir),
                "--run-pykeen",
                "--pykeen-model",
                args.pykeen_model,
                "--pykeen-epochs",
                str(epoch),
                "--pykeen-test-ratio",
                str(args.pykeen_test_ratio),
                "--pykeen-device",
                args.pykeen_device,
                "--seed",
                str(args.seed),
            ]
            print(f"[sweep] running epochs={epoch} -> {run_dir}", flush=True)
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        rows.append(load_run_row(epoch, run_dir, args.pykeen_model))

    csv_path = sweep_dir / "pykeen_epoch_sweep.csv"
    write_csv(csv_path, rows)
    svg_path = sweep_dir / "pykeen_epoch_sweep.svg"
    write_svg_plot(svg_path, rows)
    json_path = sweep_dir / "pykeen_epoch_sweep.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[sweep] wrote CSV: {csv_path}")
    print(f"[sweep] wrote plot: {svg_path}")
    print(f"[sweep] wrote JSON: {json_path}")


def load_run_row(epoch: int, run_dir: Path, model: str) -> dict[str, Any]:
    report = load_json(run_dir / "kg_quality_report.json")
    pykeen_summary = ((report.get("optional_evaluations") or {}).get("pykeen") or {})
    metrics_path = run_dir / "pykeen_metrics.json"
    flat_metrics = flatten_metrics(load_json(metrics_path)) if metrics_path.exists() else {}
    selected = extract_metrics(flat_metrics)

    row: dict[str, Any] = {
        "epochs": epoch,
        "model": model,
        "status": pykeen_summary.get("status", ""),
        "device": pykeen_summary.get("device", ""),
        "train_triples": pykeen_summary.get("train_triples", ""),
        "test_triples": pykeen_summary.get("test_triples", ""),
        "report_dir": str(run_dir),
    }
    row.update(selected)
    return row


def extract_metrics(flat_metrics: dict[str, Any]) -> dict[str, float | str]:
    extracted: dict[str, float | str] = {}
    for output_name, candidates in METRIC_KEYS.items():
        extracted[output_name] = first_metric_value(flat_metrics, candidates)
    return extracted


def first_metric_value(flat_metrics: dict[str, Any], candidates: tuple[str, ...]) -> float | str:
    for key in candidates:
        value = flat_metrics.get(key)
        if is_number(value):
            return float(value)
    for key, value in flat_metrics.items():
        normalized = key.lower()
        if any(normalized.endswith(candidate) for candidate in candidates) and is_number(value):
            return float(value)
    return ""


def flatten_metrics(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        output: dict[str, Any] = {}
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_metrics(value, child_prefix))
        return output
    return {prefix: payload}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epochs",
        "model",
        "status",
        "device",
        "train_triples",
        "test_triples",
        "realistic_mrr",
        "realistic_hits_at_1",
        "realistic_hits_at_3",
        "realistic_hits_at_10",
        "optimistic_mrr",
        "optimistic_hits_at_10",
        "report_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_svg_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    series = [
        ("realistic_mrr", "#2563eb", "MRR(realistic)"),
        ("realistic_hits_at_1", "#16a34a", "Hits@1"),
        ("realistic_hits_at_3", "#ca8a04", "Hits@3"),
        ("realistic_hits_at_10", "#dc2626", "Hits@10"),
    ]
    plotted = [(key, color, label) for key, color, label in series if any(is_number(row.get(key)) for row in rows)]
    width, height = 1100, 650
    left, right, top, bottom = 90, 40, 70, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_y = max([float(row[key]) for key, _, _ in plotted for row in rows if is_number(row.get(key))] or [1.0])
    max_y = min(1.0, max(0.05, max_y * 1.15))

    def x_at(index: int) -> float:
        if len(rows) <= 1:
            return left + chart_w / 2
        return left + chart_w * index / (len(rows) - 1)

    def y_at(value: float) -> float:
        return top + chart_h * (1 - value / max_y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="90" y="36" font-family="Arial, sans-serif" font-size="24" font-weight="700">PyKEEN Epoch Sweep</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#111827" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        value = max_y * tick / 5
        y = y_at(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{value:.3f}</text>')

    for index, row in enumerate(rows):
        x = x_at(index)
        parts.append(f'<line x1="{x:.2f}" y1="{top + chart_h}" x2="{x:.2f}" y2="{top + chart_h + 6}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + chart_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{row["epochs"]}</text>')

    for key, color, label in plotted:
        points = []
        for index, row in enumerate(rows):
            if is_number(row.get(key)):
                points.append((x_at(index), y_at(float(row[key]))))
        if len(points) >= 2:
            parts.append(
                '<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>'.format(
                    color=color,
                    points=" ".join(f"{x:.2f},{y:.2f}" for x, y in points),
                )
            )
        for x, y in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    legend_x = left + chart_w - 230
    for index, (_, color, label) in enumerate(plotted):
        y = top + index * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 22}" y="{y}" font-family="Arial, sans-serif" font-size="14">{label}</text>')

    parts.append(f'<text x="{left + chart_w / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">epochs</text>')
    parts.append(f'<text x="24" y="{top + chart_h / 2}" transform="rotate(-90 24 {top + chart_h / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">metric value</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) or (isinstance(value, str) and value.strip() != "" and can_float(value))


def can_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
