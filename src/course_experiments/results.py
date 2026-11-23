from __future__ import annotations

import html
import json
from pathlib import Path
from .manifest import digest, file_record, read_document
from .run import validate_entry, verify_cached, validation_failure
from .comparison import validate_comparisons

COLUMNS = ('course_id', 'book_set', 'method_id', 'state', 'evaluation', 'epochs', 'control', 'scope', 'comparison_id', 'comparison_role', 'layer', 'status', 'score', 'counts', 'coverage', 'cost', 'metrics', 'raw_path', 'raw_paths', 'version', 'reason')


def summarize(plan_path, root):
    plan = read_document(plan_path)
    if plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'}):
        raise ValueError('Plan changed')
    validate_comparisons(plan)
    rows = []
    for item in plan['positions']:
        row = {k: item.get(k) for k in COLUMNS}
        row['epochs'] = item.get('entry', {}).get('epochs')
        row['control'] = item.get('entry', {}).get('control', 'graph' if item['evaluation'] == 'rag' else None)
        row['scope'] = item.get('entry', {}).get('scope')
        row['comparison_id'] = item.get('identity', {}).get('comparison_id')
        row['comparison_role'] = item.get('entry', {}).get('comparison_role', 'main')
        if item['status'] == 'not_applicable':
            try:
                verified = validate_entry(item, item['entry'])
                if verified['status'] != 'not_applicable':
                    raise ValueError('N/A capability changed')
            except (ValueError, KeyError, OSError, TypeError) as exc:
                row.update(status='blocked', reason=str(exc))
        path = Path(root) / item.get('job_id', 'missing') / 'record.json'
        if item['status'] == 'ready':
            try:
                current = validate_entry(item, item['entry'])
                if current.get('job_id') != item['job_id']:
                    raise ValueError('Result cache identity mismatch')
            except (ValueError, KeyError, OSError, TypeError) as exc:
                failure = validation_failure(item, path.parent, exc)
                row.update(status='failed', reason=str(exc), raw_paths=failure['raw_paths'])
                rows.append(row)
                continue
        if item['status'] == 'ready' and path.exists():
            try:
                record = read_document(path)
                result = verify_cached(item, record)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                failure = validation_failure(item, path.parent, exc)
                row.update(status='failed', reason=str(exc), raw_paths=failure['raw_paths'])
                rows.append(row)
                continue
            row.update(status=record['status'], reason=record.get('error_message'), version=current['job_id'], raw_paths=record.get('raw_paths', []))
            if result is not None:
                metrics = result.get('metrics')
                if record.get('metrics_record'):
                    metrics = read_document(record['metrics_record']['path'])
                row.update(raw_path=record['raw_record']['path'], layer=result.get('layer'),
                           counts={k: result[k] for k in ('entity_count', 'relation_count', 'question_count', 'completed_count', 'train_triples', 'validation_triples', 'test_triples') if k in result},
                           coverage={k: result[k] for k in ('answerability_coverage', 'fixed_test_count', 'covered_test_count', 'common_test_count') if k in result},
                           cost=result.get('usage'), metrics=metrics, score=None,
                           reason=result.get('error_message') or result.get('reason') or record.get('error_message'),
                           raw_paths=result.get('raw_paths', result.get('existing_artifact_paths', [])))
        elif item['status'] == 'ready' and list(path.parent.glob('validation_failure_*.json')):
            failures = sorted(path.parent.glob('validation_failure_*.json'))
            failure = read_document(failures[-1])
            row.update(status='failed', reason=failure['error_message'], raw_paths=[str(p) for p in failures])
        elif item['status'] == 'ready':
            row['status'] = 'pending'
        rows.append(row)
    main_rows = [r for r in rows if r['comparison_role'] == 'main']
    applicable = [r for r in main_rows if not (r['status'] == 'not_applicable' and r['state'] == 'native') and not r['method_id'].lower().startswith('tree')]
    required = {'k12_kgraph', 'kggen_lm_dedup', 'graphrag_extract'}
    actual = {r['method_id'].lower() for r in rows if r['state'] == 'native'}
    groups = {(r['course_id'], r['book_set']) for r in rows}
    controls_complete = all({'vector_only', 'no_retrieval'} <= {r['control'] for r in rows if (r['course_id'], r['book_set']) == g and r['evaluation'] == 'rag'} for g in groups)
    kge_groups = {(r['course_id'], r['book_set'], r['method_id'], r['state']) for r in rows if r['evaluation'] == 'kge' and not (r['status'] == 'not_applicable' and r['state'] == 'native') and not r['method_id'].lower().startswith('tree')}
    epochs_complete = all({500, 1000} <= {r['epochs'] for r in main_rows if (r['course_id'], r['book_set'], r['method_id'], r['state']) == g and r['evaluation'] == 'kge'} for g in kge_groups)
    complete = bool(plan.get('manifest_path') and plan.get('comparisons')) and epochs_complete and controls_complete and bool(applicable) and required <= actual and all(r['status'] == 'completed' and r['scope'] == 'full' and r['comparison_id'] for r in applicable)
    return dict(core_comparison_complete=complete, comparison_positions=len(rows),
                unique_executions=len({p['job_id'] for p in plan['positions'] if 'job_id' in p}), rows=rows)


def report(plan_path, root, output):
    value = summarize(plan_path, root)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'summary.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')
    def cell(v):
        return html.escape(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) if v is not None else 'null')
    filters = ('course_id', 'book_set', 'method_id', 'state', 'evaluation', 'epochs', 'control', 'scope', 'layer', 'status')
    controls = ''.join('<label>' + k + '<select data-column="' + str(COLUMNS.index(k)) + '"><option value="">all</option>' + ''.join('<option>' + html.escape(str(v)) + '</option>' for v in sorted({str(r[k]) for r in value['rows'] if r[k] is not None})) + '</select></label> ' for k in filters)
    table = '<tr>' + ''.join('<th>' + k + '</th>' for k in COLUMNS) + '</tr>'
    table += ''.join('<tr>' + ''.join('<td>' + cell(r[k]) + '</td>' for k in COLUMNS) + '</tr>' for r in value['rows'])
    page = '<!doctype html><meta charset="utf-8"><title>Course experiment results</title><style>body{font:14px sans-serif}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px;max-width:380px;overflow-wrap:anywhere}select{margin:6px}</style><h1>Verified experiment results</h1><p>Core comparison complete: ' + str(value['core_comparison_complete']) + '. Positions: ' + str(value['comparison_positions']) + '; unique executions: ' + str(value['unique_executions']) + '. Null is unavailable, never zero. Machine source references are not human gold.</p>' + controls + '<table>' + table + '</table><script>document.querySelectorAll("select").forEach(s=>s.onchange=()=>document.querySelectorAll("table tr").forEach((r,i)=>{if(i)r.hidden=Array.from(document.querySelectorAll("select")).some(x=>x.value&&r.cells[+x.dataset.column].textContent!==x.value)}));</script>'
    (directory / 'index.html').write_text(page, 'utf-8')
    return value
