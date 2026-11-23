from __future__ import annotations

import argparse
import math
from pathlib import Path
from urllib.error import HTTPError

from .manifest import digest, file_record, read_document
from .contributions import write_immutable, export_contribution
from .matrix import generate, require_executable
from .evaluation.protocol import validate_phase, bound_input
from .evaluation.chat import account_status
from .comparison import comparison_inputs, bind_comparisons, validate_binding, validate_comparisons

STAGES = {
    'rag': ('construction', 'reference', 'index', 'graph', 'phase', 'output'),
    'topology': ('graph', 'evaluator', 'phase', 'output'),
    'kge': ('graph', 'protocol', 'patched_evaluator', 'phase', 'output'),
}


def textbook_only(course_id, schema_id, contributions, active_books, *, runtime=None, alignment=None):
    from .fusion import build_view
    carrier = export_contribution(dict(nodes={}, claims={}, evidence={}, books={}), course_id, None,
                                  'full', digest(dict(course_id=course_id, schema_id=schema_id,
                                                     role='bookkeeping_carrier/textbook_only')))
    return build_view(carrier, contributions, active_books, runtime=runtime, alignment=alignment)


def freeze_catalog(entries, output, additional=None):
    payload = dict(format_version=1, entries=entries, additional=additional or [])
    payload['catalog_id'] = digest(payload)
    return write_immutable(output, payload)


def validate_entry(row, entry):
    phase = validate_phase(entry['phase'])
    metadata = read_document(entry['metadata']['path'])
    if file_record(entry['metadata']['path']) != entry['metadata']:
        raise ValueError('Graph metadata changed')
    for key in ('course_id', 'method_id', 'state'):
        if metadata.get(key) != row[key]:
            raise ValueError('Graph provenance mismatch: ' + key)
    if metadata.get('scope') != entry['scope'] or entry['scope'] not in {'full', 'smoke'}:
        raise ValueError('Scope mismatch')
    if not metadata.get('source_records'):
        raise ValueError('Source provenance absent')
    frozen_records = list(phase['files'].values())
    for source in metadata['source_records']:
        if source not in frozen_records or file_record(source['path']) != source:
            raise ValueError('Source not frozen or changed')
    if metadata.get('status') != 'complete':
        raise ValueError('Graph not complete')
    if metadata.get('phase_id') != phase['phase_id']:
        raise ValueError('Graph phase mismatch')
    if not (row['state'] == 'fixed_g0' and row['evaluation'] != 'rag') and sorted(metadata.get('book_ids', [])) != sorted(row['book_ids']):
        raise ValueError('Graph bookset mismatch')
    baseline = None
    if row['state'] in {'native', 'schema_aligned'}:
        from .manifest import baseline_ready
        record = row['manifest_record']
        if file_record(record['path']) != record:
            raise ValueError('Canonical method manifest changed')
        baseline = row['canonical_method']
        methods = read_document(record['path'])['experiments']['methods']
        if baseline not in methods or not baseline_ready(baseline, 'rag'):
            raise ValueError('Native method is not canonically verified')
        expected_method = baseline['method_id'] + ('_schema_aligned' if row['state'] == 'schema_aligned' else '')
        if row['method_id'] != expected_method or metadata.get('native_commit') != baseline['commit']:
            raise ValueError('Native source commit/method mismatch')
    if metadata.get('graph') != file_record(entry['inputs']['graph']):
        raise ValueError('Graph hash mismatch')
    for role, path in entry['inputs'].items():
        bound_input(phase, role, read_document(path))
    if entry.get('native_semantic_kge') is False and row['evaluation'] == 'kge':
        if row['state'] != 'native' or baseline is None:
            raise ValueError('Semantic KGE N/A restricted to verified native graphs')
        artifact = read_document(phase['files']['native_artifact']['path'])
        if artifact.get('artifact_id') != digest({k:v for k,v in artifact.items() if k != 'artifact_id'}):
            raise ValueError('Native artifact digest mismatch')
        for key, value in [('course_id', row['course_id']), ('method_id', baseline['method_id']),
                           ('native_commit', baseline['commit']), ('scope', entry['scope']), ('status', 'complete')]:
            if artifact.get(key) != value:
                raise ValueError('Native artifact provenance mismatch: ' + key)
        native = artifact['native_graph']
        view = artifact['semantic_view']
        graph = read_document(entry['inputs']['graph'])
        if graph != dict(entities=view['entities'], relations=view['relations']):
            raise ValueError('Native capability artifact does not describe evaluated graph')
        reason = view.get('capability_reasons', {}).get('kge')
        untyped = any(e.get('type_status') == 'native_untyped_candidate' for e in view['entities'])
        if view.get('capabilities', {}).get('kge', True) is not None or not reason or (native.get('directed') is not False and not untyped):
            raise ValueError('Native graph does not establish semantic KGE N/A')
        return dict(status='not_applicable', reason=reason, score=None)
    expected = set(STAGES[row['evaluation']]) - {'phase', 'output'}
    if set(entry['arguments']) != expected:
        raise ValueError('Unknown or missing stage arguments')
    for flag, role in {'evaluator': 'evaluator_original', 'patched_evaluator': 'evaluator_patched'}.items():
        if flag in entry['arguments'] and file_record(entry['arguments'][flag]) != phase['files'].get(role):
            raise ValueError('Evaluator not phase bound')
    for role in expected - {'evaluator', 'patched_evaluator', 'protocol'}:
        if entry['arguments'][role] != entry['inputs'].get(role):
            raise ValueError('Argument differs from frozen input')
    if row['evaluation'] == 'kge':
        from .evaluation.kge import validate_protocol
        protocol = validate_protocol(read_document(entry['arguments']['protocol']))
        if protocol['phase_id'] != phase['phase_id'] or protocol['scope'] != entry['scope'] or entry['epochs'] not in protocol['budget']['epochs']:
            raise ValueError('KGE protocol/scope/epochs mismatch')
    if row['evaluation'] == 'rag':
        from .evaluation.rag import BUDGET
        from .evaluation.reference import validate_reference
        reference = validate_reference(read_document(entry['inputs']['reference']))
        if entry['scope'] == 'full' and (len(reference['questions']) < 100 or len(reference['facts']) < 400):
            raise ValueError('Formal reference below 100Q/400facts')
        if reference['course_id'] != row['course_id'] or reference['scope'] != entry['scope']:
            raise ValueError('Reference course/scope mismatch')
        from .evaluation.protocol import check_runtime
        construction = read_document(entry['inputs']['construction'])
        config_record = phase['files'].get('runtime_config') or construction['files']['assets']['runtime_config']
        if file_record(config_record['path']) != config_record:
            raise ValueError('Runtime changed')
        check_runtime(read_document(config_record['path']))
        if phase['parameters'] != dict(BUDGET, control=entry.get('control', 'graph'), active_books=sorted(row['book_ids'])):
            raise ValueError('RAG budget/bookset mismatch')
    evidence = comparison_inputs(row, entry, phase)
    comparison_id = validate_binding(dict(row, entry=entry), evidence)
    identity = dict(course_id=row['course_id'], method_id=row['method_id'], state=row['state'],
                    evaluation=row['evaluation'], scope=entry['scope'], phase=phase,
                    inputs={k: file_record(v) for k, v in entry['inputs'].items()},
                    options={k: entry.get(k) for k in ('control', 'epochs', 'device', 'native_undirected')},
                    protocol=file_record(entry['arguments']['protocol']) if 'protocol' in entry['arguments'] else None,
                    scheduler=file_record(__file__), baseline=baseline,
                    baseline_manifest=row.get('manifest_record') if baseline else None,
                    comparison_inputs=evidence, comparison_role=entry.get('comparison_role', 'main'))
    if comparison_id is not None:
        identity['comparison_id'] = comparison_id
    if row['evaluation'] == 'rag':
        identity['book_ids'] = sorted(row['book_ids'])
    return dict(status='ready', job_id=digest(identity), identity=identity)


def prepare(manifest_path, catalog_path, output):
    from .manifest import validate
    frozen = validate(manifest_path)
    catalog = read_document(catalog_path)
    if catalog['catalog_id'] != digest({k: v for k, v in catalog.items() if k != 'catalog_id'}):
        raise ValueError('Catalog changed')
    positions = []
    canonical_rows = generate(frozen)
    method_map = {m['method_id']: m for m in frozen['experiments']['methods']}
    def with_method(row):
        return dict(row, canonical_method=method_map[row['method_id']], manifest_record=file_record(manifest_path))
    scheduled = [(with_method(row), catalog['entries'].get(row['row_id'])) for row in canonical_rows]
    for extra in catalog.get('additional', []):
        original = next(r for r in canonical_rows if r['row_id'] == extra['row_id'])
        require_executable(original, manifest_path)
        row = with_method(original)
        if extra.get('variant') == 'schema_aligned' and original['state'] == 'native':
            row['method_id'] += '_schema_aligned'
            row['state'] = 'schema_aligned'
        elif extra.get('epochs') in {2, 500, 1000} and original['evaluation'] == 'kge' and extra['entry'].get('epochs') == extra['epochs']:
            pass
        elif extra.get('control') in {'vector_only', 'no_retrieval'} and original['evaluation'] == 'rag':
            pass
        else:
            raise ValueError('Invalid additional comparison')
        scheduled.append((row, extra['entry']))
    for row, entry in scheduled:
        item = dict(row, score=None)
        if entry is None:
            item.update(status='unavailable' if row['method_id'].lower().startswith('tree') else 'missing', reason='No frozen catalog input')
        else:
            try:
                require_executable(next(r for r in canonical_rows if r['row_id'] == row['row_id']), manifest_path)
                item.update(validate_entry(row, entry), entry=entry)
            except (ValueError, KeyError, OSError) as exc:
                item.update(status='blocked', reason=str(exc))
        positions.append(item)
    comparisons = bind_comparisons(positions, manifest_path)
    plan = dict(manifest_path=str(Path(manifest_path).resolve()), catalog_path=str(Path(catalog_path).resolve()),
                positions=positions, comparisons=comparisons)
    plan['plan_id'] = digest(plan)
    return write_immutable(output, plan)


def execute_stage(item, output):
    from .evaluation.__main__ import main
    entry = item['entry']
    argv = [item['evaluation']]
    for flag, value in dict(entry['arguments'], phase=entry['phase'], output=str(output)).items():
        argv += ['--' + flag.replace('_', '-'), str(value)]
    if item['evaluation'] == 'rag':
        argv += ['--control', entry.get('control', 'graph')]
        for book in item['book_ids']:
            argv += ['--active-book', book]
    if item['evaluation'] == 'topology' and entry.get('native_undirected'):
        argv += ['--native-undirected']
    if item['evaluation'] == 'kge':
        argv += ['--epochs', str(entry['epochs']), '--device', entry.get('device', 'cuda:0')]
    main(argv)
    candidates = [output] if output.is_file() else sorted(output.rglob('*result*.json'))
    if not candidates:
        raise ValueError('Stage produced no result')
    return read_document(candidates[-1]), candidates[-1]


def verify_result(item, result):
    if not result.get('status'):
        raise ValueError('Result status absent')
    if item['evaluation'] in {'rag', 'topology'} and result.get('input_digest') != read_document(item['entry']['phase'])['phase_id']:
        raise ValueError('Result phase mismatch')
    for key, expected in [('course_id', item['course_id']), ('scope', item['entry']['scope']),
                          ('graph_id', digest(read_document(item['entry']['inputs']['graph'])))]:
        if key in result and result[key] != expected:
            if key != 'graph_id' or result[key] != read_document(item['entry']['inputs']['graph']).get('graph_id'):
                raise ValueError('Result identity mismatch: ' + key)
    if item['evaluation'] == 'kge':
        p = read_document(item['entry']['arguments']['protocol'])
        for key, expected in [('protocol_id', p['protocol_id']), ('split_id', p['projection']['split_id']),
                              ('graph_id', p['projection']['graph_id']), ('scope', p['scope'])]:
            if result.get(key) != expected:
                raise ValueError('GPU result mismatch: ' + key)
        expected_run = dict(protocol_id=p['protocol_id'], phase_id=p['phase_id'], graph_id=p['projection']['graph_id'], split_id=p['projection']['split_id'], epochs=item['entry']['epochs'], requested_device=item['entry'].get('device', 'cuda:0'), options=p['budget'])
        if result.get('run_id') != digest(expected_run):
            raise ValueError('GPU result phase/run mismatch')
        if result.get('options') != p['budget']:
            raise ValueError('GPU result budget mismatch')
        projection = p['projection']
        mask = p['common_test_mask'] if p.get('common_test_mask') is not None else projection['covered_test_mask']
        expected_ids = [value for value, selected in zip(projection['test_ids'], mask) if selected]
        counts = dict(test_ids=expected_ids, fixed_test_count=len(projection['test']),
                      covered_test_count=sum(projection['covered_test_mask']), common_test_count=sum(mask))
        if result['status'] in {'completed', 'completed_with_diagnostic_failure'}:
            counts.update(epochs=item['entry']['epochs'], train_triples=len(projection['training']),
                          validation_triples=sum(projection['covered_validation_mask']), test_triples=sum(mask))
        elif 'epochs' in result:
            counts['epochs'] = item['entry']['epochs']
        for key, value in counts.items():
            if result.get(key) != value:
                raise ValueError('GPU result evaluated selection mismatch: ' + key)
    return result



def result_record(item, result, raw):
    record = dict(job_id=item['job_id'], identity=item['identity'], status=result['status'], score=None,
                  raw_record=file_record(raw))
    if item['evaluation'] == 'kge':
        metrics_path = result.get('metrics_file')
        if result['status'] in {'completed', 'completed_with_diagnostic_failure'} and not metrics_path:
            raise ValueError('Completed KGE result lacks actual metrics artifact')
        if metrics_path:
            path = Path(metrics_path).resolve(strict=True)
            if path.name != 'pykeen_metrics.json' or path.parent != Path(raw).resolve(strict=True).parent:
                raise ValueError('KGE metrics must be the actual sibling PyKEEN artifact')
            def metric_value(value):
                if isinstance(value, dict): return all(metric_value(v) for v in value.values())
                if isinstance(value, list): return all(metric_value(v) for v in value)
                return value is None or type(value) in (int, float) and math.isfinite(value)
            if not metric_value(read_document(path)):
                raise ValueError('KGE metrics contain nonnumeric/source data')
            record['metrics_record'] = file_record(path)
    return record


def verify_cached(item, record):
    if validate_entry(item, item['entry']).get('job_id') != item['job_id']:
        raise ValueError('Cache comparison input changed')
    if record['job_id'] != item['job_id'] or record['identity'] != item['identity']:
        raise ValueError('Cache identity changed')
    if record.get('raw_record'):
        raw = record['raw_record']
        if file_record(raw['path']) != raw:
            raise ValueError('Cached raw result changed')
        result = verify_result(item, read_document(raw['path']))
        expected = result_record(item, result, raw['path'])
        if expected.get('metrics_record') != record.get('metrics_record'):
            raise ValueError('Cached metrics artifact changed')
        return result
    return None


def validation_failure(item, directory, exc):
    record = dict(job_id=item['job_id'], identity=item['identity'], status='failed', score=None,
                  failure_stage='input_validation', error_type=type(exc).__name__, error_message=str(exc))
    path = directory / ('validation_failure_' + digest(record) + '.json')
    write_immutable(path, record)
    return dict(record, raw_paths=[str(path)])


def run(plan_path, root, executor=execute_stage):
    plan = read_document(plan_path)
    if plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'}):
        raise ValueError('Plan changed')
    validate_comparisons(plan)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stopped = False
    for item in plan['positions']:
        if item['status'] != 'ready':continue
        cache = root / item['job_id'] / 'record.json'
        try:
            if cache.exists():
                record = read_document(cache)
                if (isinstance(record, dict) and record.get('account_stop')
                        and record.get('job_id') == item['job_id'] and record.get('identity') == item['identity']):
                    stopped = True
        except (ValueError, OSError):
            # 缓存及实际文件错误由下方逐任务校验持久化，不阻断其他任务。
            continue
    records = []
    for item in plan['positions']:
        if item['status'] != 'ready':
            records.append(item)
            continue
        directory = root / item['job_id']
        directory.mkdir(exist_ok=True)
        cache = directory / 'record.json'
        try:
            current = validate_entry(item, item['entry'])
            if current.get('job_id') != item['job_id']:
                raise ValueError('Prepared identity changed')
            if cache.exists():
                record = read_document(cache)
                verify_cached(item, record)
                stopped = stopped or record.get('account_stop', False)
                records.append(dict(item, **record))
                continue
        except (ValueError, KeyError, OSError, TypeError) as exc:
            stopped = stopped or (isinstance(exc, HTTPError) and exc.code in {401, 402, 403})
            record = validation_failure(item, directory, exc)
            records.append(dict(item, **record))
            continue
        if stopped:
            records.append(dict(item, status='blocked', reason='Account/model calls stopped', score=None))
            continue
        lock = directory / 'job.lock'
        owned = False
        try:
            with lock.open('x'):
                owned = True
                try:
                    destination = directory / ('topology.json' if item['evaluation'] == 'topology' else 'raw')
                    result, raw = executor(item, destination)
                    verify_result(item, result)
                    record = result_record(item, result, raw)
                except Exception as exc:
                    status_code = account_status(exc)
                    stopped = status_code is not None
                    record = dict(job_id=item['job_id'], identity=item['identity'], status='failed', score=None,
                                  error_type=type(exc).__name__, error_message=str(exc), account_stop=stopped, status_code=status_code,
                                  raw_paths=[str(p) for p in directory.rglob('*') if p.is_file() and p != lock])
                write_immutable(cache, record)
                records.append(dict(item, **record))
        except FileExistsError:
            records.append(dict(item, status='locked', score=None))
        finally:
            if owned:
                lock.unlink(missing_ok=True)
    return records


def collect(plan_path, job_id, result_path, root):
    plan = read_document(plan_path)
    if plan['plan_id'] != digest({k:v for k,v in plan.items() if k != 'plan_id'}):
        raise ValueError('Plan changed')
    validate_comparisons(plan)
    item = next(p for p in plan['positions'] if p.get('job_id') == job_id)
    current = validate_entry(item, item['entry'])
    if current['job_id'] != job_id:
        raise ValueError('Collection input changed')
    result = verify_result(item, read_document(result_path))
    record = result_record(item, result, result_path)
    directory = Path(root) / job_id
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'job.lock'
    owned = False
    try:
        with lock.open('x'):
            owned = True
            return write_immutable(directory / 'record.json', record)
    finally:
        if owned:
            lock.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description='Frozen matrix scheduler')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    for flag in ('manifest', 'catalog', 'output'): a.add_argument('--' + flag, required=True)
    a = sub.add_parser('run')
    for flag in ('plan', 'root'): a.add_argument('--' + flag, required=True)
    a = sub.add_parser('collect')
    for flag in ('plan', 'job-id', 'result', 'root'): a.add_argument('--' + flag, required=True)
    a = sub.add_parser('report')
    for flag in ('plan', 'root', 'output'): a.add_argument('--' + flag, required=True)
    args = p.parse_args(argv)
    if args.command == 'prepare': prepare(args.manifest, args.catalog, args.output)
    elif args.command == 'run': run(args.plan, args.root)
    elif args.command == 'collect': collect(args.plan, args.job_id, args.result, args.root)
    else:
        from .results import report
        report(args.plan, args.root, args.output)


if __name__ == '__main__':
    main()
