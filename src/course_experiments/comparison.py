from __future__ import annotations

from .manifest import digest, file_record, read_document, validate


def comparison_inputs(row, entry, phase):
    role = entry.get('comparison_role', 'main')
    if role not in {'main', 'own_coverage'} or role == 'own_coverage' and row['evaluation'] != 'kge':
        raise ValueError('Comparison role must be main or KGE own_coverage')
    if row['evaluation'] == 'rag':
        from .evaluation.rag import BUDGET
        reference = read_document(entry['inputs']['reference'])
        index = read_document(entry['inputs']['index'])
        if index.get('index_id') != digest({k:v for k,v in index.items() if k != 'index_id'}):
            raise ValueError('Comparison index digest mismatch')
        if index.get('embedding_model') != BUDGET['embedding_model']:
            raise ValueError('Comparison embedding model mismatch')
        construction = read_document(entry['inputs']['construction'])
        config = phase['files'].get('runtime_config') or construction['files']['assets']['runtime_config']
        from .evaluation.reference import source_units
        books = {}; sources = {}
        for book in sorted(row['book_ids']):
            rows = [r for r in index['rows'] if r['book_id'] == book]
            if not rows:
                raise ValueError('Comparison active source missing')
            source = construction['files']['sources'][book]
            if file_record(source['path']) != source:
                raise ValueError('Comparison raw source changed')
            allowed = {digest(u) for u in source_units([read_document(source['path'])])}
            if any(digest({k:r[k] for k in ('book_id','chunk_id','evidence','text')}) not in allowed
                   or r['evidence_ids'] != [e['id'] for e in r['evidence']] for r in rows):
                raise ValueError('Comparison index is not selected raw source')
            books[book] = digest(rows)
            sources[book] = source['sha256']
        return dict(reference_digest=digest(reference), rag=dict(budget=BUDGET,
                    runtime_digest=digest(read_document(config['path'])), books=books, sources=sources))
    if row['evaluation'] == 'kge':
        protocol = read_document(entry['arguments']['protocol'])
        p = protocol['projection']
        if p['course_id'] != row['course_id']:
            raise ValueError('Comparison KGE course mismatch')
        mask = protocol.get('common_test_mask')
        if role == 'main' and mask is None:
            raise ValueError('Comparison main KGE requires declared common selection')
        return dict(reference_digest=p['reference_id'], kge=dict(split_id=p['split_id'],
                    test_ids=p['test_ids'], test_digest=digest(p['test']), budget=protocol['budget'],
                    common_mask=mask if role == 'main' else None))
    return dict(topology=dict(evaluator_sha256=phase['files']['evaluator_original']['sha256']))


def _asset(frozen, role, course, scope):
    value = read_document(frozen['files']['assets'][role]['path'])
    if isinstance(value, dict) and 'course_scopes' in value:
        record = value['course_scopes'][course][scope]
        if file_record(record['path']) != record:
            raise ValueError('Comparison course asset changed: ' + role)
        value = read_document(record['path'])
    return value


def _canonical(item, frozen):
    data = item['identity']['comparison_inputs']; course = item['course_id']; scope = item['entry']['scope']
    if 'reference_digest' in data:
        for role in ('questions', 'reference_facts'):
            if digest(_asset(frozen, role, course, scope)) != data['reference_digest']:
                raise ValueError('Comparison reference differs from canonical ' + role)
    if 'rag' in data:
        from .evaluation.reference import source_units
        actual = data['rag']['budget']; expected = frozen['experiments']['budgets']['rag']
        for key, field in [('retrieval_top_k','retrieval_top_k'), ('context_token_limit','context_token_limit'),
                           ('max_output_tokens','max_tokens'), ('temperature','temperature')]:
            if expected[key] != actual[field]:
                raise ValueError('Comparison canonical RAG budget mismatch: ' + key)
        if frozen['experiments']['models'] != dict(construction=actual['answer_model'], embedding=actual['embedding_model']):
            raise ValueError('Comparison canonical model mismatch')
        index = read_document(item['entry']['inputs']['index'])
        construction = read_document(item['entry']['inputs']['construction'])
        for book in item['book_ids']:
            source = frozen['files']['sources'][book]
            if construction['files']['sources'][book]['sha256'] != source['sha256']:
                raise ValueError('Comparison construction sources differ from canonical sources')
            units = source_units([read_document(source['path'])])
            allowed = {digest(u) for u in units}
            for row in index['rows']:
                if row['book_id'] == book and (digest({k:row[k] for k in ('book_id','chunk_id','evidence','text')}) not in allowed
                        or row['book_id'] == book and row['evidence_ids'] != [e['id'] for e in row['evidence']]):
                    raise ValueError('Comparison index is not selected canonical raw source')
    if 'kge' in data:
        split = _asset(frozen, 'fact_split', course, scope)
        if split['split_id'] != data['kge']['split_id'] or split['reference_id'] != data['reference_digest']:
            raise ValueError('Comparison canonical split mismatch')
        expected = frozen['experiments']['budgets']['kge']; actual = data['kge']['budget']
        if (expected['models'] != [actual['model']] or expected['split_ratios'] != [.8,.1,.1]
                or expected['epochs'] not in [500,1000]
                or any(expected[k] != actual[k] for k in ('batch_size','negative_samples','learning_rate'))):
            raise ValueError('Comparison canonical KGE budget mismatch')
    if 'topology' in data and data['topology']['evaluator_sha256'] != frozen['files']['assets']['evaluator_topology']['sha256']:
        raise ValueError('Comparison canonical topology evaluator mismatch')


def comparison_records(positions, manifest_path=None, canonical_binding=None):
    frozen = validate(manifest_path) if manifest_path else None
    binding = dict(manifest_id=frozen['manifest_id'], record=file_record(manifest_path)) if frozen else canonical_binding
    groups = {}
    def agree(target, key, value):
        if key in target and target[key] != value:
            raise ValueError('Comparison incompatible ' + key)
        target[key] = value
    for item in positions:
        if item['status'] != 'ready':
            continue
        data = item['identity'].get('comparison_inputs', {})
        if frozen:
            _canonical(item, frozen)
        key = (item['course_id'], item['entry']['scope'])
        group = groups.setdefault(key, dict(course_id=key[0], scope=key[1],
            manifest_id=binding['manifest_id'] if binding else None, canonical_manifest=binding['record'] if binding else None))
        if 'reference_digest' in data:
            agree(group, 'reference_digest', data['reference_digest'])
        for stage in ('rag','kge','topology'):
            if stage not in data:
                continue
            shared = group.setdefault(stage, {})
            for field, value in data[stage].items():
                if field in {'books', 'sources'}:
                    books = shared.setdefault(field, {})
                    for book, content in value.items():
                        agree(books, book, content)
                elif field == 'common_mask' and item['identity']['comparison_role'] == 'own_coverage':
                    continue
                else:
                    agree(shared, field, value)
    records = []
    for key in sorted(groups):
        group = groups[key]
        records.append(dict(group, comparison_id=digest(group)))
    return records


def bind_comparisons(positions, manifest_path=None):
    records = comparison_records(positions, manifest_path)
    for item in positions:
        if item['status'] != 'ready':
            continue
        record = next(r for r in records if (r['course_id'],r['scope']) == (item['course_id'],item['entry']['scope']))
        item['comparison'] = record
        item['identity']['comparison_id'] = record['comparison_id']
        item['job_id'] = digest(item['identity'])
    return records


def validate_binding(row, evidence):
    record = row.get('comparison')
    if record is None:
        return None
    if (record.get('comparison_id') != digest({k:v for k,v in record.items() if k != 'comparison_id'})
            or (record['course_id'],record['scope']) != (row['course_id'],row['entry']['scope'])):
        raise ValueError('Comparison identity changed')
    for stage, data in evidence.items():
        if stage == 'reference_digest':
            if data != record.get(stage):raise ValueError('Comparison reference changed')
            continue
        for key, value in data.items():
            if key == 'common_mask' and row['entry'].get('comparison_role') == 'own_coverage':continue
            expected = record.get(stage, {}).get(key)
            if key in {'books', 'sources'}:
                if not expected or any(expected.get(b) != v for b,v in value.items()):raise ValueError('Comparison book index changed')
            elif value != expected:raise ValueError('Comparison input changed: ' + key)
    return record['comparison_id']


def validate_comparisons(plan):
    records = plan.get('comparisons')
    if records is None:
        if any(p['status'] == 'ready' and p['evaluation'] in {'rag','kge'} for p in plan['positions']):
            raise ValueError('Comparison identity absent; prepare a new bound plan')
        return
    binding = None
    if plan.get('manifest_path') and records:
        record = file_record(plan['manifest_path'])
        manifest = read_document(record['path'])
        manifest_id = digest({k:v for k,v in manifest.items() if k != 'manifest_id'})
        if manifest.get('manifest_id') != manifest_id or any(r.get('canonical_manifest') != record or r.get('manifest_id') != manifest_id for r in records):
            raise ValueError('Comparison canonical manifest changed')
        binding = dict(manifest_id=manifest_id, record=record)
    expected = comparison_records(plan['positions'], canonical_binding=binding)
    if records != expected:
        raise ValueError('Comparison plan binding changed')
    for item in plan['positions']:
        if item['status'] != 'ready':continue
        if item.get('comparison') not in records:
            raise ValueError('Comparison row not bound to plan')
        comparison_id = validate_binding(item, item['identity']['comparison_inputs'])
        if item['identity'].get('comparison_id') != comparison_id or digest(item['identity']) != item['job_id']:
            raise ValueError('Comparison prepared identity changed')
