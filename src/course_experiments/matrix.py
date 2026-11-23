from __future__ import annotations

from .manifest import baseline_ready, digest, validate

PRIMARY_METHOD = 'course_g0_textbook_fusion'


def generate(frozen: dict) -> list[dict]:
    config = frozen['experiments']
    rows = []
    for course in frozen['courses']['courses']:
        books = {b['alias']: b['book_id'] for b in course['books']}
        book_sets = (config.get('course_book_sets', {}).get(course['course_id'], config['book_sets'])
                     if config.get('version', 1) == 2 else config['book_sets'])
        for aliases in book_sets:
            selected = [books[a] for a in aliases]
            for method in config['methods']:
                states = config['primary_states'] + config['additional_states'] if method['method_id'] == PRIMARY_METHOD else ['native']
                for state in states:
                    for evaluation in config['evaluations']:
                        evaluation_frozen = frozen['stage'] == 'evaluation'
                        ready = evaluation_frozen and (method['method_id'] == PRIMARY_METHOD or baseline_ready(method, evaluation))
                        inputs = {'manifest_id': frozen['manifest_id'], 'course_id': course['course_id'],
                                  'method': method, 'state': state, 'evaluation': evaluation,
                                  'books': [] if state == 'fixed_g0' and evaluation != 'rag' else selected}
                        rows.append({'row_id': digest({**inputs, 'declared_books': selected}),
                                     'execution_id': digest(inputs), 'manifest_id': frozen['manifest_id'],
                                     'course_id': course['course_id'], 'book_set': '+'.join(aliases),
                                     'book_ids': selected, 'method_id': method['method_id'],
                                     'state': state, 'evaluation': evaluation,
                                     'graph_role': 'external_reference' if state == 'native' else 'experiment',
                                     'status': 'ready' if ready else 'blocked',
                                     'blocked_reason': None if ready else ('evaluation assets and budgets not frozen' if not evaluation_frozen else 'verified commit and evaluation capability required'),
                                     'score': None})
    seen = set()
    for row in rows:
        row['deduplicated'] = row['execution_id'] in seen
        seen.add(row['execution_id'])
    return rows


def require_executable(row: dict, manifest_path: str) -> dict:
    frozen = validate(manifest_path)
    if row.get('manifest_id') != frozen['manifest_id']:
        raise ValueError('row manifest digest mismatch')
    canonical = next((r for r in generate(frozen) if r['row_id'] == row.get('row_id')), None)
    if canonical is None or row != canonical:
        raise ValueError('row differs from frozen matrix')
    if canonical['status'] == 'blocked':
        raise ValueError('execution blocked: ' + canonical['blocked_reason'])
    return canonical
