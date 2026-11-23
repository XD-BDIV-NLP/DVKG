from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

CONSTRUCTION_ASSETS = ('schema', 'g0_prompt', 'textbook_prompt')
REQUIRED_ASSETS = CONSTRUCTION_ASSETS + ('rag_prompt', 'questions',
                   'reference_facts', 'fact_split', 'gpu_environment', 'evaluator_rag',
                   'evaluator_topology', 'evaluator_kge')
MODELS = {'construction': 'deepseek-ai/DeepSeek-V4-Flash', 'embedding': 'BAAI/bge-m3'}
BUDGET_FIELDS = {
    'g0': ('max_output_tokens', 'temperature'),
    'textbook': ('chunk_chars', 'evidence_chars', 'max_output_tokens', 'temperature', 'max_retries'),
    'rag': ('retrieval_top_k', 'context_token_limit', 'max_output_tokens', 'temperature'),
    'kge': ('models', 'epochs', 'batch_size', 'negative_samples', 'learning_rate', 'split_ratios'),
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_record(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_document(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text(encoding='utf-8-sig'))
    import yaml
    return yaml.safe_load(path.read_text(encoding='utf-8-sig'))


def _explicit(value: object) -> bool:
    if value is None or value == '' or value == {} or value == []:
        return False
    if isinstance(value, dict):
        return all(_explicit(v) for v in value.values())
    if isinstance(value, list):
        return all(_explicit(v) for v in value)
    return True


def baseline_ready(method: dict, evaluation: str) -> bool:
    return bool(re.fullmatch(r'[0-9a-fA-F]{40}', str(method.get('commit', '')))
                and method.get('status') == 'verified'
                and method.get('repository')
                and evaluation in method.get('capability_scope', []))


def _valid_budget(role: str, budget: object) -> bool:
    if not isinstance(budget, dict) or not set(BUDGET_FIELDS[role]).issubset(budget) or not _explicit(budget):
        return False
    for field in BUDGET_FIELDS[role]:
        value = budget[field]
        if field == 'models':
            if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
                return False
        elif field == 'split_ratios':
            if not isinstance(value, list) or len(value) != 3 or not all(type(v) in (int, float) and 0 < v < 1 for v in value) or not math.isclose(sum(value), 1.0):
                return False
        elif field in ('temperature', 'learning_rate'):
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
            if (field == 'temperature' and not 0 <= value <= 2) or (field == 'learning_rate' and not 0 < value <= 1):
                return False
        elif type(value) is not int or value < (0 if field == 'max_retries' else 1):
            return False
    return True


def _validate_corpus(path: str | Path, book: dict) -> None:
    corpus = read_document(path)
    book_id = book['book_id']
    rows = corpus.get('books', [])
    if len(rows) != 1 or rows[0].get('id') != book_id or rows[0].get('source_fingerprint') != book.get('source_fingerprint'):
        raise ValueError(f'book/source_fingerprint mismatch: {book_id}')
    if not book.get('source_fingerprint') or not corpus.get('chunks') or not corpus.get('evidence'):
        raise ValueError(f'book corpus requires fingerprint, chunks and evidence: {book_id}')
    evidence_ids = {e['id'] for e in corpus['evidence']}
    for chunk in corpus['chunks']:
        referenced = chunk.get('evidence_ids', []) + chunk.get('context_evidence_ids', [])
        if chunk.get('book_id') != book_id or not set(referenced).issubset(evidence_ids):
            raise ValueError(f'book chunk/evidence mismatch: {book_id}')
    if any(e.get('book_id', book_id) != book_id for e in corpus['evidence']):
        raise ValueError(f'book evidence mismatch: {book_id}')
    if corpus.get('fingerprint') != digest({k: v for k, v in corpus.items() if k != 'fingerprint'}):
        raise ValueError(f'book corpus fingerprint mismatch: {book_id}')


def _validate_course_book_sets(courses: list[dict], experiments: dict) -> None:
    overrides = experiments.get('course_book_sets', {})
    if not isinstance(overrides, dict):
        raise ValueError('course book sets must be a course_id mapping')
    known = {c['course_id'] for c in courses}
    if set(overrides) - known:
        raise ValueError('course book sets contain unknown course_id')
    for course in courses:
        declared = {b['alias'] for b in course['books']}
        book_sets = overrides.get(course['course_id'], experiments['book_sets'])
        if not isinstance(book_sets, list) or not book_sets:
            raise ValueError('course book sets must be nonempty')
        seen = set()
        for aliases in book_sets:
            if (not isinstance(aliases, list) or not aliases
                    or not all(isinstance(a, str) for a in aliases)
                    or len(set(aliases)) != len(aliases)):
                raise ValueError('course book set requires nonempty unique aliases')
            selected = frozenset(aliases)
            if not selected <= declared:
                raise ValueError('course book set uses undeclared aliases')
            if selected in seen:
                raise ValueError('duplicate course book set')
            seen.add(selected)
        if len(declared) == 4 and (course['course_id'] not in overrides
                or not {frozenset('AB'), frozenset('ABCD')} <= seen):
            raise ValueError('four-book course requires explicit A/B and A/B/C/D sets')


def freeze(courses: dict, experiments: dict, source_paths: dict, asset_paths: dict,
           *, output: str | Path, config_paths: list[str | Path] | None = None,
           stage: str = 'evaluation') -> dict:
    if stage not in ('construction', 'evaluation'):
        raise ValueError('stage must be construction or evaluation')
    version = courses.get('version', 1)
    if (type(version) is not int or version not in (1, 2)
            or experiments.get('version', 1) != version):
        raise ValueError('courses and experiments require matching version 1 or 2')
    if experiments.get('seed') != 42 or experiments.get('models') != MODELS:
        raise ValueError('seed/model must match fixed experiment protocol')
    protocol = {'book_sets': [['A'], ['B'], ['A', 'B']], 'primary_states': ['fixed_g0', 'fused'],
                'additional_states': ['textbook_only'], 'evaluations': ['rag', 'topology', 'kge']}
    if any(experiments.get(k) != v for k, v in protocol.items()):
        raise ValueError('incomplete matrix protocol')
    method_ids = [m['method_id'] for m in experiments['methods']]
    if 'course_g0_textbook_fusion' not in method_ids or len(set(method_ids)) != len(method_ids):
        raise ValueError('matrix protocol requires distinct methods and primary method')
    budget_roles = ('g0', 'textbook') if stage == 'construction' else ('g0', 'textbook', 'rag', 'kge')
    for role in budget_roles:
        budget = experiments.get('budgets', {}).get(role)
        if not _valid_budget(role, budget):
            raise ValueError(f'explicit budget required: {role}')
    for role in CONSTRUCTION_ASSETS if stage == 'construction' else REQUIRED_ASSETS:
        if not asset_paths.get(role):
            raise ValueError(f'missing asset: {role}')
    course_list = courses['courses']
    ids = [c['course_id'] for c in course_list]
    book_ids = [b['book_id'] for c in course_list for b in c['books']]
    if len(set(ids)) != len(ids) or len(set(book_ids)) != len(book_ids):
        raise ValueError('course_id and book_id namespaces must be unique')
    for course in course_list:
        if course.get('g0_inputs') != ['description', 'schema'] or not course.get('description'):
            raise ValueError('G0 inputs must be description and schema only')
        if version == 2:
            aliases = [b['alias'] for b in course['books']]
            if not 2 <= len(aliases) <= 4 or set(aliases) != set('ABCD'[:len(aliases)]):
                raise ValueError('each course requires 2–4 unique consecutive aliases A through D')
        elif {b['alias'] for b in course['books']} != {'A', 'B'} or len(course['books']) != 2:
            raise ValueError('each course requires unique A/B books')
    if version == 2:
        _validate_course_book_sets(course_list, experiments)
    if set(source_paths) != set(book_ids):
        raise ValueError('source paths must contain exactly declared book_ids')
    for course in course_list:
        for book in course['books']:
            _validate_corpus(source_paths[book['book_id']], book)
    files = {'sources': {k: file_record(v) for k, v in sorted(source_paths.items())},
             'assets': {k: file_record(v) for k, v in sorted(asset_paths.items())},
             'configs': [file_record(p) for p in (config_paths or [])]}
    payload = {'version': version, 'stage': stage, 'status': 'inputs_frozen_baselines_may_be_pending',
               'courses': courses, 'experiments': experiments, 'files': files,
               'g0_identities': {c['course_id']: digest({'description': c['description'],
                                  'schema': files['assets']['schema']['sha256']}) for c in course_list}}
    payload['manifest_id'] = digest(payload)
    with Path(output).open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return payload


def require_construction(path: str | Path) -> dict:
    return validate(path)


def validate(path: str | Path) -> dict:
    payload = read_document(path)
    expected = payload['manifest_id']
    if digest({k: v for k, v in payload.items() if k != 'manifest_id'}) != expected:
        raise ValueError('manifest digest mismatch')
    records = list(payload['files']['sources'].values()) + list(payload['files']['assets'].values()) + payload['files']['configs']
    for record in records:
        if file_record(record['path'])['sha256'] != record['sha256']:
            raise ValueError(f"file digest mismatch: {record['path']}")
    return payload
