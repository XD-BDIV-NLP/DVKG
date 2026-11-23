"""ECHA持久执行器；教材、题答、核验及模型收据仅保留运行服务器。"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback

from filelock import FileLock
import numpy as np

from ..contributions import write_immutable
from ..dual_view import build_mappings
from ..manifest import digest, file_record, read_document
from . import echa_pipeline as pipeline
from . import echa_rag as rag
from . import evidence_hierarchical_alignment as core
from . import run_frozen_dual_view as frozen
from . import state_dual_view as state
from .dual_view_pipeline import store_vectors
from .dual_view_rag import asset, concept_text, STRUCTURAL
from .native_provider import AuditedNativeProvider, NativeAccountUnavailable, NativeBudgetExceeded
from .protocol import check_runtime, runtime_for
from .run_independent_holdout import LocalHTTPBudget, HoldoutBudgetExceeded
from .run_native_full import LIMITS, ReceiptTracker, validate_judgment
from .run_repaired_full import atomic, credentials


POLICY = dict(version='echa_runtime_v1', local_http_cap=8000, http_prefix='echa_v1/',
    original_started_unix=1792604536.907112, global_cumulative_http_cap=80000, max_seconds=172800,
    max_concurrency=4, edc_excluded=True, development_count=32, development_cohorts=['compABCD', 'osAB', 'netAB'],
    answer_max_tokens=2048, retrieval_top_k=5, context_bytes=3000, judge_version='native_blind_original_source_v1',
    alignment='all_frozen_state_book_concept_candidates;never_question_selected',
    development='query_selected_mappings_only;not_full_alignment_or_final_blind_freeze',
    failed_denominator='all_frozen_questions;failed_zero;interruption_pending_retained', human_audited=False)


class DependencyPending(RuntimeError):
    pass


class CheckpointError(ValueError):
    pass


class ECHABudget(LocalHTTPBudget):
    def __init__(self, provider, run_root):
        cap, name = 8000, 'echa_allowance'
        if provider.effective_http_cap == 200000:
            _validate_provider(provider)
            with closing(sqlite3.connect(provider.database, timeout=30)) as db:
                inherited = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND id<=? AND substr(stage,1,8)='echa_v1/'",
                    (provider.allocation['watermark_event_id'],)).fetchone()[0]
            cap = inherited+provider.allocation['caps']['echa']; name = 'echa_allowance_20261023'
        super().__init__(provider, Path(run_root) / name, 'echa_v1', cap=cap)
        self.prefix = 'echa_v1/'


def _checked(path, sha):
    record = file_record(path)
    if record['sha256'] != sha:
        raise ValueError('Frozen ECHA input digest changed')
    return read_document(path), record


def load_context(inventory_path, inventory_sha, tokenizer_path, tokenizer_sha, *, tokenizer_loader=None):
    inventory, inventory_record = _checked(inventory_path, inventory_sha)
    if inventory.get('version') != 'echa_asset_inventory_v1' or not inventory.get('states'):
        raise ValueError('Frozen ECHA state inventory required')
    config = asset(inventory['runtime_config']); check_runtime(config)
    assets = asset(inventory['assets'])
    manifest, tokenizer_record = _checked(tokenizer_path, tokenizer_sha)
    for record in manifest['files'].values():
        if file_record(record['path']) != {k: record[k] for k in ('path', 'sha256')}:
            raise ValueError('Frozen tokenizer resource changed')
    vendor = manifest.get('vendor')
    if vendor:
        directory = str(Path(vendor['path']).parent) if isinstance(vendor, dict) else str(Path(vendor))
        if not Path(directory).is_dir():
            raise ValueError('Tokenizer vendor directory unavailable')
        if directory not in sys.path:
            sys.path.insert(0, directory)
    tokenizer = pipeline.FrozenTokenizer(manifest, loader=tokenizer_loader)
    books = sorted(assets['unit_assets'])
    if len(books) != 8 or set(books) != set(assets['structures']):
        raise ValueError('ECHA requires the same eight source books')
    structures, units, matrices, token_counts, records = {}, {}, {}, {}, [inventory_record, tokenizer_record, inventory['assets'], inventory['runtime_config']]
    records.extend({k: r[k] for k in ('path', 'sha256')} for r in manifest['files'].values())
    for book in books:
        structure, unit = asset(assets['structures'][book]), asset(assets['unit_assets'][book])
        if structure['book_id'] != book:
            raise ValueError('Foreign original textbook structure')
        core._hierarchy(structure)
        if file_record(unit['vectors']['path']) != unit['vectors']:
            raise ValueError('Original paragraph vectors changed')
        matrix = np.load(unit['vectors']['path'], mmap_mode='r', allow_pickle=False)
        if matrix.shape != (len(unit['rows']), 1024) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise ValueError('Normalized complete original paragraph vectors required')
        ids = [row['evidence_ids'][0] for row in unit['rows']]
        if len(set(ids)) != len(ids) or set(ids) != set(structure['evidence']):
            raise ValueError('One original paragraph per source unit required')
        if any(len(r['evidence_ids']) != 1 or r['book_id'] != book or r['text'] != structure['evidence'][eid]['text'] for r, eid in zip(unit['rows'], ids)):
            raise ValueError('Original paragraph text binding changed')
        structures[book], units[book], matrices[book] = structure, dict(unit, vectors=matrix), matrix
        token_counts[book] = {eid: tokenizer.count(row['text']) for eid, row in structure['evidence'].items()}
        records.extend([assets['structures'][book], assets['unit_assets'][book], unit['vectors']])
    entries = {}
    for key, entry in sorted(inventory['states'].items()):
        if entry['state'] not in ('F_corrected', 'G0', 'T') or entry['cohort'] not in assets['cohorts']:
            raise ValueError('Unknown frozen ECHA state or cohort')
        active = entry['active_books']
        if active != assets['cohorts'][entry['cohort']]['active_books'] or not set(active) <= set(books):
            raise ValueError('Frozen state cohort pairing differs')
        graph = asset(entry['graph']); plan = entry['vector_plan']
        entities = {e.get('entity_id', e.get('id')): e for e in graph['entities'] if e.get('type') not in STRUCTURAL}
        ids = sorted(entities); texts = [concept_text(entities[cid]) for cid in ids]
        if plan['concept_ids'] != ids or plan['input_text_digest'] != digest(texts) or plan['embedding_configuration'] != config['embedding']:
            raise ValueError('State concept text or embedding configuration differs')
        if len(plan['entries']) != len(ids) or any(row['concept_id'] != cid or row.get('text_key') != state.vector_key(text, config['embedding']) for row, cid, text in zip(plan['entries'], ids, texts)):
            raise ValueError('Exact per-concept vector text identity required')
        vectors = state.materialize_reused(plan)
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError('Normalized reused concept vectors required')
        mappings = build_mappings(graph, {b: structures[b] for b in active})
        entries[key] = dict(entry, graph_data=graph, vectors=vectors, mappings=mappings)
        records.append(entry['graph'])
        records.extend(row['reuse']['vectors'] for row in plan['entries'])
    expected = {(co, st) for co in assets['cohorts'] for st in ('F_corrected', 'G0', 'T')}
    if {(v['cohort'], v['state']) for v in entries.values()} != expected or len(entries) != len(expected):
        raise ValueError('One F_corrected/G0/T state per cohort required')
    retries = config.get('service_retries', 6)
    if type(retries) is not int or retries < 0:
        raise ValueError('Frozen nonnegative service retry count required')
    code_names = ['run_echa.py', 'echa_pipeline.py', 'echa_rag.py', 'evidence_hierarchical_alignment.py', 'state_dual_view.py',
        'run_frozen_dual_view.py', 'dual_view_run.py', 'dual_view_rag.py', 'graph_rag_v2.py', 'native_provider.py',
        'run_native_full.py', 'run_independent_holdout.py', 'protocol.py', 'run_repaired_full.py', 'dual_view_pipeline.py',
        'rag_repair_v3.py', 'repaired_full.py', 'rag.py']
    code = {name: file_record(Path(__file__).with_name(name)) for name in code_names}
    for path in (Path(__file__).parents[1] / 'manifest.py', Path(__file__).parents[1] / 'contributions.py', Path(__file__).parents[1] / 'dual_view.py'):
        code[path.name] = file_record(path)
    from domain_ckg_workspace.ckg_incremental import runtime as transport_runtime
    code['model_runtime.py'] = file_record(transport_runtime.__file__)
    unique = {r['path']: r for r in records}
    contract = dict(inventory=inventory_record, tokenizer=tokenizer_record, tokenizer_files=manifest['files'], code=code,
        policy=POLICY, alignment_policy=pipeline.POLICY, retrieval_policy=rag.POLICY, source_records=list(unique.values()))
    return dict(identity=digest(contract), contract=contract, inventory=inventory, assets=assets, entries=entries,
        structures=structures, units=units, paragraph_matrices=matrices, paragraph_tokens=token_counts,
        parent=Path(inventory.get('parent_root', frozen.PARENT)), books=books,
        transport_attempts=retries+1, stat_cache={})


def _recheck(ctx, *, deep=False):
    if digest(ctx['contract']) != ctx['identity']:
        raise ValueError('Prepared ECHA contract changed')
    records = list(ctx['contract']['code'].values()) + ctx['contract']['source_records']
    for record in records:
        stat = Path(record['path']).stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        old = ctx['stat_cache'].get(record['path'])
        if old is not None and old != signature:
            raise CheckpointError('Frozen ECHA implementation or inputs changed')
        if deep or old is None:
            if file_record(record['path']) != record:
                raise CheckpointError('Frozen ECHA implementation or inputs changed')
            ctx['stat_cache'][record['path']] = signature


def _base(ctx, run_root, mode):
    return Path(run_root) / 'echa_v1' / mode / ctx['identity']


def _plans(ctx, key, book, only=None):
    entry = ctx['entries'][key]
    return list(pipeline.candidate_sets(entry['graph_data'], ctx['structures'][book], entry['mappings'],
        entry['vector_plan']['concept_ids'], entry['vectors'], ctx['units'][book]['rows'], ctx['paragraph_matrices'][book], only_concepts=only))


def preflight(ctx, run_root):
    _recheck(ctx, deep=True)
    root = _base(ctx, run_root, 'preflight'); references = {}; pairs = batches = 0
    if (root / 'result.json').exists():
        saved = read_document(root / 'result.json')
        if saved.get('context_identity') != ctx['identity'] or saved.get('checkpoint_digest') != digest({k: v for k, v in saved.items() if k != 'checkpoint_digest'}):
            raise CheckpointError('Frozen full candidate preflight result changed')
        for books in saved['plans'].values():
            for record in books.values():
                if file_record(record['path']) != record:
                    raise CheckpointError('Frozen original candidate plan changed')
        return dict(saved, budget_snapshot=_budget_snapshot(run_root))
    for key, entry in ctx['entries'].items():
        references[key] = {}
        for book in entry['active_books']:
            plans = _plans(ctx, key, book); count = sum(len(row['candidates']) for row in plans)
            path = root / 'plans' / key / (book + '.json')
            write_immutable(path, dict(context_identity=ctx['identity'], state_key=key, book_id=book, plans=plans,
                pair_count=count, concept_count=len(plans), candidate_digest=digest(plans)))
            references[key][book] = file_record(path); pairs += count; batches += (count + 7) // 8
    result = dict(status='preflight_complete_no_api', context_identity=ctx['identity'], state_count=len(ctx['entries']),
        state_book_count=sum(len(v) for v in references.values()), pair_count=pairs, verification_logical_base=batches,
        verification_logical_with_repairs=2*batches, verification_http_upper=2*batches*ctx['transport_attempts'],
        transport_attempts_per_logical_upper=ctx['transport_attempts'], plans=references, model_api_calls=0, full_alignment_complete=False,
        raw_text_exported=False)
    write_immutable(root / 'contract.json', ctx['contract']); write_immutable(root / 'tokens.json', ctx['paragraph_tokens'])
    result = frozen.checkpoint(root / 'result.json', result)
    return dict(result, budget_snapshot=_budget_snapshot(run_root))


def _budget_snapshot(run_root):
    root = Path(run_root) / 'formal_provider'
    if not (root / 'audit.sqlite').exists():
        return dict(available=False, reason='existing_formal_ledger_unavailable;never_initialized')
    limits_path = root / 'limits.json'; stored = read_document(limits_path)
    if stored['limits'] != LIMITS or stored['started_unix'] != POLICY['original_started_unix']:
        raise ValueError('Original formal limits or deadline differ')
    extension = read_document(root / 'budget_extension_authorization_20261022.json')
    if (extension.get('approved_cumulative_http_attempts') != 80000 or extension.get('original_limits_sha256') != file_record(limits_path)['sha256']
            or extension.get('original_started_unix') != stored['started_unix'] or extension.get('deadline_reset') is not False or extension.get('inherited_spend') is not True):
        raise ValueError('Original cumulative extension required')
    uri = (root / 'audit.sqlite').resolve().as_uri() + '?mode=ro'
    global_cap, seconds, local_cap = 80000, LIMITS['max_seconds'], 8000
    new_path = root/'budget_extension_authorization_20261023.json'
    allocation = None
    if new_path.exists():
        new = read_document(new_path); allocation = read_document(root/'bounded_allocations_20261023.json')
        expected = dict(version='human_authorized_cumulative_extension_v2',
            question_call='call_6b80ba3cee9a4340b6b8331f6f92d189',answer='允许累计20万次、原起点96小时，按有界子额度执行',
            original_limits_sha256=file_record(limits_path)['sha256'],
            previous_authorization_sha256=file_record(root/'budget_extension_authorization_20261022.json')['sha256'],
            original_started_unix=stored['started_unix'],original_limits=LIMITS,
            approved_cumulative_http_attempts=200000,approved_total_seconds=345600,inherited_spend=True,deadline_reset=False,edc_excluded=True)
        if (any(new.get(k)!=v or type(new.get(k)) is not type(v) for k,v in expected.items())
                or allocation.get('authorization_sha256') != file_record(new_path)['sha256']):
            raise ValueError('Approved ECHA budget extension identity differs')
        global_cap, seconds = 200000, 345600
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as db:
        total = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started'").fetchone()[0]
        local = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,8)='echa_v1/'").fetchone()[0]
        if allocation:
            inherited = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND id<=? AND substr(stage,1,8)='echa_v1/'",
                (allocation['watermark_event_id'],)).fetchone()[0]
            local_cap = inherited+allocation['caps']['echa']
    return dict(available=True, global_http_attempts=total, echa_http_attempts=local, global_remaining_http=max(0, global_cap-total),
        echa_remaining_http=max(0, local_cap-local), effective_echa_http_cap=local_cap,effective_global_http_cap=global_cap,
        deadline_unix=stored['started_unix']+seconds,
        deadline_remaining_seconds=max(0, stored['started_unix']+seconds-time.time()))


def _validate_provider(provider):
    if (provider.limits != LIMITS or provider.started != POLICY['original_started_unix'] or not provider.budget_extension
            or provider.effective_http_cap not in (80000,200000)
            or (provider.effective_http_cap == 200000 and (provider.effective_max_seconds != 345600 or not provider.allocation))):
        raise ValueError('Original formal provider with human-authorized cumulative extension required; never reset')


def _remaining(provider, budget):
    provider.raise_if_fatal()
    return min(budget.cap-budget.count(), provider.effective_http_cap-provider.statistics()['http_attempts'])


@contextmanager
def _provider_scope(ctx, run_root, provider, requests):
    _validate_provider(provider)
    if provider.runtime.config != asset(ctx['inventory']['runtime_config']):
        raise ValueError('Actual provider runtime differs from frozen ECHA configuration')
    tracker = ReceiptTracker(provider); original_chat = provider.chat_response
    try:
        with ECHABudget(provider, run_root) as budget, tracker.scope(requests):
            def scoped(messages, **kwargs):
                _recheck(ctx)
                if kwargs.get('stage') == 'native_blind_judge':
                    kwargs['stage'] = budget.stage('judge')
                return original_chat(messages, **kwargs)
            provider.chat_response = scoped
            yield budget
    finally:
        provider.chat_response = original_chat; tracker.close()


def _frozen_stage(plan, reference):
    if reference is None:
        return None
    value = asset(reference)
    if (value.get('version') != 'echa_frozen_stage_budget_v1' or value.get('evaluation_plan_identity') != plan['plan_identity']
            or not isinstance(value.get('stage_id'), str) or not value['stage_id']
            or type(value.get('actual_http_cap')) is not int or not 1 <= value['actual_http_cap'] <= 8000):
        raise ValueError('Frozen stage budget must bind the complete evaluation plan within existing 8000 cap')
    return dict(**value, manifest=reference, identity=digest(value))


def _stage_count(provider, stage):
    if stage is None:
        return None
    prefix = 'echa_v1/frozen_stage/' + stage['identity'] + '/'
    with closing(sqlite3.connect(provider.database, timeout=30)) as db:
        return db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,?)=?",
            (len(prefix), prefix)).fetchone()[0]


@contextmanager
def _stage_scope(provider, budget, stage, root):
    if stage is None:
        yield; return
    write_immutable(Path(root)/'budget_stages'/stage['identity']/'plan.json', stage)
    original_record, original_stage = provider.runtime._record, budget.stage
    prefix = 'echa_v1/frozen_stage/' + stage['identity'] + '/'
    lock = FileLock(str(provider.root/('echa_stage_'+stage['identity']+'.lock')), timeout=30)
    def bounded(key, scope, status, value):
        if status != 'request_started':
            return original_record(key, scope, status, value)
        if file_record(stage['manifest']['path']) != stage['manifest'] or not scope.startswith(prefix):
            raise CheckpointError('Frozen stage manifest or request scope changed')
        with lock:
            if _stage_count(provider, stage) >= stage['actual_http_cap']:
                raise HoldoutBudgetExceeded('Frozen ECHA stage actual HTTP cap exhausted; complete denominator retained')
            return original_record(key, scope, status, value)
    try:
        budget.stage = lambda name: prefix+name
        provider.runtime._record = bounded
        yield
    finally:
        provider.runtime._record, budget.stage = original_record, original_stage


def _dataset_questions(reference, book):
    value = asset(reference); questions = value['questions']; ids = [q['question_id'] for q in questions]
    if len(set(ids)) != len(ids) or any(q['target_book_id'] != book for q in questions):
        raise ValueError('Frozen question identity or target book differs')
    return questions


def datasets(ctx, run_root, split):
    if split == 'regression':
        index_path = ctx['parent'] / 'datasets/index.json'; index = read_document(index_path)['books']
        refs = {b: index[b]['confirmation'] for b in ctx['books']}; seal = file_record(index_path)
    elif split == 'independent':
        path = Path(run_root) / 'independent_holdout_128/sealed.json'
        if not path.exists():
            raise DependencyPending('Historical independent123 not sealed')
        sealed = read_document(path)
        if sealed.get('status') not in ('sealed_complete', 'sealed_insufficient') or sealed.get('target_count') != 128:
            raise DependencyPending('Historical independent set is not sealed')
        asset(sealed['contract']); refs = asset(sealed['dataset_index'])['books']; seal = file_record(path)
        total = sum(len(_dataset_questions(r, b)) for b, r in refs.items())
        if total != sealed['accepted_count'] or sealed['shortage'] != 128-total:
            raise ValueError('Independent actual denominator differs')
    elif split == 'blind64':
        path = Path(run_root) / 'blind64_20261022_v1/sealed.json'
        if not path.exists():
            raise DependencyPending('New blind64 is not sealed')
        sealed = read_document(path)
        if sealed.get('status') != 'sealed_complete' or sealed.get('accepted_count') != 64 or sealed.get('target_count') != 64 or sealed.get('released_for_quality_evaluation') is not True:
            raise DependencyPending('New blind64 requires full64 seal and release')
        contract = asset(sealed['contract']); freeze = contract['algorithm_freeze']
        from .blind64_pipeline import require_freeze
        algorithm, _ = require_freeze(freeze['path'], freeze['sha256'])
        methods = [m for m in algorithm['methods'] if 'echa' in m['method_id'].casefold()]
        if not methods:
            raise ValueError('Blind64 final freeze does not preregister ECHA')
        frozen_code = {(r['path'], r['sha256']) for m in methods for r in m['code']}
        if any((r['path'], r['sha256']) not in frozen_code for r in ctx['contract']['code'].values()):
            raise ValueError('Current ECHA implementation differs from pre-generation final freeze')
        refs = asset(sealed['dataset_index'])['books']; seal = file_record(path)
        if sum(len(_dataset_questions(r, b)) for b, r in refs.items()) != 64:
            raise ValueError('Blind64 actual denominator differs')
    else:
        raise ValueError('Unknown ECHA evaluation split')
    if set(refs) != set(ctx['books']):
        raise ValueError('Question pool must cover the same eight books')
    for book, ref in refs.items():
        _dataset_questions(ref, book)
    return refs, seal


def _existing_query(ctx, q, run_root, root):
    _recheck(ctx)
    path = ctx['parent'] / 'task_vectors' / q['target_book_id'] / (q['question_id'] + '.npy')
    if path.exists():
        record = read_document(path.with_suffix('.npy.meta.json'))['vectors']
        if file_record(path) != record:
            raise ValueError('Original frozen query vector changed')
        matrix = np.load(path, allow_pickle=False)
    else:
        path = Path(root) / 'query_vectors' / (digest([q['target_book_id'], q['question_id']]) + '.npy')
        config = asset(ctx['inventory']['runtime_config'])
        binding = dict(question_id=q['question_id'], question_digest=digest(q['question']), embedding=config['embedding'])
        write_immutable(path.with_suffix('.input.json'), binding)
        meta = path.with_suffix('.npy.meta.json')
        if meta.exists():
            record = read_document(meta)['vectors']
            if file_record(path) != record:
                raise ValueError('ECHA saved query vector changed')
            matrix = np.load(path, allow_pickle=False)
        else:
            body = dict(model=config['embedding']['model'], input=[q['question']], encoding_format='float')
            request_id = digest(dict(endpoint='/embeddings', body=body, config=config))
            database = Path(run_root) / 'formal_provider/audit.sqlite'
            if not database.exists():
                return None
            with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True, timeout=30)) as db:
                found = db.execute('SELECT record FROM receipts WHERE key=?', (request_id,)).fetchone()
            if found is None:
                return None
            receipt = json.loads(found[0]); value = asset(receipt)
            if value['request_id'] != request_id or value['endpoint'] != '/embeddings' or value['request'] != body or value['runtime_config'] != config:
                raise CheckpointError('Exact cached query embedding receipt differs')
            data = value['response']['data']
            if len(data) != 1 or data[0]['index'] != 0:
                raise ValueError('Cached query embedding response index differs')
            matrix = np.asarray([data[0]['embedding']], dtype=np.float32)
            norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            if matrix.shape != (1, 1024) or not np.isfinite(matrix).all() or (norm <= 0).any():
                raise ValueError('Cached query embedding response invalid')
            matrix /= norm
            record = store_vectors(path, matrix)
            write_immutable(path.with_suffix('.receipt.json'), dict(receipt=receipt, input_identity=digest(binding)))
    if matrix.shape != (1, 1024) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
        raise ValueError('Normalized frozen BGE query vector required')
    return matrix[0], record


def _query(ctx, q, provider, budget, root):
    cached = _existing_query(ctx, q, provider.root.parent, root)
    if cached is None:
        provider.embed([q['question']], stage=budget.stage('query_embedding'))
        cached = _existing_query(ctx, q, provider.root.parent, root)
    if cached is None:
        raise CheckpointError('Successful query embedding has no audited receipt')
    return cached


def _engine(ctx, key, qualified, candidates=None):
    entry = ctx['entries'][key]; books = entry['active_books']
    return rag.ECHARetriever(entry['graph_data'], {b: ctx['structures'][b] for b in books}, entry['mappings'],
        {b: ctx['units'][b] for b in books}, entry['vector_plan']['concept_ids'], entry['vectors'], qualified,
        ctx['paragraph_tokens'], candidates=candidates)


class ExactDeferredAlignment:
    def __init__(self, ctx, key, book, plan_record, root):
        self.ctx, self.key, self.book, self.plan_record, self.root = ctx, key, book, plan_record, Path(root)
        plan = asset(plan_record)
        if plan['context_identity'] != ctx['identity'] or plan['state_key'] != key or plan['book_id'] != book or plan['candidate_digest'] != digest(plan['plans']):
            raise CheckpointError('Frozen original candidate plan identity differs')
        self.flat = [(row['concept_id'], candidate) for row in plan['plans'] for candidate in row['candidates']]
        self.concepts = {row['concept_id'] for row in plan['plans']}; self.incidence = {}; self.batch_ids = []
        for first in range(0, len(self.flat), 8):
            batch = self.flat[first:first+8]; ordinal = first//8
            self.batch_ids.append(digest([digest([cid, candidate]) for cid, candidate in batch]))
            for cid, _ in batch:
                self.incidence.setdefault(cid, set()).add(ordinal)
        self.loaded = {}; self.proofs = {}
        write_immutable(self.root / 'fixed_plan.json', dict(context_identity=ctx['identity'], full_original_plan=plan_record,
            batch_size=8, original_pair_count=len(self.flat), original_batch_count=len(self.batch_ids),
            original_batch_order_digest=digest(self.batch_ids), protocol='exact_full_original_batches_no_regrouping'))

    def required_batches(self, concepts):
        if not set(concepts) <= self.concepts:
            raise ValueError('Unknown query-selected concept in fixed candidate plan')
        return sorted({i for cid in concepts for i in self.incidence.get(cid, set())})

    def ensure(self, concepts, provider, budget):
        _recheck(self.ctx)
        for ordinal in self.required_batches(concepts):
            if ordinal in self.loaded:
                continue
            batch = self.flat[ordinal*8:ordinal*8+8]; plans = []
            for cid, candidate in batch:
                if not plans or plans[-1]['concept_id'] != cid:
                    plans.append(dict(concept_id=cid, candidates=[]))
                plans[-1]['candidates'].append(candidate)
            folder = self.root / 'batches' / self.batch_ids[ordinal]
            records = pipeline.verify_candidates(provider, self.ctx['entries'][self.key]['graph_data'], self.ctx['structures'][self.book],
                plans, folder, budget.stage('exact_deferred_alignment'))
            if len(records) != len(batch) or {(r['concept_id'], r['evidence_id']) for r in records} != {(cid, c['evidence_id']) for cid, c in batch}:
                raise CheckpointError('Original batch decision coverage differs')
            path = folder / 'batch_result.json'
            write_immutable(path, dict(full_original_plan=self.plan_record, original_batch_ordinal=ordinal,
                original_batch_id=self.batch_ids[ordinal], pair_keys=[digest([cid, c]) for cid, c in batch], records=records))
            self.loaded[ordinal] = records; self.proofs[ordinal] = file_record(path)
        return [row for ordinal in sorted(self.loaded) for row in self.loaded[ordinal]]

    def candidates(self, concepts):
        return [dict(concept_id=cid, **candidate) for cid, candidate in self.flat if cid in concepts]

    def status(self):
        return dict(protocol='exact_full_original_batches_no_regrouping', total_original_batches=len(self.batch_ids),
            verified_original_batches=len(self.loaded), verified_pair_count=sum(map(len, self.loaded.values())),
            full_alignment_complete=False, all_original_batches_cached=len(self.loaded) == len(self.batch_ids))


def _methods(entry):
    return rag.ECHARetriever.methods if entry['state'] == 'F_corrected' else ('echa',)


def deferred_preflight(ctx, run_root, *, pools=None):
    planned = preflight(ctx, run_root); root = _base(ctx, run_root, 'deferred_preflight'); root.mkdir(parents=True, exist_ok=True)
    if pools is None:
        pools = {split: datasets(ctx, run_root, split)[0] for split in ('regression', 'independent')}
    index = read_document(ctx['parent'] / 'datasets/index.json')['books']
    pools = dict(pools, development32={b: index[b]['development'] for b in ctx['books']})
    questions = {name: {b: _dataset_questions(ref, b) for b, ref in refs.items()} for name, refs in pools.items()}
    queries, missing, query_proofs = {}, [], {}
    for name, books in questions.items():
        for book, values in books.items():
            for q in values:
                qkey = (book, q['question_id'])
                if qkey in queries:
                    continue
                cached = _existing_query(ctx, q, run_root, root)
                if cached is None:
                    missing.append(dict(split=name, target_book_id=book, question_id=q['question_id'])); continue
                queries[qkey], query_proofs[book+'/'+q['question_id']] = cached
    totals = {name: dict(unique_questions=sum(map(len, books.values())), question_evaluations=0, verification_original_batches=0,
        verification_logical_with_repairs=0, verification_http_upper=0, answer_judge_logical_bodies=0, baseline_cache_deductions=0)
        for name, books in questions.items()}
    union = {}; development_union = {}; total_union = 0; total_development = 0
    no_verifier = {'keyword', 'vector', 'hybrid', 'no_mapping', 'no_role_verification'}
    for key, entry in ctx['entries'].items():
        union[key] = {}; development_union[key] = {}
        engine = _engine(ctx, key, {b: [] for b in entry['active_books']})
        for book in entry['active_books']:
            dispatcher = ExactDeferredAlignment(ctx, key, book, planned['plans'][key][book], root / 'fixed_batches' / key / book)
            old = set(); dev = set()
            for name, refs in pools.items():
                batches = set()
                for method in _methods(entry):
                    totals[name]['question_evaluations'] += len(questions[name][book])
                    if method in no_verifier:
                        continue
                    for q in questions[name][book]:
                        query = queries.get((book, q['question_id']))
                        if query is None:
                            continue
                        seeds, neighbors, _ = engine._concepts(q['question'], query, method != 'no_semantic', method != 'no_relations')
                        batches.update(dispatcher.required_batches(seeds+neighbors))
                totals[name]['verification_original_batches'] += len(batches)
                if name != 'development32':
                    old.update(batches)
                elif entry['state'] == 'F_corrected' and entry['cohort'] in POLICY['development_cohorts']:
                    for q in questions[name][book]:
                        query = queries.get((book, q['question_id']))
                        if query is not None:
                            seeds, neighbors, _ = engine._concepts(q['question'], query)
                            dev.update(dispatcher.required_batches(seeds+neighbors))
            union[key][book] = sorted(old); development_union[key][book] = sorted(dev)
            total_union += len(old); total_development += len(dev)
    for row in totals.values():
        row['verification_logical_base'] = row['verification_original_batches']
        row['format_repair_logical_upper'] = row['verification_original_batches']
        row['verification_logical_with_repairs'] = 2*row['verification_original_batches']
        row['verification_http_upper'] = row['verification_logical_with_repairs']*ctx['transport_attempts']
        row['answer_judge_logical_bodies'] = 2*row['question_evaluations']
        row['answer_judge_http_upper'] = row['answer_judge_logical_bodies']*ctx['transport_attempts']
    new_evaluations = sum(8*len(entry['active_books'])*len(_methods(entry)) for entry in ctx['entries'].values())
    index_path = root / 'selected_original_batches.json'
    write_immutable(index_path, dict(context_identity=ctx['identity'], full_plans=planned['plans'], pools=pools,
        original_batches_union_all_old=union, original_batches_development32=development_union, query_vector_records=query_proofs,
        selection_inputs='question_string_target_book_query_vector_only;no_answers_or_support_labels'))
    result = dict(status='deferred_preflight_complete_no_api' if not missing else 'query_vectors_pending', context_identity=ctx['identity'],
        exact_old_query_costs=not bool(missing), missing_query_vectors=missing, pools=totals,
        all_old_original_batches_union=total_union, all_old_verification_logical_with_repairs=2*total_union,
        all_old_verification_http_upper=2*total_union*ctx['transport_attempts'],
        all_old_question_evaluations=sum(v['question_evaluations'] for k, v in totals.items() if k != 'development32'),
        all_old_answer_judge_logical_bodies=sum(v['answer_judge_logical_bodies'] for k, v in totals.items() if k != 'development32'),
        development32=dict(question_evaluations=32, original_batches_union=total_development,
            verification_logical_with_repairs=2*total_development, answer_judge_logical_bodies=64,
            total_http_upper=(2*total_development+64)*ctx['transport_attempts']),
        unknown_new64=dict(exact=False, unique_questions=64, question_evaluations=new_evaluations,
            answer_judge_logical_bodies=2*new_evaluations, query_embedding_logical_upper=64,
            answer_judge_http_upper=2*new_evaluations*ctx['transport_attempts'],
            additional_verification_original_batches_upper=max(0, planned['verification_logical_base']-total_union),
            additional_total_http_upper=(2*max(0, planned['verification_logical_base']-total_union)+2*new_evaluations+64)*ctx['transport_attempts'],
            reserve_policy='conservative_all_remaining_original_batches;no_generated_questions_read'),
        fixed_batch_index=file_record(index_path), current_trial_http_cap=8000, full_matrix_funded=False,
        model_api_calls=0, raw_text_exported=False)
    write_immutable(root / 'result.json', result)
    return result


def _completion(raw):
    choices = raw.get('choices')
    if not isinstance(choices, list) or not choices or choices[0].get('finish_reason') != 'stop':
        raise ValueError('ECHA model response incomplete; original raw retained')
    value = json.loads(choices[0]['message']['content'])
    if not isinstance(value, dict):
        raise ValueError('ECHA model response must be an object')
    return value


def _checkpoint_asset(record):
    try:
        return asset(record)
    except (ValueError, KeyError, OSError) as exc:
        raise CheckpointError('Referenced ECHA checkpoint changed') from exc


def _bound_raw_answer(provider, question, context, path):
    payload = dict(question=question['question'], target_book_id=question['target_book_id'], context=context)
    body = dict(model=provider.runtime.config['llm']['model'], temperature=0, seed=42, enable_thinking=False,
        max_tokens=2048, response_format={'type': 'json_object'}, messages=[dict(role='system', content=frozen.old.ANSWER),
        dict(role='user', content=json.dumps(payload, ensure_ascii=False))])
    key = digest(dict(endpoint='/chat/completions', body=body, config=provider.runtime.config))
    try:
        original = provider._cached(key, '/chat/completions', body)
        saved = read_document(path)['response']
        if original is None or {k: v for k, v in saved.items() if k != '_native_cache_hit'} != original:
            raise CheckpointError('Raw ECHA answer differs from exact original provider receipt')
    except (ValueError, KeyError, OSError) as exc:
        raise CheckpointError('Raw ECHA answer receipt changed or missing') from exc
    return _completion(saved), file_record(provider.root/'receipts'/(key+'.json'))


def _answer(provider, q, context, folder, stage):
    path = folder / 'candidate.json'
    binding = dict(question=q['question'], target_book_id=q['target_book_id'], context=context,
        runtime=provider.runtime.config, prompt=frozen.old.ANSWER, max_tokens=2048)
    write_immutable(folder / 'answer_contract.json', binding)
    if path.exists():
        saved = read_document(path)
        if saved.get('input_identity') != digest(binding) or saved.get('checkpoint_digest') != digest({k: v for k, v in saved.items() if k != 'checkpoint_digest'}):
            raise CheckpointError('ECHA candidate checkpoint changed')
        _checkpoint_asset(saved['receipt']['source'])
        if 'provider_receipt' in saved['receipt']:
            _checkpoint_asset(saved['receipt']['provider_receipt'])
        return saved['candidate'], saved['receipt']
    raw_path = folder / 'answer_raw.json'
    if raw_path.exists():
        candidate, proof = _bound_raw_answer(provider, q, context, raw_path)
        receipt = dict(reused=False, source=file_record(raw_path), provider_receipt=proof)
    else:
        candidate, receipt = frozen.answer(provider, q, context, folder, stage)
        if raw_path.exists():
            candidate, proof = _bound_raw_answer(provider, q, context, raw_path)
            receipt = dict(receipt, provider_receipt=proof)
    if not isinstance(candidate.get('answer'), str) or not isinstance(candidate.get('citations'), list):
        raise ValueError('Invalid common ECHA candidate answer')
    frozen.checkpoint(path, dict(candidate=candidate, receipt=receipt, input_identity=digest(binding)))
    return candidate, receipt


def _judge(provider, q, context, candidate, folder, identity):
    path = folder / 'judgment.json'
    binding = dict(question=q, context=context, candidate=candidate, result_id=identity, prompt=frozen.JUDGE, runtime=provider.runtime.config)
    write_immutable(folder / 'judge_contract.json', binding)
    if path.exists():
        saved = read_document(path)
        if saved.get('input_identity') != digest(binding) or saved.get('checkpoint_digest') != digest({k: v for k, v in saved.items() if k != 'checkpoint_digest'}):
            raise CheckpointError('ECHA judge checkpoint changed')
        for record in saved['raw_records']:
            _checkpoint_asset(record)
        return validate_judgment(saved['judgment'])
    raw_path = folder / 'judge_raw.json'
    if raw_path.exists():
        saved = read_document(raw_path)
        if saved.get('job_identity') != identity or saved.get('checkpoint_digest') != digest({k: v for k, v in saved.items() if k != 'checkpoint_digest'}):
            raise CheckpointError('Common judge raw checkpoint changed')
        judgment = validate_judgment(_completion(saved['response']))
    else:
        judgment = frozen.judge(provider, q, context, candidate, folder, identity)
    frozen.checkpoint(path, dict(judgment=judgment, input_identity=digest(binding),
        raw_records=[file_record(folder / name) for name in ('judge_raw.json', 'judge.json') if (folder / name).exists()]))
    return judgment


def _evaluate_questions(ctx, key, book, method, reference, queries, engine, provider, budget, root, identity, deferred=None):
    records = []; questions = _dataset_questions(reference, book)
    for q in questions:
        _recheck(ctx); folder = root / 'questions' / digest(q['question_id']); path = folder / 'record.json'
        if path.exists():
            saved = read_document(path)
            if saved.get('result_id') != identity or saved.get('checkpoint_digest') != digest({k: v for k, v in saved.items() if k != 'checkpoint_digest'}):
                raise ValueError('ECHA question checkpoint changed')
            records.append(saved); continue
        folder.mkdir(parents=True, exist_ok=True)
        try:
            query = queries[q['question_id']] if queries is not None else _query(ctx, q, provider, budget, root)[0]
            retrieval_path = folder / 'retrieval.json'
            if retrieval_path.exists():
                retrieval = read_document(retrieval_path)
                if retrieval.get('result_id') != identity or retrieval.get('checkpoint_digest') != digest({k: v for k, v in retrieval.items() if k != 'checkpoint_digest'}):
                    raise CheckpointError('ECHA retrieval checkpoint changed')
                engine.load_book(book); engine.candidate_ids = tuple(retrieval['candidate_ids'])
                context, trace = retrieval['context'], retrieval['trace']
                for record in trace.get('deferred_original_batches', []):
                    _checkpoint_asset(record)
            else:
                if deferred is not None and method not in {'keyword', 'vector', 'hybrid', 'no_mapping'}:
                    seeds, neighbors, _ = engine._concepts(q['question'], query, method != 'no_semantic', method != 'no_relations')
                    if method == 'no_role_verification':
                        engine.unverified[book] = deferred.candidates(set(seeds+neighbors))
                    else:
                        engine.qualified[book] = deferred.ensure(seeds+neighbors, provider, budget)
                selected, trace = engine.retrieve(q['question'], book, query, method)
                if deferred is not None:
                    trace = dict(trace, alignment_protocol='exact_full_original_batches_no_regrouping',
                        deferred_original_batches=[deferred.proofs[i] for i in sorted(deferred.proofs)],
                        deferred_cache=deferred.status())
                context = frozen.old.pack_context(selected, q['question'])
                retrieval = dict(result_id=identity, context=context, selected_ids=[r['evidence_ids'][0] for r in selected],
                    candidate_ids=list(engine.candidate_ids), trace=trace)
                frozen.checkpoint(retrieval_path, retrieval)
            frozen.old.validate_context(context, book)
            candidate, receipt = _answer(provider, q, context, folder, budget.stage('answer'))
            judgment = _judge(provider, q, context, candidate, folder, identity)
            coarse = trace.get('coarse_evidence_ids', engine.candidate_ids)
            scores = dict(frozen.old.score_answer(candidate['answer'], q['answer']),
                **frozen.old.citations_score(candidate['citations'], context, engine.rows),
                **frozen.old.reference_metrics(q, context, retrieval['selected_ids'], coarse, engine.structure))
            scores['fallback_inclusive_candidate_recall'] = len(set(q['evidence_ids']) & set(engine.candidate_ids))/len(set(q['evidence_ids']))
            record = dict(result_id=identity, status='completed', question_id=q['question_id'], question_type=q['type'],
                target_book_id=book, candidate=candidate, context=context, retrieval=trace, machine_judge=judgment, scores=scores, answer_receipt=receipt)
        except (NativeAccountUnavailable, NativeBudgetExceeded, HoldoutBudgetExceeded):
            raise
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            if isinstance(exc, CheckpointError):
                raise
            error = folder / 'error.log'; error.write_text(traceback.format_exc(), encoding='utf-8')
            record = dict(result_id=identity, status='failed', question_id=q['question_id'], question_type=q['type'],
                target_book_id=book, error_type=type(exc).__name__, error_record=file_record(error))
        records.append(frozen.checkpoint(path, record))
        atomic(root / 'state.json', dict(status='running', records=len(records), total=len(questions), raw_text_exported=False))
    result = dict(frozen.old.summarize(records, [q['question_id'] for q in questions]), state_key=key, target_book_id=book,
        method=method, failed_count=sum(r['status'] == 'failed' for r in records), result_id=identity, raw_text_exported=False)
    if not questions:
        result.update(status='insufficient_empty_sealed_book', empty_sealed_book=True, reason='zero_legal_sealed_questions;metrics_NA')
    write_immutable(root / 'result.json', result)
    return result


def _development(ctx, run_root, provider, alignment_policy='exhaustive'):
    _recheck(ctx, deep=True)
    if alignment_policy not in ('exhaustive', 'exact-deferred'):
        raise ValueError('Unknown ECHA alignment policy')
    fixed = preflight(ctx, run_root) if alignment_policy == 'exact-deferred' else None
    root = _base(ctx, run_root, 'development') / alignment_policy; root.mkdir(parents=True, exist_ok=True)
    index_path = ctx['parent'] / 'datasets/index.json'; index = read_document(index_path)['books']
    references = {b: index[b]['development'] for b in ctx['books']}
    questions = {b: _dataset_questions(ref, b) for b, ref in references.items()}
    if any(len(v) != 4 for v in questions.values()) or sum(map(len, questions.values())) != 32:
        raise ValueError('Development requires exactly the existing32 questions, four per book')
    selected = [k for k, e in ctx['entries'].items() if e['state'] == 'F_corrected' and e['cohort'] in POLICY['development_cohorts']]
    if len(selected) != 3 or sorted(b for k in selected for b in ctx['entries'][k]['active_books']) != ctx['books']:
        raise ValueError('Development requires compABCD/osAB/netAB F_corrected exactly once per book')
    base_contract = dict(context_identity=ctx['identity'], datasets=references, dataset_index=file_record(index_path), mode='development', policy=POLICY,
        alignment_protocol='exact_full_original_batches_no_regrouping' if fixed else 'development_selected_subset_batches_not_full_protocol',
        frozen_full_preflight=file_record(_base(ctx, run_root, 'preflight') / 'result.json') if fixed else None)
    write_immutable(root / 'input_contract.json', base_contract)
    with FileLock(str(root / 'execution.lock'), timeout=0), _provider_scope(ctx, run_root, provider, root / 'requests.jsonl') as budget:
        chosen, plans, queries, query_records, dispatchers = {}, {}, {}, {}, {}
        for key in selected:
            entry = ctx['entries'][key]; engine = _engine(ctx, key, {b: [] for b in entry['active_books']})
            for book in entry['active_books']:
                concepts = set(); queries[book] = {}; query_records[book] = {}
                for q in questions[book]:
                    query, rec = _query(ctx, q, provider, budget, root); queries[book][q['question_id']] = query; query_records[book][q['question_id']] = rec
                    seeds, neighbors, _ = engine._concepts(q['question'], query)
                    concepts.update(seeds + neighbors)
                chosen[book] = sorted(concepts)
                if fixed:
                    dispatchers[book] = ExactDeferredAlignment(ctx, key, book, fixed['plans'][key][book], root / 'alignment' / key / book)
                    plans[book] = dict(full_plan=fixed['plans'][key][book], selected_original_batches=dispatchers[book].required_batches(chosen[book]))
                else:
                    plans[book] = _plans(ctx, key, book, chosen[book])
        contract = dict(base_contract, query_vectors=query_records, selected_concepts=chosen,
            candidate_digests={b: digest(p) for b, p in plans.items()})
        identity = digest(contract); write_immutable(root / 'contract.json', dict(result_id=identity, **contract))
        if (root / 'result.json').exists():
            saved = read_document(root / 'result.json')
            if saved['result_id'] != identity:
                raise ValueError('Development result identity changed')
            return saved
        logical = 2*sum(len(p['selected_original_batches']) for p in plans.values())+64 if fixed else 2*sum((sum(len(r['candidates']) for r in p)+7)//8 for p in plans.values())+64
        cost = logical*ctx['transport_attempts']
        batches = (logical-64)//2
        projection = dict(verification_logical_base=batches, format_repair_logical_upper=batches, answer_judge_logical_base=64,
            logical_with_format_repairs=logical, transport_attempts_per_logical_upper=ctx['transport_attempts'], theoretical_http_upper=cost,
            admission='bounded32_pilot;actual_shared8000_cap_hard;all_retries_not_reserved_simultaneously')
        write_immutable(root / 'allowance_projection.json', projection)
        if _remaining(provider, budget) <= 0:
            result = dict(status='budget_preflight_blocked', required_http_upper=cost, remaining_http=_remaining(provider, budget),
                full_alignment_complete=False, question_count=32, completed_count=0, failed_count=0, pending_count=32,
                allowance_projection=projection, result_id=identity, model_api_calls=0, raw_text_exported=False)
            atomic(root / 'state.json', result); return result
        results = []; mappings_index = {}
        for key in selected:
            entry = ctx['entries'][key]; qualified = {}
            for book in entry['active_books']:
                path = root / 'alignment' / key / book
                qualified[book] = dispatchers[book].ensure(chosen[book], provider, budget) if fixed else pipeline.verify_candidates(provider,
                    entry['graph_data'], ctx['structures'][book], plans[book], path, budget.stage('development_alignment'))
                out = path / 'partial_mappings.json'
                write_immutable(out, dict(context_identity=ctx['identity'], result_id=identity, records=qualified[book],
                    selected_concepts=chosen[book], pair_count=len(qualified[book]), full_alignment_complete=False,
                    alignment_protocol=base_contract['alignment_protocol']))
                mappings_index[book] = file_record(out)
            engine = _engine(ctx, key, qualified)
            for book in entry['active_books']:
                results.append(_evaluate_questions(ctx, key, book, 'echa', references[book], queries[book], engine, provider, budget,
                    root / 'evaluations' / key / book, identity))
        _recheck(ctx, deep=True)
        completed = sum(r['completed_count'] for r in results)
        metrics = {name: sum((r['metrics'].get(name) or 0)*r['question_count'] for r in results)/32
            for name in ('f1', 'em', 'machine_judge_mean', 'machine_grounding_mean', 'coarse_evidence_recall', 'selected_evidence_recall', 'evidence_recall', 'fallback_rate')}
        result = dict(status='completed_development_only' if completed == 32 else 'partial_development_only', question_count=32,
            completed_count=completed, failed_count=sum(r['failed_count'] for r in results), full_alignment_complete=False,
            cohorts=POLICY['development_cohorts'], selected_concepts=chosen, partial_mappings=mappings_index,
            alignment_protocol=base_contract['alignment_protocol'],
            allowance_projection=projection,
            results=results, metrics=metrics, denominator=POLICY['failed_denominator'], result_id=identity,
            context_identity=ctx['identity'], local_http_attempts=budget.count(), raw_text_exported=False)
        write_immutable(root / 'result.json', result); atomic(root / 'state.json', result); return result


def development(ctx, run_root, provider, alignment_policy='exhaustive'):
    try:
        return _development(ctx, run_root, provider, alignment_policy)
    except (HoldoutBudgetExceeded, NativeBudgetExceeded) as exc:
        root = _base(ctx, run_root, 'development') / alignment_policy; records = []
        for path in sorted((root / 'evaluations').rglob('record.json')):
            row = read_document(path)
            if row.get('checkpoint_digest') != digest({k: v for k, v in row.items() if k != 'checkpoint_digest'}):
                raise CheckpointError('Interrupted pilot question checkpoint changed')
            records.append(row)
        index = read_document(ctx['parent'] / 'datasets/index.json')['books']
        expected = [q['question_id'] for b in ctx['books'] for q in _dataset_questions(index[b]['development'], b)]
        summary = frozen.old.summarize(records, expected)
        result = dict(summary, status='partial_development_budget_exhausted', failed_count=sum(r['status']=='failed' for r in records),
            pending_count=32-len(records), full_alignment_complete=False, context_identity=ctx['identity'],
            budget_error=type(exc).__name__, local_http_attempts=ECHABudget(provider, run_root).count(),
            trial_http_cap=8000, raw_text_exported=False)
        atomic(root / 'state.json', result)
        return result


def alignment(ctx, run_root, provider):
    _recheck(ctx, deep=True); planned = preflight(ctx, run_root); root = _base(ctx, run_root, 'alignment'); root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'execution.lock'), timeout=0), _provider_scope(ctx, run_root, provider, root / 'requests.jsonl') as budget:
        if (root / 'alignment_complete.json').exists():
            return _full_alignment(ctx, run_root)[1]
        remaining = _remaining(provider, budget)
        if planned['verification_logical_base'] > remaining:
            result = dict(status='budget_preflight_blocked', verification_logical_base=planned['verification_logical_base'],
                format_repair_logical_upper=planned['verification_logical_base'], theoretical_http_upper=planned['verification_http_upper'], remaining_http=remaining,
                full_alignment_complete=False, context_identity=ctx['identity'], model_api_calls=0, raw_text_exported=False)
            atomic(root / 'state.json', result); return result
        output = {}
        for key, references in planned['plans'].items():
            entry = ctx['entries'][key]; output[key] = {}
            for book, ref in references.items():
                plan = asset(ref); folder = root / key / book
                rows = pipeline.verify_candidates(provider, entry['graph_data'], ctx['structures'][book], plan['plans'], folder, budget.stage('full_alignment'))
                if len(rows) != plan['pair_count']:
                    raise ValueError('Full alignment decision coverage incomplete')
                path = folder / 'qualified_mappings.json'
                write_immutable(path, dict(context_identity=ctx['identity'], planned=ref, pair_count=len(rows), records=rows, full_alignment_complete=True))
                output[key][book] = file_record(path)
        _recheck(ctx, deep=True)
        result = dict(status='completed_full_alignment', context_identity=ctx['identity'], full_alignment_complete=True,
            planned=file_record(_base(ctx, run_root, 'preflight') / 'result.json'), mappings=output, pair_count=planned['pair_count'],
            verification_logical_base=planned['verification_logical_base'], format_repair_logical_upper=planned['verification_logical_base'],
            theoretical_http_upper=planned['verification_http_upper'],
            local_http_attempts=budget.count(), raw_text_exported=False)
        write_immutable(root / 'alignment_complete.json', result); atomic(root / 'state.json', result); return result


def _full_alignment(ctx, run_root):
    path = _base(ctx, run_root, 'alignment') / 'alignment_complete.json'
    if not path.exists():
        raise DependencyPending('Complete all-concept ECHA alignment required; development mappings are partial')
    result = read_document(path)
    if result.get('context_identity') != ctx['identity'] or result.get('full_alignment_complete') is not True or result.get('status') != 'completed_full_alignment':
        raise ValueError('Frozen complete alignment identity differs')
    planned = asset(result['planned']); qualified = {}; total = 0
    if set(result['mappings']) != set(ctx['entries']):
        raise ValueError('Full alignment states incomplete')
    for key, entry in ctx['entries'].items():
        if set(result['mappings'][key]) != set(entry['active_books']):
            raise ValueError('Full alignment book conditions incomplete')
        qualified[key] = {}
        for book, ref in result['mappings'][key].items():
            value = asset(ref); plan = asset(planned['plans'][key][book]); rows = value['records']
            expected = {(p['concept_id'], c['evidence_id']) for p in plan['plans'] for c in p['candidates']}
            if value.get('full_alignment_complete') is not True or value['context_identity'] != ctx['identity'] or len(rows) != plan['pair_count'] or {(r['concept_id'], r['evidence_id']) for r in rows} != expected:
                raise ValueError('Qualified full alignment pair inventory differs')
            core.lift_coverage(entry['graph_data'], ctx['structures'][book], rows)
            qualified[key][book] = rows; total += len(rows)
    if total != result['pair_count'] or total != planned['pair_count']:
        raise ValueError('Complete ECHA alignment denominator differs')
    return qualified, result


def deferred_evaluation_plan(ctx, run_root, split):
    _recheck(ctx, deep=True); fixed = preflight(ctx, run_root); references, seal = datasets(ctx, run_root, split)
    root = _base(ctx, run_root, 'deferred_evaluation_plan') / split
    questions = {b: _dataset_questions(ref, b) for b, ref in references.items()}; queries = {}; proofs = {}; missing = []
    for book, values in questions.items():
        queries[book] = {}; proofs[book] = {}
        for q in values:
            cached = _existing_query(ctx, q, run_root, root)
            if cached is None:
                missing.append(dict(target_book_id=book, question_id=q['question_id'])); continue
            queries[book][q['question_id']], proofs[book][q['question_id']] = cached
    jobs = []; selected_batches = {}; batch_count = evaluations = 0
    for key, entry in ctx['entries'].items():
        selected_batches[key] = {}; engine = _engine(ctx, key, {b: [] for b in entry['active_books']})
        for book in entry['active_books']:
            dispatcher = ExactDeferredAlignment(ctx, key, book, fixed['plans'][key][book], root / 'fixed_batches' / key / book)
            chosen = set()
            for method in _methods(entry):
                jobs.append(dict(state_key=key, book=book, method=method)); evaluations += len(questions[book])
                if method in {'keyword', 'vector', 'hybrid', 'no_mapping', 'no_role_verification'}:
                    continue
                for q in questions[book]:
                    query = queries[book].get(q['question_id'])
                    if query is not None:
                        seeds, neighbors, _ = engine._concepts(q['question'], query, method != 'no_semantic', method != 'no_relations')
                        chosen.update(dispatcher.required_batches(seeds+neighbors))
            selected_batches[key][book] = sorted(chosen); batch_count += len(chosen)
    plan = dict(status='deferred_evaluation_plan_ready' if not missing else 'query_vectors_pending', context_identity=ctx['identity'],
        split=split, datasets=references, seal=seal, full_plans=fixed['plans'],
        frozen_full_preflight=file_record(_base(ctx, run_root, 'preflight')/'result.json'), jobs=jobs,
        query_vector_records=proofs, selected_original_batches=selected_batches, missing_query_vectors=missing,
        unique_question_count=sum(map(len, questions.values())), question_evaluations=evaluations,
        verification_logical_base=batch_count, verification_logical_with_repairs=2*batch_count,
        answer_judge_logical_bodies=2*evaluations, baseline_cache_deductions=0,
        logical_base_bodies=batch_count+2*evaluations, format_repair_logical_upper=batch_count,
        theoretical_http_upper=(2*batch_count+2*evaluations)*ctx['transport_attempts'],
        transport_attempts_per_logical_upper=ctx['transport_attempts'],
        alignment_protocol='exact_full_original_batches_no_regrouping', full_alignment_complete=False,
        budget_scope='requested_complete_frozen_split;full_matrix_funding_not_claimed', model_api_calls=0, raw_text_exported=False)
    plan['plan_identity'] = digest(plan)
    write_immutable(root/'plan.json', plan)
    return plan, queries


def _deferred_jobs(ctx, plan, root, queries, provider, budget, identity, split):
    groups = [(key, book) for key, entry in ctx['entries'].items() for book in entry['active_books']]
    def run(group):
        key, book = group; entry = ctx['entries'][key]
        engine = _engine(ctx, key, {b: [] for b in entry['active_books']})
        dispatcher = ExactDeferredAlignment(ctx, key, book, plan['full_plans'][key][book], root/'alignment'/key/book)
        return [_evaluate_questions(ctx, key, book, method, plan['datasets'][book], queries[book],
            engine, provider, budget, root/'evaluations'/key/book/method, identity, deferred=dispatcher)
            for method in _methods(entry)]
    workers = 4 if provider.effective_http_cap == 200000 else 1
    if workers == 1:
        return [row for group in groups for row in run(group)]
    # 同一书图内仍串行；不同书图只共用有锁的收据与原四个请求槽。
    pool = ThreadPoolExecutor(max_workers=workers); completed = {}
    try:
        futures = {pool.submit(run, group): index for index, group in enumerate(groups)}
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
            atomic(root/'state.json', dict(status='running', result_id=identity, split=split,
                groups_finished=len(completed), groups=len(groups), worker_count=workers, raw_text_exported=False))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return [row for index in range(len(groups)) for row in completed[index]]


def _evaluate_deferred(ctx, run_root, provider, split, stage_budget=None):
    plan, queries = deferred_evaluation_plan(ctx, run_root, split)
    if plan['missing_query_vectors']:
        raise DependencyPending('Exact frozen query vectors required for complete joint budget preflight; no API issued')
    stage = _frozen_stage(plan, stage_budget)
    identity = plan['plan_identity']; root = _base(ctx, run_root, 'evaluate') / split / 'exact-deferred' / identity
    root.mkdir(parents=True, exist_ok=True)
    write_immutable(root/'contract.json', dict(result_id=identity, plan=file_record(_base(ctx, run_root, 'deferred_evaluation_plan')/split/'plan.json')))
    if (root/'result.json').exists():
        saved = read_document(root/'result.json')
        if saved.get('result_id') != identity or saved.get('checkpoint_digest') != digest({k: v for k, v in saved.items() if k != 'checkpoint_digest'}):
            raise CheckpointError('Deferred evaluation completed checkpoint changed')
        return saved
    if provider is None:
        raise DependencyPending('Original shared provider required for actual exact-deferred evaluation')
    try:
        with FileLock(str(root/'execution.lock'), timeout=0), _provider_scope(ctx, run_root, provider, root/'requests.jsonl') as budget, _stage_scope(provider, budget, stage, root):
            remaining = _remaining(provider, budget)
            if remaining <= 0 or (stage is None and plan['logical_base_bodies'] > remaining):
                blocked = dict(status='budget_preflight_blocked', result_id=identity, split=split,
                    logical_base_bodies=plan['logical_base_bodies'], format_repair_logical_upper=plan['format_repair_logical_upper'],
                    theoretical_http_upper=plan['theoretical_http_upper'], remaining_http=remaining,
                    admission='base_logical_projection_exceeds_available;explicit_frozen_bounded_stage_optional;no_unproven_cache_deduction',
                    question_evaluations=plan['question_evaluations'], unique_question_count=plan['unique_question_count'],
                    full_alignment_complete=False, full_matrix_funded=False, model_api_calls=0, raw_text_exported=False)
                atomic(root/'state.json', blocked); return blocked
            results = _deferred_jobs(ctx, plan, root, queries, provider, budget, identity, split)
            _recheck(ctx, deep=True)
            result = dict(status='completed_evaluation' if all(r['status']=='completed' for r in results) else 'partial_evaluation',
                result_id=identity, alignment_protocol=plan['alignment_protocol'], split=split, job_count=len(plan['jobs']),
                results=results, unique_question_count=plan['unique_question_count'], question_evaluations=plan['question_evaluations'],
                completed_evaluations=sum(r['completed_count'] for r in results), failed_evaluations=sum(r['failed_count'] for r in results),
                full_alignment_complete=False, full_matrix_funded=False, logical_base_bodies=plan['logical_base_bodies'],
                format_repair_logical_upper=plan['format_repair_logical_upper'], theoretical_http_upper=plan['theoretical_http_upper'],
                stage_budget=stage, stage_http_attempts=_stage_count(provider, stage),
                local_http_attempts=budget.count(), raw_text_exported=False)
            result = frozen.checkpoint(root/'result.json', result); atomic(root/'state.json', result); return result
    except (HoldoutBudgetExceeded, NativeBudgetExceeded) as exc:
        records = [read_document(path) for path in (root/'evaluations').rglob('record.json')]
        if any(r.get('result_id') != identity or r.get('checkpoint_digest') != digest({k: v for k, v in r.items() if k != 'checkpoint_digest'}) for r in records):
            raise CheckpointError('Deferred evaluation partial checkpoints changed')
        result = dict(status='partial_evaluation_budget_exhausted', result_id=identity, split=split,
            question_evaluations=plan['question_evaluations'], completed_evaluations=sum(r['status']=='completed' for r in records),
            failed_evaluations=sum(r['status']=='failed' for r in records), pending_evaluations=plan['question_evaluations']-len(records),
            full_alignment_complete=False, full_matrix_funded=False, budget_error=type(exc).__name__,
            local_http_attempts=ECHABudget(provider, run_root).count(), raw_text_exported=False)
        result.update(stage_budget=stage, stage_http_attempts=_stage_count(provider, stage))
        atomic(root/'state.json', result); return result


def evaluate(ctx, run_root, provider, split, alignment_policy='exhaustive', stage_budget=None):
    if alignment_policy == 'exact-deferred':
        return _evaluate_deferred(ctx, run_root, provider, split, stage_budget)
    if stage_budget is not None:
        raise ValueError('Frozen bounded stages require exact-deferred complete-plan execution')
    if alignment_policy != 'exhaustive':
        raise ValueError('Unknown ECHA evaluation alignment policy')
    _recheck(ctx, deep=True); qualified, alignment_record = _full_alignment(ctx, run_root)
    refs, seal = datasets(ctx, run_root, split)
    jobs = [dict(state_key=key, book=book, method=method) for key, entry in ctx['entries'].items() for book in entry['active_books']
        for method in (rag.ECHARetriever.methods if entry['state'] == 'F_corrected' else ('echa',))]
    contract = dict(context_identity=ctx['identity'], full_alignment=file_record(_base(ctx, run_root, 'alignment') / 'alignment_complete.json'),
        datasets=refs, split=split, seal=seal, jobs=jobs, policy=POLICY)
    identity = digest(contract); root = _base(ctx, run_root, 'evaluate') / split / identity; root.mkdir(parents=True, exist_ok=True)
    write_immutable(root / 'contract.json', dict(result_id=identity, **contract))
    if provider is None:
        raise DependencyPending('Qualified ECHA evaluation needs the original shared provider')
    with FileLock(str(root / 'execution.lock'), timeout=0), _provider_scope(ctx, run_root, provider, root / 'requests.jsonl') as budget:
        results = []
        for job in jobs:
            key, book, method = job['state_key'], job['book'], job['method']
            engine = _engine(ctx, key, qualified[key])
            results.append(_evaluate_questions(ctx, key, book, method, refs[book], None, engine, provider, budget,
                root / key / book / method, identity))
        _recheck(ctx, deep=True)
        result = dict(status='completed_evaluation' if all(r['status'] == 'completed' for r in results) else 'partial_evaluation',
            result_id=identity, split=split, job_count=len(jobs), results=results,
            unique_question_count=sum(len(_dataset_questions(ref, b)) for b, ref in refs.items()),
            question_evaluations=sum(r['question_count'] for r in results), full_alignment_complete=True,
            local_http_attempts=budget.count(), raw_text_exported=False)
        write_immutable(root / 'result.json', result); atomic(root / 'state.json', result); return result


def _safe(value):
    return {k: v for k, v in value.items() if k not in {'plans', 'mappings', 'partial_mappings', 'results', 'selected_concepts',
        'datasets', 'jobs', 'query_vector_records', 'selected_original_batches', 'full_plans'}}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True); p.add_argument('--inventory-sha', required=True)
    p.add_argument('--tokenizer-manifest', type=Path, required=True); p.add_argument('--tokenizer-sha', required=True)
    p.add_argument('--mode', choices=['preflight', 'development', 'alignment', 'evaluate'], required=True)
    p.add_argument('--alignment-policy', choices=['exhaustive', 'exact-deferred'], default='exhaustive')
    p.add_argument('--stage-budget-plan', type=Path); p.add_argument('--stage-budget-sha')
    p.add_argument('--split', choices=['regression', 'independent', 'blind64'], default='regression'); p.add_argument('--check-only', action='store_true')
    args = p.parse_args(); ctx = load_context(args.inventory, args.inventory_sha, args.tokenizer_manifest, args.tokenizer_sha)
    stage_budget = None
    if args.stage_budget_plan or args.stage_budget_sha:
        if not (args.stage_budget_plan and args.stage_budget_sha and args.mode == 'evaluate' and args.alignment_policy == 'exact-deferred'):
            raise ValueError('Frozen stage path and SHA required together for exact-deferred evaluation')
        _, stage_budget = _checked(args.stage_budget_plan, args.stage_budget_sha)
    if args.mode == 'development' and args.split != 'regression':
        raise ValueError('Development mode may only read the existing32 development questions')
    if args.mode == 'preflight':
        result = deferred_preflight(ctx, args.run_root) if args.alignment_policy == 'exact-deferred' else preflight(ctx, args.run_root)
        print(json.dumps(_safe(result))); return 0 if result['status'] != 'query_vectors_pending' else 2
    if args.check_only:
        if args.alignment_policy == 'exact-deferred' and args.mode == 'evaluate':
            planned, _ = deferred_evaluation_plan(ctx, args.run_root, args.split)
            _frozen_stage(planned, stage_budget)
            print(json.dumps(_safe(planned))); return 0 if not planned['missing_query_vectors'] else 2
        if args.alignment_policy == 'exact-deferred' and args.mode == 'alignment':
            print(json.dumps(_safe(preflight(ctx, args.run_root)))); return 0
        if args.mode == 'evaluate':
            _full_alignment(ctx, args.run_root); refs, _ = datasets(ctx, args.run_root, args.split)
            count = sum(len(_dataset_questions(r, b)) for b, r in refs.items())
        else:
            count = 32 if args.mode == 'development' else None
        print(json.dumps(dict(status='checked_no_api', mode=args.mode, split=args.split, context_identity=ctx['identity'],
            question_count=count, model_api_calls=0, full_alignment_complete=False, raw_text_exported=False))); return 0
    if args.mode == 'alignment' and args.alignment_policy == 'exact-deferred':
        prepared = preflight(ctx, args.run_root)
        print(json.dumps(dict(status='frozen_original_batches_ready_no_api', context_identity=ctx['identity'],
            original_pair_count=prepared['pair_count'], original_logical_batches=prepared['verification_logical_base'],
            full_alignment_complete=False, model_api_calls=0, raw_text_exported=False))); return 0
    stored = read_document(args.run_root / 'formal_provider/limits.json')
    if stored['limits'] != LIMITS or stored['started_unix'] != POLICY['original_started_unix'] or not (args.run_root / 'formal_provider/audit.sqlite').exists():
        raise ValueError('Existing original formal ledger required; initialization or reset forbidden')
    credentials(); construction = {'files': {'assets': {'runtime_config': ctx['inventory']['runtime_config']}}}
    transport = _base(ctx, args.run_root, 'transport')
    provider = AuditedNativeProvider(runtime_for(construction, transport), args.run_root / 'formal_provider', **LIMITS)
    try:
        result = development(ctx, args.run_root, provider, args.alignment_policy) if args.mode == 'development' else alignment(ctx, args.run_root, provider) if args.mode == 'alignment' else evaluate(ctx, args.run_root, provider, args.split, args.alignment_policy, stage_budget)
        print(json.dumps(_safe(result))); return 0 if result['status'].startswith('completed') else 2
    except Exception as exc:
        root = _base(ctx, args.run_root, args.mode); root.mkdir(parents=True, exist_ok=True)
        path = root / ('interruption_' + str(time.time_ns()) + '.log'); path.write_text(traceback.format_exc(), encoding='utf-8')
        result = dict(status='interrupted_preserved_checkpoints', mode=args.mode, split=args.split, error_type=type(exc).__name__,
            error_record=file_record(path), expected_development_count=32 if args.mode == 'development' else None, raw_text_exported=False)
        atomic(root / 'state.json', result); print(json.dumps(result)); return 2
    finally:
        provider.close()


if __name__ == '__main__':
    raise SystemExit(main())
