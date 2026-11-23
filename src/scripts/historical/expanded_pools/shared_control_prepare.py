"""原353题五方法共享精排候选；仅CPU/既有缓存，永不生成答案或补候选。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

METHODS = ('GraphRAG', 'LightRAG', 'HippoRAG', 'DVKG', 'K12-KGraph')
SCOPES = ('same_subject_pool', 'mixed_six_book_pool')
CANONICAL_SHA = '43cb0cb57a2288f1b39ebde8d77575c76b4b847764d279392f0e4b02922e1fcb'
N = Path(os.environ.get("DVKG_HISTORICAL_ROOT") or str(Path("outputs/complete_course_benchmark_20261022_v1").resolve()))
POLICY = dict(version='own_candidates_shared_RRF60_top64_control_v1', candidate_limit=64,
    rrf_constant=60, per_channel_limit=64, paragraph_dedup='book_and_original_source_id',
    global_fill=False, route_boost=False, retrieval_top_k=5, context_bytes=3000,
    baseline_answer_API_authorized=False, answer_generation=False, model_API_calls=0,
    source='each_method_actual_pre_common_RRF_candidates', graph_SOTA_claim=False)


class PendingCandidates(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def record(path):
    path = Path(path).resolve(strict=True); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(reference):
    raw = Path(reference['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Frozen shared-control dependency changed')
    return json.loads(raw)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value: raise ValueError('Existing output differs; use a new private output root')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)


def load(reference, name):
    if record(reference['path']) != reference: raise ValueError('Frozen executable dependency changed')
    spec = importlib.util.spec_from_file_location(name, reference['path'])
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def zero_api_guard():
    def deny(*args, **kwargs): raise RuntimeError('Shared-control preparation denies all network access')
    with patch.object(socket.socket, 'connect', deny), patch.object(socket.socket, 'connect_ex', deny), \
            patch.object(socket, 'create_connection', deny), patch.object(socket, 'getaddrinfo', deny):
        yield


def preselect(pool, question, vector, keys):
    """与原共同排序相同的两路RRF，只将段落输出上限从5扩到64。"""
    import numpy as np
    if not isinstance(question, str) or not question.strip(): raise ValueError('Nonempty exact question required')
    keys = {tuple(k) for k in keys}
    if not keys: raise PendingCandidates('empty_actual_candidate_set')
    if not keys <= pool.source_positions.keys(): raise ValueError('Candidate outside its declared source pool')
    indices = {pool.source_positions[k] for k in keys}
    query = pool._query(vector); lexical = pool.lexical.lexical(question); dense = pool.vectors @ query
    if (len(lexical) != len(pool.rows) or not np.isfinite(dense).all()
            or any(not math.isfinite(float(v)) for v in lexical)):
        raise ValueError('Finite complete common ranking scores required')
    tie = lambda i: (pool.rows[i]['book_id'], pool.rows[i]['source_evidence_id'])
    left = sorted((i for i in indices if lexical[i] > 0), key=lambda i: (-float(lexical[i]), tie(i)))[:64]
    right = sorted(indices, key=lambda i: (-float(dense[i]), tie(i)))[:64]
    scores = {}
    for channel in (left, right):
        for rank, i in enumerate(channel, 1): scores[i] = scores.get(i, 0.) + 1. / (60 + rank)
    rows = []; seen = set()
    for i in sorted(scores, key=lambda i: (-scores[i], tie(i))):
        row = pool.rows[i]; key = row['book_id'], row['original_source_id']
        if key in seen: continue
        seen.add(key); rows.append(row)
        if len(rows) == 64: break
    return rows, dict(input_candidate_count=len(keys), input_candidate_digest=digest(sorted(keys)),
        output_paragraph_count=len(rows), output_paragraph_digest=digest(sorted(seen)),
        rrf_constant=60, per_channel_limit=64, global_fill=False, route_boost=False)


def native_candidate_keys(saved):
    trace = saved.get('trace', {})
    if 'native_candidate_keys' not in trace: raise PendingCandidates('full_native_candidate_keys_not_saved')
    keys = {tuple(k) for k in trace['native_candidate_keys']}
    if any(len(k) != 2 or not all(isinstance(x, str) and x for x in k) for k in keys):
        raise ValueError('Invalid native evidence key')
    if len(keys) != trace.get('candidate_count'): raise ValueError('Native candidate count differs')
    return keys


def dvkg_candidate_keys(task, pool):
    trace = task.get('candidate_trace', {})
    if 'original_candidate_evidence_ids' not in trace:
        raise PendingCandidates('original_DVKG_pre_RRF_evidence_ids_not_saved')
    ids = trace['original_candidate_evidence_ids']
    if len(set(ids)) != len(ids) or len(ids) != trace.get('original_candidate_count'):
        raise ValueError('Original DVKG candidate evidence count differs')
    if any(uid not in pool.native_sources for uid in ids): raise ValueError('DVKG evidence outside source pool')
    return {tuple(pool.native_sources[uid]) for uid in ids}


def original_dvkg_keys(saved, pool, question, vector):
    trace = saved.get('trace', {})
    if 'routing' not in trace or type(trace.get('fallback')) is not bool:
        raise PendingCandidates('canonical_original_DVKG_routing_trace_missing')
    keys = {tuple(k) for k in trace['routing'].get('evidence_ids', [])}
    if not keys <= pool.source_positions.keys(): raise ValueError('Original DVKG route leaves source pool')
    if trace['fallback']:
        lexical = pool.lexical.lexical(question); dense = pool.vectors @ pool._query(vector)
        tie = lambda i: (pool.rows[i]['book_id'], pool.rows[i]['source_evidence_id'])
        left = sorted((i for i in range(len(pool.rows)) if lexical[i] > 0),
            key=lambda i: (-float(lexical[i]), tie(i)))[:64]
        right = sorted(range(len(pool.rows)), key=lambda i: (-float(dense[i]), tie(i)))[:64]
        keys.update((pool.rows[i]['book_id'], pool.rows[i]['source_evidence_id']) for i in set(left + right))
    if len(keys) != trace.get('candidate_count'): raise ValueError('Canonical original DVKG candidate count differs')
    return keys


def replay_k12_candidates(engines, question, vector, allowed, cache):
    result = set()
    for book in sorted(allowed):
        key = question['question_id'], book
        if key not in cache:
            if book not in engines: raise PendingCandidates('K12_frozen_book_engine_missing')
            engine = engines[book]
            engine.retrieve(question['question'], book, vector, 'dual')
            cache[key] = {(book, eid) for eid in engine.candidate_ids}
        result.update(cache[key])
    return result


def answer_body(answer, runtime, question, context):
    return dict(model=runtime['llm']['model'], messages=answer.answer_messages(dict(question=question, context=context)),
        temperature=0, seed=42, enable_thinking=False, max_tokens=2048, response_format=dict(type='json_object'))


def verify_k12_replay(saved, q, scope, pool, vector, keys, answer, runtime):
    """读取旧body做相等核验；不创建调用方或执行该body。"""
    record_value = saved.get('record', {}); retrieval = saved.get('retrieval', {})
    chosen = pool._rank(q['question'], pool._query(vector), {pool.source_positions[k] for k in keys}, 'hybrid')
    context = pool.pack_context(chosen, q['question'])
    selected = [[r['book_id'], r['source_evidence_id']] for r in chosen]
    if (record_value.get('question_id') != q['question_id'] or record_value.get('book') != q['target_book_id']
            or record_value.get('scope') != scope or record_value.get('candidate_count') != len(keys)
            or record_value.get('selected_digest') != digest(selected)
            or record_value.get('context_sha256') != digest(context)
            or retrieval.get('question') != q['question'] or retrieval.get('selected_source_keys') != selected
            or json.dumps(retrieval.get('context'), ensure_ascii=False) != json.dumps(context, ensure_ascii=False)
            or saved.get('body') != answer_body(answer, runtime, q['question'], context)):
        raise PendingCandidates('K12_exact_count_selected_context_body_replay_mismatch')
    return dict(exact_replay=True, candidate_count=len(keys), selected_digest=digest(selected),
        context_sha256=digest(context), old_answer_body_digest=digest(saved['body']), new_model_calls=0)


def score_packet(tasks, builder):
    packet = builder(tasks)
    if packet.get('answers_included') is not False: raise ValueError('Score packet may not contain labels')
    for item in packet['packets']:
        if set(item) != {'task_id', 'question', 'candidates'}: raise ValueError('Unexpected scoring input')
        for c in item['candidates']:
            if set(c) != {'key', 'text', 'char_start', 'char_end', 'input_sha256', 'source_sha256'}:
                raise ValueError('Unexpected candidate scoring field')
    return packet


def prepare_case(method, q, prepared, deps):
    scope = prepared['scope']; label = q['evaluation_course_id'] if scope == SCOPES[0] else 'mixed_six_books'
    pool = deps['pools'][label]; vector = deps['queries'][q['question_id']]
    if method == 'K12-KGraph' and deps.get('k12_pending'):
        raise PendingCandidates(deps['k12_pending'])
    anchor = deps['canonical'].get((method, scope, q['question_id']))
    if anchor is None: raise ValueError('Missing canonical original observation')
    canonical_key = digest([q['question_id'], q['target_book_id'], q['question']])
    if anchor['canonical_question_key'] != canonical_key: raise ValueError('Canonical original question bridge differs')
    source = None; replay = None
    if method == 'DVKG':
        keys = dvkg_candidate_keys(prepared, pool)
        source = anchor['source_retrieval']; saved = deps['read_anchor'](source)
        if (saved.get('question_id') != q['question_id'] or saved.get('question') != q['question']
                or saved.get('scope') != scope or saved.get('method') != 'dvkg'
                or saved.get('checkpoint_digest') != digest({k: v for k, v in saved.items() if k != 'checkpoint_digest'})):
            raise ValueError('Canonical original DVKG retrieval identity differs')
        if keys != original_dvkg_keys(saved, pool, q['question'], vector):
            raise ValueError('Prepared DVKG candidates differ from original canonical route/fallback')
    else:
        source = anchor['source_retrieval']; saved = deps['read_anchor'](source)
        if method == 'K12-KGraph':
            try:
                keys = replay_k12_candidates(deps['k12_engines'], q, vector, pool.allowed_books, deps['k12_cache'])
                replay = verify_k12_replay(saved, q, scope, pool, vector, keys, deps['k12_answer'], deps['k12_runtime'])
            except PendingCandidates:
                raise
            except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
                raise PendingCandidates('K12_CPU_replay_failed:' + type(exc).__name__ + ':' +
                    hashlib.sha256(str(exc).encode('utf-8')).hexdigest()) from None
        else:
            if (saved.get('question_id') != q['question_id'] or saved.get('question') != q['question']
                    or saved.get('scope') != scope or saved.get('status') != 'ready'
                    or saved.get('method') != {'GraphRAG': 'graphrag', 'LightRAG': 'lightrag', 'HippoRAG': 'hipporag'}[method]
                    or saved.get('checkpoint_digest') != digest({k: v for k, v in saved.items() if k != 'checkpoint_digest'})):
                raise ValueError('Native task identity/checkpoint differs from canonical source')
            keys = native_candidate_keys(saved)
    rows, trace = preselect(pool, q['question'], vector, keys)
    return dict(task_id=digest(dict(method=method, scope=scope, question=q['question_id'])), method=method,
        scope=scope, question_id=q['question_id'], canonical_question_key=canonical_key,
        question=q['question'], book_id=q['target_book_id'], course_id=q['evaluation_course_id'],
        candidates=rows, candidate_count=len(rows), candidate_trace=trace,
        source_retrieval=source, K12_replay=replay, policy=POLICY,
        original353_observed_questions=True, independent_test=False, model_API_calls=0)


def make_pool(allowed, units, structures, joint, lexical_class):
    """仅装载段落词法索引和既有向量；不实例化语义图或原生模型。"""
    import numpy as np
    rows = []; vectors = {}; native = {}
    for book in sorted(allowed):
        value = units[book]; matrix = value['vectors']
        if isinstance(matrix, dict):
            if record(matrix['path']) != matrix: raise ValueError('Frozen paragraph matrix changed')
            matrix = np.load(matrix['path'], allow_pickle=False, mmap_mode='r')
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (len(value['rows']), 1024) or not np.isfinite(matrix).all():
            raise ValueError('Frozen paragraph matrix shape differs')
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1., atol=1e-4):
            raise ValueError('Frozen paragraph vectors must remain normalized')
        for source, vector in zip(value['rows'], matrix):
            eid = source['evidence_ids'][0]; evidence = structures[book]['evidence'][eid]
            if source['book_id'] != book or len(source['evidence_ids']) != 1 or source['text'] != evidence['text']:
                raise ValueError('Frozen paragraph identity/text differs')
            key = book, eid; uid = json.dumps(key, ensure_ascii=False, separators=(',', ':'))
            if key in vectors: raise ValueError('Duplicate source slice identity')
            item = dict(source, source_evidence_id=eid, original_source_id=evidence['source']['source_id'],
                evidence_ids=[uid], evidence=[dict(id=uid, text=source['text'])])
            rows.append(item); vectors[key] = vector; native[uid] = key
    pool = joint.JointPoolRetriever.__new__(joint.JointPoolRetriever)
    pool.allowed_books = sorted(allowed); pool.paragraph_dedup = True
    pool.lexical = lexical_class(rows, dict(entities=[], relations=[])); pool.rows = pool.lexical.rows
    pool.source_positions = {(r['book_id'], r['source_evidence_id']): i for i, r in enumerate(pool.rows)}
    pool.native_sources = native
    pool.vectors = np.asarray([vectors[(r['book_id'], r['source_evidence_id'])] for r in pool.rows], dtype=np.float32)
    return pool


def cached_k12_vectors(entry, roots, materialize, proofs=None):
    """沿用旧16条批次身份，只读meta和向量；不存在时明确pending。"""
    plan = entry['vector_plan']; groups = {}
    for key in plan['missing_texts']:
        groups.setdefault(digest(plan['embedding_configuration']), []).append(key)
    for root in roots:
        reuse = {}; complete = True; used = []
        for keys in groups.values():
            keys = sorted(keys)
            for first in range(0, len(keys), 16):
                chosen = keys[first:first + 16]; path = Path(root) / 'new_vectors' / (digest(chosen) + '.npy')
                meta = path.with_suffix('.meta.json')
                if not meta.is_file(): complete = False; break
                saved = read(meta)
                if saved['keys'] != chosen or record(path) != saved['vectors']:
                    raise ValueError('Frozen K12 vector batch changed')
                used.extend([record(meta), saved['vectors']])
                for i, key in enumerate(chosen): reuse[key] = dict(vectors=saved['vectors'], row=i)
            if not complete: break
        if not complete: continue
        frozen = deepcopy(plan)
        frozen['entries'] = [dict(r, reuse=r['reuse'] or reuse[r['text_key']]) for r in plan['entries']]
        frozen['missing_count'] = 0
        result = materialize(frozen)
        if proofs is not None:
            used.extend(r['reuse']['vectors'] for r in frozen['entries'])
            proofs.extend({(r['path'], r['sha256']): r for r in used}.values())
        return result
    raise PendingCandidates('K12_complete_frozen_concept_vectors_missing')


def initialize_k12(deps, canonical):
    """直接读旧registry，不调用原prepare/prepare_entry/模型工厂。"""
    protocol_ref = canonical['frozen_legacy_source_refs']['k12_protocol']; protocol = checked(protocol_ref)
    if protocol.get('variant') != 'occurrence_mapping_v2': raise ValueError('K12 frozen bridge variant differs')
    bridge = load(protocol['bridge'], 'course_experiments.evaluation.baseline_rag_bridge')
    from course_experiments.evaluation import state_dual_view as state
    options_path = N / 'network_supplement_20261103/retrieval_options.json'; options = read(options_path)
    configuration = read(options['old_configuration']); release = checked(configuration['release'])
    controlled = checked(release['registry']['controlled_K12']); inventory = checked(controlled['inventory'])
    assets = checked(inventory['asset_manifest'])
    entries = {c: inventory['ready']['k12_kgraph:' + c] for c in ['compAB', 'osAB']}
    net_path = N / 'network_supplement_20261103/nov9/k12_rag32/entry.json'; entries['netAB'] = read(net_path)
    roots = [Path(options['old_configuration']).parent / 'jobs/K12_KGraph',
        Path(configuration['release']['path']).parent / 'jobs/K12_KGraph',
        *[Path(r) for r in options['baseline_vector_roots']], net_path.parent]
    matrices = {}; proofs = [protocol_ref, protocol['bridge'], record(options_path), record(net_path),
        record(options['old_configuration']), configuration['release'], release['registry']['controlled_K12'],
        controlled['inventory'], inventory['asset_manifest']]
    for cohort, entry in entries.items():
        if entry['artifact'] != protocol['artifacts'][cohort]: raise ValueError('K12 own graph artifact differs')
        original = bridge.load_complete_artifact(entry['artifact'], assets, cohort, 'k12_kgraph')
        if digest(original['graph']) != digest(entry['graph']) or digest(original['mappings']) != digest(entry['mappings']):
            raise ValueError('K12 registry graph/mapping differs from exact completed construction')
        matrices[cohort] = cached_k12_vectors(entry, roots, state.materialize_reused, proofs)
        proofs.append(entry['artifact'])
    cohorts = dict(computer_organization='compAB', operating_systems='osAB', computer_networks='netAB')
    engines = {}
    for source in deps['cfg']['sources']:
        book = source['book_id']; cohort = cohorts[source['course_id']]; entry = entries[cohort]
        allowed = {s['book_id'] for s in deps['cfg']['sources'] if s['course_id'] == source['course_id']}
        if set(entry['active_books']) != allowed: raise ValueError('K12 own full two-book graph scope differs')
        for b in allowed:
            if assets['structures'][b] != deps['source_refs'][b]['structure'] or assets['unit_assets'][b] != deps['source_refs'][b]['units']:
                raise ValueError('K12 source corpus differs from common reranking corpus')
        engines[book] = bridge.BaselineRetriever('k12_kgraph', entry['graph'],
            {b: assets['structures'][b] for b in allowed}, entry['mappings'],
            {b: assets['unit_assets'][b] for b in allowed}, entry['vector_plan']['concept_ids'], matrices[cohort],
            variant=protocol['variant'])
    execution = checked(canonical['frozen_legacy_source_refs']['k12_execution_contract'])
    deps.update(k12_engines=engines, k12_cache={}, k12_proofs=proofs,
        k12_answer=load(execution['original_answer_worker'], 'shared_control_frozen_answer'),
        k12_runtime=checked(execution['runtime']))
    deps['k12_proofs'].extend([execution['original_answer_worker'], execution['runtime'], record(state.__file__),
        record(sys.modules[bridge.CompletedMappingsRetriever.__module__].__file__)])


def initialize(full_root):
    if os.name != 'posix': raise RuntimeError('Real original textbook inputs stay on PRIMARY')
    finish_path = Path(__file__).with_name('full_rerank_finish.py')
    finish = load(record(finish_path), 'shared_control_full_finish')
    contract = read(full_root / 'input_contract.json'); prepared = read(full_root / 'private_prepared_tasks.json')
    if prepared.get('contract_digest') != digest(contract): raise ValueError('Full prepared input contract changed')
    deps = finish.initialize_dependencies(contract)
    canonical = deps['canonical']
    if deps['canonical_ref']['sha256'] != CANONICAL_SHA: raise ValueError('Canonical original result differs')
    from course_experiments.evaluation import joint_pool_rag as joint
    from course_experiments.evaluation.graph_rag_v2 import EvidenceRetriever
    sources = deps['cfg']['sources']; courses = {s['course_id'] for s in sources}
    units = {s['book_id']: checked(deps['source_refs'][s['book_id']]['units']) for s in sources}
    deps['pools'] = {c: make_pool([s['book_id'] for s in sources if s['course_id'] == c],
        units, deps['structures'], joint, EvidenceRetriever) for c in courses}
    deps['pools']['mixed_six_books'] = make_pool([s['book_id'] for s in sources], units, deps['structures'], joint, EvidenceRetriever)
    import numpy as np
    queries = {}; tasks = {}
    questions = {q['question_id']: dict(q, evaluation_course_id=s['course_id']) for s in sources
        for q in deps['datasets'][s['book_id']]}
    for task in prepared['tasks']:
        qid = task['question_id']; q = questions.get(qid); key = task['scope'], qid
        if (q is None or key in tasks or task['scope'] not in SCOPES or task['question'] != q['question']
                or task['book_id'] != q['target_book_id']): raise ValueError('Prepared original question/scope differs')
        tasks[key] = task
        reference = task['query_vector_ref']
        if record(reference['path']) != reference: raise ValueError('Frozen original query vector changed')
        matrix = np.load(reference['path'], allow_pickle=False)
        if matrix.shape != (1, 1024) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1., atol=1e-4):
            raise ValueError('Exact normalized frozen query vector required')
        if qid in queries and not np.array_equal(queries[qid], matrix[0]): raise ValueError('Two scope query vectors differ')
        queries[qid] = matrix[0]
    if len(questions) != 353 or len(tasks) != 706: raise ValueError('All353 original questions and both scopes required')
    deps.update(queries=queries, questions=questions, prepared=tasks, canonical=finish.canonical_rows(deps, questions),
        read_anchor=checked, full_contract=contract, full_source_refs=[record(full_root / name)
            for name in ['input_contract.json', 'private_prepared_tasks.json']], k12_pending=None)
    try: initialize_k12(deps, canonical)
    except (PendingCandidates, FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        deps['k12_pending'] = 'K12_initialization_pending:' + type(exc).__name__
        deps['k12_initialization_error'] = dict(error_type=type(exc).__name__,
            message_sha256=hashlib.sha256(str(exc).encode('utf-8')).hexdigest(),
            missing_path=getattr(exc, 'filename', None))
    return deps


def prepare(full_root, output, deps=None):
    full_root = Path(full_root).resolve(); output = Path(output).resolve()
    if output == full_root or output in full_root.parents or full_root in output.parents:
        raise ValueError('Shared-control output must be separate from old full-comparison outputs')
    deps = deps or initialize(full_root)
    owner = dict(worker=record(__file__), full_sources=deps['full_source_refs'], policy=POLICY)
    if output.exists() and not (output / 'owner.json').exists():
        raise ValueError('Existing directory is not owned by this new candidate preparation')
    immutable(output / 'owner.json', owner)
    counts = Counter(); pending = []; artifacts = []; packets = []; ready_total = 0
    for method in METHODS:
        ready = []
        for qid, q in deps['questions'].items():
            for scope in SCOPES:
                try: task = prepare_case(method, q, deps['prepared'][scope, qid], deps)
                except PendingCandidates as exc:
                    item = dict(method=method, scope=scope, question_id=qid, status='pending_actual_candidates',
                        reason=str(exc), baseline_API_calls=0, candidate_substitution=False)
                    pending.append(item); counts[method + ':pending'] += 1
                    immutable(output / 'pending' / method / scope / (digest(qid) + '.json'), item)
                    continue
                path = output / 'tasks' / method / scope / (digest(qid) + '.json')
                immutable(path, task); artifacts.append(record(path)); ready.append(task)
                counts[method + ':ready'] += 1
            if method == 'K12-KGraph': deps.get('k12_cache', {}).clear()
        if ready:
            packet = score_packet(ready, deps['build_score_packet']); path = output / 'score_packets' / (method + '.json')
            immutable(path, packet); packets.append(record(path)); ready_total += len(ready)
        del ready
    result = dict(version=POLICY['version'], status='all3530_shared_candidates_prepared_NO_API'
        if ready_total == 3530 else 'shared_candidates_partially_ready_explicit_pending_NO_API',
        expected=3530, ready=ready_total, pending=len(pending), by_method=dict(counts),
        canonical_SHA=CANONICAL_SHA, owner=record(output / 'owner.json'), task_records=artifacts,
        score_packets=packets, pending_records=pending, K12_source_proofs=deps.get('k12_proofs', []),
        K12_initialization_error=deps.get('k12_initialization_error'),
        baseline_answer_API_calls=0, answer_generation=False, GPU_inference_started=False,
        old_outputs_modified=False, graph_SOTA_claim=False)
    immutable(output / 'preparation_manifest.json', result)
    return {k: v for k, v in result.items() if k not in {'task_records', 'pending_records'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    with zero_api_guard(): result = prepare(args.full_root, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
