"""完整双书原文的冷建库；预检不创建模型客户端。"""
from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback

BOOKS = ('book_25f718562685', 'book_49bb8d31ddb9')
METHODS = ('dvkg', 'graphrag', 'lightrag', 'hipporag')
CHUNKS_PER_BOOK = None
PARALLEL_ECHA_FILE = Path(__file__).with_name('parallel_echa.py')
HTTP_CAP = 50000
NATIVE_PREFIX = 'native_cost_full_20261030/'
DVKG_PREFIX = 'echa_v1/full_cost_20261030/'
PINS = dict(graphrag='56a865bff0cac9fbbc24f343cf52d0a1850abba6',
            lightrag='aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a',
            hipporag='d5c8329422e0a0b834a15874545cb6a74b4f9b26')
SOURCE_SHA = dict(book_25f718562685='0fec2f25bf6824a0b4fc2769215a80369007f67c28bfd6328294362fbe4bf792',
                  book_49bb8d31ddb9='83ff926b71772b72de57301273eb17e46fc634bdd36dcc09f6d2385c961fe3d0')
POLICY = dict(version='cold_postparse_full_two_book_index_parallel4_20261030_v2',
    selection='all_original_chunks_and_paragraphs_no_question_or_score_access',
    selected_unit_text='complete_original_evidence_text_no_summary_or_truncation',
    boundary='complete_original_source_context_retained',
    common_input='same_ordered_source_paragraphs_grouped_by_original_chunk_for_every_method',
    start='parsed_text_corpus_available; excludes_PDF_OCR_common_upstream_work',
    stop='fresh_method_native_graph_and_all_required_index_artifacts_ready; no_QA_or_judge',
    cold='new_empty_index_and_runtime_per_method; isolated_provider_cache_namespace; intra_method_dedup_retained',
    comparison='complete_two_book_parsed_sources; one_cold_run_per_method; no_repeat_variance_claim',
    resource_limit='four shared request slots across methods; ECHA four unchanged original eight-pair batches in flight',
    dvkg='fresh_prior_60_100 + source_construction + semantic_fusion + structures + fresh_vectors + exhaustive_ECHA + materialization',
    dvkg_difference='fresh_pipeline_not_frozen_paper_corrected_graph; no_historical_human_or_model_review_replayed',
    budget='50000_actual_HTTP_attempts_across_all_cost_methods;native_capped_at_20000_DVKG_at_30000; inherits_global_and_native_or_ECHA_allocation; no_reset')


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_record(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(ref):
    if file_record(ref['path']) != ref:
        raise ValueError('Frozen input changed: ' + ref['path'])
    path = Path(ref['path'])
    if path.suffix.lower() == '.json':
        return read(path)
    import yaml
    return yaml.safe_load(path.read_text('utf-8-sig'))


def seal(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Refusing to overwrite a frozen cost artifact: ' + str(path))
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)
    return value


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_prefix(corpus, book):
    if len(corpus['books']) != 1 or corpus['books'][0]['id'] != book:
        raise ValueError('Expected exactly the declared original source book')
    if len({e['id'] for e in corpus['evidence']}) != len(corpus['evidence']):
        raise ValueError('Repeated full-source paragraph identity')
    if any(e['book_id'] != book or not isinstance(e['text'],str) or not e['text'].strip()
           or type(e.get('paragraph_order')) is not int for e in corpus['evidence']):
        raise ValueError('Complete ordered original paragraphs required')
    ids={e['id'] for e in corpus['evidence']}
    if {eid for c in corpus['chunks'] for eid in c['evidence_ids']} != ids:
        raise ValueError('Full corpus chunk-to-paragraph coverage differs')
    result=copy.deepcopy(corpus)
    result['construction_scope']='full'
    result['source_corpus_fingerprint']=corpus.get('fingerprint')
    result['cost_selection_audit']=dict(rule='all_original_chunks_and_paragraphs_no_source_prefix_cutoff',
        selected_original_positions=list(range(len(corpus['chunks']))),
        selected_chunk_ids=[c['id'] for c in corpus['chunks']],question_or_method_score_access=False)
    result['fingerprint']=digest({k:v for k,v in result.items() if k!='fingerprint'})
    return result, []



def common_workload(corpus, book):
    from course_experiments.dual_view import build_structure
    from course_experiments.evaluation.native_corpus import native_passages
    selected, clipped = select_prefix(corpus, book)
    structure = build_structure(selected)
    rows = [dict(book_id=book, chunk_id=row['chunk_id'], text=row['text'], evidence_ids=[eid])
            for eid, row in sorted(structure['evidence'].items(),
                key=lambda pair: (pair[1]['source']['paragraph_order'], pair[0]))]
    passages, audit = native_passages(rows, structure, book)
    order = {c['id']: i for i, c in enumerate(selected['chunks'])}
    passages.sort(key=lambda row: order[row['chunk_id']])
    if len(passages) != len(selected['chunks']):
        raise ValueError('Full workload must preserve all original chunks')
    binding = [dict(chunk_id=p['chunk_id'], text_sha256=hashlib.sha256(p['text'].encode()).hexdigest(),
                    evidence_spans=p['evidence_spans']) for p in passages]
    return dict(book=book, corpus=selected, structure=structure, rows=rows, passages=passages,
        binding=binding, source_digest=digest(binding), native_restoration=audit,
        boundary_clipping=clipped, selection_audit=selected['cost_selection_audit'])


def echa_admission(concept_count, paragraph_count, candidate_pairs=None):
    if candidate_pairs is not None:
        if not 0 <= candidate_pairs <= concept_count * paragraph_count:
            raise ValueError('Impossible ECHA candidate-pair count')
        base = math.ceil(candidate_pairs / 8)
        return dict(candidate_pairs=candidate_pairs, batch_size=8, logical_base_calls=base,
                    logical_with_format_repair_upper=2 * base, transport_retry_factor_upper=7,
                    actual_HTTP_hard_limit_external=True)
    return dict(candidate_pairs=None, logical_base_calls=None,
        reason='fresh_graph_required; concept_count_and_origin_anchors_are_not_known_before_construction',
        conditional_pair_upper_formula='fresh_concept_count * selected_paragraph_count',
        admission='after_graph_and_vectors_recompute_exact_candidate_pairs_before_any_ECHA_call')


def code_assets(repo):
    names = ['course_experiments/construction.py', 'course_experiments/textbook.py',
        'course_experiments/prior.py', 'course_experiments/fusion.py', 'course_experiments/views.py',
        'course_experiments/contributions.py', 'course_experiments/dual_view.py',
        'course_experiments/evaluation/native_corpus.py', 'course_experiments/evaluation/native_provider.py',
        'course_experiments/evaluation/native_microsoft_graphrag.py',
        'course_experiments/evaluation/native_microsoft_full.py',
        'course_experiments/evaluation/native_lightrag.py', 'course_experiments/evaluation/native_hipporag.py',
        'course_experiments/evaluation/run_native_full.py', 'course_experiments/evaluation/run_native_rag.py',
        'course_experiments/evaluation/echa_pipeline.py', 'course_experiments/evaluation/echa_rag.py',
        'course_experiments/evaluation/evidence_hierarchical_alignment.py',
        'course_experiments/evaluation/echa_structured_fast.py',
        'course_experiments/evaluation/echa_structured_focus.py',
        'course_experiments/manifest.py', 'course_experiments/splits.py',
        'course_experiments/evaluation/protocol.py', 'course_experiments/evaluation/run_repaired_full.py',
        'course_experiments/evaluation/repaired_full.py', 'course_experiments/evaluation/rag_repair_v3.py',
        'course_experiments/evaluation/rag_attribution.py', 'course_experiments/evaluation/rag.py',
        'course_experiments/evaluation/reference.py', 'course_experiments/evaluation/chat.py',
        'course_experiments/evaluation/dual_view_rag.py', 'course_experiments/evaluation/graph_rag_v2.py']
    paths = [repo / name for name in names]
    paths += list((repo / 'domain_ckg_workspace/ckg_incremental').glob('*.py'))
    paths += list((repo / 'domain_ckg_workspace/ckg_build').glob('*.py'))
    paths += list((repo / 'prompts/incremental_v5').glob('*.txt'))
    paths.extend([Path(__file__).resolve(),PARALLEL_ECHA_FILE])
    return [file_record(path) for path in sorted(set(paths))]


def token_forecast(workload):
    try:
        import tiktoken
    except ImportError:
        return dict(status='tokenizer_unavailable', provider_requests=0,
                    native_chunks_not_guessed=True)
    output = {}
    for model, size, overlap in [('graphrag', 600, 100), ('lightrag', 1200, 100)]:
        encoder = tiktoken.get_encoding('cl100k_base') if model == 'graphrag' else tiktoken.encoding_for_model('gpt-4o-mini')
        lengths = [len(encoder.encode(p['text'])) for p in workload['passages']]
        # 预估仅用于准入，最终使用原生实际切分与收据。
        count = sum(math.ceil(n / (size - overlap)) for n in lengths if n)
        output[model] = dict(source_tokens_by_document=lengths,
            predicted_native_chunks=count, chunk_size=size, overlap=overlap,
            extraction_base_call_prediction=count,
            gleaning_and_summaries_and_embeddings='data_dependent_not_a_total_bound',
            prediction_is_not_HTTP_upper_bound=True)
    output['hipporag'] = dict(original_passages=len(workload['passages']),
        openie_base_call_prediction=2 * len(workload['passages']),
        embeddings='chunk_entity_and_fact_counts_determined_after_fresh_OpenIE',
        prediction_is_not_HTTP_upper_bound=True)
    return dict(status='local_tokenization_only', methods=output, provider_requests=0)


def preflight(config, root):
    root = Path(root)
    repo = Path(config['repo']).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if set(config['corpora']) != set(BOOKS):
        raise ValueError('Exactly the two frozen CO sources are required')
    summaries = {}
    for book in BOOKS:
        if config['corpora'][book]['sha256'] != SOURCE_SHA[book]:
            raise ValueError('Corpus SHA differs from the fixed original CO corpus')
        workload = common_workload(checked(config['corpora'][book]), book)
        seal(root / 'frozen' / book / 'workload.json', workload)
        summaries[book] = dict(source= config['corpora'][book], source_digest=workload['source_digest'],
            chunks=len(workload['passages']), paragraphs=len(workload['rows']),
            characters=sum(len(p['text']) for p in workload['passages']),
            utf8_bytes=sum(len(p['text'].encode()) for p in workload['passages']),
            selected_chunk_ids=[p['chunk_id'] for p in workload['passages']],
            selection_audit=workload['selection_audit'],
            selected_source_metadata=[dict(chunk_id=c['id'], chapter_id=c.get('chapter_id'),
                title_path=c.get('title_path'), section_id=c.get('section_id'),
                source_type=c.get('type', c.get('content_type')),
                paragraphs=[dict(evidence_id=eid,
                    type=workload['structure']['evidence'][eid]['source'].get('type',
                         workload['structure']['evidence'][eid]['source'].get('content_type')),
                    title_path=workload['structure']['evidence'][eid]['source'].get('title_path'),
                    chapter_id=workload['structure']['evidence'][eid]['chapter_id'],
                    characters=len(workload['structure']['evidence'][eid]['text']))
                    for eid in c['evidence_ids']]) for c in workload['corpus']['chunks']],
            selected_evidence_ids_digest=digest([r['evidence_ids'][0] for r in workload['rows']]),
            boundary_clipping=workload['boundary_clipping'], forecast=token_forecast(workload),
            echa=echa_admission(None, len(workload['rows'])))
    dependencies = {}
    for method in METHODS[1:]:
        source = config.get('native_sources', {}).get(method)
        path = Path(source) if source else None
        if path is None or not path.is_dir() or not (path / '.git').exists():
            dependencies[method] = dict(status='checkout_missing', required_commit=PINS[method])
            continue
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        status = ['status', '--porcelain', '--untracked-files=all', '--', 'src/hipporag', 'pyproject.toml'] if method == 'hipporag' else ['status', '--porcelain', '--untracked-files=no']
        dirty = subprocess.check_output(['git', '-C', str(path), *status], text=True).strip()
        dependencies[method] = dict(status='ready' if head == PINS[method] and not dirty else 'pin_or_cleanliness_mismatch',
            checkout=str(path), commit=head, required_commit=PINS[method], clean=not bool(dirty))
    assets = code_assets(repo)
    optional = {}
    for key in ['construction_manifest', 'tokenizer_manifest']:
        if config.get(key):
            value = checked(config[key]); assets.append(config[key]); optional[key] = 'sha_verified'
            if key == 'construction_manifest':
                for name in ('runtime_config', 'schema'):
                    checked(value['files']['assets'][name])
                    assets.append(value['files']['assets'][name])
                from course_experiments.construction import check_config
                check_config(checked(value['files']['assets']['runtime_config']))
            else:
                for ref in value['files'].values():
                    if file_record(ref['path']) != {k: ref[k] for k in ('path', 'sha256')}:
                        raise ValueError('Frozen tokenizer resource changed')
                    assets.append({k: ref[k] for k in ('path', 'sha256')})
        else:
            optional[key] = 'missing_required_for_dvkg_execution'
    contract = dict(policy=POLICY, sources=summaries, native_dependencies=dependencies,
        dvkg_dependencies=optional, code=assets, methods=list(METHODS),
        execution_API_enabled=False, actual_model_calls=0, cost_HTTP_cap=HTTP_CAP,
        same_input_per_method={m: {b: summaries[b]['source_digest'] for b in BOOKS} for m in METHODS},
        unforecastable_costs='semantic_identity_fusion_and_native_communities_summaries_repairs_embeddings_are_data_dependent',
        total_admission='hard_cumulative_50000_actual_HTTP_attempts;native20000_DVKG30000; exact_ECHA_second_gate; stop_and_preserve_incomplete_if_exhausted')
    contract['identity'] = digest(contract)
    seal(root / 'preflight.json', contract)
    return contract


def verify_preflight(config, root):
    frozen = read(Path(root) / 'preflight.json')
    if frozen['identity'] != digest({k: v for k, v in frozen.items() if k != 'identity'}):
        raise ValueError('Cost preflight identity changed')
    for ref in frozen['code']:
        if file_record(ref['path']) != ref:
            raise ValueError('Cost implementation changed after input seal')
    for book in BOOKS:
        checked(config['corpora'][book])
        workload = read(Path(root) / 'frozen' / book / 'workload.json')
        if workload['source_digest'] != frozen['sources'][book]['source_digest'] or digest(workload['binding']) != workload['source_digest']:
            raise ValueError('Frozen common source binding changed')
    return frozen


def isolated_config(config, namespace):
    result = copy.deepcopy(config)
    if '_measurement_cache_namespace' in result:
        raise ValueError('Input model runtime already has measurement namespace')
    result['_measurement_cache_namespace'] = namespace
    restored = copy.deepcopy(result)
    restored.pop('_measurement_cache_namespace')
    if restored != config:
        raise AssertionError('Cache isolation changed model or HTTP parameters')
    return result


def provider_for(config, root, method, frozen):
    from course_experiments.evaluation import native_provider
    from course_experiments.evaluation.protocol import runtime_for
    from course_experiments.evaluation.run_native_full import LIMITS
    from course_experiments.evaluation.run_repaired_full import credentials
    factory = config['provider_factory']
    if file_record(factory['path']) != factory:
        raise ValueError('Authorized provider factory changed')
    checked(config['provider_authorization'])
    ops = load_module(factory['path'], 'cost_authorized_provider_' + method)
    Provider = ops.provider_type(native_provider, config['provider_authorization'])
    construction = checked(config['construction_manifest'])
    transport = Path(root) / 'methods' / method / 'transport'
    if transport.exists():
        raise ValueError('Cold measurement refuses an existing transport/runtime directory')
    credentials()
    runtime = runtime_for(construction, transport)
    runtime.config = isolated_config(runtime.config, frozen['identity'] + '/' + method)
    return Provider(runtime, Path(config['parent_run_root']) / 'formal_provider', **LIMITS)


def cost_scope(config, root, provider, frozen, method):
    from course_experiments.evaluation.run_native_full import ReceiptTracker
    from filelock import FileLock
    checked(config['native_budget_authorization'])
    root = Path(root)
    stack = ExitStack()
    original = provider.runtime._record
    prefix = (DVKG_PREFIX if method == 'dvkg' else NATIVE_PREFIX) + frozen['identity'] + '/' + method + '/'
    lock = FileLock(str(provider.root / 'cold_cost_full_20261030_budget.lock'), timeout=30)
    seal(root / 'cost_budget.json', dict(cap=HTTP_CAP, stage_prefixes=[NATIVE_PREFIX, DVKG_PREFIX],
        global_budget_reset=False, native_allocation_reset=False, ECHA_allocation_reset=False,
        actual_HTTP_transport_retries_counted=True, authorization=config['native_budget_authorization']))
    def bounded(key, stage, status, value):
        scope = stage if stage.startswith(prefix) else prefix + stage
        if status == 'request_started':
            with lock:
                if cost_http_count(provider) >= HTTP_CAP:
                    raise RuntimeError('Fixed combined 50000-HTTP full-cost budget exhausted')
                family_prefix=DVKG_PREFIX if method=='dvkg' else NATIVE_PREFIX
                with sqlite3.connect(provider.database,timeout=30) as db:
                    used=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,?)=?",(len(family_prefix),family_prefix)).fetchone()[0]
                if used>=(30000 if method=='dvkg' else 20000):
                    raise RuntimeError('Full-cost family cap exhausted; preserve partial artifacts')
                return original(key, scope, status, value)
        return original(key, scope, status, value)
    provider.runtime._record = bounded
    stack.callback(setattr, provider.runtime, '_record', original)
    tracker = ReceiptTracker(provider)
    stack.callback(tracker.close)
    stack.enter_context(tracker.scope(root / 'methods' / method / 'requests.jsonl'))
    return stack


def cost_http_count(provider, method=None, identity=None):
    prefixes = [NATIVE_PREFIX, DVKG_PREFIX]
    if method is not None:
        prefixes = [(DVKG_PREFIX if method == 'dvkg' else NATIVE_PREFIX) + identity + '/' + method + '/']
    with sqlite3.connect(provider.database, timeout=30) as db:
        return sum(db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,?)=?",
            (len(prefix), prefix)).fetchone()[0] for prefix in prefixes)


def request_summary(path):
    rows = [json.loads(line) for line in Path(path).read_text('utf8').splitlines() if line.strip()] if Path(path).exists() else []
    completed = [r for r in rows if r['status'] == 'completed']
    usage = {}
    for row in completed:
        if not row['cache_hit']:
            for key, value in (row.get('usage') or {}).items():
                if type(value) in {int, float}:
                    usage[key] = usage.get(key, 0) + value
    stages = {}
    for row in rows:
        counts = stages.setdefault(row['stage'], dict(logical_calls=0, cache_hits=0, failed_calls=0))
        counts['logical_calls'] += 1
        counts['cache_hits'] += int(row.get('cache_hit', False))
        counts['failed_calls'] += int(row['status'] != 'completed')
    return dict(logical_calls=len(rows), within_method_cache_hits=sum(r['cache_hit'] for r in completed),
        completed_new_requests=sum(not r['cache_hit'] for r in completed),
        failed_calls=len(rows)-len(completed), usage_new_requests=usage, stages=stages,
        HTTP_retry_counts='use_shared_provider_request_started_events_with_exact_cost_stage_prefix')


async def native_ready(config, root, method, provider):
    from course_experiments.evaluation.run_native_full import open_adapter, _index
    from course_experiments.evaluation.native_microsoft_full import open_microsoft
    output = {}
    for book in BOOKS:
        workload = read(Path(root) / 'frozen' / book / 'workload.json')
        plan = dict(method=method, source=Path(config['native_sources'][method]), book=book,
                    allowed_books=[book], passages=workload['passages'], structure=workload['structure'])
        folder = Path(root) / 'methods' / method / book
        if folder.exists():
            raise ValueError('Cold indexing refuses an existing per-book output directory')
        start = time.monotonic()
        adapter = None
        try:
            if method == 'graphrag':
                adapter = await open_microsoft(plan, folder, provider)
                receipt = adapter.index_receipt
            else:
                adapter = await open_adapter(plan, folder, provider)
                receipt = await _index(adapter, plan)
            provider.raise_if_fatal()
            output[book] = dict(status='ready_index', source_digest=workload['source_digest'],
                elapsed_seconds=time.monotonic()-start, receipt=receipt,
                scope='all_original_documents_in_complete_parsed_book')
            seal(folder / 'cost_ready.json', output[book])
        finally:
            if adapter is not None and hasattr(adapter, 'close'):
                value = adapter.close()
                if inspect.isawaitable(value):
                    await value
    return output


def dvkg_ready(config, root, provider):
    import numpy as np
    from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
    from course_experiments.construction import PROMPTS, check_config
    from course_experiments.contributions import export_contribution
    from course_experiments.textbook import TextbookEngine
    from course_experiments.prior import generate_prior
    from course_experiments.fusion import build_view
    from course_experiments.dual_view import build_mappings
    from course_experiments.evaluation.dual_view_rag import concept_text
    from course_experiments.evaluation.echa_pipeline import FrozenTokenizer, candidate_sets, verify_candidates as original_verify_candidates
    parallel=load_module(PARALLEL_ECHA_FILE,'full_cost_exact_batch_parallel')
    def verify_candidates(provider,graph,structure,plans,root,stage):
        return parallel.verify(original_verify_candidates,provider,graph,structure,plans,root,stage,workers=4)
    from course_experiments.evaluation.echa_structured_focus import StructuredFocusRetriever
    construction = checked(config['construction_manifest'])
    assets = construction['files']['assets']
    base_config = checked(assets['runtime_config'])
    check_config(base_config)
    schema = checked(assets['schema'])
    course = next(c for c in construction['courses']['courses'] if c['course_id'] == 'computer_organization')
    model_config = dict(base_config, domain=course['description'])
    folder = Path(root) / 'methods' / 'dvkg'
    if (folder / 'prior').exists():
        raise ValueError('Cold DVKG refuses an existing prior or graph runtime')
    times = {}
    class RoutedRuntime(ModelRuntime):
        def __init__(self, path, stage):
            super().__init__(path, PROMPTS, model_config)
            self.cost_stage = stage
        def _post(self, endpoint, body, options, key, stage):
            body = copy.deepcopy(body)
            if endpoint == '/chat/completions':
                body.setdefault('seed', 42)
            raw, _ = provider.request_executor(endpoint, body, self.cost_stage + '/' + stage)
            self._record(key, stage, 'response', raw)
            return raw
    def timed(stage, action):
        start = time.monotonic()
        wall_start = time.time()
        value = action()
        times[stage] = time.monotonic() - start
        seal(folder / 'timings' / (stage + '.json'),
             dict(stage=stage, elapsed_seconds=times[stage], started_unix=wall_start, finished_unix=time.time()))
        return value
    prior_runtime = RoutedRuntime(folder / 'prior/runtime', 'prior')
    try:
        prior = timed('prior', lambda: generate_prior('computer_organization', course['description'],
            schema, model_config, prior_runtime, folder / 'prior', digest(['cost_prior', digest(construction)])))
    finally:
        prior_runtime.close()
    contributions = []
    structures = {}; units = {}
    for book in BOOKS:
        workload = read(Path(root) / 'frozen' / book / 'workload.json')
        # 重新从同一原始文本路径生成结构，不能恢复旧结构图冒充构建。
        from course_experiments.dual_view import build_structure
        structures[book] = timed('structure_' + book, lambda: build_structure(workload['corpus']))
        runtime = RoutedRuntime(folder / book / 'runtime', 'textbook/' + book)
        engine = None
        try:
            engine = TextbookEngine(workload['corpus'], folder / book / 'construction', schema, model_config, runtime)
            timed('concept_' + book, engine.run)
            incomplete = [c['id'] for c in workload['corpus']['chunks']
                if (engine.store.chunk_state(c['id']) or {}).get('status') != 'complete']
            if incomplete:
                raise RuntimeError('Fresh sample extraction incomplete; not a ready index')
            contribution = export_contribution(engine.store.snapshot(), 'computer_organization', book,
                'smoke', digest(['cost_source', workload['source_digest']]))
            seal(folder / book / 'contribution.json', contribution)
            contributions.append(contribution)
        finally:
            if engine is not None:
                engine.store.close()
            runtime.close()
    runtime = RoutedRuntime(folder / 'fusion/runtime', 'semantic_fusion')
    try:
        fused = timed('semantic_fusion', lambda: build_view(prior, contributions, list(BOOKS), runtime=runtime))
    finally:
        runtime.close()
    graph = dict(entities=fused['entities'], relations=fused['relations'])
    if not graph['entities']:
        raise RuntimeError('Fresh extraction returned no semantic concepts')
    seal(folder / 'fused.json', fused)
    origins = build_mappings(graph, structures)
    ids = sorted(e['entity_id'] for e in graph['entities'])
    entities = {e['entity_id']: e for e in graph['entities']}
    vectors = timed('concept_vectors', lambda: provider.embed([concept_text(entities[cid]) for cid in ids], stage='concept_vectors'))
    qualified = {}; candidates = {}; token_counts = {}
    tokenizer = FrozenTokenizer(checked(config['tokenizer_manifest']))
    for book in BOOKS:
        workload = read(Path(root) / 'frozen' / book / 'workload.json')
        rows = workload['rows']
        pv = timed('paragraph_vectors_' + book, lambda: provider.embed([r['text'] for r in rows], stage='paragraph_vectors/' + book))
        units[book] = dict(rows=rows, vectors=pv)
        plans = list(candidate_sets(graph, structures[book], origins, ids, vectors, rows, pv))
        pairs = sum(len(p['candidates']) for p in plans)
        admission = echa_admission(len(ids), len(rows), pairs)
        seal(folder / book / 'echa_admission.json', admission)
        # 精确候选形成后第二次准入；不缩减候选来挤进额度。
        if admission['logical_with_format_repair_upper'] > config['cost_remaining_HTTP']():
            raise RuntimeError('Full ECHA repair allowance exceeds remaining cost budget; preserved as incomplete')
        qualified[book] = timed('ECHA_' + book, lambda: verify_candidates(provider, graph,
            structures[book], plans, folder / book / 'ECHA', 'ECHA/' + book))
        candidates[book] = [dict(concept_id=p['concept_id'], **c) for p in plans for c in p['candidates']]
        token_counts[book] = {eid: tokenizer.count(r['text']) for eid, r in structures[book]['evidence'].items()}
        np.save(folder / book / 'paragraph_vectors.npy', pv, allow_pickle=False)
    engine = StructuredFocusRetriever(graph, structures, origins, units, ids, vectors,
        qualified, token_counts, candidates=candidates)
    for book in BOOKS:
        timed('materialize_' + book, lambda: engine.materialize(book))
    np.save(folder / 'concept_vectors.npy', vectors, allow_pickle=False)
    seal(folder / 'ready_manifest.json', dict(graph=graph, structures=structures, origins=origins,
        concept_ids=ids, qualified=qualified, paragraph_tokens=token_counts, candidates=candidates,
        unit_rows={book: units[book]['rows'] for book in BOOKS},
        vector_paths={book: str(folder / book / 'paragraph_vectors.npy') for book in BOOKS},
        concept_vector_path=str(folder / 'concept_vectors.npy'), ready_books=list(BOOKS)))
    return dict(status='ready_index', stage_seconds=times, concepts=len(ids),
        semantic_relations=len(graph['relations']), all_concepts_all_source_paragraphs_aligned=True,
        mapping_counts={book: dict(candidates=len(qualified[book]),
            accepted=sum(r['status'] == 'accepted' for r in qualified[book]),
            rejected_or_unresolved=sum(r['status'] != 'accepted' for r in qualified[book])) for book in BOOKS},
        historical_graph_or_mapping_or_vector_reused=False, prior_charged_once_for_two_books=True,
        correction_review_cost_included=False, original_PDF_parsing_cost_included=False)


def execute(config, root, method):
    from filelock import FileLock
    root = Path(root)
    frozen = verify_preflight(config, root)
    if method not in METHODS:
        raise ValueError('Unknown cost method')
    if config.get('isolated_timing_confirmed') is not True:
        raise ValueError('Cost timing requires heavy CPU/indexing jobs to be stopped first')
    folder = root / 'methods' / method
    if (folder / 'result.json').exists():
        raise ValueError('Cold cost method has already been measured')
    if method != 'dvkg' and frozen['native_dependencies'][method]['status'] != 'ready':
        raise ValueError('Native source pin unavailable at preflight')
    provider = None
    environment_before = dict(unix=time.time(), load_average=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None)
    start = time.monotonic()
    with FileLock(str(root / 'cost_execution.lock'), timeout=0):
        try:
            provider = provider_for(config, root, method, frozen)
            before = provider.statistics()
            with cost_scope(config, root, provider, frozen, method):
                def remaining():
                    family_prefix=DVKG_PREFIX if method=='dvkg' else NATIVE_PREFIX
                    with sqlite3.connect(provider.database,timeout=30) as db:
                        used=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,?)=?",(len(family_prefix),family_prefix)).fetchone()[0]
                    return min(HTTP_CAP-cost_http_count(provider),(30000 if method=='dvkg' else 20000)-used)
                active = dict(config, cost_remaining_HTTP=remaining)
                result = dvkg_ready(active, root, provider) if method == 'dvkg' else asyncio.run(native_ready(active, root, method, provider))
                provider.raise_if_fatal()
            after = provider.statistics()
            summary = dict(status='completed_cold_postparse_full_book_index_measurement', method=method,
                identity=frozen['identity'], elapsed_seconds=time.monotonic()-start, result=result,
                before_shared_ledger=before, after_shared_ledger=after,
                requests=request_summary(folder / 'requests.jsonl'),
                per_method_actual_HTTP=cost_http_count(provider, method, frozen['identity']),
                shared_ledger_HTTP_delta=after['http_attempts']-before['http_attempts'],
                HTTP_delta_scope='per_method_actual_HTTP_is_exact_stage_count; shared_ledger_delta_may_include_other_tasks',
                cache_namespace=frozen['identity']+'/'+method,
                timing_environment=dict(heavy_CPU_jobs_stopped_confirmed=True, before=environment_before,
                    after=dict(unix=time.time(), load_average=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None)),
                source_digests={b: frozen['sources'][b]['source_digest'] for b in BOOKS}, policy=POLICY)
            seal(folder / 'result.json', summary)
            return summary
        except Exception as exc:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'error.log').write_text(traceback.format_exc(), encoding='utf8')
            seal(folder / 'incomplete.json', dict(status='incomplete_not_full_construction', method=method,
                 error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-start,
                 requests=request_summary(folder / 'requests.jsonl')))
            raise
        finally:
            if provider is not None:
                provider.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'execute'), default='preflight')
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    config = read(args.config)
    sys.path.insert(0, str(Path(config['repo']).resolve()))
    if args.mode == 'execute' and not args.method:
        parser.error('--method is required for an explicitly authorized execute call')
    value = preflight(config, args.root) if args.mode == 'preflight' else execute(config, args.root, args.method)
    print(json.dumps(dict(status=value.get('status', 'sealed_zero_API_preflight'), identity=value.get('identity'),
                         output_root=str(args.root), model_calls=0 if args.mode == 'preflight' else 'see_audited_receipts')))


if __name__ == '__main__':
    main()
