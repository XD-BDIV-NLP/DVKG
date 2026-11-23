"""六书联合检索的零API执行器；缺失查询或原始核验批次时先记录需求再停止。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time


VERSION = 'joint_pool_six_book_paragraph_retrieval_v2'
METHODS = ('bm25', 'bge', 'hybrid', 'dvkg')


def paragraph_keys(structure, book, evidence_ids):
    """评估只按原始段落身份计数；多个切片不重复计分。"""
    result = set()
    for eid in evidence_ids:
        if eid not in structure['evidence']:
            raise ValueError('Selected evidence is outside its declared original textbook')
        row = structure['evidence'][eid]
        source = row.get('source', {}).get('source_id')
        if not isinstance(source, str) or not source:
            raise ValueError('Every evidence slice requires its original source_id')
        result.add((book, source))
    return result


def source_audit(structure, book):
    identities = paragraph_keys(structure, book, structure['evidence'])
    n = len(structure['evidence'])
    return dict(evidence_rows=n, original_paragraphs=len(identities),
                repeated_source_rows=n-len(identities))


def accept_original_batch(dispatcher, batch, reference):
    ordinal = batch['original_batch_ordinal']
    if (type(ordinal) is not int or not 0 <= ordinal < len(dispatcher.batch_ids)
            or batch['full_original_plan'] != dispatcher.plan_record
            or batch['original_batch_id'] != dispatcher.batch_ids[ordinal]):
        raise ValueError('Cached mapping batch differs from the exact original plan')
    expected = {(cid, c['evidence_id']) for cid, c in dispatcher.flat[ordinal*8:ordinal*8+8]}
    actual = {(r['concept_id'], r['evidence_id']) for r in batch['records']}
    if expected != actual or len(batch['records']) != len(expected):
        raise ValueError('Cached mapping decisions do not cover the original batch pairs')
    if ordinal in dispatcher.loaded and dispatcher.loaded[ordinal] != batch['records']:
        raise ValueError('Conflicting saved original mapping decisions')
    dispatcher.loaded[ordinal] = batch['records']
    dispatcher.proofs[ordinal] = reference


def mapping_preflight(sources, required):
    reports = []
    for source in sources:
        dispatcher = source['dispatcher']
        concepts = sorted(required.get(source['graph_id'], set()))
        batches = dispatcher.required_batches(concepts)
        missing = [i for i in batches if i not in dispatcher.loaded]
        reports.append(dict(book_id=source['book_id'], graph_id=source['graph_id'],
                            required_concept_ids=concepts, full_original_plan=dispatcher.plan_record,
                            required_original_batch_count=len(batches),
                            saved_original_batch_count=len(dispatcher.loaded),
                            missing_original_batch_ordinals=missing,
                            missing_original_batch_ids=[dispatcher.batch_ids[i] for i in missing]))
    return dict(status='mapping_dependencies_missing' if any(r['missing_original_batch_ordinals'] for r in reports)
                else 'all_required_original_mapping_batches_available',
                books=reports, missing_batch_count=sum(len(r['missing_original_batch_ordinals']) for r in reports),
                model_calls=0, missing_mapping_policy='stop_before_retrieval_no_silent_partial_mapping')


def replay_one_book(engine, book, questions, queries, saved_selected):
    from course_experiments.evaluation.joint_pool_rag import JointPoolRetriever
    source = [dict(course_id='replay', book_id=book, engine=engine)]
    pool = JointPoolRetriever(source, paragraph_dedup=False)
    adapted = JointPoolRetriever(source, paragraph_dedup=True)
    changed = []; representatives = []
    for question in questions:
        qid = question['question_id']; vector = queries[qid]
        original, _ = engine.retrieve(question['question'], book, vector)
        rows, _ = pool.retrieve(question['question'], vector, 'dvkg')
        native = [r['evidence_ids'][0] for r in original]
        if native != saved_selected(question) or native != [r['source_evidence_id'] for r in rows]:
            raise ValueError('Single-book pooled retrieval differs from saved Full or the frozen retriever')
        unique_rows, _ = adapted.retrieve(question['question'], vector, 'dvkg')
        ids = [r['source_evidence_id'] for r in unique_rows]
        if ids != native:
            changed.append(qid)
        representatives.append(dict(question_id=qid, raw_selected_native_ids=native,
            paragraph_adapted_selected_native_ids=ids,
            paragraph_sources=[r['original_source_id'] for r in unique_rows]))
    return dict(matched_question_count=len(questions), selected_native_ids_equal=True,
                paragraph_adapted_changed_question_count=len(changed), changed_question_ids=changed,
                paragraph_adapter_records=representatives,
                paragraph_adapted_selection_claimed_native_equal=False,
                packed_context_identity_claimed_equal=False, model_calls=0)


def retrieve_and_score(pool, structures, question, vector, method):
    started = time.monotonic()
    rows, trace = pool.retrieve(question['question'], vector, method)
    context = pool.pack_context(rows, question['question'])
    chosen = {pair for row in rows for pair in paragraph_keys(
        structures[row['book_id']], row['book_id'], [row['source_evidence_id']])}
    packed = set()
    for row in context:
        for uid in row['evidence_ids']:
            book, eid = pool.native_sources[uid]
            packed.update(paragraph_keys(structures[book], book, [eid]))
    target = question['target_book_id']; course = question['evaluation_course_id']
    gold = paragraph_keys(structures[target], target, question['evidence_ids'])
    if not gold:
        raise ValueError('Original question has no paragraph-level gold evidence')
    return dict(selected_native_sources=trace['selected_source_keys'],
        selected_paragraph_sources=sorted(chosen), packed_paragraph_sources=sorted(packed),
        context=context, context_bytes=len(json.dumps(context, ensure_ascii=False).encode('utf-8')),
        trace=trace, retrieval_seconds=time.monotonic()-started, model_calls=0,
        metrics=dict(paragraph_recall_at5=len(gold & chosen)/len(gold),
                     packed_paragraph_recall=len(gold & packed)/len(gold),
                     selected_foreign_book_paragraphs=sum(b != target for b, _ in chosen),
                     selected_foreign_course_paragraphs=sum(pool.book_courses[b] != course for b, _ in chosen)))


def _read(path):
    return json.loads(Path(path).read_bytes())


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _checked(reference):
    path = Path(reference['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Pinned experiment dependency changed')
    return _read(path)


def _load_mapping_proofs(m, source, dispatcher):
    references = {}
    for manifest in source.get('mapping_manifests', []):
        for reference in m.asset(manifest)['proofs']:
            references[reference['path']] = reference
    for root in source.get('alignment_roots', []):
        for path in Path(root).rglob('batch_result.json'):
            references[str(path)] = m.file_record(path)
    for reference in references.values():
        accept_original_batch(dispatcher, m.asset(reference), reference)


def execute(configuration, configuration_ref, *, preflight_only=False):
    if configuration.get('version') != VERSION:
        raise ValueError('Explicit six-book joint retrieval configuration required')
    original = _checked(configuration['original_contract'])
    sys.path.insert(0, str(Path(original['code']['run_echa.py']['path']).parents[2]))
    from course_experiments.evaluation import run_echa as m
    for reference in original['code'].values():
        if m.file_record(reference['path']) != reference:
            raise ValueError('Original execution code changed')
    freeze = m.asset(configuration['algorithm_freeze'])
    if freeze['identity'] != m.digest({k: v for k, v in freeze.items() if k != 'identity'}):
        raise ValueError('Frozen structured retrieval identity changed')
    for name, reference in [('base', freeze['base']), ('focus', freeze['focus'])]:
        if m.file_record(reference['path']) != reference:
            raise ValueError('Frozen structured retrieval module changed')
        _load(reference['path'], 'course_experiments.evaluation.echa_structured_' + ('fast' if name == 'base' else 'focus'))
    from course_experiments.evaluation.echa_structured_focus import StructuredFocusRetriever
    if m.file_record(configuration['pool_module']['path']) != configuration['pool_module']:
        raise ValueError('Joint pool implementation changed')
    joint = _load(configuration['pool_module']['path'], 'course_experiments.evaluation.joint_pool_rag')
    plan = m.asset(configuration['deferred_plan'])
    if plan['plan_identity'] != m.digest({k: v for k, v in plan.items() if k != 'plan_identity'}):
        raise ValueError('Original deferred mapping plan changed')
    ctx = m.load_context(original['inventory']['path'], original['inventory']['sha256'],
                         original['tokenizer']['path'], original['tokenizer']['sha256'])
    if ctx['identity'] != plan['context_identity']:
        raise ValueError('Frozen context and deferred mapping plan differ')
    m._recheck(ctx, deep=True)
    specs = configuration['sources']; books = [s['book_id'] for s in specs]
    if (len(books) != 6 or len(set(books)) != 6 or len({s['course_id'] for s in specs}) != 3
            or any(sum(s['course_id'] == course for s in specs) != 2 for course in {s['course_id'] for s in specs})):
        raise ValueError('This stage requires three courses with exactly two books each')
    root = Path(configuration['output_root']); root.mkdir(parents=True, exist_ok=True)
    identity = m.digest(dict(configuration=configuration_ref, worker=m.file_record(__file__), policy=joint.POLICY))
    m.write_immutable(root/'execution_contract.json', dict(identity=identity, version=VERSION,
        configuration=configuration_ref, worker=m.file_record(__file__), policy=joint.POLICY,
        context_identity=ctx['identity'], model_calls=0, paid_generation_enabled=False,
        paragraph_dedup=True, paragraph_adapter=joint.POLICY['paragraph_adapter'],
        single_book_raw_replayed_before_adapter=True, vector_aggregation=False,
        all_original_questions=True, target_book_used_only_for_scoring=True,
        cross_source_necessity_claim=False, new_blind_test_claim=False))
    sources = []; questions = []; queries = {}; query_refs = {}; missing_queries = []; audits = {}; seen_qids = set()
    for spec in specs:
        book = spec['book_id']; key = spec['state_key']; entry = ctx['entries'][key]
        if (entry['state'] != 'F_corrected' or set(entry['active_books']) !=
                {s['book_id'] for s in specs if s['course_id'] == spec['course_id']}):
            raise ValueError('Course graph must use exactly its declared two-book frozen release')
        dataset = m._dataset_questions(spec['dataset'], book)
        if not dataset or len(dataset) != spec['expected_question_count']:
            raise ValueError('All-original-question denominator differs')
        audits[book] = source_audit(ctx['structures'][book], book)
        dispatcher = m.ExactDeferredAlignment(ctx, key, book, plan['full_plans'][key][book],
                                             root/'preparation'/book/'alignment')
        _load_mapping_proofs(m, spec, dispatcher)
        qualified = [row for i in sorted(dispatcher.loaded) for row in dispatcher.loaded[i]]
        engine = StructuredFocusRetriever(entry['graph_data'], {book: ctx['structures'][book]}, entry['mappings'],
            {book: ctx['units'][book]}, entry['vector_plan']['concept_ids'], entry['vectors'],
            {book: qualified}, ctx['paragraph_tokens'])
        sources.append(dict(spec, engine=engine, dispatcher=dispatcher, graph_id=engine.graph_id))
        for question in dataset:
            if question['question_id'] in seen_qids:
                raise ValueError('Question IDs must be unique across the declared original datasets')
            seen_qids.add(question['question_id'])
            cached = m._existing_query(ctx, question, configuration['model_cache_root'], root/'query_cache')
            if cached is None:
                missing_queries.append(dict(book_id=book, question_id=question['question_id'],
                                            question_digest=m.digest(question['question'])))
            else:
                queries[question['question_id']] = cached[0]; query_refs[question['question_id']] = cached[1]
            questions.append(dict(question, evaluation_course_id=spec['course_id']))
    m.write_immutable(root/'source_audit.json', dict(books=audits, slice_ids_used_as_paragraphs=False))
    if missing_queries:
        m.write_immutable(root/'missing_query_vectors.json', dict(status='cached_query_dependencies_missing',
            missing=missing_queries, all_question_count=len(questions), model_calls=0))
        return dict(status='cached_query_dependencies_missing', missing_count=len(missing_queries), model_calls=0)
    # 只对最终排序做共同段落适配；原始切片、引用、向量和原方法均保持冻结。
    pools = {course: joint.JointPoolRetriever([s for s in sources if s['course_id'] == course], paragraph_dedup=True)
             for course in sorted({s['course_id'] for s in sources})}
    pools['mixed_six_books'] = joint.JointPoolRetriever(sources, paragraph_dedup=True)
    required = defaultdict(set)
    for question in questions:
        vector = queries[question['question_id']]
        for name in [question['evaluation_course_id'], 'mixed_six_books']:
            for graph, concepts in pools[name].query_concepts(question['question'], vector).items():
                required[graph].update(concepts)
        source = next(s for s in sources if s['book_id'] == question['target_book_id'])
        seeds, neighbors, _ = source['engine']._concepts(question['question'], vector)
        required[source['graph_id']].update(seeds+neighbors)
    preparation = mapping_preflight(sources, required)
    preparation.update(question_count=len(questions), query_vector_records=query_refs,
        mapping_proofs={s['book_id']: list(s['dispatcher'].proofs.values()) for s in sources},
        frozen_source_records=ctx['contract']['source_records'])
    m.write_immutable(root/'mapping_preflight.json', preparation)
    if preparation['missing_batch_count'] or preflight_only:
        return dict(status=preparation['status'], missing_batch_count=preparation['missing_batch_count'],
                    question_count=len(questions), preflight_only=preflight_only, model_calls=0)
    replay = {}
    for source in sources:
        book = source['book_id']; saved_refs = []
        def saved(question):
            path = Path(source['full_answers_root'])/'questions'/m.digest(question['question_id'])/'retrieval.json'
            value = _read(path)
            if value.get('checkpoint_digest') != m.digest({k: v for k, v in value.items() if k != 'checkpoint_digest'}):
                raise ValueError('Saved Full retrieval checkpoint changed')
            if source.get('full_result_id') and value['result_id'] != source['full_result_id']:
                raise ValueError('Saved Full retrieval result identity differs')
            saved_refs.append(m.file_record(path)); return value['selected_ids']
        replay[book] = dict(replay_one_book(source['engine'], book,
            [q for q in questions if q['target_book_id'] == book], queries, saved), saved_full_records=saved_refs)
    m.write_immutable(root/'single_book_replay.json', replay)
    stats = []
    for scope in ['same_subject_pool', 'mixed_six_book_pool']:
        for method in METHODS:
            for question in questions:
                qid = question['question_id']; target = question['target_book_id']
                pool_name = question['evaluation_course_id'] if scope == 'same_subject_pool' else 'mixed_six_books'
                pool = pools[pool_name]
                scored = retrieve_and_score(pool, ctx['structures'], question, queries[qid], method)
                record = dict(result_id=identity, scope=scope, method=method, question_id=qid,
                    question=question['question'], target_book_for_scoring=target,
                    course_for_scoring=question['evaluation_course_id'], dataset=next(s['dataset'] for s in sources if s['book_id']==target),
                    allowed_books=pool.allowed_books, **scored)
                m.frozen.checkpoint(root/'retrieval'/scope/method/m.digest(qid)/'retrieval.json', record)
                stats.append(dict(scope=scope, method=method, target_book_id=target,
                    course_id=question['evaluation_course_id'], question_id=qid,
                    retrieval_seconds=record['retrieval_seconds'], context_bytes=record['context_bytes'], **record['metrics']))
    summaries = {}
    for scope in ['same_subject_pool', 'mixed_six_book_pool']:
        for method in METHODS:
            values = [r for r in stats if r['scope'] == scope and r['method'] == method]
            by_book = {b: [r for r in values if r['target_book_id'] == b] for b in books}
            fields = ['paragraph_recall_at5', 'packed_paragraph_recall', 'selected_foreign_book_paragraphs',
                      'selected_foreign_course_paragraphs', 'context_bytes']
            summaries[scope+'/'+method] = dict(question_count=len(values),
                question_micro={key: statistics.mean(r[key] for r in values) for key in fields},
                equal_book_macro={key: statistics.mean(statistics.mean(r[key] for r in rows) for rows in by_book.values()) for key in fields},
                by_course={c: dict(question_count=sum(len(rows) for b, rows in by_book.items() if next(s['course_id'] for s in sources if s['book_id']==b)==c),
                    equal_book_macro={key: statistics.mean(statistics.mean(r[key] for r in rows) for b, rows in by_book.items()
                        if next(s['course_id'] for s in sources if s['book_id']==b)==c) for key in fields}) for c in pools if c!='mixed_six_books'},
                median_retrieval_seconds=statistics.median(r['retrieval_seconds'] for r in values))
    result = dict(status='all_original_questions_retrieval_complete_NO_answers_generated',
        result_id=identity, original_question_count=len(questions), retrieval_task_count=len(stats),
        methods=list(METHODS), pool_scopes=['same_subject_pool','mixed_six_book_pool'],
        paragraph_dedup=True, paragraph_adapter=joint.POLICY['paragraph_adapter'],
        book_count=6, course_count=3, model_calls=0, answer_generation_completed=False,
        cross_source_necessity_claim=False, rows=summaries)
    m.write_immutable(root/'retrieval_summary.json', result)
    m.write_immutable(root/'retrieval_metrics.json', dict(records=stats, model_calls=0))
    m._recheck(ctx, deep=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--configuration-sha', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    reference = dict(path=str(Path(args.configuration).resolve()), sha256=args.configuration_sha)
    result = execute(_checked(reference), reference, preflight_only=args.preflight_only)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))
    return 0 if result.get('missing_batch_count', 0) == 0 and result['status'] != 'cached_query_dependencies_missing' else 2


if __name__ == '__main__':
    raise SystemExit(main())
