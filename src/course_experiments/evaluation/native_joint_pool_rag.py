"""真实原生来源候选的联合段落适配；原生回调由执行器绑定已完成索引。"""
from __future__ import annotations

import inspect
import hashlib
from pathlib import Path

from ..manifest import digest, file_record, read_document
from .native_source_paragraph_projection import project_sources


POLICY = dict(version='native_source_projected_joint_pool_v1',
    methods=['GraphRAG', 'LightRAG', 'HippoRAG'],
    source_scope='query_every_declared_book_index_with_identical_question_only',
    source_projection='validate_all_exact_native_source_positions_before_candidate_union',
    candidate_scope='only_original_evidence_slices_intersecting_actual_native_sources',
    candidate_expansion='native_fragments_project_to_full_frozen_original_evidence_slices',
    ranking='unchanged_joint_pool_global_BM25_dense_equal_RRF60_within_native_candidates',
    paragraph_dedup='highest_ranked_exact_slice_per_book_and_original_source_id',
    retrieval_top_k=5, context_bytes=3000,
    native_order='preserved_in_source_audit_not_claimed_as_final_paragraph_ranking',
    native_end_to_end=False, global_candidate_substitution=False,
    provider_scope='caller_binds_audited_context_only_callbacks_no_index_build_entrypoint',
    embedding_reuse='exact_native_input_body_and_runtime_config_only_no_query_rewriting')


class NativeJointPoolRetriever:
    """sources含book_id、evidence、passages、retrieve(question)原生来源映射回调。"""

    def __init__(self, pool, sources, *, method):
        books = [source['book_id'] for source in sources]
        courses = list(pool.book_courses.values())
        if (method not in POLICY['methods'] or len(books) not in {2, 6} or len(books) != len(set(books))
                or set(books) != set(pool.allowed_books) or not pool.paragraph_dedup):
            raise ValueError('Exact declared native book indexes and paragraph-deduplicated pool required')
        if (len(set(courses)) != len(books) // 2
                or any(courses.count(course) != 2 for course in set(courses))):
            raise ValueError('Only a same-course two-book pool or the three-course six-book pool is allowed')
        self.pool, self.method = pool, method
        self.sources = sorted(sources, key=lambda source: source['book_id'])
        for source in self.sources:
            book, evidence = source['book_id'], source['evidence']
            expected = {eid for b, eid in pool.source_positions if b == book}
            if set(evidence) != expected or not callable(source['retrieve']):
                raise ValueError('Native source binding must cover the exact original joint corpus')
            for eid in expected:
                original = evidence[eid]
                row = pool.rows[pool.source_positions[book, eid]]
                if (original['book_id'] != book or original['text'] != row['text']
                        or original['chunk_id'] != row['chunk_id']
                        or original['source']['source_id'] != row['original_source_id']):
                    raise ValueError('Native source text, book, chunk or paragraph differs from the joint corpus')

    async def retrieve(self, question, query_vector):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Only nonempty question text may enter native pool retrieval')
        self.pool._query(query_vector)
        candidates = set()
        audits = []
        for source in self.sources:
            book = source['book_id']
            result = source['retrieve'](question)
            packet = await result if inspect.isawaitable(result) else result
            if packet.get('book_id') != book or packet.get('question') != question:
                raise ValueError('Native mapped context belongs to a different book or exact question')
            rows = packet['rows']
            projection = project_sources(rows, book, source['evidence'], source['passages'])
            # 只接纳真实原生窗口触及的原始片段；不以共同稠密检索补候选。
            keys = {(book, eid) for row in rows for eid in row['evidence_ids']}
            if not keys <= self.pool.source_positions.keys():
                raise ValueError('Native source mapping includes evidence outside the declared pool')
            candidates.update(keys)
            audits.append(dict(book_id=book, mapped_source_digest=digest(rows),
                candidate_keys=[list(key) for key in sorted(keys)], projection=projection))
        rows, trace = self.pool.retrieve_candidates(question, query_vector, sorted(candidates),
            method_name=self.method)
        trace.update(native_pool_policy=POLICY, queried_books=[source['book_id'] for source in self.sources],
            native_candidate_keys=[list(key) for key in sorted(candidates)], native_source_audits=audits,
            native_end_to_end=False, global_candidate_substitution=False,
            comparison_variant='controlled_native_source_projected_common_paragraph')
        return rows, trace

    def pack_context(self, rows, question):
        return self.pool.pack_context(rows, question)


def _checked(reference):
    if file_record(reference['path']) != reference:
        raise ValueError('Pinned native joint-pool asset bytes changed')
    return read_document(reference['path'])


def graph_document_binding(path, units, structure, book, original, receipt):
    import pandas as pd
    from .native_corpus import native_passages
    passages, audit = native_passages(units['rows'], structure, book)
    if original.get('corpus_audit', audit) != audit:
        raise ValueError('Original GraphRAG full source corpus audit changed')
    expected = {digest(digest([book, row['chunk_id']])) + '.txt': row['text'] for row in passages}
    documents = pd.read_parquet(path).to_dict(orient='records')
    if (not documents or any(row.get('title') not in expected or expected[row['title']] != row.get('text') for row in documents)
            or {row['text'] for row in documents} != {row['text'] for row in passages}):
        raise ValueError('GraphRAG documents do not exactly cover the full original source passages')
    binding = [dict(chunk_id=row['chunk_id'], document_id=digest([book, row['chunk_id']]),
        text_sha256=hashlib.sha256(row['text'].encode('utf-8')).hexdigest(), evidence_spans=row['evidence_spans']) for row in passages]
    if receipt['native_index'].get('source_binding_digest', digest(binding)) != digest(binding):
        raise ValueError('Original GraphRAG document/source binding digest differs')
    return dict(documents=file_record(path), source_binding_digest=digest(binding),
        source_passage_count=len(passages), actual_native_document_count=len(documents),
        all_original_passage_texts_covered=True, corpus_audit=audit)


def preflight_native_jobs(configuration):
    """只读核验显式18索引与353题合同；不导入原生依赖、provider或凭据。"""
    retrieval = _checked(configuration['retrieval_configuration'])
    sources = retrieval['sources']
    books = {source['book_id'] for source in sources}
    courses = {source['course_id'] for source in sources}
    if (len(sources) != 6 or len(books) != 6 or len(courses) != 3
            or any(sum(source['course_id'] == course for source in sources) != 2 for course in courses)):
        raise ValueError('Exact six original books and three two-book courses required')
    denominators = {}
    question_ids = []
    for source in sources:
        dataset = _checked(source['dataset'])
        questions = dataset['questions'] if isinstance(dataset, dict) else dataset
        if len(questions) != source['expected_question_count']:
            raise ValueError('Original fixed dataset denominator changed')
        denominators[source['book_id']] = len(questions)
        question_ids.extend(question['question_id'] for question in questions)
    if len(question_ids) != 353 or len(set(question_ids)) != 353:
        raise ValueError('All 353 unique fixed original questions required')
    jobs = configuration['native_jobs']
    expected = {(method, book) for method in ['graphrag', 'lightrag', 'hipporag'] for book in books}
    if len(jobs) != 18 or {(job['method'], job['book']) for job in jobs} != expected:
        raise ValueError('All eighteen explicitly pinned completed original native indexes required')
    corpus_pins = {}
    verified = []
    for job in jobs:
        method, book = job['method'], job['book']
        original, index, receipt = (_checked(job[key]) for key in ['source_contract', 'index_contract', 'index_receipt'])
        identity = index['index_identity']
        root = Path(job['index_contract']['path']).parent
        if (index['method'] != method or index['book'] != book or original['method'] != method
                or original['book'] != book or original['index_identity'] != identity
                or digest({k: v for k, v in index.items() if k != 'index_identity'}) != identity
                or original['job_identity'] != digest({k: v for k, v in original.items() if k != 'job_identity'})
                or receipt['job_identity'] != identity
                or receipt['checkpoint_digest'] != digest({k: v for k, v in receipt.items() if k != 'checkpoint_digest'})):
            raise ValueError('Original native source, index or completed receipt identity differs')
        if receipt.get('receipt_type') == 'derived_completed_index_evidence':
            if (method != 'graphrag' or receipt.get('original_index_receipt_absent') is not True
                    or (root / 'index_receipt.json').exists() or receipt['source_contract'] != job['source_contract']
                    or receipt['index_contract'] != job['index_contract']):
                raise ValueError('Derived completion proof cannot replace an existing original receipt')
            ready = _checked(receipt['original_ready'])
            marker = _checked(receipt['completion_marker'])
            manifest = _checked(receipt['native_manifest'])
            ready_source = _checked(ready['original_contract'])
            if (ready['status'] != 'complete_original_network_native_index' or ready['method'] != method
                    or ready['book'] != book or ready['index_identity'] != identity
                    or Path(ready['index_root']).resolve() != root.resolve()
                    or ready['native_receipt'] != receipt['native_index']
                    or ready_source['index_identity'] != identity or ready_source['source_units'] != original['source_units']
                    or ready_source['structure'] != original['structure']
                    or marker['manifest_digest'] != digest(manifest) or manifest['target_book_id'] != book):
                raise ValueError('Derived native completion evidence differs from actual original recovery assets')
            if 'independent_index_audit' in ready:
                _checked(ready['independent_index_audit'])
        units = _checked(original['source_units'])
        structure = _checked(original['structure'])
        pin = (original['source_units']['sha256'], original['structure']['sha256'])
        if (index['source_units_sha256'] != pin[0] or index['structure_sha256'] != pin[1]
                or structure['book_id'] != book or index['runtime_config'] != original['runtime_config']
                or corpus_pins.setdefault(book, pin) != pin):
            raise ValueError('Native methods must use the exact same original book corpus and model configuration')
        for reference in original['code'].values():
            if isinstance(reference, dict) and set(reference) == {'path', 'sha256'}:
                if file_record(reference['path']) != reference:
                    raise ValueError('Frozen native implementation bytes changed')
        if method == 'hipporag':
            corpus = read_document(root / 'storage/native_corpus_identity.json')
            if corpus.get('target_book_id') != book or receipt['native_index']['corpus_identity'] != corpus:
                raise ValueError('Completed original HippoRAG corpus binding differs; rebuilding is prohibited')
            completion = file_record(root / 'storage/native_corpus_identity.json')
        else:
            complete = list((root / 'storage').glob('*/native_index_complete.json'))
            if len(complete) != 1:
                raise ValueError('One completed exact native index required; building is prohibited')
            marker = read_document(complete[0])
            manifest = read_document(complete[0].parent / 'native_adapter_manifest.json')
            if (receipt.get('receipt_type') == 'derived_completed_index_evidence'
                    and (file_record(complete[0]) != receipt['completion_marker'] or
                         file_record(complete[0].parent / 'native_adapter_manifest.json') != receipt['native_manifest'])):
                raise ValueError('Derived completed index marker or manifest points outside this exact index')
            if manifest['target_book_id'] != book or marker['manifest_digest'] != digest(manifest):
                raise ValueError('Completed native manifest binding differs')
            if method == 'graphrag':
                required = {'text_units', 'entities', 'relationships', 'communities', 'community_reports'}
                if not required <= set(marker['tables']) <= required | {'documents'}:
                    raise ValueError('GraphRAG requires the completed Standard derived-table index')
                for name, reference in marker['tables'].items():
                    if file_record(complete[0].parent / 'output' / (name + '.parquet'))['sha256'] != reference['sha256']:
                        raise ValueError('Completed original GraphRAG table changed')
                documents = graph_document_binding(complete[0].parent / 'output/documents.parquet',
                    units, structure, book, original, receipt)
            completion = file_record(complete[0])
        verified.append(dict(method=method, book_id=book, index_contract=job['index_contract'],
            source_contract=job['source_contract'], index_receipt=job['index_receipt'], completion=completion,
            **(dict(document_binding=documents) if method == 'graphrag' else {})))
    return dict(status='zero_API_native_asset_preflight_complete', model_calls=0,
        indexes_verified=len(verified), question_count=353, pool_evaluations_per_method=706,
        question_denominators=denominators, native_jobs=verified, native_callbacks_invoked=False,
        native_runtime_loading_tested=False, index_build_allowed=False, raw_source_text_exported=False)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(preflight_native_jobs(read_document(arguments.preflight)), ensure_ascii=False, indent=2))
