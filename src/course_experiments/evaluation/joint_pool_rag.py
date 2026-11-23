"""冻结检索器的联合语料适配：全局路由、排序和上下文预算；不发起模型请求。"""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from .dual_view_rag import concept_text, reciprocal
from .echa_structured_fast import POLICY as ROUTING
from .echa_structured_focus import StructuredFocusRetriever
from .graph_rag_v2 import BUDGET, EvidenceRetriever, pack_context


POLICY = dict(version='frozen_structured_joint_pool_paragraph_adapter_v2',
              question_scope='all_fixed_original_questions_no_score_based_selection',
              source_scope='all_declared_books_no_target_book_or_course_filter',
              concept_namespace='frozen_graph_digest_no_new_cross_course_identity_merges',
              ranking='global_BM25_dense_equal_RRF60',
              paragraph_adapter='highest_ranked_exact_slice_per_book_and_original_source_id_no_vector_aggregation',
              routing_tokens=4096, max_regions=8, retrieval_top_k=5, context_bytes=3000,
              claim='mixed_corpus_robustness_not_cross_source_necessity_or_new_blind_test')


class JointPoolRetriever(StructuredFocusRetriever):
    """sources的每项包含course_id、book_id和已绑定冻结资产的engine。"""

    def __init__(self, sources, *, paragraph_dedup=True):
        if not sources or len({s['book_id'] for s in sources}) != len(sources):
            raise ValueError('A nonempty collection of unique textbook identities is required')
        self.sources = sorted(sources, key=lambda s: s['book_id'])
        self.paragraph_dedup = bool(paragraph_dedup)
        self.allowed_books = [s['book_id'] for s in self.sources]
        self.book_courses = {s['book_id']: s['course_id'] for s in self.sources}
        self.entities = {}; concept_vectors = {}; self.adjacency = defaultdict(list)
        source_rows = {}; source_vectors = {}; seen_graphs = set()
        self.native_sources = {}; self.source_ids = {}
        for source in self.sources:
            book, engine = source['book_id'], source['engine']
            engine.load_book(book)
            for row, vector in zip(engine.rows, engine.vectors):
                if row['book_id'] != book or len(row['evidence_ids']) != 1:
                    raise ValueError('Each pool row must identify one original paragraph in its book')
                eid = row['evidence_ids'][0]; key = (book, eid)
                if key in source_rows:
                    raise ValueError('Duplicate original paragraph identity')
                # 派生引用身份带教材命名空间，原始资产与原始证据ID保持不变。
                uid = json.dumps(key, ensure_ascii=False, separators=(',', ':'))
                original = engine.structure['evidence'][eid].get('source', {}).get('source_id')
                if self.paragraph_dedup and (not isinstance(original, str) or not original):
                    raise ValueError('Original paragraph source_id required for explicit paragraph deduplication')
                item = dict(row, source_evidence_id=eid, original_source_id=original, evidence_ids=[uid],
                            evidence=[dict(id=uid, text=row['text'])])
                source_rows[key] = item; source_vectors[key] = np.asarray(vector)
                self.native_sources[uid] = key; self.source_ids[key] = uid
            graph_id = engine.graph_id
            if graph_id in seen_graphs:
                continue
            seen_graphs.add(graph_id)
            for cid, vector in zip(engine.ids, engine.concept_vectors):
                key = self._concept_id(graph_id, cid)
                self.entities[key] = engine.entities[cid]
                concept_vectors[key] = vector
            for cid, adjacent in engine.adjacency.items():
                self.adjacency[self._concept_id(graph_id, cid)] = [
                    (self._concept_id(graph_id, target), edge, direction)
                    for target, edge, direction in adjacent]
        self.ids = sorted(self.entities)
        self.position = {cid: i for i, cid in enumerate(self.ids)}
        self.concept_vectors = np.asarray([concept_vectors[cid] for cid in self.ids], dtype=np.float32)
        self.concept_lexical = EvidenceRetriever([
            dict(book_id='concept', chunk_id=cid, text=concept_text(self.entities[cid]), evidence_ids=[])
            for cid in self.ids], dict(entities=[], relations=[]))
        self.lexical = EvidenceRetriever(list(source_rows.values()), dict(entities=[], relations=[]))
        self.rows = self.lexical.rows
        self.source_positions = {(row['book_id'], row['source_evidence_id']): i
                                 for i, row in enumerate(self.rows)}
        self.vectors = np.asarray([source_vectors[key] for key, _ in
                                  sorted(self.source_positions.items(), key=lambda pair: pair[1])],
                                 dtype=np.float32)
        if (self.vectors.ndim != 2 or self.concept_vectors.ndim != 2
                or self.vectors.shape[1] != self.concept_vectors.shape[1]
                or not np.isfinite(self.vectors).all() or not np.isfinite(self.concept_vectors).all()
                or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1., atol=1e-4)
                or not np.allclose(np.linalg.norm(self.concept_vectors, axis=1), 1., atol=1e-4)):
            raise ValueError('Compatible finite normalized frozen concept and paragraph vectors required')

    @staticmethod
    def _concept_id(graph_id, cid):
        return json.dumps([graph_id, cid], ensure_ascii=False, separators=(',', ':'))

    def query_concepts(self, question, query_vector):
        """供外层核验执行器预取原始concept ID；不读取参考答案或目标书标签。"""
        query = self._query(query_vector)
        seeds, neighbors, _ = self._concepts(question, query)
        result = defaultdict(list)
        for value in seeds + neighbors:
            graph_id, cid = json.loads(value)
            result[graph_id].append(cid)
        return dict(result)

    def _query(self, vector):
        query = np.asarray(vector, dtype=np.float32)
        if (query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all()
                or np.linalg.norm(query) <= 0):
            raise ValueError('Finite nonzero query vector with the frozen embedding dimension required')
        return query / np.linalg.norm(query)

    def _pool_index(self):
        regions = {}; costs = {}; ancestors = {}; descendants = {}
        for source in self.sources:
            book, engine = source['book_id'], source['engine']
            index, _ = engine.materialize(book)
            for node, values in index['regions'].items():
                key = (book, node)
                regions[key] = {self._concept_id(engine.graph_id, cid): weight
                                for cid, weight in values.items()}
                costs[key] = index['costs'][node]
                ancestors[key] = {(book, ancestor) for ancestor in index['ancestors'][node]}
                descendants[key] = {(book, eid) for eid in index['descendants'][node]}
        return dict(regions=regions, costs=costs, ancestors=ancestors, descendants=descendants)

    def _rank(self, question, query, candidates, method):
        lexical = self.lexical.lexical(question); dense = self.vectors @ query
        tie = lambda i: (self.rows[i]['book_id'], self.rows[i]['source_evidence_id'])
        left = sorted((i for i in candidates if lexical[i] > 0),
                      key=lambda i: (-lexical[i], tie(i)))[:64]
        right = sorted(candidates, key=lambda i: (-float(dense[i]), tie(i)))[:64]
        if method == 'bm25':
            order = left
        elif method == 'bge':
            order = right
        else:
            scores = reciprocal([left, right])
            order = sorted(scores, key=lambda i: (-scores[i], tie(i)))
        result = []; seen = set()
        for i in order:
            row = self.rows[i]; key = (row['book_id'], row['original_source_id'])
            if self.paragraph_dedup and key in seen:
                continue
            seen.add(key); result.append(row)
            if len(result) == 5:
                break
        return result

    def retrieve(self, question, query_vector, method='dvkg'):
        if not isinstance(question, str) or method not in {'dvkg', 'bm25', 'bge', 'hybrid'}:
            raise ValueError('Question text and a declared joint-pool method are required')
        query = self._query(query_vector); all_rows = set(range(len(self.rows)))
        route = {}; reasons = []; routed = set(); seeds = []; neighbors = []
        candidates = all_rows
        if method == 'dvkg':
            seeds, neighbors, weights = self._concepts(question, query)
            # 复用冻结贪心过程，仅将区域及概念身份扩展为全池命名空间。
            route = self._regions(self._pool_index(), weights)
            routed = {self.source_positions[tuple(key)] for key in route['evidence_ids']}
            lexical = self.lexical.lexical(question); dense = self.vectors @ query
            tie = lambda i: (self.rows[i]['book_id'], self.rows[i]['source_evidence_id'])
            global_left = sorted((i for i in all_rows if lexical[i] > 0),
                                 key=lambda i: (-lexical[i], tie(i)))[:64]
            global_right = sorted(all_rows, key=lambda i: (-float(dense[i]), tie(i)))[:64]
            if not routed:
                reasons.append('mapping_coverage_missing')
            elif route['mapped_query_weight'] < ROUTING['coverage_floor']:
                reasons.append('query_concept_coverage_low')
            if (routed and global_right and dense[global_right[0]] >
                    max(float(dense[i]) for i in routed) + ROUTING['dense_gap']):
                reasons.append('region_semantic_gap')
            if (routed and global_left and lexical[global_left[0]] >
                    ROUTING['lexical_gap_ratio'] * max(float(lexical[i]) for i in routed)):
                reasons.append('region_lexical_gap')
            if len(routed) < 5:
                reasons.append('too_few_local_candidates')
            candidates = set(routed)
            if reasons:
                candidates.update(global_left); candidates.update(global_right)
        rows = self._rank(question, query, candidates, method)
        trace = dict(policy=POLICY, method=method, allowed_books=self.allowed_books,
                     book_courses=self.book_courses, paragraph_dedup=self.paragraph_dedup,
                     corpus_slice_count=len(self.rows),
                     corpus_paragraph_count=len({(r['book_id'], r['original_source_id']) for r in self.rows}),
                     candidate_count=len(candidates), routed_candidate_count=len(routed),
                     routing=route, fallback=bool(reasons), backup_reasons=reasons,
                     seed_ids=seeds, expanded_concept_ids=seeds + neighbors,
                     selected_source_keys=[self.native_sources[r['evidence_ids'][0]] for r in rows])
        return rows, trace

    def retrieve_candidates(self, question, query_vector, candidate_keys, *, method_name='K12-KGraph'):
        """接受外部构图器实际召回的原始书/段落键，再按统一预算排序。"""
        query = self._query(query_vector)
        keys = {tuple(key) for key in candidate_keys}
        if not keys <= self.source_positions.keys():
            raise ValueError('External candidates must belong to the declared joint corpus')
        candidates = {self.source_positions[key] for key in keys}
        rows = self._rank(question, query, candidates, 'hybrid')
        return rows, dict(policy=POLICY, method=method_name, allowed_books=self.allowed_books,
                          paragraph_dedup=self.paragraph_dedup, corpus_slice_count=len(self.rows),
                          corpus_paragraph_count=len({(r['book_id'], r['original_source_id']) for r in self.rows}),
                          candidate_count=len(candidates),
                          routing={}, selected_source_keys=[self.native_sources[r['evidence_ids'][0]] for r in rows])

    def pack_context(self, rows, question):
        if len(rows) > 5 or any(row['evidence_ids'][0] not in self.native_sources for row in rows):
            raise ValueError('Context must use at most five selected joint-pool paragraphs')
        if self.paragraph_dedup and len({(r['book_id'], r['original_source_id']) for r in rows}) != len(rows):
            raise ValueError('Paragraph-adapted context cannot repeat the same original source')
        if BUDGET['context_token_limit'] != 3000:
            raise ValueError('Frozen context byte budget changed')
        result = pack_context(rows, question)
        if len(json.dumps(result, ensure_ascii=False).encode('utf-8')) > 3000:
            raise ValueError('Joint context exceeds the shared byte budget')
        return result
