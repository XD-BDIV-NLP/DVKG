"""冻结联合池的证据选择包装；原检索器及图谱保持不变。"""
from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np

from ..manifest import digest
from .evidence_window_packing import pack_evidence_windows
from .query_evidence_rerank import paragraph_key, rerank_evidence


POLICY = dict(version='question_evidence_selection_pool_v1', global_channel_limit=64,
              candidate_limit=64, rrf_constant=60, route_rank_weight=.25,
              retrieval_top_k=5, context_bytes=3000,
              variants=['baseline', 'rerank', 'windows', 'combined'],
              global_bypass_claims_verified_support=False,
              reference_answers_used=False, target_book_or_course_oracle=False,
              baseline_equivalence='selected_rows_and_serialized_context_only', model_calls=0)


class EvidenceSelectionPool:
    def __init__(self, base_pool):
        self.base = base_pool
        allowed = base_pool.allowed_books
        if (not isinstance(allowed, (list, tuple)) or not allowed
                or len(set(allowed)) != len(allowed)
                or any(not isinstance(book, str) or not book for book in allowed)):
            raise ValueError('The base pool must declare unique allowed textbooks')
        self.allowed_books = tuple(allowed)
        self._bindings = {}
        positions = base_pool.source_positions
        if set(positions.values()) != set(range(len(base_pool.rows))):
            raise ValueError('Source positions must cover every original pool row once')
        for native_key, i in positions.items():
            row = base_pool.rows[i]
            key = paragraph_key(row)
            if (row['book_id'] not in self.allowed_books or tuple(native_key) !=
                    (row['book_id'], row.get('source_evidence_id'))):
                raise ValueError('Pool source belongs to a foreign textbook or identity')
            evidence_key = row['book_id'], row['evidence_ids'][0]
            if evidence_key in self._bindings:
                raise ValueError('Duplicate book-specific source evidence identity')
            self._bindings[evidence_key] = key, row['chunk_id'], row['text']

    def _validate_rows(self, rows, *, top_five=False):
        if top_five and len(rows) > POLICY['retrieval_top_k']:
            raise ValueError('Frozen selection exceeds five paragraphs')
        seen = set()
        for row in rows:
            key = paragraph_key(row)
            expected = self._bindings.get((row['book_id'], row['evidence_ids'][0]))
            if (row['book_id'] not in self.allowed_books or expected !=
                    (key, row['chunk_id'], row['text'])):
                raise ValueError('Selected row differs from its declared original source')
            if top_five and key in seen:
                raise ValueError('Frozen selection repeats an original paragraph')
            seen.add(key)

    def candidates(self, question, vector):
        """召回只接收问题及其向量，不读取目标教材标签或支持答案。"""
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Nonempty question text is required')
        baseline, old = self.base.retrieve(question, vector, 'dvkg')
        self._validate_rows(baseline, top_five=True)
        query = self.base._query(vector)
        lexical = list(self.base.lexical.lexical(question))
        dense = np.asarray(self.base.vectors @ query)
        if (len(lexical) != len(self.base.rows) or dense.shape != (len(self.base.rows),)
                or any(not math.isfinite(float(s)) for s in lexical)
                or not np.isfinite(dense).all()):
            raise ValueError('Finite per-row lexical and dense scores are required')
        indices = set(range(len(self.base.rows)))
        tie = lambda i: (self.base.rows[i]['book_id'], self.base.rows[i]['source_evidence_id'])
        size = POLICY['global_channel_limit']
        left = sorted((i for i in indices if lexical[i] > 0),
                      key=lambda i: (-float(lexical[i]), tie(i)))[:size]
        right = sorted(indices, key=lambda i: (-float(dense[i]), tie(i)))[:size]
        routed = set()
        for native in old.get('routing', {}).get('evidence_ids', []):
            if (not isinstance(native, (list, tuple)) or len(native) != 2
                    or tuple(native) not in self.base.source_positions
                    or native[0] not in self.allowed_books):
                raise ValueError('Frozen route references an undeclared textbook source')
            routed.add(self.base.source_positions[tuple(native)])
        route_left = sorted((i for i in routed if lexical[i] > 0),
                            key=lambda i: (-float(lexical[i]), tie(i)))[:size]
        route_right = sorted(routed, key=lambda i: (-float(dense[i]), tie(i)))[:size]

        def reciprocal(channels):
            scores = {}
            for channel in channels:
                for rank, i in enumerate(channel, 1):
                    scores[i] = scores.get(i, 0.) + 1. / (POLICY['rrf_constant'] + rank)
            return scores

        scores = reciprocal([left, right])
        routed_scores = reciprocal([route_left, route_right])
        route_order = sorted(routed_scores, key=lambda i: (-routed_scores[i], tie(i)))[:size]
        route_boost = {i: POLICY['route_rank_weight'] / (POLICY['rrf_constant'] + rank)
                       for rank, i in enumerate(route_order, 1)}
        union = routed | set(left) | set(right)
        for i in union:
            scores[i] = scores.get(i, 0.) + route_boost.get(i, 0.)
        order = sorted(union, key=lambda i: (-scores[i], tie(i)))
        selected = []; selected_indices = []; seen = set(); routes = {}
        for i in order:
            row = self.base.rows[i]
            key = paragraph_key(row)
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            selected_indices.append(i)
            if len(selected) == POLICY['candidate_limit']:
                break
        self._validate_rows(selected)
        # 同一段落的不同原始切片可以共享导航先验，不据此声明证据支持。
        for i, score in route_boost.items():
            key = paragraph_key(self.base.rows[i])
            if key in seen:
                routes[key] = max(routes.get(key, 0.), score)
        original = routed | set(left) | set(right) if old.get('fallback', False) else set(routed)
        if 'candidate_count' in old and old['candidate_count'] != len(original):
            raise ValueError('Reconstructed original candidates differ from frozen baseline trace')
        paragraph_ids = lambda values: sorted({paragraph_key(self.base.rows[i]) for i in values})
        candidate_ids = [list(paragraph_key(row)) for row in selected]
        original_ids = [list(key) for key in paragraph_ids(original)]
        trace = dict(policy=deepcopy(POLICY), allowed_books=list(self.allowed_books),
            baseline_selected=deepcopy(baseline), baseline_trace=deepcopy(old),
            old_baseline_selected=[list(paragraph_key(row)) for row in baseline],
            original_candidate_count=len(original), original_candidate_ids=original_ids,
            original_candidate_paragraph_count=len(original_ids),
            original_candidate_evidence_ids=[self.base.rows[i]['evidence_ids'][0]
                                             for i in sorted(original, key=tie)],
            original_candidate_digest=digest(original_ids),
            union_candidate_count=len(union), union_paragraph_count=len(paragraph_ids(union)),
            union_candidate_digest=digest([list(key) for key in paragraph_ids(union)]),
            candidate_count=len(selected), candidate_ids=candidate_ids,
            candidate_digest=digest(candidate_ids), soft_route_prior=True,
            global_bypass_claims_verified_support=False,
            global_lexical_candidates=[self.base.rows[i]['evidence_ids'][0] for i in left],
            global_dense_candidates=[self.base.rows[i]['evidence_ids'][0] for i in right],
            candidate_scores=[dict(candidate_id=list(paragraph_key(self.base.rows[i])),
                                   evidence_id=self.base.rows[i]['evidence_ids'][0], rrf_score=scores[i],
                                   route_boost=route_boost.get(i, 0.)) for i in selected_indices],
            model_calls=0)
        return selected, routes, trace

    def run(self, question, vector, *, variant='combined', pair_scores=None):
        if variant not in POLICY['variants']:
            raise ValueError('Unknown evidence selection variant')
        if variant in {'baseline', 'windows'} and pair_scores is not None:
            raise ValueError('Baseline and window-only variants do not consume pair scores')
        candidates, routes, trace = self.candidates(question, vector)
        reranking = None
        if variant in {'baseline', 'windows'}:
            selected = trace['baseline_selected']
        else:
            selected, reranking = rerank_evidence(question, candidates, route_scores=routes,
                                                 pair_scores=pair_scores, limit=5)
        self._validate_rows(selected, top_five=True)
        if variant in {'baseline', 'rerank'}:
            context = self.base.pack_context(selected, question)
            packing = dict(policy='frozen_original_pack_context', model_calls=0)
        else:
            context, packing = pack_evidence_windows(selected, question, byte_budget=3000, top_k=5)
        context_bytes = len(json.dumps(context, ensure_ascii=False).encode('utf-8'))
        if context_bytes > 3000 or len(context) > 5:
            raise ValueError('Packed context exceeds the common evidence budget')
        seen = set()
        for item in context:
            if (item.get('book_id') not in self.allowed_books
                    or not isinstance(item.get('evidence_ids'), list) or len(item['evidence_ids']) != 1):
                raise ValueError('Context contains an undeclared source')
            source = self._bindings.get((item['book_id'], item['evidence_ids'][0]))
            if (source is None or source[1] != item.get('chunk_id') or not item.get('text')
                    or item['text'] not in source[2] or source[0] in seen):
                raise ValueError('Context must contain distinct exact original paragraph windows')
            seen.add(source[0])
        trace.update(variant=variant, reranking=reranking, packing=packing,
                     selected_paragraph_ids=[list(paragraph_key(row)) for row in selected],
                     selected_evidence_ids=[row['evidence_ids'][0] for row in selected],
                     context_bytes=context_bytes, context_rows=len(context), model_calls=0)
        return selected, context, trace
