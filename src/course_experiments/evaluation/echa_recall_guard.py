"""ECHA同书召回保护；独立版本，不改原路由与来源核验。"""
from __future__ import annotations

import time
import numpy as np

from ..manifest import digest
from .dual_view_rag import reciprocal
from .echa_rag import ECHARetriever


POLICY = dict(version='echa_recall_guard_v1', global_channel_limit=64,
              route_rank_weight=0.25, rrf_constant=60, retrieval_top_k=5,
              context_bytes=3000, guard='always_union_same_book_lexical_dense_candidates',
              development='one_fixed_policy_no_observed_regression_score_selection',
              provenance='global_guard_candidates_are_navigation_only_not_verified_concept_support')


class RecallGuardRetriever(ECHARetriever):
    def retrieve(self, question, target_book_id, query_vector, method='echa'):
        original, trace = super().retrieve(question, target_book_id, query_vector, method)
        if method != 'echa':
            return original, trace
        started = time.monotonic()
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / np.linalg.norm(query)
        lexical = self.lexical.lexical(question)
        dense = self.vectors @ query
        tie = lambda i: self.rows[i]['evidence_ids'][0]
        limit = POLICY['global_channel_limit']
        global_terms = sorted((i for i, score in enumerate(lexical) if score > 0),
                              key=lambda i: (-lexical[i], tie(i)))[:limit]
        global_dense = sorted(range(len(self.rows)), key=lambda i: (-float(dense[i]), tie(i)))[:limit]
        routed = {self.evidence_index[eid] for eid in trace['coarse_evidence_ids']}
        route_terms = sorted((i for i in routed if lexical[i] > 0),
                             key=lambda i: (-lexical[i], tie(i)))[:limit]
        route_dense = sorted(routed, key=lambda i: (-float(dense[i]), tie(i)))[:limit]
        route_scores = reciprocal([route_terms, route_dense])
        route_order = sorted(route_scores, key=lambda i: (-route_scores[i], tie(i)))[:limit]
        scores = reciprocal([global_terms, global_dense])
        for rank, i in enumerate(route_order, 1):
            scores[i] += POLICY['route_rank_weight'] / (POLICY['rrf_constant'] + rank)
        candidates = routed | set(global_terms) | set(global_dense)
        self.candidate_ids = tuple(sorted(tie(i) for i in candidates))
        order = sorted(scores, key=lambda i: (-scores[i], tie(i)))[:POLICY['retrieval_top_k']]
        rows = [self.rows[i] for i in order]
        if any(row['book_id'] != target_book_id for row in rows):
            raise ValueError('Recall guard crossed target textbook boundary')
        trace = dict(trace, policy=POLICY['version'], recall_guard_policy=POLICY,
                     recall_guard_applied=True, routed_candidate_count=len(routed),
                     routed_evidence_ids=trace['coarse_evidence_ids'],
                     coarse_candidate_count=len(candidates), coarse_evidence_ids=list(self.candidate_ids),
                     candidate_count=len(candidates), candidate_digest=digest(self.candidate_ids),
                     global_lexical_candidates=[tie(i) for i in global_terms],
                     global_dense_candidates=[tie(i) for i in global_dense],
                     guard_candidates_claim_verified_mapping=False, source_support_claimed=False,
                     qualified_routing_retained=True,
                     retrieval_seconds=trace['retrieval_seconds'] + time.monotonic() - started)
        return rows, trace
