"""复用本地GPU精排分数；绑定问题、候选原文和教材身份。"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from ..manifest import digest
from .dual_view_rag import reciprocal


POLICY = dict(version='same64_RRF_candidates_v1', pool=64, retrieval_top_k=5,
              model='BAAI/bge-reranker-v2-m3', max_length=512,
              score='raw_cross_encoder_logit_not_probability')


def rerank_pool(engine, question, query_vector):
    query = np.asarray(query_vector, dtype=np.float32)
    query = query / np.linalg.norm(query)
    positions = [engine.evidence_index[eid] for eid in engine.candidate_ids]
    lexical = engine.lexical.lexical(question); dense = engine.vectors @ query
    tie = lambda i: engine.rows[i]['evidence_ids'][0]
    left = sorted((i for i in positions if lexical[i] > 0),
                  key=lambda i: (-lexical[i], tie(i)))[:POLICY['pool']]
    right = sorted(positions, key=lambda i: (-float(dense[i]), tie(i)))[:POLICY['pool']]
    scores = reciprocal([left, right])
    return [tie(i) for i in sorted(scores, key=lambda i: (-scores[i], tie(i)))[:POLICY['pool']]]


class CachedLocalReranker:
    def __init__(self, base, variant, table):
        self.base, self.variant, self.table = base, variant, table

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def candidate_ids(self):
        return self.base.candidate_ids

    @candidate_ids.setter
    def candidate_ids(self, value):
        self.base.candidate_ids = value

    def retrieve(self, question, target_book_id, query_vector, method='echa'):
        _, trace = self.base.retrieve(question, target_book_id, query_vector, method)
        started = time.monotonic(); key = digest([target_book_id, question])
        if key not in self.table:
            raise ValueError('No bound local reranking receipt for this question and book')
        packet = self.table[key]; ids = rerank_pool(self.base, question, query_vector)
        if (packet['book'] != target_book_id or packet['question'] != question
                or packet['variants'].get(self.variant) != ids):
            raise ValueError('Local reranking candidate identity or order differs')
        for eid in ids:
            row = self.base.rows[self.base.evidence_index[eid]]
            if (eid not in packet['scores'] or not math.isfinite(packet['scores'][eid])
                    or hashlib.sha256(row['text'].encode()).hexdigest() != packet['source_sha256'].get(eid)):
                raise ValueError('Local reranking source changed or score missing')
        order = sorted(ids, key=lambda eid: (-packet['scores'][eid], eid))
        rows = [self.base.rows[self.base.evidence_index[eid]] for eid in order[:POLICY['retrieval_top_k']]]
        return rows, dict(trace, local_reranker_used=True, local_reranker_policy=POLICY,
                          reranked_candidate_count=len(ids), rerank_receipt_identity=digest(packet),
                          score_replay_seconds=time.monotonic()-started,
                          gpu_inference_seconds_reported_separately=True)
