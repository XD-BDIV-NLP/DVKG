"""降低弱邻居对章节路由的干扰，保留原始候选与证据核验。"""
from __future__ import annotations

import math

from .echa_structured_fast import StructuredFastRetriever
from .graph_rag_v2 import matches


POLICY = dict(version='echa_structured_focus_v2', concept_temperature=.1,
              exact_query_concept_boost=3., neighbor_weight=.25,
              development_only=True, test_scores_used=False)


class StructuredFocusRetriever(StructuredFastRetriever):
    def _concepts(self, question, query, semantic=True, relations=True):
        seeds, neighbors, old = super()._concepts(question, query, semantic, relations)
        maximum = max(old.values(), default=0.)
        weights = {}
        for cid, score in old.items():
            entity = self.entities[cid]
            exact = any(matches(name, question) for name in
                        [entity.get('canonical_name', ''), *(entity.get('aliases') or [])] if name)
            value = math.exp((score - maximum) / POLICY['concept_temperature'])
            if exact:
                value *= POLICY['exact_query_concept_boost']
            if cid in neighbors:
                value *= POLICY['neighbor_weight']
            weights[cid] = value
        return seeds, neighbors, weights

    def retrieve(self, *args, **kwargs):
        rows, trace = super().retrieve(*args, **kwargs)
        if trace['method'] == 'echa':
            trace = dict(trace, policy=POLICY['version'], query_concept_policy=POLICY)
        return rows, trace
