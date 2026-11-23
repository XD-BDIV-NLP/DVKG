"""摘要导向的物化对齐、原生区域检索与有界同书补证。"""
from __future__ import annotations

import time

import numpy as np

from ..manifest import digest
from . import evidence_hierarchical_alignment as core
from .dual_view_rag import reciprocal
from .echa_rag import ECHARetriever


POLICY = dict(version='echa_structured_fast_v1', region_tokens=4096, max_regions=8,
              local_channel_limit=64, backup_channel_limit=64, coverage_floor=.8,
              dense_gap=.04, lexical_gap_ratio=1.5, retrieval_top_k=5,
              context_bytes=3000, backup='conditional_same_book_only',
              mapping='pre_materialized_verified_original_batches',
              novelty_claim=False, accuracy_guarantee=False)


class StructuredFastRetriever(ECHARetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._materialized = {}

    def materialize(self, book):
        self.load_book(book)
        signature = digest(dict(graph=self.graph, structure=self.structure,
                                qualified=self.qualified[book], tokens=self.paragraph_tokens[book]))
        if book in self._materialized:
            index = self._materialized[book]
            if signature != index['signature']:
                raise ValueError('Materialized graph, source or mapping changed')
            return index, True
        coverage = core.lift_coverage(self.graph, self.structure, self.qualified[book])
        ancestors, descendants, _ = core._hierarchy(self.structure)
        regions = {k: values for k, values in coverage.items()
                   if self.structure['nodes'][k]['type'] not in {'Book', 'Evidence'}}
        # 无原生小节时保留叶证据，不能凭空制造章节。
        if not regions:
            regions = {k: values for k, values in coverage.items()
                       if self.structure['nodes'][k]['type'] != 'Book'}
        costs = {k: sum(self.paragraph_tokens[book][eid] for eid in descendants[k]) for k in regions}
        index = dict(signature=signature, regions=regions, costs=costs,
                     ancestors=ancestors, descendants=descendants)
        self._materialized[book] = index
        return index, False

    def _regions(self, index, weights):
        total = sum(weights.values())
        weights = {c: v / total for c, v in weights.items()} if total else {}
        values = index['regions']; costs = index['costs']; ancestors = index['ancestors']
        selected = []; covered = {}; spent = 0
        for _ in range(POLICY['max_regions']):
            choices = []
            for key, coverage in values.items():
                cost = costs[key]
                if cost + spent > POLICY['region_tokens'] or key in selected:
                    continue
                if any(key in ancestors[old] or old in ancestors[key] for old in selected):
                    continue
                gain = sum(w * max(0., coverage.get(c, 0.) - covered.get(c, 0.))
                           for c, w in weights.items())
                if gain > 0:
                    choices.append((gain / max(1, cost), gain, cost, key))
            if not choices:
                break
            _, _, cost, key = min(choices, key=lambda v: (-v[0], -v[1], v[2], v[3]))
            selected.append(key); spent += cost
            for c in weights:
                covered[c] = max(covered.get(c, 0.), values[key].get(c, 0.))
        ids = sorted({eid for key in selected for eid in index['descendants'][key]})
        mapped = sum(w for c, w in weights.items() if covered.get(c, 0.) > 0)
        return dict(region_ids=selected, evidence_ids=ids, tokens=spent,
                    mapped_query_weight=mapped, concept_coverage=covered,
                    budget=POLICY['region_tokens'], antichain=True)

    def retrieve(self, question, target_book_id, query_vector, method='echa'):
        if method != 'echa':
            return super().retrieve(question, target_book_id, query_vector, method)
        if not isinstance(question, str):
            raise ValueError('Question text required')
        started = time.monotonic(); load_seconds = self.load_book(target_book_id)
        query = np.asarray(query_vector, dtype=np.float32)
        if (query.shape != (self.concept_vectors.shape[1],) or not np.isfinite(query).all()
                or np.linalg.norm(query) <= 0):
            raise ValueError('Finite nonzero query vector with matching dimensions required')
        query = query / np.linalg.norm(query)
        index, reused = self.materialize(target_book_id)
        seeds, neighbors, weights = self._concepts(question, query)
        route = self._regions(index, weights)
        routed = {self.evidence_index[eid] for eid in route['evidence_ids']}
        lexical = self.lexical.lexical(question); dense = self.vectors @ query
        tie = lambda i: self.rows[i]['evidence_ids'][0]
        global_terms = sorted((i for i, v in enumerate(lexical) if v > 0),
                              key=lambda i: (-lexical[i], tie(i)))[:POLICY['backup_channel_limit']]
        global_dense = sorted(range(len(self.rows)), key=lambda i: (-float(dense[i]), tie(i)))[:POLICY['backup_channel_limit']]
        reasons = []
        if not routed:
            reasons.append('mapping_coverage_missing')
        elif route['mapped_query_weight'] < POLICY['coverage_floor']:
            reasons.append('query_concept_coverage_low')
        if routed and global_dense and dense[global_dense[0]] > max(float(dense[i]) for i in routed) + POLICY['dense_gap']:
            reasons.append('region_semantic_gap')
        if routed and global_terms and lexical[global_terms[0]] > POLICY['lexical_gap_ratio'] * max(float(lexical[i]) for i in routed):
            reasons.append('region_lexical_gap')
        if len(routed) < POLICY['retrieval_top_k']:
            reasons.append('too_few_local_candidates')
        candidates = set(routed)
        if reasons:
            candidates.update(global_terms); candidates.update(global_dense)
        left = sorted((i for i in candidates if lexical[i] > 0), key=lambda i: (-lexical[i], tie(i)))[:POLICY['local_channel_limit']]
        right = sorted(candidates, key=lambda i: (-float(dense[i]), tie(i)))[:POLICY['local_channel_limit']]
        scores = reciprocal([left, right])
        order = sorted(scores, key=lambda i: (-scores[i], tie(i)))
        rows = [self.rows[i] for i in order[:POLICY['retrieval_top_k']]]
        if any(row['book_id'] != target_book_id for row in rows):
            raise ValueError('Structured retrieval crossed target textbook')
        self.candidate_ids = tuple(sorted(tie(i) for i in candidates))
        paths = [dict(seed=seed, target=target, relation=edge.get('relation'),
                      direction=direction, conditions=edge.get('conditions', ''))
                 for seed in seeds for target, edge, direction in self.adjacency.get(seed, [])
                 if target in seeds + neighbors]
        return rows, dict(policy=POLICY['version'], method=method, target_book_id=target_book_id,
                          concept_graph_id=self.graph_id, seed_ids=seeds,
                          expanded_concept_ids=seeds + neighbors, paths=paths, routing=route,
                          graph_navigation_used=bool(routed), fallback=bool(reasons),
                          backup_reasons=reasons, materialized_index_reused=reused,
                          routed_candidate_count=len(routed), candidate_count=len(candidates),
                          coarse_candidate_count=len(routed), coarse_evidence_ids=route['evidence_ids'],
                          candidate_digest=digest(self.candidate_ids), total_target_units=len(self.rows),
                          source_support_claimed=not bool(reasons),
                          loaded_structure_books=[target_book_id], structure_load_seconds=load_seconds,
                          retrieval_seconds=time.monotonic() - started)
