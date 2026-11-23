"""独立离线归因；参考引用只用于检索完成后的度量。"""
from __future__ import annotations

import sys
from collections import Counter

from .graph_rag_v2 import EvidenceRetriever
from ..views import semantic_view


def traced_retrieve(retriever, question, control='graph', query_vector=None):
    detail = {}
    previous = sys.getprofile()
    def capture(frame, event, arg):
        if event == 'return' and frame.f_code is EvidenceRetriever.retrieve.__code__:
            values = frame.f_locals
            detail['candidate_indices'] = list(values['order'])
            detail['lexical'] = values['lexical']
            detail['seed_coverage'] = {i:len(v['seeds']) for i,v in values.get('candidates',{}).items()}
        if previous is not None:
            previous(frame, event, arg)
    sys.setprofile(capture)
    try:
        selected, trace = retriever.retrieve(question, control, query_vector)
    finally:
        sys.setprofile(previous)
    return selected, trace, detail


def citation_stages(question, corpus, candidates, selected, context):
    def hit(citation, rows, *, source_only=False):
        return any(citation['evidence_id'] in row['evidence_ids'] and
                   (source_only or citation['quote'] in row['text']) for row in rows)
    counts = Counter(dict(total=0,corpus=0,candidates=0,selected=0,packed_source=0,packed_quote=0))
    losses = Counter(dict(corpus_missing=0,candidate_missing=0,ranking=0,source_unit_selection=0,window=0,survived=0))
    details = []
    for citation in question['citations']:
        flags = dict(corpus=hit(citation,corpus), candidates=hit(citation,candidates),
                     selected=hit(citation,selected), packed_source=hit(citation,context,source_only=True),
                     packed_quote=hit(citation,context))
        if not flags['corpus']: loss='corpus_missing'
        elif not flags['candidates']: loss='candidate_missing'
        elif not flags['selected']: loss='ranking'
        elif not flags['packed_source']: loss='source_unit_selection'
        elif not flags['packed_quote']: loss='window'
        else: loss='survived'
        counts['total'] += 1
        counts.update({k:int(v) for k,v in flags.items()})
        losses[loss] += 1
        details.append(dict(evidence_id=citation['evidence_id'],quote_chars=len(citation['quote']),
                            quote_bytes=len(citation['quote'].encode('utf-8')),loss=loss,**flags))
    return dict(counts=dict(counts),loss_counts=dict(losses),citations=details)


def native_graph(view):
    nodes = {key:row for value in view['native_contributions'].values()
             for key,row in value['graph']['nodes'].items()}
    facts = []
    for value in view['native_contributions'].values():
        for key,claim in value['graph']['claims'].items():
            if claim['status'] != 'active':
                continue
            facts.append(dict(fact_id=key,content=claim['content'],evidence_ids=claim['evidence_ids'],
                              supports=[],source_state='unmerged_native'))
    return semantic_view(nodes,{key:key for key in nodes},facts,view['evidence'])
