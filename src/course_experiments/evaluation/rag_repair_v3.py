"""针对候选硬筛选与单来源打包损失的独立探索版本。"""
from __future__ import annotations

import json

from .graph_rag_v2 import BUDGET, EvidenceRetriever, window
from .rag_attribution import traced_retrieve

POLICY = dict(version='rag_repair_v3',lexical_chunk_pool=20,
    ranking='original_question_unit_BM25_then_graph_seed_tiebreak',
    source_policy='exact_individual_evidence_units',packing='balanced_then_redistribute_UTF8_bytes',
    development='synthetic_fixtures_before_new_source_disjoint_questions')


def source_units(rows):
    units = {}
    for row in rows:
        for e in row.get('evidence',[]):
            if e['id'] in row['evidence_ids'] and e.get('text'):
                key=(row['book_id'],e['id'])
                units.setdefault(key,dict(book_id=row['book_id'],chunk_id=row['chunk_id'],
                    evidence_ids=[e['id']],evidence=[e],text=e['text'],vector=row.get('vector')))
    return list(units.values())


def pack_context(rows, question, *, limit=BUDGET['context_token_limit']):
    units=source_units(rows)
    if not units:return []
    ranker=EvidenceRetriever(units,dict(entities=[],relations=[]))
    scores=ranker.lexical(question)
    units=[r for _,r in sorted(zip(scores,ranker.rows),key=lambda x:(-x[0],x[1]['book_id'],x[1]['chunk_id'],x[1]['evidence_ids'][0]))[:BUDGET['retrieval_top_k']]]
    clean=[dict(book_id=r['book_id'],chunk_id=r['chunk_id'],evidence_ids=r['evidence_ids'],text='') for r in units]
    def size():return len(json.dumps(clean,ensure_ascii=False).encode('utf-8'))
    # 先平分真实字节，短证据剩余额度再给长证据。
    if size()>=limit:return []
    def fit(i, allowance):
        text=units[i]['text'];lo,hi=0,len(text)
        while lo<hi:
            mid=(lo+hi+1)//2
            candidate=window(text,question,mid)
            if len(json.dumps(candidate,ensure_ascii=False).encode('utf-8'))-2<=allowance:lo=mid
            else:hi=mid-1
        clean[i]['text']=window(text,question,lo) if lo else ''
    free=limit-size()
    for i in range(len(clean)):fit(i,free//len(clean))
    for i in range(len(clean)):
        free=limit-size()
        if free>0:fit(i,len(json.dumps(clean[i]['text'],ensure_ascii=False).encode('utf-8'))-2+free)
    clean=[r for r in clean if r['text']]
    assert len(json.dumps(clean,ensure_ascii=False).encode('utf-8'))<=limit
    return clean


class RepairedRetriever:
    def __init__(self,index,graph):
        self.graph=EvidenceRetriever(index,graph)
        self.units=EvidenceRetriever(source_units(index),dict(entities=[],relations=[]))
        self.row_index={(r['book_id'],r['chunk_id']):i for i,r in enumerate(self.graph.rows)}

    def retrieve(self,question):
        _,trace,detail=traced_retrieve(self.graph,question)
        lexical=self.graph.lexical(question)
        lexical_pool=sorted((i for i,s in enumerate(lexical) if s>0),key=lambda i:(-lexical[i],self.graph.keys[i]))[:POLICY['lexical_chunk_pool']]
        pool=set(detail['candidate_indices']) | set(lexical_pool)
        scores=self.units.lexical(question)
        candidates=[]
        for i,row in enumerate(self.units.rows):
            index=self.row_index[(row['book_id'],row['chunk_id'])]
            if index in pool:
                candidates.append((scores[i],detail['seed_coverage'].get(index,0),row))
        candidates.sort(key=lambda x:(-x[0],-x[1],x[2]['book_id'],x[2]['chunk_id'],x[2]['evidence_ids'][0]))
        selected=[r for _,_,r in candidates[:BUDGET['retrieval_top_k']]]
        return selected,dict(version=POLICY['version'],graph_trace=trace,candidates=len(candidates),
            lexical_candidate_chunk_ids=[self.graph.rows[i]['chunk_id'] for i in lexical_pool])
