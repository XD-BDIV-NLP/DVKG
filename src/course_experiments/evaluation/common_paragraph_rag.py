"""共同段落适配；只使用已有关键词和稠密分数，不调用精排模型。"""
from __future__ import annotations
import hashlib
import math
import time
from ..manifest import digest

POLICY = dict(version='common_paragraph_no_cross_encoder_v1',pool=64,retrieval_top_k=5,
              context_bytes=3000,cross_encoder=False,rrf_constant=60,
              paragraph_identity='target_book_and_original_source_id',
              duplicate_slices='highest_ranked_exact_source_slice_represents_paragraph')


def paragraph_pool(ids,lexical,dense,paragraphs,mode='rrf',limit=64):
    ids=list(dict.fromkeys(ids))
    if mode not in {'keyword','vector','rrf'} or any(e not in lexical or e not in dense or e not in paragraphs for e in ids):
        raise ValueError('Invalid paragraph candidate identity or ranking mode')
    if any(not math.isfinite(float(lexical[e])) or not math.isfinite(float(dense[e])) for e in ids):
        raise ValueError('Nonfinite paragraph retrieval score')
    left=sorted((e for e in ids if lexical[e]>0),key=lambda e:(-lexical[e],e))
    right=sorted(ids,key=lambda e:(-dense[e],e))
    if mode=='keyword':order=left or sorted(ids)
    elif mode=='vector':order=right
    else:
        scores={}
        for channel in [left[:limit],right[:limit]]:
            for rank,eid in enumerate(channel,1):scores[eid]=scores.get(eid,0.)+1/(POLICY['rrf_constant']+rank)
        order=sorted(scores,key=lambda e:(-scores[e],e))
    result=[];seen=set()
    for eid in order:
        pid=paragraphs[eid]
        if pid not in seen:result.append(eid);seen.add(pid)
        if len(result)==limit:break
    return result


class CommonParagraphRetriever:
    def __init__(self,base,variant,table):
        self.base,self.variant,self.table=base,variant,table

    def __getattr__(self,name):return getattr(self.base,name)

    @property
    def candidate_ids(self):return self.base.candidate_ids

    @candidate_ids.setter
    def candidate_ids(self,value):self.base.candidate_ids=value

    def retrieve(self,question,target_book_id,query_vector,method='echa'):
        started=time.monotonic();key=digest([target_book_id,question])
        if key not in self.table:raise ValueError('Unregistered question or target book')
        packet=self.table[key]
        if packet['book']!=target_book_id or packet['question']!=question or self.variant not in packet['variants']:
            raise ValueError('Frozen paragraph source identity changed')
        self.base.load_book(target_book_id);ids=packet['variants'][self.variant]
        if len(ids)!=len(set(ids)) or any(e not in self.base.evidence_index for e in ids):
            raise ValueError('Unknown or duplicate paragraph source')
        for eid in ids:
            row=self.base.rows[self.base.evidence_index[eid]]
            if row['book_id']!=target_book_id or hashlib.sha256(row['text'].encode()).hexdigest()!=packet['source_sha256'].get(eid):
                raise ValueError('Paragraph source text or book changed')
        if not ids:raise ValueError('No preserved method retrieval sources; no global substitution')
        self.candidate_ids=tuple(ids);rows=[self.base.rows[self.base.evidence_index[e]] for e in ids[:POLICY['retrieval_top_k']]]
        return rows,dict(method=self.variant,target_book_id=target_book_id,policy=POLICY,
                         candidate_count=len(ids),total_target_units=len(self.base.rows),
                         candidate_digest=digest(self.candidate_ids),fallback=False,
                         source_candidate_audit=packet['candidate_audit'][self.variant],
                         paragraph_packet_identity=digest(packet),retrieval_seconds=time.monotonic()-started,
                         offline_candidate_preparation_reported_separately=True,cross_encoder_used=False)
