"""只在开发集检验细排假设，保留旧版全部结果。"""
import copy
import time
import numpy as np

from .dual_view_rag import DualViewRetriever,POLICY,reciprocal
from .graph_rag_v2 import matches


HYPOTHESIS = dict(id='dense_fine_v2',
    observation='reference-support loss is larger at local ranking than context packing',
    change='preserve C/S/M navigation and coarse candidate pool; fine rank by frozen BGE cosine',
    controls='same model/source/5 units/3000 serialized UTF8 bytes; original context packer',
    selection='development-only retrieval diagnostic followed by same-rule development answers',
    search_space=['unchanged_v1','dense_fine_v2'],confirmation_score_selection=False)

ALIGNMENT_HYPOTHESIS=dict(id='occurrence_mapping_v2',
    observation='development coarse recall .823; dense-only fine ranking reduced exact quote recall',
    change='add exact canonical/alias occurrence links for expanded concepts; union with original coarse pool',
    interpretation='occurrence is a navigation link, never a new supported semantic relation',
    guard='target book only; discard one-character Chinese and <3-character ASCII aliases',
    retained='original concept graph/relations/structural hierarchy/RRF/5 units/3000 byte packer',
    selection='development_only; no confirmation answer or score enters this implementation')


def dense_fine_order(rows,vectors,candidates,query):
    positions=sorted(candidates)
    scores=np.asarray(vectors)[positions]@np.asarray(query)
    return [i for _,i in sorted(zip(scores,positions),
        key=lambda item:(-float(item[0]),rows[item[1]]['evidence_ids'][0]))]


class DenseFineRetriever(DualViewRetriever):
    def retrieve(self,question,target_book_id,query_vector,method='dual'):
        selected,trace=super().retrieve(question,target_book_id,query_vector,method)
        if method not in {'dual','no_structure','no_mapping','no_semantic','no_relations'}:
            return selected,trace
        start=time.monotonic()
        query=np.asarray(query_vector,dtype=np.float32)
        query=query/(np.linalg.norm(query) or 1.)
        candidates={self.evidence_index[e] for e in self.candidate_ids}
        order=dense_fine_order(self.rows,self.vectors,candidates,query)
        selected=[self.rows[i] for i in order[:POLICY['retrieval_top_k']]]
        trace=dict(trace,policy='dvkg_development_dense_fine_v2',hypothesis=HYPOTHESIS,
            retrieval_seconds=trace['retrieval_seconds']+time.monotonic()-start)
        return selected,trace


def occurrence_evidence_ids(rows,book,names):
    if any(r['book_id']!=book for r in rows):raise ValueError('Foreign source in occurrence mapping')
    valid={n.strip() for n in names if isinstance(n,str) and len(n.strip())>=(3 if n.strip().isascii() else 2)}
    return sorted({r['evidence_ids'][0] for r in rows if any(matches(n,r['text']) for n in valid)})


class CompletedMappingsRetriever(DualViewRetriever):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.original_mappings=self.mappings
        self.completed_mappings=copy.deepcopy(self.mappings)
        self.completed_pairs=set()

    def retrieve(self,question,target_book_id,query_vector,method='dual'):
        if method in {'keyword','vector','hybrid','no_mapping'}:
            self.mappings=self.original_mappings
            return super().retrieve(question,target_book_id,query_vector,method)
        started=time.monotonic()
        self.mappings=self.original_mappings
        _,original=super().retrieve(question,target_book_id,query_vector,method)
        original_ids=set(self.candidate_ids)
        added=0
        for key in original['expanded_concept_ids']:
            if (target_book_id,key) in self.completed_pairs:continue
            entity=self.entities[key]
            names=[entity.get('canonical_name',entity.get('name','')),*(entity.get('aliases') or [])]
            found=occurrence_evidence_ids(self.rows,target_book_id,names)
            maps=self.completed_mappings['concepts'].setdefault(key,{})
            before=set(maps.get(target_book_id,[]));maps[target_book_id]=sorted(before|set(found))
            added+=len(set(found)-before);self.completed_pairs.add((target_book_id,key))
        self.mappings=self.completed_mappings
        _,trace=super().retrieve(question,target_book_id,query_vector,method)
        candidates={self.evidence_index[e] for e in original_ids|set(self.candidate_ids)}
        self.candidate_ids=tuple(sorted(original_ids|set(self.candidate_ids)))
        lexical=self.lexical.lexical(question)
        term_order=sorted((i for i in candidates if lexical[i]>0),key=lambda i:(-lexical[i],self.rows[i]['evidence_ids'][0]))
        query=np.asarray(query_vector,dtype=np.float32);query=query/(np.linalg.norm(query) or 1.)
        dense_order=dense_fine_order(self.rows,self.vectors,candidates,query)
        combined=reciprocal([term_order,dense_order])
        order=sorted(combined,key=lambda i:(-combined[i],self.rows[i]['evidence_ids'][0]))
        selected=[self.rows[i] for i in order[:POLICY['retrieval_top_k']]]
        trace=dict(trace,policy='dvkg_development_occurrence_mapping_v2',hypothesis=ALIGNMENT_HYPOTHESIS,
            original_coarse_count=len(original_ids),candidate_count=len(candidates),
            candidate_digest=__import__('hashlib').sha256('\n'.join(self.candidate_ids).encode()).hexdigest(),
            added_occurrence_links_this_call=added,retrieval_seconds=time.monotonic()-started)
        return selected,trace
