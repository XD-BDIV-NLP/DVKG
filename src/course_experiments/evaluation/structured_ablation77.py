"""固定77题抽样及摘要导向检索的独立模块消融。"""
from __future__ import annotations

from ..manifest import digest
from . import evidence_hierarchical_alignment as core
from .echa_structured_focus import StructuredFocusRetriever

SEED='AB77_stratified_20261026_v1'
QUOTAS={
    'book_25f718562685':dict(definition=10,paraphrase=10,conditions=10,cross_chapter=9),
    'book_49bb8d31ddb9':dict(definition=10,paraphrase=9,conditions=9,cross_chapter=10),
}


def select77(metadata):
    if set(metadata)!=set(QUOTAS):
        raise ValueError('Exact computer AB textbooks required')
    selected={};seen=set()
    for book,quota in QUOTAS.items():
        rows=[]
        for q in metadata[book]:
            if (q['target_book_id']!=book or q['type'] not in quota
                    or not isinstance(q['question_id'],str) or q['question_id'] in seen):
                raise ValueError('Invalid book, question type or duplicate identity')
            seen.add(q['question_id'])
            rows.append({k:q[k] for k in ['question_id','type','target_book_id']})
        selected[book]=[]
        for kind,count in quota.items():
            candidates=[q for q in rows if q['type']==kind]
            if len(candidates)<count:
                raise ValueError('Insufficient fixed question stratum')
            selected[book].extend(sorted(candidates,key=lambda q:digest([SEED,book,kind,q['question_id']]))[:count])
    return selected


class FlatStructuredRetriever(StructuredFocusRetriever):
    def materialize(self,book):
        index,reused=super().materialize(book)
        if index.get('ablation')=='without_textbook_hierarchy':
            return index,reused
        coverage=core.lift_coverage(self.graph,self.structure,self.qualified[book])
        regions={k:v for k,v in coverage.items() if self.structure['nodes'][k]['type']=='Evidence'}
        index=dict(index,regions=regions,costs={k:sum(self.paragraph_tokens[book][eid]
                   for eid in index['descendants'][k]) for k in regions},ablation='without_textbook_hierarchy')
        self._materialized[book]=index
        return index,reused


class NoBudgetStructuredRetriever(StructuredFocusRetriever):
    def _regions(self,index,weights):
        route=super()._regions(dict(index,costs={k:0 for k in index['costs']}),weights)
        route.update(tokens=sum(index['costs'][k] for k in route['region_ids']),
                     routing_token_budget_applied=False,ablation='without_budgeted_coverage',
                     actual_token_cost_reported=True)
        return route


class NoRoleStructuredRetriever(StructuredFocusRetriever):
    def materialize(self,book):
        index,reused=super().materialize(book)
        candidates=self.unverified.get(book)
        if candidates is None:
            raise ValueError('Frozen original candidate set required')
        signature=digest(candidates)
        if index.get('ablation')=='without_semantic_role_verification':
            if index['candidate_signature']!=signature:
                raise ValueError('Unverified frozen candidate set changed')
            return index,reused
        coverage={};ancestors,descendants,owners=core._hierarchy(self.structure)
        for row in candidates:
            cid=row['concept_id'];eid=row['evidence_id']
            if row['book_id']!=book or cid not in self.entities or eid not in owners:
                raise ValueError('Candidate graph or source book boundary differs')
            score=core._score(row['semantic_score'])
            for key in ancestors[owners[eid]]|{owners[eid]}:
                values=coverage.setdefault(key,{})
                values[cid]=max(values.get(cid,0.),score)
        regions={k:v for k,v in coverage.items() if self.structure['nodes'][k]['type'] not in {'Book','Evidence'}}
        if not regions:
            regions={k:v for k,v in coverage.items() if self.structure['nodes'][k]['type']!='Book'}
        index=dict(index,regions=regions,costs={k:sum(self.paragraph_tokens[book][eid]
                   for eid in descendants[k]) for k in regions},ablation='without_semantic_role_verification',
                   candidate_signature=signature)
        self._materialized[book]=index
        return index,reused

    def retrieve(self,*args,**kwargs):
        rows,trace=super().retrieve(*args,**kwargs)
        trace=dict(trace,source_support_claimed=False,semantic_role_verification_applied=False)
        return rows,trace
