"""只保留原始抽取出处的已核验映射，其他检索机制保持不变。"""
from __future__ import annotations

from ..manifest import digest
from . import evidence_hierarchical_alignment as core
from .echa_structured_focus import StructuredFocusRetriever


class OriginAlignmentRetriever(StructuredFocusRetriever):
    def materialize(self,book):
        index,reused=super().materialize(book)
        origins={cid:set(self.mappings['concepts'].get(cid,{}).get(book,[])) for cid in self.entities}
        if any(ids-self.structure['evidence'].keys() for ids in origins.values()):
            raise ValueError('Original source anchor outside target textbook')
        signature=digest({cid:sorted(ids) for cid,ids in origins.items()})
        if index.get('alignment_policy')=='verified_original_extraction_sources_only':
            if index['origin_mapping_digest']!=signature:
                raise ValueError('Original source anchors changed after materialization')
            return index,reused
        # 先由父类检查全部核验记录，再筛选；不修改核验状态与权重。
        rows=[row for row in self.qualified[book] if row['evidence_id'] in origins[row['concept_id']]]
        coverage=core.lift_coverage(self.graph,self.structure,rows)
        regions={k:v for k,v in coverage.items() if self.structure['nodes'][k]['type'] not in {'Book','Evidence'}}
        if not regions:
            regions={k:v for k,v in coverage.items() if self.structure['nodes'][k]['type']!='Book'}
        accepted=[r for r in rows if r['status']=='accepted']
        counts=dict(full_records=len(self.qualified[book]),full_accepted=sum(r['status']=='accepted' for r in self.qualified[book]),
            original_source_records=len(rows),original_source_accepted=len(accepted),
            original_source_positive_concepts=len({r['concept_id'] for r in accepted if r['weight']>0}))
        index=dict(index,regions=regions,costs={k:sum(self.paragraph_tokens[book][eid]
                   for eid in index['descendants'][k]) for k in regions},
                   alignment_policy='verified_original_extraction_sources_only',origin_mapping_digest=signature,
                   alignment_counts=counts)
        self._materialized[book]=index
        return index,reused

    def retrieve(self,*args,**kwargs):
        rows,trace=super().retrieve(*args,**kwargs)
        book=trace['target_book_id'];index=self._materialized[book]
        return rows,dict(trace,alignment_policy=index['alignment_policy'],alignment_counts=index['alignment_counts'],
                         original_source_mapping_digest=index['origin_mapping_digest'],verification_unchanged=True)
