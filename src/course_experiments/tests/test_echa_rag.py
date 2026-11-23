import copy
import unittest

import numpy as np

from course_experiments.manifest import digest
from course_experiments.evaluation import evidence_hierarchical_alignment as core
from course_experiments.evaluation.echa_rag import ECHARetriever,qualify_verdict,MODEL_PROMPT
from course_experiments.tests.test_dual_view_rag import retriever


def aligned_engine():
    old=retriever();qualified={};tokens={}
    for book,structure in old.structures.items():
        qualified[book]=[];tokens[book]={eid:20 for eid in structure['evidence']}
        for cid,books in old.mappings['concepts'].items():
            for eid in books.get(book,[]):
                text=structure['evidence'][eid]['text']
                verdict=dict(decision='accept',role='definition',same_sense=True,scope_consistent=True,
                    evidence_supported=True,reason='synthetic',citations=[dict(evidence_id=eid,start=0,end=len(text),quote=text)])
                qualified[book].append(core.qualify_mapping(digest(old.graph),cid,structure,
                    dict(book_id=book,evidence_id=eid,semantic_score=1.,channels=['origin']),verdict))
    return ECHARetriever(old.graph,old.structures,old.mappings,old.unit_assets,old.ids,
                         old.concept_vectors,qualified,tokens)


class ECHATests(unittest.TestCase):
    def test_verifier_cannot_admit_fabricated_quote_or_missing_semantics(self):
        old=retriever();s=old.structures['A'];candidate=dict(book_id='A',evidence_id='e1',semantic_score=1)
        good=dict(decision='accept',role='definition',same_sense=True,scope_consistent=True,evidence_supported=True,
                  reason='synthetic',citations=[dict(evidence_id='e1',quote=s['evidence']['e1']['text'])])
        row=qualify_verdict(digest(old.graph),'cache',s,candidate,good)
        self.assertEqual(row['status'],'accepted');self.assertEqual(row['verification']['citations'][0]['start'],0)
        for field,value in [('citations',[dict(evidence_id='e1',quote='fabricated')]),('same_sense',None)]:
            bad=dict(good);bad[field]=value
            with self.assertRaises(ValueError):qualify_verdict(digest(old.graph),'cache',s,candidate,bad)
        self.assertIn('提及',MODEL_PROMPT)

    def test_every_control_keeps_original_text_and_selected_book(self):
        r=aligned_engine()
        for method in r.methods:
            rows,trace=r.retrieve('缓存','A',np.array([1.,0.]),method)
            self.assertEqual({row['book_id'] for row in rows},{'A'})
            self.assertTrue(trace['policy'].startswith('echa_'))
            self.assertEqual(trace['concept_graph_id'],digest(r.graph))
            self.assertTrue(set(r.candidate_ids)<=set(r.structure['evidence']))

    def test_budget_fallback_and_switch_do_not_change_shared_graph(self):
        r=aligned_engine();before=copy.deepcopy(r.graph)
        a,t=r.retrieve('缓存','A',np.array([1.,0.]),'echa')
        self.assertEqual(t['routing']['tokens'],20)
        b,_=r.retrieve('缓存','B',np.array([1.,0.]),'echa')
        again,_=r.retrieve('缓存','A',np.array([1.,0.]),'echa')
        self.assertEqual(a,again);self.assertEqual(r.graph,before);self.assertEqual(b[0]['book_id'],'B')
        r.qualified['A']=[];r.coverage.pop('A',None)
        _,trace=r.retrieve('缓存','A',np.array([1.,0.]),'echa')
        self.assertTrue(trace['fallback'])

    def test_unverified_occurrence_is_explicit_ablation_not_full_method(self):
        r=aligned_engine();r.qualified['A']=[]
        _,full=r.retrieve('缓存','A',np.array([1.,0.]),'echa')
        _,ablation=r.retrieve('缓存','A',np.array([1.,0.]),'no_role_verification')
        self.assertEqual(full['coarse_candidate_count'],0)
        self.assertGreater(ablation['coarse_candidate_count'],0)
        self.assertFalse(ablation['source_support_claimed'])

    def test_query_and_mapping_drift_are_rejected(self):
        r=aligned_engine()
        for query in [np.array([float('nan'),0]),np.array([0.,0.]),np.array([1.,0.,0.])]:
            with self.assertRaises(ValueError):r.retrieve('缓存','A',query,'echa')
        r.qualified['A'][0]['source_text_sha256']='changed'
        with self.assertRaises(ValueError):r.retrieve('缓存','A',np.array([1.,0.]),'echa')

    def test_no_semantic_neighbor_navigation_does_not_use_query_vector(self):
        r=aligned_engine();r.adjacency['cache']=[('virtual',{},'outgoing')]
        first=r._concepts('缓存',np.array([1.,0.]),semantic=False)
        second=r._concepts('缓存',np.array([0.,1.]),semantic=False)
        self.assertEqual(first,second)

    def test_flat_alignment_removes_hierarchy_but_keeps_routing_budget(self):
        r=aligned_engine();r.paragraph_tokens['A']={'e1':5000,'e2':5000}
        _,trace=r.retrieve('缓存','A',np.array([1.,0.]),'flat_alignment')
        self.assertIsNotNone(trace['routing'])
        self.assertEqual(trace['routing']['tokens'],0)
        self.assertEqual(trace['coarse_candidate_count'],0)

    def test_relation_navigation_keeps_direction_and_condition_in_trace_only(self):
        r=aligned_engine();r.adjacency['cache']=[('virtual',dict(relation_id='r',relation='prerequisite_of',conditions='test condition'),'outgoing')]
        rows,trace=r.retrieve('缓存','A',np.array([1.,1.]),'echa')
        self.assertEqual(trace['paths'][0]['direction'],'outgoing')
        self.assertEqual(trace['paths'][0]['conditions'],'test condition')
        self.assertTrue(all('test condition' not in row['text'] for row in rows))


if __name__=='__main__':unittest.main()
