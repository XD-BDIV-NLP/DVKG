import copy
import unittest
import numpy as np
from course_experiments.dual_view import build_structure,build_mappings
from course_experiments.tests.test_dual_view import corpus
from course_experiments.evaluation.dual_view_rag import DualViewRetriever,concept_text


def retriever():
    structures={b:build_structure(corpus(b)) for b in ['A','B']}
    graph=dict(entities=[dict(entity_id='cache',canonical_name='缓存',aliases=None,source_node_ids=['s/A/evidence/e1','s/B/evidence/e1']),
                         dict(entity_id='virtual',canonical_name='虚拟内存',aliases=[],source_node_ids=['s/A/evidence/e2','s/B/evidence/e2'])],relations=[])
    assets={}
    for book in ['A','B']:
        rows=[dict(book_id=book,chunk_id=e['chunk_id'],evidence_ids=[eid],evidence=[dict(id=eid,text=e['text'])],text=e['text']) for eid,e in structures[book]['evidence'].items()]
        assets[book]=dict(rows=rows,vectors=np.asarray([[1.,0.],[0.,1.]]))
    return DualViewRetriever(graph,structures,build_mappings(graph,structures),assets,['cache','virtual'],np.eye(2))


class DualViewRagTests(unittest.TestCase):
    def test_every_method_has_target_book_boundary(self):
        r=retriever()
        for method in ['dual','keyword','vector','hybrid','no_structure','no_mapping','no_semantic','no_relations']:
            rows,trace=r.retrieve('缓存','A',np.asarray([1.,0.]),method)
            self.assertTrue(rows)
            self.assertEqual({row['book_id'] for row in rows},{'A'})

    def test_semantic_concept_navigation_changes_scope_without_keyword_match(self):
        r=retriever();rows,trace=r.retrieve('高速暂存','A',np.asarray([1.,0.]),'dual')
        self.assertEqual(trace['seed_ids'][0],'cache')
        self.assertEqual(trace['candidate_count'],1)
        self.assertFalse(trace['fallback'])
        self.assertEqual(rows[0]['evidence_ids'],['e1'])

    def test_switch_is_reversible_and_does_not_mutate_shared_graph(self):
        r=retriever();before=copy.deepcopy(r.graph)
        a,_=r.retrieve('缓存','A',np.array([1.,0.]),'dual')
        b,t=r.retrieve('缓存','B',np.array([1.,0.]),'dual')
        again,_=r.retrieve('缓存','A',np.array([1.,0.]),'dual')
        self.assertEqual(a,again);self.assertEqual(r.graph,before)
        self.assertEqual(b[0]['book_id'],'B');self.assertEqual(t['loaded_structure_books'],['B'])

    def test_missing_mapping_falls_back_only_inside_selected_book(self):
        r=retriever();r.mappings['concepts']={}
        rows,t=r.retrieve('高速暂存','B',np.array([1.,0.]),'dual')
        self.assertTrue(t['fallback']);self.assertEqual({row['book_id'] for row in rows},{'B'})

    def test_unknown_book_and_reference_input_rejected(self):
        r=retriever()
        with self.assertRaises(ValueError):r.retrieve('x','foreign',np.array([1.,0.]),'dual')
        with self.assertRaises(TypeError):r.retrieve('x','A',np.array([1.,0.]),'dual',reference='answer')

    def test_concept_embedding_text_handles_nullable_fields(self):
        self.assertIn('缓存',concept_text(dict(canonical_name='缓存',aliases=None,definition=None)))


if __name__=='__main__':unittest.main()
