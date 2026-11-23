import copy
import unittest
from course_experiments.dual_view import build_structure,build_mappings


def corpus(book='A'):
    evidence=[dict(id='e1',book_id=book,text='缓存降低访存延迟。',section_id='s1',section_path_ids=['c1','s1'],title_path=['存储系统','缓存'],chapter_id='c1',paragraph_order=0,start=0,end=9,page_start=1,page_end=1),
              dict(id='e2',book_id=book,text='虚拟内存使用地址映射。',section_id='s2',section_path_ids=['c2','s2'],title_path=['操作系统','地址'],chapter_id='c2',paragraph_order=1,start=0,end=11,page_start=2,page_end=2)]
    return dict(books=[dict(id=book,title=book)],evidence=evidence,chunks=[dict(id='chunk1',book_id=book,evidence_ids=['e1']),dict(id='chunk2',book_id=book,evidence_ids=['e2'])])


class DualViewTests(unittest.TestCase):
    def test_structure_preserves_paths_offsets_and_book_namespace(self):
        c=corpus();before=copy.deepcopy(c);a=build_structure(c);b=build_structure(corpus('B'))
        self.assertEqual(c,before)
        self.assertFalse(set(a['nodes'])&set(b['nodes']))
        self.assertEqual(a['evidence']['e1']['source']['page_start'],1)
        self.assertEqual(a['evidence']['e1']['section_path_ids'],['c1','s1'])
        self.assertIn(('A/section/c1','A/section/s1','contains'),{tuple(e) for e in a['edges']})

    def test_foreign_evidence_rejected(self):
        c=corpus();c['evidence'][0]['book_id']='B'
        with self.assertRaises(ValueError):build_structure(c)

    def test_mapping_resolves_book_scoped_anchor_and_retains_relation_support(self):
        structures={b:build_structure(corpus(b)) for b in ['A','B']}
        graph=dict(entities=[dict(entity_id='x',source_node_ids=['course/A/evidence/e1']),dict(entity_id='y',source_node_ids=['course/B/evidence/e2'])],
                   relations=[dict(relation_id='r',head_entity_id='x',tail_entity_id='y',relation='related_to',source_node_ids=['course/A/evidence/e2'])])
        before=copy.deepcopy(graph);m=build_mappings(graph,structures)
        self.assertEqual(graph,before)
        self.assertEqual(m['concepts']['x']['A'],['e1'])
        self.assertNotIn('B',m['concepts']['x'])
        self.assertEqual(m['relations']['r']['A'],['e2'])

    def test_ambiguous_unscoped_anchor_not_assigned_to_wrong_book(self):
        s={b:build_structure(corpus(b)) for b in ['A','B']}
        m=build_mappings(dict(entities=[dict(entity_id='x',source_node_ids=['e1'])],relations=[]),s)
        self.assertEqual(m['concepts']['x'],{})
        self.assertEqual(m['audit']['ambiguous_anchors'],1)


if __name__=='__main__':unittest.main()
