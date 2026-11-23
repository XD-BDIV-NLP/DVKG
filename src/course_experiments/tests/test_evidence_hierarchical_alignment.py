import copy
import unittest

from course_experiments.dual_view import build_structure
from course_experiments.manifest import digest
from course_experiments.evaluation import evidence_hierarchical_alignment as alignment


def fixture(book='book'):
    texts={'a':'高速缓存用于缩短访存时间。','b':'进程调度选择下一个运行进程。','c':'其他介绍。'}
    rows=[dict(id=eid,book_id=book,text=text,section_path_ids=['chapter',eid],
               chapter_id='chapter',paragraph_order=i) for i,(eid,text) in enumerate(texts.items())]
    corpus=dict(books=[dict(id=book)],chunks=[dict(id='chunk-'+r['id'],book_id=book,
        evidence_ids=[r['id']]) for r in rows],evidence=rows)
    structure=build_structure(corpus)
    graph=dict(entities=[dict(entity_id='cache',canonical_name='高速缓存',aliases=['Cache']),
                         dict(entity_id='scheduler',canonical_name='进程调度',aliases=[])],relations=[])
    return graph,structure


def verdict(structure,eid,role='definition',**changes):
    value=dict(decision='accept',role=role,same_sense=True,scope_consistent=True,
        evidence_supported=True,reason='synthetic verification only',
        citations=[dict(evidence_id=eid,start=0,end=len(structure['evidence'][eid]['text']),
                        quote=structure['evidence'][eid]['text'])])
    value.update(changes)
    return value


class EvidenceAlignmentTests(unittest.TestCase):
    def mapping(self,graph,structure,cid,eid,weight=1,role='definition'):
        candidate=dict(book_id=structure['book_id'],evidence_id=eid,semantic_score=weight,
                       channels=['origin'])
        return alignment.qualify_mapping(digest(graph),cid,structure,candidate,
                                          verdict(structure,eid,role))

    def test_three_candidate_channels_preserve_origins_without_creating_support(self):
        graph,s=fixture()
        rows=alignment.candidate_paragraphs(graph['entities'][0],s,{'a':.6,'b':.9,'c':.1},origins=['c'],limit=1)
        self.assertEqual({r['evidence_id'] for r in rows},{'a','b','c'})
        self.assertTrue(all(r['status']=='candidate_not_verified' for r in rows))
        self.assertEqual(next(r for r in rows if r['evidence_id']=='c')['channels'],['origin'])

    def test_real_graph_nullable_aliases_keep_name_channel(self):
        _,s=fixture()
        rows=alignment.candidate_paragraphs(dict(canonical_name='高速缓存',aliases=None),s,{'a':.5})
        self.assertIn('lexical',rows[0]['channels'])

    def test_wrong_book_unknown_source_and_nonfinite_score_rejected(self):
        g,s=fixture();candidate=dict(book_id='other',evidence_id='a',semantic_score=.9)
        with self.assertRaises(ValueError):alignment.qualify_mapping(digest(g),'cache',s,candidate,verdict(s,'a'))
        for scores,origins in [({'unknown':.8},[]),({'a':float('nan')},[]),({'a':.8},['unknown'])]:
            with self.assertRaises(ValueError):alignment.candidate_paragraphs(g['entities'][0],s,scores,origins=origins)

    def test_exact_quotation_role_and_navigation_are_separate(self):
        g,s=fixture();definition=self.mapping(g,s,'cache','a',.7)
        mention=self.mapping(g,s,'cache','a',.7,'mention')
        self.assertFalse(definition['navigation_only']);self.assertTrue(mention['navigation_only'])
        self.assertEqual(mention['weight'],.7)
        candidate=dict(book_id='book',evidence_id='a',semantic_score=.7)
        bad=verdict(s,'a');bad['citations'][0]['quote']='fabricated'
        with self.assertRaises(ValueError):alignment.qualify_mapping(digest(g),'cache',s,candidate,bad)

    def test_negative_or_uncertain_verdict_does_not_become_mapping(self):
        g,s=fixture();candidate=dict(book_id='book',evidence_id='a',semantic_score=.9)
        for changes in [dict(decision='unresolved'),dict(same_sense=False),dict(evidence_supported=False),dict(scope_consistent=False)]:
            row=alignment.qualify_mapping(digest(g),'cache',s,candidate,verdict(s,'a',**changes))
            self.assertEqual(row['status'],'rejected_or_unresolved');self.assertEqual(row['weight'],0)

    def test_ancestor_coverage_is_maximum_not_mention_count(self):
        g,s=fixture();rows=[self.mapping(g,s,'cache','a',.6),self.mapping(g,s,'cache','b',.8)]
        coverage=alignment.lift_coverage(g,s,rows)
        self.assertEqual(coverage['book/book']['cache'],.8)
        self.assertEqual(coverage,alignment.lift_coverage(g,s,rows+rows))

    def test_changed_source_graph_or_mapping_digest_invalidates_alignment(self):
        g,s=fixture();row=self.mapping(g,s,'cache','a')
        for changed_graph,changed_s,changed_row in [(dict(g,version=2),s,row),(g,copy.deepcopy(s),row),(g,s,dict(row,weight=.1))]:
            if changed_s is not s:changed_s['evidence']['a']['text']='changed text'
            with self.assertRaises(ValueError):alignment.lift_coverage(changed_graph,changed_s,[changed_row])

    def test_budget_and_antichain_prevent_duplicate_hierarchy_access(self):
        g,s=fixture();maps=[self.mapping(g,s,'cache','a'),self.mapping(g,s,'scheduler','b')]
        coverage=alignment.lift_coverage(g,s,maps)
        out=alignment.select_regions(s,coverage,{'cache':1,'scheduler':1},{'a':6,'b':6,'c':6},budget=12,max_regions=2)
        self.assertEqual(set(out['evidence_ids']),{'a','b'});self.assertEqual(out['tokens'],12)
        self.assertAlmostEqual(out['objective'],1)
        for node in out['region_ids']:
            parent=s['nodes'][node].get('parent')
            while parent:
                self.assertNotIn(parent,out['region_ids']);parent=s['nodes'][parent].get('parent')

    def test_best_single_region_can_recover_from_greedy_ratio_trap(self):
        g,s=fixture();maps=[self.mapping(g,s,'cache','a'),self.mapping(g,s,'cache','b',.9),self.mapping(g,s,'scheduler','b')]
        out=alignment.select_regions(s,alignment.lift_coverage(g,s,maps),{'cache':.7,'scheduler':.3},
                                      {'a':6,'b':10,'c':20},budget=10,max_regions=2)
        self.assertEqual(out['evidence_ids'],['b']);self.assertAlmostEqual(out['objective'],.93)
        self.assertTrue(out['best_single_replaced_greedy'])

    def test_empty_coverage_or_zero_budget_reports_fallback(self):
        g,s=fixture()
        for budget in [0,10]:
            out=alignment.select_regions(s,{}, {'cache':1},{'a':6,'b':6,'c':6},budget=budget,max_regions=2)
            self.assertTrue(out['fallback_required']);self.assertEqual(out['region_ids'],[])

    def test_book_switch_does_not_mutate_concept_graph_or_previous_mapping(self):
        g,s=fixture();_,other=fixture('other');before=copy.deepcopy(g)
        first=self.mapping(g,s,'cache','a');second=self.mapping(g,other,'cache','a')
        self.assertEqual(first['concept_graph_id'],second['concept_graph_id'])
        self.assertNotEqual(first['structure_id'],second['structure_id']);self.assertEqual(g,before)
        with self.assertRaises(ValueError):alignment.lift_coverage(g,other,[first])

    def test_cyclic_or_unknown_parent_hierarchy_rejected(self):
        g,s=fixture()
        for parent in ['unknown','book/section/a']:
            bad=copy.deepcopy(s);bad['nodes']['book/section/chapter']['parent']=parent
            with self.assertRaises(ValueError):alignment.select_regions(bad,{}, {},{'a':1,'b':1,'c':1},budget=2,max_regions=2)

    def test_candidate_ties_are_stable_and_weak_occurrence_cannot_add_relation(self):
        g,s=fixture();before=copy.deepcopy(g)
        first=alignment.candidate_paragraphs(g['entities'][0],s,{'b':.8,'a':.8},limit=1)
        second=alignment.candidate_paragraphs(g['entities'][0],s,{'a':.8,'b':.8},limit=1)
        self.assertEqual(first,second);self.assertEqual(g,before)


if __name__=='__main__':unittest.main()
