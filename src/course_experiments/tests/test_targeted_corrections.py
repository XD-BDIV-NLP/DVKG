import copy
import unittest
from course_experiments import targeted_corrections as c


class TargetedCorrectionTests(unittest.TestCase):
    def nodes(self):
        return {k:dict(entity_type='Concept',name=k) for k in ['a','b','c']}

    def test_transitive_union_cannot_bypass_distinction(self):
        mapping,blocked=c.replay(self.nodes(),set(),{'b':'a','c':'b'},[('a','c')])
        self.assertEqual(mapping['a'],mapping['b'])
        self.assertNotEqual(mapping['a'],mapping['c'])
        self.assertEqual(len(blocked),1)

    def test_original_prior_representative_preserved(self):
        mapping,_=c.replay(self.nodes(),{'c'},{'a':'b','b':'c'},[])
        self.assertEqual(set(mapping.values()),{'c'})

    def test_rebuild_quarantines_only_target_claim_and_keeps_source(self):
        content=dict(kind='relation',head='a',tail='b',relation='requires')
        view=dict(native_contributions={'A':dict(book_id='A',artifact_id='native',graph=dict(nodes=self.nodes(),claims={
            'one':dict(status='active',content=content,evidence_ids=['e']),
            'two':dict(status='active',content=content,evidence_ids=['e'])},events=[]))})
        before=copy.deepcopy(view)
        facts=c.facts_from_native(view,dict(a='a',b='b',c='c'),{'one'})
        self.assertEqual([s['claim_id'] for s in facts[0]['supports']],['two'])
        self.assertEqual(view,before)

    def test_invalid_quote_does_not_authorize_correction(self):
        payload=dict(sources=[dict(evidence_id='e',text='exact source')])
        self.assertFalse(c.quotes_valid(dict(quotes=[dict(evidence_id='e',quote='invented')]),payload))
        self.assertTrue(c.quotes_valid(dict(quotes=[dict(evidence_id='e',quote='exact')]),payload))

    def test_empty_correction_reconstructs_original_semantics(self):
        from course_experiments.tests.test_fusion import contribution,mapping
        from course_experiments.fusion import build_view
        view=build_view(contribution(),[contribution('A')],['A'],alignment=mapping('A'))
        before=copy.deepcopy(view)
        revised=c.derive(view,[],set(),{'path':'review','sha256':'frozen'})
        self.assertEqual(revised['facts'],view['facts'])
        self.assertEqual(revised['entities'],view['entities'])
        self.assertEqual(revised['relations'],view['relations'])
        self.assertEqual(view,before)

if __name__=='__main__':unittest.main()
