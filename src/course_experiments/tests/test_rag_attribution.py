import sys
import unittest

from course_experiments.evaluation.graph_rag_v2 import EvidenceRetriever
from course_experiments.tests.test_graph_rag_v2 import edge, entity, row

try:
    from course_experiments.evaluation import rag_attribution as audit
except ImportError:
    audit = None


class RagAttributionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, 'Missing independent attribution module')

    def test_stage_distinguishes_ranking_source_unit_and_window_loss(self):
        q = dict(citations=[dict(evidence_id='a', quote='目标原句'),
                            dict(evidence_id='b', quote='正确原句'),
                            dict(evidence_id='c', quote='关键原句'),
                            dict(evidence_id='d', quote='最后原句')])
        corpus = [row(k, text) for k, text in [('a','目标原句'),('b','正确原句'),
                                               ('c','关键原句'),('d','最后原句')]]
        selected = corpus[1:]
        context = [dict(corpus[2], text='关键'), corpus[3]]
        result = audit.citation_stages(q, corpus, corpus, selected, context)
        self.assertEqual(result['counts'], dict(total=4, corpus=4, candidates=4,
            selected=3, packed_source=2, packed_quote=1))
        self.assertEqual(result['loss_counts'], dict(corpus_missing=0, candidate_missing=0,
            ranking=1, source_unit_selection=1, window=1, survived=1))

    def test_original_retrieval_and_profiler_preserved(self):
        r = EvidenceRetriever([row('e1','缓存命中率'), row('e2','命中率影响时间')],
            dict(entities=[entity('x','缓存',sources=['e1']), entity('y','时间',sources=['e2'])],
                 relations=[edge('r','x','y',['e2'])]))
        expected = r.retrieve('缓存的时间','graph')
        previous = sys.getprofile()
        selected, trace, detail = audit.traced_retrieve(r, '缓存的时间')
        self.assertIs(sys.getprofile(), previous)
        self.assertEqual((selected,trace), expected)
        self.assertEqual(set(detail['candidate_indices']), {0,1})

    def test_no_merge_preserves_original_endpoints_and_source_identity(self):
        native = dict(graph=dict(nodes={'x':dict(id='x',name='缓存',entity_type='Concept',evidence_ids=['e']),
                                       'y':dict(id='y',name='主存',entity_type='Concept',evidence_ids=['f'])},
            claims={'r':dict(id='r',status='active',evidence_ids=['e'],content=dict(kind='relation',head='x',tail='y',relation='part_of'))}))
        g = audit.native_graph(dict(native_contributions={'b':native}, evidence={'e':dict(id='e',text='原文'),'f':dict(id='f',text='原文')}))
        self.assertEqual({e['entity_id'] for e in g['entities']}, {'x','y'})
        self.assertEqual(g['relations'][0]['head_entity_id'],'x')
        self.assertEqual(g['relations'][0]['tail_entity_id'],'y')

    def test_quote_in_different_evidence_is_not_a_hit(self):
        q = dict(citations=[dict(evidence_id='a',quote='同一原句')])
        r = [row('b','同一原句')]
        result = audit.citation_stages(q,r,r,r,r)
        self.assertEqual(result['counts']['packed_quote'],0)
        self.assertEqual(result['loss_counts']['corpus_missing'],1)


if __name__ == '__main__':
    unittest.main()
