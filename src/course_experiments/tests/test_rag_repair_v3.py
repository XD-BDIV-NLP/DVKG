import json
import unittest

from course_experiments.tests.test_graph_rag_v2 import row,entity

try:
    from course_experiments.evaluation import rag_repair_v3 as repair
except ImportError:
    repair=None


class RagRepairTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(repair,'Missing independent repair implementation')

    def test_multiple_relevant_units_in_one_chunk_can_survive(self):
        value=row('c','缓存映射的方式。缓存替换的策略。')
        value['evidence_ids']=['a','b']
        value['evidence']=[dict(id='a',text='缓存映射的方式。'),dict(id='b',text='缓存替换的策略。')]
        context=repair.pack_context([value],'缓存映射与缓存替换')
        self.assertEqual({eid for r in context for eid in r['evidence_ids']},{'a','b'})

    def test_actual_byte_budget_is_used_without_prefix_cut(self):
        value=row('c','无关内容。'*300+'缓存一致性依靠写失效机制。'+'无关内容。'*300)
        context=repair.pack_context([value],'缓存一致性写失效',limit=700)
        self.assertLessEqual(len(json.dumps(context,ensure_ascii=False).encode()),700)
        self.assertIn('缓存一致性依靠写失效机制。',context[0]['text'])

    def test_graph_cannot_exclude_lexically_relevant_source(self):
        rows=[row('wrong','缓存背景介绍'),row('correct','缓存一致性通过写失效保持')]
        graph=dict(entities=[entity('x','缓存',sources=['wrong'])],relations=[])
        retriever=repair.RepairedRetriever(rows,graph)
        selected,trace=retriever.retrieve('缓存一致性写失效')
        self.assertEqual(selected[0]['chunk_id'],'correct')
        self.assertIn('correct',trace['lexical_candidate_chunk_ids'])

    def test_no_reference_fields_in_retriever_interface(self):
        import inspect
        self.assertEqual(list(inspect.signature(repair.RepairedRetriever.retrieve).parameters),['self','question'])

    def test_escaped_source_text_counts_serialized_bytes(self):
        value=row('c',('缓存\\\n"'*500))
        context=repair.pack_context([value],'缓存',limit=700)
        self.assertLessEqual(len(json.dumps(context,ensure_ascii=False).encode()),700)


if __name__=='__main__':unittest.main()
