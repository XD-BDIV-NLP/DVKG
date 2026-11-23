import asyncio
import unittest

from course_experiments.evaluation.run_native_rag import select_gate_rows, light_callbacks


class GateTests(unittest.TestCase):
    def test_gate_keeps_complete_development_support_and_deterministic_extra_rows(self):
        rows = [dict(book_id='a',text=str(i),evidence_ids=[str(i)]) for i in range(20)]
        q = dict(target_book_id='a',question_id='dev',evidence_ids=['18','19'])
        result = select_gate_rows(rows,q,6)
        self.assertEqual(len(result),6)
        self.assertTrue({'18','19'} <= {r['evidence_ids'][0] for r in result})
        self.assertEqual(result,select_gate_rows(list(reversed(rows)),q,6))

    def test_missing_or_cross_book_source_cannot_be_a_successful_gate(self):
        with self.assertRaises(ValueError):
            select_gate_rows([dict(book_id='a',text='text',evidence_ids=['1'])],
                dict(target_book_id='b',evidence_ids=['1']),1)
        with self.assertRaises(ValueError):
            select_gate_rows([dict(book_id='a',text='text',evidence_ids=['1'])],
                dict(target_book_id='a',evidence_ids=['missing']),1)

    def test_sample_budget_does_not_silently_drop_support(self):
        rows = [dict(book_id='a',text=str(i),evidence_ids=[str(i)]) for i in range(3)]
        with self.assertRaises(ValueError):
            select_gate_rows(rows,dict(target_book_id='a',evidence_ids=['0','1']),1)

    def test_light_callbacks_preserve_native_messages_and_frozen_controls(self):
        class Provider:
            async def achat(self,messages,**kwargs):
                self.messages=messages;self.kwargs=kwargs;return 'native'
            async def aembed(self,texts,**kwargs):
                return (texts,kwargs)
        p=Provider();chat,embed=light_callbacks(p)
        value=asyncio.run(chat('question',system_prompt='native system',history_messages=[dict(role='user',content='history')],
            purpose='keyword',response_format=dict(type='json_object'),enable_cot=True))
        self.assertEqual(value,'native')
        self.assertEqual([r['content'] for r in p.messages],['native system','history','question'])
        self.assertNotIn('enable_cot',p.kwargs)
        self.assertEqual(p.kwargs['max_tokens'],4096)
        with self.assertRaises(ValueError):
            asyncio.run(chat('q',purpose='extract',model='other'))


if __name__ == '__main__':
    unittest.main()
