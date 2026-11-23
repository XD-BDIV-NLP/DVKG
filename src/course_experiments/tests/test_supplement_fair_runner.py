import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from course_experiments.manifest import digest,file_record
from course_experiments.evaluation.supplement_fair_runner import evaluate_all,POLICY


class FairSupplementTests(unittest.TestCase):
    def test_missing_chosen_reference_source_still_answers_from_other_valid_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def save(name,value):
                p=root/name;p.write_text(json.dumps(value,ensure_ascii=False),'utf-8');return file_record(p)
            graph=save('graph.json',dict(entities=[],relations=[]))
            row=dict(book_id='A',chunk_id='a',evidence_ids=['a'],evidence=[dict(id='a',text='缓存命中返回数据。')],text='缓存命中返回数据。',vector=[1.,0.])
            index=save('index.json',dict(rows=[row]))
            question=dict(question_id='q',question='缓存命中返回什么？',answer='数据',type='new_book_coverage',evidence_ids=['c'],citations=[dict(evidence_id='c',quote='返回数据')])
            reference=save('reference.json',dict(reference_id='ref',questions=[question]))
            phase=dict(files=dict(graph=graph,index=index,reference=reference,construction=save('construction.json',{})),parameters=dict(reference_coverage_policy=POLICY))
            phase['phase_id']=digest(phase)
            job=dict(name='supp_test',active_books=['A'],control='keyword')
            candidate=dict(answer='数据',citations=[dict(evidence_id='a',quote='返回数据')])
            def model(runtime,prompt,payload,max_tokens,*args):return (candidate if max_tokens==2048 else dict(score=1.)),{}
            with patch('course_experiments.evaluation.protocol.bind_runtime'),patch('course_experiments.evaluation.repaired_full.shared_call',side_effect=model) as calls:
                result=evaluate_all(job,phase,root/'out',SimpleNamespace(config={}),root)
            self.assertEqual(calls.call_count,2)
            self.assertEqual(result['completed_count'],1)
            self.assertEqual(result['metrics']['f1'],1.)
            self.assertEqual(result['metrics']['machine_judge_mean'],1.)
            self.assertEqual(result['reference_source_coverage'],0.)
            self.assertEqual(result['answer_attempt_coverage'],1.)
            self.assertEqual(result['unanswerable_count'],0)
            self.assertNotIn('answerability_coverage',result)

if __name__=='__main__':unittest.main()
