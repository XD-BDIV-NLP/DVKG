import tempfile
from pathlib import Path
import unittest
from course_experiments.evaluation.native_qa_length_recovery import BoundedQaLengthRecovery
from course_experiments.tests.test_native_hipporag import config,completion

class QaLengthRecoveryTests(unittest.TestCase):
 def test_one_qa_length_retry_preserves_messages_and_other_parameters(self):
  with tempfile.TemporaryDirectory() as root:
   calls=[]
   def execute(endpoint,body,stage):
    calls.append((body,stage));return completion(finish='length' if len(calls)==1 else 'stop'),False
   model=BoundedQaLengthRecovery(config(),execute,Path(root)/'calls.jsonl',role='qa_llm')
   messages=[dict(role='user',content='synthetic original source')]
   _,metadata,_=model.infer(messages,max_new_tokens=2048)
   self.assertEqual(metadata['finish_reason'],'stop');self.assertEqual(len(calls),2)
   self.assertEqual(calls[1][0],dict(calls[0][0],max_tokens=4096))
   self.assertEqual(metadata['bounded_length_recovery']['initial_finish_reason'],'length')
 def test_second_length_still_incomplete_and_extraction_never_retries(self):
  for role,count in [('qa_llm',2),('extraction_llm',1)]:
   with self.subTest(role=role),tempfile.TemporaryDirectory() as root:
    calls=[]
    def execute(endpoint,body,stage):calls.append(body);return completion(finish='length'),False
    model=BoundedQaLengthRecovery(config(),execute,Path(root)/'calls.jsonl',role=role)
    _,metadata,_=model.infer([dict(role='user',content='synthetic')])
    self.assertEqual(metadata['finish_reason'],'length');self.assertEqual(len(calls),count)

if __name__=='__main__':unittest.main()
