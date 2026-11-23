import asyncio
from types import SimpleNamespace
import unittest
from course_experiments.evaluation.lightrag_exact_length_recovery import callbacks
from course_experiments.manifest import digest

class ExactLengthRecoveryTests(unittest.TestCase):
 def provider(self,reasons):
  class Provider:
   def __init__(self):self.runtime=SimpleNamespace(config={'llm':{'model':'synthetic'}});self.calls=[]
   def chat_response(self,messages,**options):
    self.calls.append((messages,options));return dict(choices=[dict(message=dict(content='synthetic'),finish_reason=reasons[len(self.calls)-1])])
   async def aembed(self,*args,**kwargs):return 'vectors'
  return Provider()
 def request_id(self,p,prompt):
  body=dict(model='synthetic',messages=[dict(role='user',content=prompt)],temperature=0,seed=42,enable_thinking=False,max_tokens=8192)
  return digest(dict(endpoint='/chat/completions',body=body,config=p.runtime.config))
 def test_only_exact_known_8192_length_gets_one_16384_request(self):
  p=self.provider(['length','length','stop']);chat,_=callbacks(p,{self.request_id(p,'known source')})
  self.assertEqual(asyncio.run(chat('known source',purpose='lightrag.book.extract')),'synthetic')
  self.assertEqual([o['max_tokens'] for m,o in p.calls],[4096,8192,16384])
  self.assertEqual(p.calls[2][0],p.calls[1][0])
 def test_foreign_message_and_second_length_remain_failures(self):
  for prompt,count in [('foreign source',2),('known source',3)]:
   p=self.provider(['length','length','length']);chat,_=callbacks(p,{self.request_id(p,'known source')})
   with self.assertRaises(ValueError):asyncio.run(chat(prompt,purpose='lightrag.book.extract'))
   self.assertEqual(len(p.calls),count)

if __name__=='__main__':unittest.main()
