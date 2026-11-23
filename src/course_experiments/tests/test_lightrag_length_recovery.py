import asyncio,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from course_experiments.evaluation.lightrag_length_recovery import callbacks

class LengthRecoveryTests(unittest.TestCase):
 def test_authorized_manual_resume_does_not_enqueue_processed_corpus_again(self):
  from course_experiments.evaluation.native_lightrag import NativeLightRAG,SourceDocument,_digest
  with tempfile.TemporaryDirectory() as root:
   folder=Path(root);insert=AsyncMock(return_value='original-track')
   core=SimpleNamespace(ainsert=insert,doc_status=SimpleNamespace(get_by_ids=AsyncMock(return_value=[dict(status='processed')])))
   adapter=NativeLightRAG(core,None,folder,'book');docs=[SourceDocument('doc','book','synthetic source')]
   asyncio.run(adapter.index_book('book',docs));manifest=json.loads((folder/'native_adapter_manifest.json').read_text())
   (folder/'native_resume_authorization.json').write_text(json.dumps(dict(kind='official_manual_retry_after_callback_timeout',target_book_id='book',manifest_digest=_digest(manifest),document_ids=['doc-'+_digest(['book','doc'])])))
   adapter._retry_failed_documents=AsyncMock(return_value='manual-track')
   result=asyncio.run(adapter.index_book('book',docs))
   self.assertEqual(adapter._retry_failed_documents.await_count,1);self.assertEqual(insert.await_count,1);self.assertEqual(result['status'],'processed')
 def provider(self,reasons):
  class Provider:
   calls=[]
   def chat_response(self,messages,**kwargs):
    self.calls.append((messages,kwargs));return dict(choices=[dict(message=dict(content='synthetic'),finish_reason=reasons[len(self.calls)-1])])
   async def aembed(self,*args,**kwargs):return 'synthetic-vector'
  return Provider()
 def test_only_extract_length_retries_same_messages_once_at8192(self):
  p=self.provider(['length','stop']);chat,_=callbacks(p)
  self.assertEqual(asyncio.run(chat('source',purpose='lightrag.book.extract')),'synthetic')
  self.assertEqual([k['max_tokens'] for _,k in p.calls],[4096,8192]);self.assertEqual(p.calls[0][0],p.calls[1][0])
 def test_second_length_stops_instead_of_claiming_processed(self):
  p=self.provider(['length','length']);chat,_=callbacks(p)
  with self.assertRaises(ValueError):asyncio.run(chat('source',purpose='lightrag.book.extract'))
  self.assertEqual(len(p.calls),2)
 def test_query_length_and_other_finish_do_not_get_extra_attempts(self):
  for stage,reason in [('lightrag.book.query','length'),('lightrag.book.extract','content_filter')]:
   p=self.provider([reason]);chat,_=callbacks(p)
   with self.assertRaises(ValueError):asyncio.run(chat('source',purpose=stage))
   self.assertEqual(len(p.calls),1)
