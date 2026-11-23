import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from course_experiments.manifest import digest,file_record
from course_experiments.evaluation import run_frozen_dual_view as run


class FrozenRunTests(unittest.TestCase):
    def test_scope_keeps_two_and_four_book_target_comparison(self):
        assets={'cohorts':{'compAB':{'active_books':['a','b']},'compABCD':{'active_books':['a','b','c','d']},
            'osAB':{'active_books':['o','p']},'netAB':{'active_books':['n','m']}}}
        jobs=run.jobs(assets)
        self.assertEqual(len(jobs),80)
        self.assertEqual(len({j['book'] for j in jobs}),8)
        self.assertEqual({j['book'] for j in jobs if j['cohort']=='compAB'},{'a','b'})

    def test_selection_is_frozen_before_independent_release(self):
        selection={'status':'frozen_development_choice','confirmation_questions_used_for_selection':0,
                   'new_holdout_not_opened':True,'selected_variant':'occurrence_mapping_v2','created_unix':10}
        self.assertEqual(run.selected_variant(selection),'occurrence_mapping_v2')
        with self.assertRaises(ValueError):run.selected_variant(dict(selection,confirmation_questions_used_for_selection=1))
        with self.assertRaises(ValueError):run.selected_variant(dict(selection,selected_variant='best_test_score'))

    def test_reuse_requires_exact_body_and_immutable_complete_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);body={'messages':[{'role':'user','content':'same'}]}
            raw={'request':body,'response':{'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answer':'yes','citations':[]})}}]}}
            path=root/'raw.json';path.write_text(json.dumps(raw));source=file_record(path)
            stored={'source':source,'request_digest':digest(body),'response_digest':digest(raw['response'])}
            with closing(sqlite3.connect(root/'request_cache.sqlite')) as db,db:
                db.execute('CREATE TABLE responses(key TEXT,record TEXT)');db.execute('INSERT INTO responses VALUES(?,?)',('key',json.dumps(stored)))
            self.assertEqual(run.old_answer(root,'key',body)[0]['answer'],'yes')
            self.assertIsNone(run.old_answer(root,'missing',body))
            with self.assertRaises(ValueError):run.old_answer(root,'key',{'messages':[]})
            path.write_text('{}')
            with self.assertRaises(ValueError):run.old_answer(root,'key',body)

    def test_unified_judge_sees_answer_and_source_text_without_method_label(self):
        class Provider:
            def chat_response(self,messages,**kwargs):
                self.messages=messages
                return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'score':.7,'grounding':.8,'reason':'ok'})}}]}
        provider=Provider()
        with tempfile.TemporaryDirectory() as folder:
            question={'question':'q','target_book_id':'b','answer':'a','citations':[]}
            judge=run.judge(provider,question,[{'text':'quoted source','book_id':'b'}],{'answer':'candidate','citations':[]},Path(folder),'id')
        self.assertEqual(judge['score'],.7)
        payload=json.loads(provider.messages[1]['content'])
        self.assertEqual(payload['candidate'],'candidate')
        self.assertEqual(payload['provided_context'],[{'context_id':'P1','text':'quoted source'}])
        self.assertNotIn('method',payload)

    def test_fixed_denominator_keeps_failed_question_and_resume_needs_no_new_answer(self):
        class Provider:
            def embed(self,*args,**kwargs):return np.ones((1,2))
        class Engine:
            rows=[];candidate_ids=['e'];structure={}
            def retrieve(self,*args):
                return [],{'fallback':False,'candidate_count':1,'total_target_units':1,'retrieval_seconds':.1}
        class Budget:
            def stage(self,name):return 'frozen_dual/test/'+name
        questions=[{'question_id':i,'question':'q','target_book_id':'b','answer':'yes','type':'definition'} for i in ['one','two']]
        metrics={k:1 for k in ['coarse_evidence_recall','selected_evidence_recall','selected_chapter_recall','evidence_recall']}
        def candidate(provider,q,*args):
            if q['question_id']=='two':raise ValueError('failed generation')
            return {'answer':'yes','citations':[]},{}
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'PARENT',Path(folder)/'old'),patch.object(run,'retriever',return_value=Engine()),\
                patch.object(run.old,'pack_context',return_value=[]),patch.object(run.old,'reference_metrics',return_value=metrics),\
                patch.object(run,'answer',side_effect=candidate) as answer,patch.object(run,'judge',return_value={'score':1,'grounding':1}):
            root=Path(folder)/'new';job={'cohort':'compAB','book':'b','method':'dual'}
            result=run.evaluate(job,{},dict(questions=questions),Provider(),root,'id','unchanged_v1',Budget())
            self.assertEqual(result['question_count'],2)
            self.assertEqual(result['metrics']['f1'],.5)
            self.assertEqual(result['failed_count'],1)
            self.assertEqual(result['status'],'partial')
            again=run.evaluate(job,{},dict(questions=questions),Provider(),root,'id','unchanged_v1',Budget())
            self.assertEqual(again,result)
            self.assertEqual(answer.call_count,2)


if __name__=='__main__':unittest.main()
