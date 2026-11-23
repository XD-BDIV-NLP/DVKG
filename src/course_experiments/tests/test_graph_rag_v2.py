import json
import math
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from course_experiments.manifest import digest
from course_experiments.evaluation.protocol import freeze_phase

try:
    from course_experiments.evaluation import graph_rag_v2 as v2
except ImportError:
    v2 = None


def row(key, text, book='b', vector=None):
    return dict(book_id=book,chunk_id=key,text=text,evidence_ids=[key],
                evidence=[dict(id=key,text=text)],vector=vector or [1.,0.])


def entity(key, name, aliases=None, sources=None):
    return dict(entity_id=key,canonical_name=name,type='Concept',
                aliases=aliases or [],source_node_ids=sources or [])


def edge(key, head, tail, sources=None):
    return dict(relation_id=key,head_entity_id=head,tail_entity_id=tail,
                relation='affects',source_node_ids=sources or [])


class EvidenceGraphRagTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(v2, 'Evidence graph retriever is not implemented')

    def test_alias_links_without_canonical_substring(self):
        r=v2.EvidenceRetriever([row('e','命中率影响访问时间')],dict(
            entities=[entity('x','高速缓冲存储器',['Cache'],['course/b/evidence/e'])],relations=[]))
        rows,trace=r.retrieve('CACHE 有什么作用','graph')
        self.assertEqual(trace['seed_ids'],['x'])
        self.assertEqual(rows[0]['chunk_id'],'e')

    def test_latin_alias_does_not_match_inside_word(self):
        r=v2.EvidenceRetriever([row('e','program')],dict(entities=[entity('x','RAM')],relations=[]))
        _,trace=r.retrieve('program 的功能','graph')
        self.assertEqual(trace['seed_ids'],[])
        self.assertEqual(trace['fallback'],'keyword_no_seed')

    def test_two_hop_navigation_uses_relation_evidence(self):
        g=dict(entities=[entity('x','缓存'),entity('y','命中率'),entity('z','访问时间'),entity('w','额外节点')],
               relations=[edge('r1','x','y',['e1']),edge('r2','y','z',['e2']),edge('r3','z','w',['e3'])])
        r=v2.EvidenceRetriever([row('e1','缓存命中率'),row('e2','命中率影响访问时间'),row('e3','第三跳')],g)
        rows,trace=r.retrieve('缓存如何影响访问时间','graph')
        self.assertIn('e2',[x['chunk_id'] for x in rows])
        self.assertTrue(any(len(p['relation_ids'])==2 for p in trace['paths']))
        self.assertTrue(all(len(p['relation_ids'])<=2 for p in trace['paths']))
        self.assertTrue(all(p['directions'] for p in trace['paths']))

    def test_graph_namespace_requires_matching_book(self):
        r=v2.EvidenceRetriever([row('e','缓存解释')],dict(
            entities=[entity('x','缓存',sources=['course/foreign/evidence/e'])],relations=[]))
        _,trace=r.retrieve('缓存','graph')
        self.assertEqual(trace['source_anchored_candidates'],0)

    def test_prior_without_sources_is_explicit_lexical_navigation(self):
        g=dict(entities=[entity('x','缓存'),entity('y','命中率')],relations=[edge('r','x','y')])
        r=v2.EvidenceRetriever([row('e','命中率影响时间')],g)
        rows,trace=r.retrieve('缓存','graph')
        self.assertEqual(trace['source_anchored_candidates'],0)
        self.assertEqual(trace['navigation_candidates'],1)
        self.assertEqual(rows[0]['chunk_id'],'e')
        self.assertNotIn('relations',v2.pack_context(rows,'缓存')[0])

    def test_no_graph_has_exact_keyword_fallback(self):
        r=v2.EvidenceRetriever([row('a','缓存原理'),row('b','网络协议')],dict(entities=[],relations=[]))
        graph,trace=r.retrieve('缓存','graph')
        keyword,_=r.retrieve('缓存','keyword')
        self.assertEqual([x['chunk_id'] for x in graph],[x['chunk_id'] for x in keyword])
        self.assertEqual(trace['fallback'],'keyword_no_seed')

    def test_vector_does_not_receive_graph_bonus(self):
        r=v2.EvidenceRetriever([row('a','direct answer',vector=[.9,math.sqrt(.19)]),
                               row('b','ALPHA BETA background',vector=[.8,.6])],
                              dict(entities=[entity('x','ALPHA')],relations=[]))
        rows,_=r.retrieve('ALPHA','vector',[1.,0.])
        self.assertEqual(rows[0]['chunk_id'],'a')

    def test_balanced_context_includes_later_passages(self):
        rows=[row(str(i),'前文'*2000+'缓存的证据'+str(i)+'后文'*2000) for i in range(5)]
        c=v2.pack_context(rows,'缓存')
        self.assertEqual(len(c),5)
        self.assertLessEqual(len(json.dumps(c,ensure_ascii=False).encode()),3000)
        self.assertTrue(all('缓存的证据' in x['text'] for x in c))
        self.assertTrue(all(x['text'] in rows[i]['text'] for i,x in enumerate(c)))

    def test_deterministic_on_graph_permutation(self):
        rows=[row('a','缓存甲'),row('b','缓存乙')]
        ents=[entity('x','缓存',sources=['a']),entity('y','甲',sources=['b'])]
        r1=v2.EvidenceRetriever(rows,dict(entities=ents,relations=[edge('r','x','y',['b'])]))
        r2=v2.EvidenceRetriever(list(reversed(rows)),dict(entities=list(reversed(ents)),relations=[edge('r','x','y',['b'])]))
        a,t1=r1.retrieve('缓存','graph');b,t2=r2.retrieve('缓存','graph')
        self.assertEqual([x['chunk_id'] for x in a],[x['chunk_id'] for x in b])
        self.assertEqual(t1,t2)

    def test_retriever_accepts_question_string_only(self):
        r=v2.EvidenceRetriever([row('a','原文')],dict(entities=[],relations=[]))
        with self.assertRaises(TypeError):r.retrieve(dict(question='问题',answer='答案'),'graph')

    def test_audit_measures_missing_evidence_after_retrieval(self):
        refs=dict(citations=[dict(evidence_id='a',quote='正确证据')])
        rows=[row('a','无关前文'*1500+'正确证据')]
        audit=v2.retrieval_audit(refs,rows,v2.pack_context(rows,'无关'))
        self.assertEqual(audit['evidence_recall_before_packing'],1.)
        self.assertEqual(audit['evidence_recall'],0.)

    def test_many_source_ids_do_not_consume_entire_context_quota(self):
        rows=[]
        for i in range(5):
            r=row('chunk'+str(i),'缓存是证据')
            r['evidence']=[dict(id=str(i)+'_'+str(j)+'x'*64,text='背景材料' if j else '缓存是证据') for j in range(30)]
            r['evidence_ids']=[e['id'] for e in r['evidence']]
            rows.append(r)
        context=v2.pack_context(rows,'缓存')
        self.assertEqual(len(context),5)
        self.assertTrue(all(len(c['evidence_ids'])==1 for c in context))
        self.assertTrue(all(c['text']=='缓存是证据' for c in context))
        self.assertLessEqual(len(json.dumps(context,ensure_ascii=False).encode('utf-8')),3000)


class EvaluationTests(unittest.TestCase):
    def run_evaluation(self, folder, responses, *, stop_after=None):
        folder=Path(folder)
        index=dict(rows=[row('e','缓存是教材证据')],embedding_model='BAAI/bge-m3')
        index['index_id']=digest(index)
        graph=dict(entities=[entity('x','缓存',sources=['e'])],relations=[])
        qs=[dict(question_id='q'+str(i),question='缓存是什么'+str(i),answer='缓存是教材证据',
                 type='definition',evidence_ids=['e'],citations=[dict(evidence_id='e',quote='缓存是教材证据')]) for i in range(2)]
        reference=dict(status='complete',human_audited=False,targets=dict(questions=1,facts=1),
                       course_id='synthetic',facts=[{}],questions=qs,failed_chunks=[])
        reference['reference_id']=digest(reference)
        config=dict(llm=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',extra_body=dict(enable_thinking=False)),
                    embedding=dict(base_url='https://api.siliconflow.cn/v1',model='BAAI/bge-m3'))
        files={}
        for name,value in [('index',index),('graph',graph),('reference',reference),('runtime_config',config)]:
            p=folder/(name+'.json');p.write_text(json.dumps(value,ensure_ascii=False),'utf-8');files[name]=p
        phase=folder/'phase.json'
        if not phase.exists():freeze_phase(files,dict(v2.BUDGET,control='graph',active_books=['b']),phase)
        runtime=SimpleNamespace(config=config)
        with patch.object(v2,'call_json',side_effect=responses) as call,patch.object(v2.time,'sleep'):
            result=v2.evaluate(reference,index,graph,['b'],'graph',phase,folder/'out',runtime,stop_after=stop_after)
        return result,call.call_count

    def answer(self):
        return dict(answer='缓存是教材证据',citations=[dict(evidence_id='e',quote='缓存是教材证据')]),{}

    def judge(self):
        return dict(score=.8),{}

    def test_transient_exhaustion_records_failure_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.run_evaluation(tmp,[RuntimeError('Model service returned HTTP 503')]*3+[self.answer(),self.judge()])
        self.assertEqual(result['completed_count'],1)
        self.assertEqual(result['failed_questions'],['q0'])
        self.assertEqual(calls,5)
        self.assertEqual(result['status'],'partial')

    def test_account_failure_stops_without_consuming_next_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'HTTP 402'):
                self.run_evaluation(tmp,[RuntimeError('Model service returned HTTP 402')])

    def test_smoke_reuses_checkpoint_for_full_run_and_summary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            smoke,calls=self.run_evaluation(tmp,[self.answer(),self.judge()],stop_after=1)
            self.assertEqual(smoke['status'],'smoke_completed')
            self.assertEqual(calls,2)
            result,calls=self.run_evaluation(tmp,[self.answer(),self.judge()])
            self.assertEqual(calls,2)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['metrics']['citation_correctness'],1.)
            self.assertEqual(result['by_type']['definition']['seed_hit_count'],2)

    def test_modified_checkpoint_is_rejected_before_new_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_evaluation(tmp,[self.answer(),self.judge()],stop_after=1)
            p=next((Path(tmp)/'out').glob('*/questions/q0.json'))
            r=json.loads(p.read_text('utf-8'));r['machine_judge']['score']=1.
            p.write_text(json.dumps(r,ensure_ascii=False),'utf-8')
            with self.assertRaisesRegex(ValueError,'Checkpoint content changed'):
                self.run_evaluation(tmp,[])


class RunnerGateTests(unittest.TestCase):
    def test_incomplete_matrix_rejected_before_prepare_mutations(self):
        from course_experiments.evaluation.run_graph_rag_v2 import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'sources.json';source.write_text('[]','utf-8')
            with self.assertRaisesRegex(ValueError,'twenty-unit matrix'):
                prepare(root,source)
            self.assertFalse((root/'contract.json').exists())

    def test_shared_http_cap_blocks_next_request(self):
        from course_experiments.evaluation import run_graph_rag_v2 as driver
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);db=sqlite3.connect(root/'limits.sqlite')
            db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,created REAL,status TEXT,stage TEXT,code INTEGER)');db.close()
            runtime=SimpleNamespace(_record=lambda *args:None)
            with patch.dict(driver.LIMITS,max_http_attempts=1):
                driver.limit_runtime(root,runtime)
                runtime._record('a','test','request_started',{})
                with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                    runtime._record('b','test','request_started',{})

    def test_account_error_stops_other_worker_requests(self):
        from course_experiments.evaluation import run_graph_rag_v2 as driver
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);db=sqlite3.connect(root/'limits.sqlite')
            db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,created REAL,status TEXT,stage TEXT,code INTEGER)');db.close()
            runtime=SimpleNamespace(_record=lambda *args:None);driver.limit_runtime(root,runtime)
            runtime._record('a','test','http_error',dict(status_code=402))
            with self.assertRaisesRegex(RuntimeError,'halted'):
                runtime._record('b','test','request_started',{})


if __name__=='__main__':unittest.main()
