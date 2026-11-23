from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from course_experiments.evaluation import run_native_full as run
from course_experiments.evaluation.native_provider import NativeBudgetExceeded
from course_experiments.manifest import digest


def question(identifier):
    return dict(question_id=identifier, question='question_' + identifier,
        target_book_id='book_a', type='definition', answer='reference_secret',
        evidence_ids=['e1'], chapter_ids=['chapter'], citations=[dict(evidence_id='e1', quote='source')])


def plan(method='hipporag', count=2):
    passages=[dict(book_id='book_a',chunk_id='source_chunk',text='source evidence',
        evidence_ids=['e1'],evidence_spans=[dict(evidence_id='e1',start=0,end=15)],chapter_id='chapter')]
    contract=dict(method=method,book='book_a',fingerprint='frozen',policy=run.POLICY)
    return dict(method=method,book='book_a',passages=passages,
        structure=dict(evidence={'e1':dict(text='source evidence',chunk_id='source_chunk',
            chapter_id='chapter',source=dict(paragraph_order=0))}),
        datasets=dict(development=dict(questions=[question('dev')]),
            confirmation=dict(questions=[question(str(i)) for i in range(count)])),
        source=Path('synthetic_source'),contract=contract,index_contract=dict(method=method,book='book_a'),
        cohort_memberships=['compAB','compABCD'],allowed_books=['book_a'])


class FakeProvider:
    def __init__(self,events, fatal_after=None):
        self.events=events;self.fatal_after=fatal_after;self.calls=[];self.fatal=False
    def statistics(self):return dict(http_attempts=len(self.calls))
    def raise_if_fatal(self):
        if self.fatal:raise NativeBudgetExceeded('synthetic budget exhausted')
    def chat_response(self,messages,**kwargs):
        self.calls.append((messages,kwargs))
        if self.fatal_after is not None and len(self.calls)>self.fatal_after:
            self.fatal=True;raise NativeBudgetExceeded('synthetic budget exhausted')
        return dict(choices=[dict(message=dict(content=json.dumps(dict(score=.75,grounding=1.,reason='synthetic'))),
            finish_reason='stop')],usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2))
    def close(self):self.events.append('provider_close')


@dataclass
class SyntheticSolution:
    question: str
    docs: list
    doc_scores: list
    doc_metadata: list
    graph_seeds: list
    answer: str | None = None


class FakeHippo:
    QuerySolution=SyntheticSolution
    config=SimpleNamespace(qa_top_k=5,retrieval_top_k=200)
    def __init__(self,events,fail=None,length=None):
        self.events=events;self.fail=fail;self.length=length;self.retrieved=[]
    def index(self,rows):
        self.events.append('index');self.rows=copy.deepcopy(rows)
        return dict(status='indexed',upstream_commit='synthetic')
    def retrieve(self,queries):
        self.retrieved.extend(queries)
        if queries[0]==self.fail:raise TimeoutError('synthetic timeout')
        return [SyntheticSolution(question=queries[0],docs=['source evidence'],doc_scores=[1.],
            doc_metadata=[dict(book_id='book_a',evidence_ids=['e1'])],graph_seeds=[['a','r','b']])]
    def rag_qa(self,solutions):
        q=solutions[0];q.answer='candidate answer'
        return [q],['Answer: candidate answer'],[dict(finish_reason='length' if q.question==self.length else 'stop')]
    def metadata(self):return dict(native_retrieval_top_k=200,native_qa_top_k=5)
    def close(self):self.events.append('adapter_close')


class FakeLight:
    def __init__(self,events):self.events=events;self.queries=[]
    async def index_book(self,book,documents):
        self.events.append('index');self.documents=documents
        return dict(source_map={'native_path':digest(['e1'])},status='processed')
    async def query_native(self,book,q):
        self.queries.append(q)
        return dict(track='native_end_to_end',native_result=dict(status='success',
            data=dict(chunks=[dict(chunk_id='light_chunk',file_path='native_path',content='source')]),
            llm_response=dict(content='candidate answer')))
    async def close(self):self.events.append('adapter_close')


class LightBoundaryTests(unittest.TestCase):
    def test_official_split_utf8_window_maps_only_complete_original_characters(self):
        from unittest.mock import patch
        class Encoding:
            def encode(self, text): return list(text.encode('utf8'))
            def decode_single_token_bytes(self, token): return bytes([token])
            def decode(self, tokens): return bytes(tokens).decode('utf8', errors='replace')
        encoding = Encoding()
        source = 'a' * 1199 + '汉' + 'b' * 1200
        native = encoding.decode(encoding.encode(source)[1100:2300]).strip()
        work = plan('lightrag'); work['passages'][0].update(text=source,
            evidence_spans=[dict(evidence_id='e1', start=0, end=len(source))])
        receipt = dict(source_map={'native_path':digest(['e1'])})
        result = dict(native_result=dict(data=dict(chunks=[dict(chunk_id='c', file_path='native_path', content=native)])))
        # 首窗口在汉字中间结束，官方文本保留替代字符；引用只能取完整字符。
        result['native_result']['data']['chunks'][0]['content'] = encoding.decode(encoding.encode(source)[:1200])
        with patch('tiktoken.encoding_for_model', return_value=encoding):
            rows = run.light_context(work, result, receipt)
        self.assertEqual(rows[0]['text'], 'a' * 1199)
        self.assertEqual(rows[0]['native_content'], 'a' * 1199 + '\ufffd')
        self.assertTrue(rows[0]['native_token_windows'][0]['right_partial_character'])
        self.assertEqual(rows[0]['positions'][0]['passage_end'],1199)

    def test_official_trailing_overlap_window_is_also_mapped(self):
        from unittest.mock import patch
        class Encoding:
            def encode(self,text):return list(text.encode('utf8'))
            def decode_single_token_bytes(self,token):return bytes([token])
            def decode(self,tokens):return bytes(tokens).decode('utf8',errors='replace')
        encoding=Encoding();source='a'*1099+chr(0x6c49)+'b'*48
        native=encoding.decode(encoding.encode(source)[1100:])
        work=plan('lightrag');work['passages'][0].update(text=source,
            evidence_spans=[dict(evidence_id='e1',start=0,end=len(source))])
        result=dict(native_result=dict(data=dict(chunks=[dict(chunk_id='tail',file_path='native_path',content=native)])))
        with patch('tiktoken.encoding_for_model',return_value=encoding):
            rows=run.light_context(work,result,dict(source_map={'native_path':digest(['e1'])}))
        self.assertEqual(rows[0]['text'],'b'*48)
        self.assertEqual(rows[0]['native_token_windows'][0]['token_start'],1100)
        self.assertTrue(rows[0]['native_token_windows'][0]['left_partial_character'])

    def test_ambiguous_window_interiors_are_rejected(self):
        from unittest.mock import patch
        work=plan('lightrag');receipt=dict(source_map={'native_path':digest(['e1'])})
        result=dict(native_result=dict(data=dict(chunks=[dict(chunk_id='c',file_path='native_path',content='unmatched')])) )
        windows=[dict(native_text='unmatched',passage_start=0,passage_end=6),
                 dict(native_text='unmatched',passage_start=7,passage_end=15)]
        with patch('course_experiments.evaluation.native_microsoft_spans.token_windows',return_value=windows):
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                run.light_context(work,result,receipt)

    def test_unproved_middle_replacement_and_foreign_source_are_rejected(self):
        work=plan('lightrag');receipt=dict(source_map={'native_path':digest(['e1'])})
        for path,text in [('native_path','source \ufffdvidence'),('foreign','source')]:
            result=dict(native_result=dict(data=dict(chunks=[dict(chunk_id='c',file_path=path,content=text)])))
            with self.assertRaises(ValueError): run.light_context(work,result,receipt)


class NativeFullTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def execute(self,work,adapter,provider):
        async def factory(*args):return adapter
        return asyncio.run(run.run_book(work,self.root,provider,adapter_factory=factory))

    def test_all_questions_native_only_gold_is_blind_judge_only_and_resume_skips_success(self):
        work=plan();events=[];adapter=FakeHippo(events);provider=FakeProvider(events)
        result=self.execute(work,adapter,provider)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['splits']['confirmation']['question_count'],2)
        self.assertEqual(result['splits']['development']['question_count'],1)
        self.assertEqual(len(adapter.retrieved),3)
        self.assertNotIn('reference_secret',json.dumps(adapter.rows))
        self.assertTrue(all('reference_secret' in r[0][-1]['content'] for r in provider.calls))
        self.assertTrue(all('hipporag' not in r[0][-1]['content'].lower() for r in provider.calls))
        self.assertEqual(events[-2:],['adapter_close','provider_close'])
        records=list((Path(result['root'])/'confirmation').glob('*/record.json'))
        record=json.loads(records[0].read_text('utf-8'))
        self.assertIsNone(record['scores']['citation_correctness'])
        self.assertEqual(record['provided_context'][0]['positions'][0]['chunk_id'],'source_chunk')
        second_events=[];second_adapter=FakeHippo(second_events);second_provider=FakeProvider(second_events)
        reused=self.execute(work,second_adapter,second_provider)
        self.assertEqual(reused['status'],'completed')
        self.assertEqual(second_adapter.retrieved,[])
        self.assertEqual(second_provider.calls,[])
        self.assertNotIn('index',second_events)

    def test_failed_and_length_questions_remain_in_fixed_denominator(self):
        events=[];adapter=FakeHippo(events,fail='question_0',length='question_1')
        result=self.execute(plan(),adapter,FakeProvider(events))
        split=result['splits']['confirmation']
        self.assertEqual(result['status'],'partial')
        self.assertEqual(split['question_count'],2)
        self.assertEqual(split['failed_count'],2)
        self.assertEqual(split['metrics']['machine_judge_mean'],0.)
        self.assertEqual(split['metrics']['f1'],0.)
        self.assertEqual(len(list(Path(result['root']).rglob('*.log'))),2)

    def test_fatal_budget_preserves_state_and_closes_adapter_before_provider(self):
        events=[];provider=FakeProvider(events,fatal_after=1)
        with self.assertRaises(NativeBudgetExceeded):self.execute(plan(),FakeHippo(events),provider)
        states=list(self.root.rglob('state.json'))
        state=json.loads(states[-1].read_text('utf-8'))
        self.assertEqual(state['status'],'halted')
        self.assertEqual(state['summary']['splits']['confirmation']['question_count'],2)
        self.assertEqual(events[-2:],['adapter_close','provider_close'])

    def test_light_uses_full_native_chunks_without_a_second_retrieval_or_budget_truncation(self):
        events=[];adapter=FakeLight(events)
        result=self.execute(plan('lightrag',1),adapter,FakeProvider(events))
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(adapter.queries),2)
        record=json.loads(next((Path(result['root'])/'confirmation').glob('*/record.json')).read_text('utf-8'))
        self.assertEqual(record['provided_context'][0]['text'],'source')
        self.assertEqual(record['provided_context'][0]['positions'][0]['passage_start'],0)
        self.assertEqual(record['provided_context'][0]['positions'][0]['passage_end'],6)
        self.assertEqual(record['native_result']['native_result']['data']['chunks'][0]['chunk_id'],'light_chunk')

    def test_completed_checkpoint_tampering_is_detected_before_new_calls(self):
        events=[];result=self.execute(plan(),FakeHippo(events),FakeProvider(events))
        path=next((Path(result['root'])/'confirmation').glob('*/record.json'))
        value=json.loads(path.read_text('utf-8'));value['answer']='changed';path.write_text(json.dumps(value))
        other_events=[];provider=FakeProvider(other_events)
        with self.assertRaises(ValueError):self.execute(plan(),FakeHippo(other_events),provider)
        self.assertEqual(provider.calls,[])

    def test_judge_validation_rejects_nonfinite_scores(self):
        with self.assertRaises(ValueError):run.validate_judgment(dict(score=float('nan'),grounding=1.,reason='x'))
        with self.assertRaises(ValueError):run.validate_judgment(dict(score=1.,grounding=2.,reason='x'))

    def test_index_identity_uses_content_not_deployment_paths(self):
        def signature(prefix):
            return run.index_contract_for('hipporag','book_a',
                dict(path=prefix+'/sources/HippoRAG',commit='a'*40,repository='official'),
                dict(path=prefix+'/units.json',sha256='b'*64),
                dict(path=prefix+'/structure.json',sha256='c'*64),dict(all_evidence_covered=True),
                dict(llm=dict(model='same')),
                {'native_hipporag.py':dict(path=prefix+'/native_hipporag.py',sha256='d'*64),
                 'unrelated_graph.py':dict(path=prefix+'/unrelated_graph.py',sha256=prefix)})
        self.assertEqual(digest(signature('/deployment/one')),digest(signature('/deployment/two')))

    def test_interrupted_initial_hippo_index_uses_official_graph_rebuild_only(self):
        adapter=FakeHippo([])
        store=lambda texts:SimpleNamespace(get_all_ids=lambda:list(range(len(texts))),get_all_texts=lambda:texts)
        adapter.native=SimpleNamespace(_graph_state_available=False,_graph_edge_schema_available=True,
            chunk_embedding_store=store(['source evidence']),entity_embedding_store=store([]),
            fact_embedding_store=store([]),graph=SimpleNamespace(vcount=lambda:0),
            global_config=SimpleNamespace(force_index_from_scratch=False,force_openie_from_scratch=False))
        original=adapter.index
        def index(rows):
            if not adapter.native.global_config.force_index_from_scratch:
                raise RuntimeError('official missing graph guard')
            return original(rows)
        adapter.index=index
        receipt=asyncio.run(run._index(adapter,plan(),recovery_path=self.root/'recovery.json'))
        self.assertTrue(receipt['index_recovery']['official_force_index_from_scratch'])
        self.assertEqual(receipt['index_recovery']['cached_chunk_count'],1)
        self.assertFalse(adapter.native.global_config.force_index_from_scratch)
        self.assertTrue((self.root/'recovery.json').exists())
        adapter.native.entity_embedding_store=store(['already indexed entity'])
        with self.assertRaisesRegex(RuntimeError,'official missing graph guard'):
            asyncio.run(run._index(adapter,plan()))
        adapter.native.entity_embedding_store=store([])
        adapter.native.chunk_embedding_store=store(['foreign source'])
        with self.assertRaisesRegex(ValueError,'outside the frozen target corpus'):
            asyncio.run(run._index(adapter,plan()))


@unittest.skipUnless(os.environ.get('HIPPORAG_NATIVE_SOURCE'), 'Pinned upstream dependency integration is opt-in')
class NativeUpstreamIndexRecoveryTests(unittest.TestCase):
    def test_actual_upstream_missing_graph_guard_and_cached_vector_rebuild(self):
        from course_experiments.evaluation.native_hipporag import NativeHippoRAG
        calls=[]
        fail_ner=True
        def execute(endpoint,body,stage):
            calls.append((endpoint,body))
            if endpoint=='/embeddings':
                return dict(data=[dict(index=i,embedding=[1.]+[0.]*1023)
                    for i in range(len(body['input']))],usage=dict(prompt_tokens=1,total_tokens=1)),False
            if fail_ner:
                raise TimeoutError('synthetic interruption after chunk vectors')
            content='{"triples":[["source","supports","evidence"]]}' if body['max_tokens']==4096 else '{"named_entities":["source","evidence"]}'
            return dict(choices=[dict(message=dict(content=content),finish_reason='stop')],
                usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2)),False
        with tempfile.TemporaryDirectory() as folder:
            kwargs=dict(source_root=os.environ['HIPPORAG_NATIVE_SOURCE'],save_dir=folder,
                target_book_id='book_a',allowed_book_ids={'book_a'},request_executor=execute)
            first=NativeHippoRAG(**kwargs)
            try:
                with self.assertRaises(TimeoutError):first.index(plan()['passages'])
            finally:first.close()
            second=NativeHippoRAG(**kwargs)
            try:
                with self.assertRaisesRegex(Exception,'Graph state is missing'):
                    second.index(plan()['passages'])
                fail_ner=False
                receipt=asyncio.run(run._index(second,plan(),recovery_path=Path(folder)/'recovery.json'))
                self.assertTrue(receipt['index_recovery']['official_force_index_from_scratch'])
                self.assertTrue(second.native._graph_state_available)
                self.assertGreater(second.native.graph.vcount(),0)
                self.assertEqual(sum(body['input']==['source evidence'] for endpoint,body in calls
                    if endpoint=='/embeddings'),1)
                self.assertFalse(second.native.global_config.force_index_from_scratch)
            finally:second.close()


if __name__=='__main__':unittest.main()
