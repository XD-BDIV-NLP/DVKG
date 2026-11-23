from __future__ import annotations
import tempfile
import unittest
import ast
import asyncio
import json
import copy
import sys
import threading
from pathlib import Path
from unittest.mock import patch

from course_experiments.baselines.protocol import source_tree, validate_parameters, semantic_view, check_source, freeze_phase, validate_phase
from course_experiments.baselines.runner import process_units, corpus_units, source_support, aggregate_complete
from course_experiments.baselines.native import K12Driver, NativeChat, BgeEncoder

ROOT=Path(__file__).resolve().parents[2]
PARAMS=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',embedding_model='BAAI/bge-m3',
            max_tokens=12000,temperature=0,concurrency=2,retries=2,max_gleanings=1,seed=42)


class BaselineTests(unittest.TestCase):
    def alignment_request_fixture(self):
        graph,schema,text,response=self.alignment_fixture()
        graph['nodes'][0]['id']='a'*64; graph['nodes'][1]['id']='b'*64
        graph['edges'][0].update(source='a'*64,target='b'*64,predicate='contains')
        for obj in graph['nodes']+graph['edges']:
            obj.update(definition='GENERATED_DEFINITION',description='GENERATED_DESCRIPTION',
                       source_texts=['OTHER_UNIT_TEXT'],sidecar={'text':'SIDECAR_TEXT'})
        return graph,schema,text

    def alignment_test_runtime(self, responder):
        class Runtime:
            config={'llm':{}}; gate=threading.BoundedSemaphore(1)
            def __init__(self): self.requests=[]; self.prompts=[]
            def stats(self): return {}
            def _raw_cached(self,key): return None
            def _post(self,path,body,*args):
                prompt=body['messages'][0]['content']; self.prompts.append(prompt)
                req=json.loads(prompt.split('输入：\n',1)[1]); self.requests.append(req)
                return dict(choices=[dict(message={'content':json.dumps(responder(req))},finish_reason='stop')])
        return Runtime()

    def alignment_short_answer(self,req):
        return dict(entities=[dict(id=n['id'],include=True,entity_type='Concept',quote=req['source_text'],reason='source') for n in req['nodes']],
                    edges=[dict(id=e['id'],predicate='part_of',head=e['target'],tail=e['source'],
                                quote=req['source_text'],conditions='运行时',reason='source') for e in req['edges']])

    def test_alignment_request_excludes_generated_and_other_unit_fields(self):
        from course_experiments.baselines.alignment import align
        graph,schema,text=self.alignment_request_fixture(); original=copy.deepcopy(graph)
        runtime=self.alignment_test_runtime(self.alignment_short_answer)
        with tempfile.TemporaryDirectory() as tmp:
            result=align(graph,[dict(unit_id='u',text=text)],schema,'k12_kgraph',dict(PARAMS,retries=0),runtime,Path(tmp),'phase')
            sent=runtime.requests[0]
            for marker in ['GENERATED_DEFINITION','GENERATED_DESCRIPTION','OTHER_UNIT_TEXT','SIDECAR_TEXT']:
                self.assertNotIn(marker,json.dumps(sent))
            self.assertEqual(set(sent['nodes'][0]),{'id','name','type'})
            self.assertEqual(set(sent['edges'][0]),{'id','source','target','predicate'})
            self.assertEqual(sent['source_text'],text)
            self.assertEqual(result['coverage']['aligned_edges'],1); self.assertEqual(graph,original)

    def test_alignment_short_handles_restore_long_ids_and_retain_raw_audit(self):
        from course_experiments.baselines.alignment import align
        from course_experiments.manifest import digest
        graph,schema,text=self.alignment_request_fixture(); runtime=self.alignment_test_runtime(self.alignment_short_answer)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=align(graph,[dict(unit_id='u',text=text)],schema,'k12_kgraph',dict(PARAMS,retries=0),runtime,root,'phase')
            self.assertEqual([n['id'] for n in runtime.requests[0]['nodes']],['n0','n1'])
            self.assertEqual(runtime.requests[0]['edges'][0]['id'],'e0')
            audit=json.loads(next(root.glob('*/complete.json')).read_text('utf-8'))
            self.assertEqual(audit['request_mapping'],dict(nodes={'n0':'a'*64,'n1':'b'*64},edges={'e0':digest(graph['edges'][0])}))
            self.assertEqual(audit['raw_response']['edges'][0]['head'],'n1')
            self.assertEqual(audit['response']['edges'][0]['head'],'b'*64)
            raw=json.loads(next(root.glob('*/attempts/*/responses/*.json')).read_text('utf-8'))
            self.assertEqual(json.loads(raw['content']),audit['raw_response'])
            request_record=json.loads(next(root.glob('*/request.json')).read_text('utf-8'))
            self.assertEqual(request_record['request_mapping'],audit['request_mapping'])
            self.assertEqual(request_record['request_policy_digest'],audit['request_policy_digest'])
            self.assertEqual(result['graph']['edges'][0]['source'],'b'*64)
            self.assertEqual(result,align(graph,[dict(unit_id='u',text=text)],schema,'k12_kgraph',dict(PARAMS,retries=0),runtime,root,'phase'))
            self.assertEqual(len(runtime.requests),1)

    def test_alignment_handles_reject_unknown_original_missing_duplicate_and_wrong_types(self):
        from course_experiments.baselines.alignment import align
        graph,schema,text=self.alignment_request_fixture()
        cases=[('entities',0,'id','n99'),('entities',0,'id','a'*64),('entities',0,'id',None),
               ('entities',0,'id',[]),('entities',0,'id',True),('entities',0,'id','n1'),
               ('edges',0,'id','n0'),('edges',0,'id',{}),('edges',0,'head','e0'),
               ('edges',0,'head','b'*64),('edges',0,'head',[]),('edges',0,'head','n0'),
               ('entities',0,'quote','GENERATED_DEFINITION'),('edges',0,'quote','OTHER_UNIT_TEXT'),
               ('edges',0,'conditions','OTHER_UNIT_TEXT')]
        with tempfile.TemporaryDirectory() as tmp:
            for i,(field,index,key,value) in enumerate(cases):
                def responder(req):
                    response=self.alignment_short_answer(req); response[field][index][key]=value; return response
                runtime=self.alignment_test_runtime(responder); root=Path(tmp)/str(i)
                with self.subTest(case=(field,key,value)),self.assertRaisesRegex(RuntimeError,'Unfinished'):
                    align(graph,[dict(unit_id='u',text=text)],schema,'k12_kgraph',dict(PARAMS,retries=0),runtime,root,'phase')
                self.assertFalse((root/'aligned_artifact.json').exists()); self.assertEqual(list(root.glob('*/complete.json')),[])
                error=json.loads(next(root.glob('*/attempts/*/error.json')).read_text('utf-8'))
                self.assertEqual(error['error_type'],'ValueError')
                self.assertNotEqual(error['validation_category'],'request_failure')
            for i,kind in enumerate(['missing','duplicate','nonobject','empty']):
                def responder(req):
                    response=self.alignment_short_answer(req)
                    if kind=='missing': response['entities'].pop()
                    elif kind=='duplicate': response['edges'].append(copy.deepcopy(response['edges'][0]))
                    elif kind=='nonobject': response['entities'][0]=[]
                    else: response={'entities':[],'edges':[]}
                    return response
                runtime=self.alignment_test_runtime(responder); root=Path(tmp)/('coverage'+str(i))
                with self.subTest(kind=kind),self.assertRaisesRegex(RuntimeError,'Unfinished'):
                    align(graph,[dict(unit_id='u',text=text)],schema,'k12_kgraph',dict(PARAMS,retries=0),runtime,root,'phase')
                self.assertEqual(list(root.glob('*/complete.json')),[])

    def test_alignment_request_bijection_is_order_independent_and_original_handle_collision_safe(self):
        from course_experiments.baselines.alignment import model_batch, rebind_response, validate_response
        nodes=[dict(id='n1',name='CPU',type='CONCEPT'),dict(id='n0',name='寄存器',type='CONCEPT')]
        edges=[dict(id='e0',source='n1',target='n0',predicate='contains')]
        original=copy.deepcopy([nodes,edges])
        batch,mapping=model_batch(nodes,edges)
        self.assertEqual(mapping,dict(nodes={'n0':'n0','n1':'n1'},edges={'e0':'e0'}))
        self.assertEqual(model_batch(list(reversed(nodes)),edges),(batch,mapping))
        schema=dict(entity_types={'Concept':''},relation_types={'part_of':''}); text='运行时寄存器是CPU的组成部分'
        response=self.alignment_short_answer(dict(batch,source_text=text)); raw=copy.deepcopy(response)
        rebound=rebind_response(response,mapping); validate_response(rebound,nodes,edges,text,schema)
        self.assertEqual([nodes,edges],original); self.assertEqual(response,raw)
        self.assertEqual(rebound['edges'][0]['head'],'n0')
        nodes[0]['id']='e0'; edges[0]['source']='e0'
        batch,mapping=model_batch(nodes,edges)
        self.assertEqual(mapping['nodes']['n0'],'e0'); self.assertEqual(mapping['edges']['e0'],'e0')
        for badnodes,badedges in [(nodes+nodes[:1],edges),(nodes,edges+edges),(nodes,[dict(edges[0],target='absent')])]:
            with self.assertRaises(ValueError): model_batch(badnodes,badedges)
        nodes[0]['name']={'definition':'do not leak'}
        with self.assertRaises(ValueError): model_batch(nodes,edges)

    def test_alignment_policy_and_hidden_native_changes_invalidate_checkpoint(self):
        from course_experiments.baselines.alignment import align, BATCH_POLICY
        graph,schema,text=self.alignment_request_fixture(); runtime=self.alignment_test_runtime(self.alignment_short_answer)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ids=[]; requests=[]
            for i in range(3):
                if i==1: graph['nodes'][0]['definition']='CHANGED_HIDDEN_DEFINITION'
                with patch.dict(BATCH_POLICY,version=BATCH_POLICY['version']+(1 if i==2 else 0)):
                    result=align(graph,[dict(unit_id='u',text=text)],schema,'k12_kgraph',dict(PARAMS,retries=0),runtime,root/str(i),'phase')
                    ids.append(result['audit_files'][0]); requests.append(runtime.requests[-1])
            self.assertEqual(len(set(ids)),3)
            self.assertEqual(requests[0],requests[1]); self.assertNotEqual(requests[1],requests[2])

    def test_alignment_decoder_rejects_prose_multiple_objects_and_incomplete_fences(self):
        from course_experiments.baselines.alignment import decode_response
        for value in ['{}\n{}','prefix {}','```json\n{}','```json\n{}\n``` trailing',
                      '```json\n{}\n```\n```json\n{}\n```','[]','{"entities":']:
            with self.subTest(value=value),self.assertRaises(ValueError): decode_response(value)
        self.assertEqual(decode_response(' \n```\n{"entities":[],"edges":[]}\n``` \n'),dict(entities=[],edges=[]))

    def test_alignment_format_is_opt_in_and_changes_actual_cache_identity(self):
        requests=[]; keys=[]
        class Runtime:
            config={'llm':{}}; gate=threading.BoundedSemaphore(1)
            def _raw_cached(self,key): keys.append(key); return None
            def _post(self,path,body,*args):
                requests.append(copy.deepcopy(body))
                return dict(choices=[dict(message={'content':'{}'},finish_reason='stop')])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); runtime=Runtime()
            NativeChat(runtime,PARAMS,root).generate('same JSON prompt')
            NativeChat(runtime,PARAMS,root,response_format={'type':'json_object'}).generate('same JSON prompt')
            self.assertNotIn('response_format',requests[0])
            self.assertEqual(requests[1]['response_format'],{'type':'json_object'})
            self.assertNotEqual(keys[0],keys[1])
            with patch.dict('os.environ',{'SILICONFLOW_API_KEY':'fixture-secret'}):
                with self.assertRaisesRegex(RuntimeError,'credential'):
                    NativeChat(runtime,PARAMS,root/'guard').generate('fixture-secret')
            self.assertFalse((root/'guard').exists())

    def test_alignment_retry_feedback_keeps_quote_coverage_rules_and_validates_cached_payload(self):
        from course_experiments.baselines.alignment import align
        from course_experiments.manifest import digest
        graph,schema,text,good=self.alignment_fixture(wire=True); bad_quote=copy.deepcopy(good); missing=copy.deepcopy(good)
        bad_quote['entities'][0]['quote']='made-up evidence'; missing['entities'].pop()
        responses=[bad_quote,missing,good]; prompts=[]
        class Runtime:
            config={'llm':{}}; gate=threading.BoundedSemaphore(1)
            def stats(self): return {}
            def _raw_cached(self,key): return None
            def _post(self,path,body,*args):
                prompts.append(body['messages'][0]['content'])
                return dict(choices=[dict(message={'content':json.dumps(responses.pop(0))},finish_reason='stop')])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); runtime=Runtime()
            result=align(graph,[dict(unit_id='u',text=text)],schema,'graphrag_extract',PARAMS,runtime,root,'phase')
            self.assertEqual(result['coverage']['aligned_edges'],1)
            feedback=[json.loads(p.split('Previous attempt validation feedback: ',1)[1].split('\n',1)[0]) for p in prompts[1:]]
            self.assertEqual([f['validation_category'] for f in feedback],['source_quote','id_coverage'])
            self.assertEqual(feedback[1]['missing_entities_ids'],['n1'])
            checkpoint=next(root.glob('*/complete.json')); saved=json.loads(checkpoint.read_text('utf-8'))
            saved['response']['edges'][0]['head']='new'; saved.pop('artifact_id'); saved['artifact_id']=digest(saved)
            checkpoint.write_text(json.dumps(saved),'utf-8')
            with self.assertRaisesRegex(ValueError,'endpoints'):
                align(graph,[dict(unit_id='u',text=text)],schema,'graphrag_extract',PARAMS,runtime,root,'phase')
            self.assertEqual(len(prompts),3)

    def test_alignment_batch_order_identity_source_guard_and_empty_no_api(self):
        from course_experiments.baselines.alignment import align, alignment_batches
        from course_experiments.manifest import digest
        graph,schema,text,good=self.alignment_fixture(wire=True); requests=[]
        class Runtime:
            config={'llm':{}}; gate=threading.BoundedSemaphore(1)
            def stats(self): return {}
            def _raw_cached(self,key): return None
            def _post(self,path,body,*args):
                requests.append(body)
                return dict(choices=[dict(message={'content':json.dumps(good)},finish_reason='stop')])
        edges=[dict(e,id=digest(e)) for e in graph['edges']]
        self.assertEqual(alignment_batches(graph['nodes'],edges),alignment_batches(list(reversed(graph['nodes'])),list(reversed(edges))))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); runtime=Runtime(); ids=[]
            for i,(phase,source) in enumerate([('phase',text),('phase',text+' extra source'),('phase2',text)]):
                result=align(graph,[dict(unit_id='u',text=source)],schema,'graphrag_extract',PARAMS,runtime,root/str(i),phase)
                ids.append(result['audit_files'][0])
            self.assertEqual(len(set(ids)),3)
            with self.assertRaisesRegex(ValueError,'source unit'):
                align(graph,[dict(unit_id='wrong',text=text)],schema,'graphrag_extract',PARAMS,runtime,root/'unknown','phase')
            empty=align(dict(nodes=[],edges=[]),[dict(unit_id='u',text=text)],schema,'graphrag_extract',PARAMS,runtime,root/'empty','phase')
            self.assertEqual(len(requests),3); self.assertEqual(empty['batch_coverage'][0]['batch_count'],0)

    def test_alignment_retries_are_bounded_and_unknown_sdk_errors_are_not_saved(self):
        from course_experiments.baselines.alignment import align
        graph,schema,text,good=self.alignment_fixture()
        class Runtime:
            config={'llm':{}}; gate=threading.BoundedSemaphore(1); calls=0; fail=False
            def stats(self): return {}
            def _raw_cached(self,key): return None
            def _post(self,*args):
                self.calls+=1
                if self.fail: raise RuntimeError('unknown SDK params secret-fixture')
                return dict(choices=[dict(message={'content':'{"entities":'},finish_reason='stop')])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); runtime=Runtime()
            with self.assertRaisesRegex(RuntimeError,'Unfinished'):
                align(graph,[dict(unit_id='u',text=text)],schema,'graphrag_extract',dict(PARAMS,retries=6),runtime,root/'bad','phase')
            self.assertEqual(runtime.calls,7); self.assertEqual(list((root/'bad').glob('*/complete.json')),[])
            self.assertEqual(len(list((root/'bad').glob('*/attempts/*/responses/*.json'))),7)
            runtime.fail=True
            with self.assertRaisesRegex(RuntimeError,'Unfinished'):
                align(graph,[dict(unit_id='u',text=text)],schema,'graphrag_extract',dict(PARAMS,retries=6),runtime,root/'sdk','phase')
            self.assertEqual(runtime.calls,8)
            self.assertNotIn('secret-fixture',''.join(p.read_text('utf-8') for p in (root/'sdk').rglob('*.json')))

    def test_kggen_missing_native_endpoints_become_audited_candidates_without_raw_mutation(self):
        from types import SimpleNamespace
        from course_experiments.baselines.native import normalized_kggen, kggen_json
        graph=SimpleNamespace(entities={'CPU'},edges={'contains'},
                              relations={('CPU','contains','register'),('cache','near','CPU')},
                              entity_clusters=None,edge_clusters=None)
        original=kggen_json(graph)
        result=normalized_kggen(graph)
        self.assertEqual({n['id'] for n in result['nodes']},{'CPU','register','cache'})
        self.assertEqual(result['raw'],original); self.assertEqual(kggen_json(graph),original)
        self.assertEqual({(e['source'],e['predicate'],e['target']) for e in result['edges']},graph.relations)
        for node in result['nodes']:
            self.assertEqual(node['type_status'],'native_untyped_candidate')
            self.assertNotIn('evidence_ids',node); self.assertNotIn('definition',node)
        audit=result['endpoint_closure_audit']
        self.assertEqual(audit['native_entity_table_count'],1); self.assertEqual(audit['closed_entity_count'],3)
        self.assertEqual(audit['added_endpoint_candidate_count'],2)
        self.assertEqual(audit['relations_with_missing_endpoint'],2)
        self.assertEqual(audit['missing_edge_labels'],['near'])

    def test_kggen_dedup_receives_closed_graph_and_preserves_preclosure_audit(self):
        from course_experiments.baselines.native import kggen_json
        driver,history=self.kggen_output_fixture(['stop'])
        graph=driver.kg.from_dict({}); graph.entities.remove('缓存'); graph.edges.clear()
        original=kggen_json(graph); calls=[]
        def dedup(lm,value,method,retrieval_model):
            calls.append((set(value.entities),set(value.edges),method,type(retrieval_model).__name__))
            self.assertEqual(value.entities,{'CPU','缓存'})
            value.entities.remove('缓存')
            return value
        driver.dedup.run_deduplication=dedup
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=driver.aggregate([dict(native={'raw':original})],root)
            self.assertEqual(calls,[({'CPU','缓存'},{'part_of'},'lm_based','BgeEncoder')])
            self.assertEqual(json.loads((root/'native_before_dedup.json').read_text('utf-8')),original)
            self.assertEqual(json.loads((root/'native_endpoint_closure.json').read_text('utf-8'))['added_endpoint_candidate_count'],1)
            self.assertEqual(calls,[({'CPU','缓存'},{'part_of'},'lm_based','BgeEncoder')])
            self.assertEqual({n['id'] for n in result['nodes']},{'CPU','缓存'})
            self.assertEqual(result['raw']['entities'],['CPU'])

    def test_alignment_json_format_fence_feedback_and_raw_truncation_preserved(self):
        from course_experiments.baselines.alignment import align
        graph,schema,text,response=self.alignment_fixture(wire=True); requests=[]
        contents=[('length','{"entities":'),('stop','not JSON'),('stop','```json\n'+json.dumps(response)+'\n```')]
        class Runtime:
            config={'llm':{'extra_body':{'enable_thinking':False,'model':'wrong','max_tokens':1}}}
            gate=threading.BoundedSemaphore(1)
            def _raw_cached(self,key): return None
            def stats(self): return {}
            def _post(self,path,body,*args):
                requests.append(copy.deepcopy(body)); reason,content=contents.pop(0)
                return dict(choices=[dict(message={'content':content},finish_reason=reason)],usage={})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            try: result=align(graph,[dict(unit_id='u',text=text)],schema,'graphrag_extract',PARAMS,Runtime(),root,'phase')
            except RuntimeError: result=None
            self.assertEqual(requests[0].get('response_format'),{'type':'json_object'})
            self.assertIsNotNone(result)
            self.assertEqual(result['coverage']['aligned_edges'],1)
            self.assertEqual(len(requests),3)
            self.assertEqual(len({r['messages'][0]['content'] for r in requests}),3)
            for req in requests:
                self.assertEqual(req['response_format'],{'type':'json_object'})
                self.assertEqual(req['model'],PARAMS['model']); self.assertEqual(req['max_tokens'],12000)
            raw=[json.loads(p.read_text('utf-8')) for p in root.glob('*/attempts/*/responses/*.json')]
            self.assertEqual(len(raw),3); self.assertIn('length',{r['finish_reason'] for r in raw})
            errors=[json.loads(p.read_text('utf-8')) for p in root.glob('*/attempts/*/error.json')]
            self.assertEqual({r['validation_category'] for r in errors},{'output_limit','json_format'})

    def test_alignment_greedy_batches_cover_100_nodes_80_edges_and_resume_failures(self):
        from course_experiments.baselines.alignment import align
        from course_experiments.manifest import digest
        nodes=[dict(id=f'n{i:03}',name=f'node{i}',evidence_ids=['u']) for i in range(100)]
        edges=[dict(source=f'n{i:03}',target=f'n{(i+1)%100:03}',predicate='near',evidence_ids=['u']) for i in range(80)]
        graph=dict(nodes=nodes,edges=edges,directed=True); original=copy.deepcopy(graph)
        schema=dict(entity_types={'Concept':''},relation_types={'part_of':''}); text='exact original source unit text'
        requests=[]
        class Runtime:
            config={'llm':{}}; gate=threading.BoundedSemaphore(1); failing=True
            def stats(self): return {}
            def _raw_cached(self,key): return None
            def _post(self,path,body,*args):
                req=json.loads(body['messages'][0]['content'].split('输入：\n',1)[1]); requests.append(req)
                if self.failing and req['batch_index']==1: content='broken'
                else: content=json.dumps(dict(entities=[dict(id=n['id'],include=True,entity_type='Concept',quote=text,reason='source') for n in req['nodes']],
                                              edges=[dict(id=e['id'],predicate=None,head=None,tail=None,reason='uncertain') for e in req['edges']]))
                return dict(choices=[dict(message={'content':content},finish_reason='stop')])
        runtime=Runtime()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'Unfinished'):
                align(graph,[dict(unit_id='u',text=text)],schema,'kggen_lm_dedup',dict(PARAMS,retries=1),runtime,root,'phase')
            self.assertLessEqual(max(len(r['nodes']) for r in requests),32)
            self.assertFalse((root/'aligned_artifact.json').exists())
            first=copy.deepcopy(requests); requests.clear(); runtime.failing=False
            result=align(graph,[dict(unit_id='u',text=text)],schema,'kggen_lm_dedup',dict(PARAMS,retries=1),runtime,root,'phase')
            self.assertEqual([r['batch_index'] for r in requests],[1])
            self.assertEqual(sum(r['batch_index']==1 for r in first),2)
            audits=[json.loads(p.read_text('utf-8')) for p in root.glob('*/complete.json')]
            self.assertEqual({n['id'] for a in audits for n in a['response']['entities']},{n['id'] for n in nodes})
            self.assertEqual(sorted(e['id'] for a in audits for e in a['response']['edges']),sorted(digest(e) for e in edges))
            for req in first:
                self.assertLessEqual(len(req['nodes']),32); self.assertLessEqual(len(req['edges']),24)
                self.assertEqual(req['source_text'],text); self.assertEqual(req['unit_id'],'u')
                self.assertTrue({end for e in req['edges'] for end in (e['source'],e['target'])}<={n['id'] for n in req['nodes']})
            self.assertEqual(graph,original); self.assertEqual(result['coverage']['aligned_nodes'],100)
            self.assertEqual(result['batch_coverage'][0]['covered_edges'],80)
            requests.clear()
            again=align(graph,[dict(unit_id='u',text=text)],schema,'kggen_lm_dedup',dict(PARAMS,retries=1),runtime,root,'phase')
            self.assertEqual(requests,[]); self.assertEqual(again,result)

    def test_unknown_and_structural_edges_never_become_semantic_facts(self):
        graph = dict(nodes=[dict(id='a', name='CPU', type='Concept', definition='处理器'),
                            dict(id='b', name='寄存器', type='Concept'),
                            dict(id='s', name='第一章', type='Section')],
                     edges=[dict(source='b', target='a', predicate='part_of'),
                            dict(source='a', target='b', predicate='relates_to'),
                            dict(source='s', target='a', predicate='part_of')], directed=True)
        view = semantic_view(graph, {'relation_types': {'part_of': ''}}, 'kggen_lm_dedup')
        self.assertEqual(len(view['entities']), 2)
        self.assertEqual([r['relation'] for r in view['relations']], ['part_of'])
        self.assertEqual(len(view['unmapped_relations']), 2)
        self.assertEqual(view['entities'][0]['definition'], '处理器')

    def test_graphrag_undirected_is_not_typed_kge(self):
        graph = dict(nodes=[dict(id='a',name='CPU',type='CONCEPT'),dict(id='b',name='寄存器',type='CONCEPT')],
                     edges=[dict(source='a',target='b',description='包含',weight=3)],directed=False)
        view = semantic_view(graph, {'relation_types': {'consists_of': ''}}, 'graphrag_extract')
        self.assertEqual(view['relations'], [])
        self.assertIsNone(view['capabilities']['kge'])
        self.assertIn('undirected', view['capability_reasons']['kge'])
        self.assertEqual(view['unmapped_relations'][0]['description'], '包含')

    def test_k12_known_direction_mapping_only(self):
        graph = dict(nodes=[dict(id='a',name='缓存',type='Concept'),dict(id='b',name='局部性',type='Concept')],
                     edges=[dict(source='b',target='a',predicate='prerequisites_for')],directed=True)
        view=semantic_view(graph, {'relation_types': {'prerequisite_of': ''}}, 'k12_kgraph')
        self.assertEqual(view['relations'][0]['head_entity_id'], 'b')
        self.assertEqual(view['relations'][0]['relation'], 'prerequisite_of')

    def test_source_tree_hashes_all_code_and_ignores_only_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'native.py').write_text('original')
            before=source_tree(root)
            (root/'native.py').write_text('changed')
            self.assertNotEqual(before, source_tree(root))
            (root/'__pycache__').mkdir(); (root/'__pycache__/native.pyc').write_bytes(b'cache')
            self.assertEqual(len(source_tree(root)['files']),1)

    def test_budget_and_provider_are_explicit(self):
        params=dict(base_url='https://api.siliconflow.cn/v1', model='deepseek-ai/DeepSeek-V4-Flash',
                    embedding_model='BAAI/bge-m3', max_tokens=12000,temperature=0,concurrency=2,
                    retries=2,max_gleanings=1,seed=42)
        validate_parameters(params)
        for key,value in [('base_url','https://other.example/v1'),('concurrency',0),('seed',99)]:
            with self.assertRaises(ValueError): validate_parameters(dict(params,**{key:value}))

    def test_partial_checkpoint_resume_and_402_no_retries(self):
        class Fake:
            def __init__(self): self.calls=[]; self.fail=True
            def extract(self,unit,root):
                self.calls.append(unit['chunk_id'])
                if unit['chunk_id']=='b' and self.fail: raise RuntimeError('Model service returned HTTP 402')
                return dict(nodes=[],edges=[],directed=True)
        units=[dict(chunk_id=x,book_id='book',text_digest=x,text='text',evidence=[]) for x in ['a','b','c']]
        with tempfile.TemporaryDirectory() as tmp:
            driver=Fake()
            with self.assertRaises(RuntimeError): process_units(units,driver,Path(tmp),'identity',2)
            self.assertEqual(driver.calls,['a','b'])
            self.assertEqual(len(list(Path(tmp).glob('*/complete.json'))),1)
            driver.fail=False; driver.calls=[]
            results=process_units(units,driver,Path(tmp),'identity',2)
            self.assertEqual(driver.calls,['b','c']); self.assertEqual(len(results),3)
            with self.assertRaises(ValueError): process_units(units,driver,Path(tmp),'different',2)

    def test_other_independent_chunks_continue_after_format_error(self):
        class Fake:
            def extract(self,unit,root):
                if unit['chunk_id']=='b': raise ValueError('bad native response')
                return dict(nodes=[],edges=[],directed=True)
        units=[dict(chunk_id=x,book_id='book',text_digest=x,text='text',evidence=[]) for x in ['a','b','c']]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'Unfinished'): process_units(units,Fake(),Path(tmp),'id',1)
            self.assertEqual(len(list(Path(tmp).glob('*/complete.json'))),2)

    def test_missing_and_wrong_source_cannot_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'native.py').write_text('fake')
            with self.assertRaisesRegex(ValueError,'source tree mismatch'): check_source('k12_kgraph',Path(tmp))
            with self.assertRaisesRegex(ValueError,'unavailable'): check_source('tree_kg',Path(tmp))

    def test_phase_guard_changed_packages_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); lock=root/'requirements.txt'; lock.write_text('dependency==1')
            phase=root/'phase.json'; construction=root/'construction.json'
            from course_experiments.manifest import digest
            payload=dict(manifest_id='construction',files={})
            with patch('course_experiments.baselines.protocol.validate',return_value=payload), \
                 patch('course_experiments.baselines.protocol.check_source',return_value=dict(commit='fixed',path=str(root))), \
                 patch('course_experiments.baselines.protocol.environment_versions',return_value={'package':'1'}):
                freeze_phase(construction,'k12_kgraph',root,PARAMS,lock,phase)
                validate_phase(construction,phase,'k12_kgraph',root)
                with self.assertRaisesRegex(ValueError,'method phase'): validate_phase(construction,phase,'graphrag_extract',root)
                with patch('course_experiments.baselines.protocol.environment_versions',return_value={'package':'2'}):
                    with self.assertRaisesRegex(ValueError,'package versions'): validate_phase(construction,phase,'k12_kgraph',root)
                lock.write_text('dependency==2')
                with self.assertRaisesRegex(ValueError,'Dependency lock'): validate_phase(construction,phase,'k12_kgraph',root)

    def test_full_smoke_units_separate_even_if_same_chunk_count(self):
        from course_experiments.manifest import digest
        with tempfile.TemporaryDirectory() as tmp:
            corpus=dict(books=[dict(id='b')],chunks=[dict(id='c',book_id='b',evidence_ids=['e'])],
                        evidence=[dict(id='e',book_id='b',text='教材原文')])
            corpus['fingerprint']=digest(corpus); path=Path(tmp)/'corpus.json'; path.write_text(json.dumps(corpus))
            manifest={'files':{'sources':{'b':{'path':str(path)}}}}; course={'books':[{'alias':'A','book_id':'b'}]}
            full=corpus_units(manifest,course,['A'],None); smoke=corpus_units(manifest,course,['A'],1)
            self.assertNotEqual(full[0]['unit_id'],smoke[0]['unit_id'])
            self.assertEqual(full[0]['text'],smoke[0]['text'])

    def test_native_k12_section_and_merge_real_official_functions(self):
        source=ROOT/'.runtime_cache/baselines/K12-KGraph'
        if not source.exists(): self.skipTest('Public pinned native fixture source not present')
        class Runtime:
            gate=threading.BoundedSemaphore(2)
            config={'llm':{}}
            def _raw_cached(self,key): return None
            def _post(self,*args):
                raw=dict(nodes=[dict(id='cpt_1',label='Concept',name='CPU',definition='处理器'),dict(id='cpt_2',label='Concept',name='计算机')],
                         edges=[dict(source='cpt_1',target='cpt_2',type='is_a')])
                return dict(choices=[dict(message=dict(content=json.dumps(raw)),finish_reason='stop')],usage={'total_tokens':5})
        previous=set(sys.modules); old_path=list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); driver=K12Driver(source,Runtime(),PARAMS)
                units=[dict(book_id=b,chunk_id='c',unit_id=b,text='CPU是一种处理器',text_digest=b,evidence=[]) for b in ['A','B']]
                rows=process_units(units,driver,root/'chunks','id',0)
                native=driver.aggregate(rows,root/'aggregate')
                self.assertEqual(len(native['nodes']),2); self.assertEqual(len(native['edges']),1)
                graph,evidence=source_support(native,rows,units,'k12_kgraph')
                self.assertEqual(set(graph['edges'][0]['evidence_ids']),{'A','B'})
                self.assertEqual(len(evidence),2)
        finally:
            for name in set(sys.modules)-previous:
                if name.split('.')[0] in {'kg','utils'}: sys.modules.pop(name,None)
            sys.path[:]=old_path

    def test_actual_graphrag_parser_preserves_description_and_undirectedness(self):
        path=ROOT/'.runtime_cache/baselines/graphrag-pinned/graphrag/index/operations/extract_graph/graph_extractor.py'
        if not path.exists(): self.skipTest('Public pinned native parser source not present')
        import networkx as nx
        import re
        import html
        from collections.abc import Mapping
        tree=ast.parse(path.read_text('utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GraphExtractor')
        method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_process_results')
        helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_unpack_')]
        selected=ast.Module(body=[method,*helpers],type_ignores=[])
        scope=dict(nx=nx,re=re,Mapping=Mapping,clean_str=lambda s:html.unescape(s.strip().strip('"')))
        exec(compile(selected,str(path),'exec'),scope)
        records={0:'("entity"<|>CPU<|>CONCEPT<|>处理器)##("entity"<|>缓存<|>CONCEPT<|>缓冲存储)##("relationship"<|>CPU<|>缓存<|>访问缓存<|>3)'}
        obj=type('Parser',(),{'_join_descriptions':True})()
        graph=asyncio.run(scope['_process_results'](obj,records,'<|>','##'))
        self.assertFalse(graph.is_directed()); self.assertEqual(graph['CPU']['缓存']['description'],'访问缓存')
        self.assertNotIn('predicate',graph['CPU']['缓存'])

    def test_bge_shape_order_and_invalid_matrix(self):
        import numpy as np
        runtime=type('Runtime',(),{'embed':lambda self,texts:np.asarray([[i,1] for i,_ in enumerate(texts)])})()
        result=BgeEncoder(runtime).encode(['b','a'])
        self.assertEqual(result.tolist(),[[0,1],[1,1]])
        with self.assertRaises(ValueError): BgeEncoder(runtime).encode('text')

    def test_native_chat_history_and_budget_are_preserved_without_credentials(self):
        class Runtime:
            gate=threading.BoundedSemaphore(1)
            config={'llm':{}}
            def _raw_cached(self,key): return None
            def _post(self,*args): return {'choices':[{'message':{'content':'native text'},'finish_reason':'stop'}],'usage':{'total_tokens':9}}
        with tempfile.TemporaryDirectory() as tmp:
            chat=NativeChat(Runtime(),PARAMS,Path(tmp))
            response=asyncio.run(chat.achat('continue',history=[{'role':'user','content':'source'}],name='extract-continuation-0'))
            self.assertEqual(len(response.history),3)
            with self.assertRaises(ValueError): chat.generate('prompt',max_tokens=10)
            self.assertEqual(chat.records,['native text'])

    def test_existing_evaluator_reads_native_projection_with_real_context(self):
        from argparse import Namespace
        from scripts.evaluate_kg_quality import load_inputs,evaluate_entities,evaluate_relations,build_relation_context
        unit=dict(unit_id='unit',book_id='book',chunk_id='c',text='寄存器是CPU的组成部分',text_digest='digest',evidence=[])
        graph=dict(nodes=[dict(id='a',name='CPU',type='Concept',definition='处理器'),dict(id='b',name='寄存器',type='Concept',definition='存储部件')],
                   edges=[dict(source='b',target='a',predicate='part_of')],directed=True)
        rows=[dict(provenance={'unit_id':'unit'},native=graph)]
        graph,evidence=source_support(graph,rows,[unit],'kggen_lm_dedup')
        view=semantic_view(graph,{'entity_types':{'Concept':''},'relation_types':{'part_of':''}},'kggen_lm_dedup')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'concept_kg.json').write_text(json.dumps(dict(entities=view['entities'],relations=view['relations'])))
            (root/'extractable_doc_nodes.json').write_text(json.dumps(evidence))
            inputs=load_inputs(Namespace(kg_output_dir=root,schema=ROOT/'config/schema.yaml'))
            metrics,issues=evaluate_entities(inputs)
            self.assertEqual(metrics['missing_definition_count'],0); self.assertEqual(metrics['missing_source_reference_count'],0)
            metrics,issues,risks=evaluate_relations(inputs)
            self.assertEqual(metrics['unresolved_endpoint_count'],0)
            self.assertIn(unit['text'],build_relation_context(inputs,view['relations'][0],1000))

    def test_kggen_native_swallowed_cluster_error_cannot_be_published(self):
        from types import SimpleNamespace
        from course_experiments.baselines.native import KGGenDriver
        import logging
        graph=SimpleNamespace(entities={'CPU','缓存'},edges={'part_of'},relations={('缓存','part_of','CPU')},entity_clusters=None,edge_clusters=None)
        driver=KGGenDriver.__new__(KGGenDriver); driver.params=PARAMS; driver.runtime=None
        driver.history=lambda root,**kwargs:None
        driver.kg=SimpleNamespace(from_dict=lambda raw:graph,aggregate=lambda values:graph,lm='lm')
        logger=logging.getLogger('baseline_mock_native')
        driver.llmdedup=SimpleNamespace(ThreadPoolExecutor=None,LLMDeduplicate=SimpleNamespace(logger=logger))
        def dedup(*args,**kwargs):
            pool=driver.llmdedup.ThreadPoolExecutor(max_workers=64)
            def fail(): raise RuntimeError('HTTP 402')
            future=pool.submit(fail)
            try: future.result()
            except RuntimeError: pass
            return graph
        driver.dedup=SimpleNamespace(run_deduplication=dedup,DeduplicateMethod=SimpleNamespace(LM_BASED='lm_based'))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'402'): driver.aggregate([dict(native={'raw':{}})],Path(tmp))

    def test_aggregate_checkpoint_prevents_repeated_model_aggregation(self):
        class Fake:
            calls=0
            def aggregate(self,rows,root):
                self.calls+=1
                if self.calls==1: raise ValueError('native format failure')
                return dict(nodes=[],edges=[],directed=True)
        with tempfile.TemporaryDirectory() as tmp:
            fake=Fake(); root=Path(tmp)
            result=aggregate_complete(fake,[],root,'id',2)
            self.assertEqual(fake.calls,2)
            self.assertEqual(result,aggregate_complete(fake,[],root,'id',2))
            self.assertEqual(fake.calls,2)
            with self.assertRaises(ValueError): aggregate_complete(fake,[],root,'wrong',2)

    def test_kggen_untyped_candidates_are_not_claimed_concept_evaluation(self):
        from types import SimpleNamespace
        from course_experiments.baselines.native import normalized_kggen
        graph=normalized_kggen(SimpleNamespace(entities={'CPU','book'},edges={'part_of'},relations={('CPU','part_of','book')},
                                                 entity_clusters=None,edge_clusters=None))
        view=semantic_view(graph,{'entity_types':{'Concept':''},'relation_types':{'part_of':''}},'kggen_lm_dedup')
        self.assertIsNone(view['capabilities']['topology']); self.assertIsNone(view['capabilities']['kge'])
        self.assertTrue(view['capabilities']['native_topology']); self.assertTrue(view['capabilities']['rag'])

    def alignment_fixture(self,wire=False):
        from course_experiments.manifest import digest
        nodes=[dict(id='a',name='CPU',type='CONCEPT',evidence_ids=['u']),dict(id='b',name='寄存器',type='CONCEPT',evidence_ids=['u'])]
        edge=dict(source='a',target='b',description='有关',evidence_ids=['u'])
        eid=digest(edge)
        schema=dict(entity_types={'Concept':''},relation_types={'part_of':''})
        text='运行时寄存器是CPU的组成部分'
        response=dict(entities=[dict(id=n['id'],include=True,entity_type='Concept',quote=text,reason='原文概念') for n in nodes],
                      edges=[dict(id=eid,predicate='part_of',head='b',tail='a',conditions='运行时',quote=text,reason='原文组成关系')])
        if wire:
            response['entities'][0]['id']='n0'; response['entities'][1]['id']='n1'
            response['edges'][0].update(id='e0',head='n1',tail='n0')
        return dict(nodes=nodes,edges=[edge],directed=False),schema,text,response

    def test_alignment_rejects_new_elements_unquoted_or_wrong_predicates(self):
        from course_experiments.baselines.alignment import validate_response
        from course_experiments.manifest import digest
        graph,schema,text,response=self.alignment_fixture()
        edges=[dict(e,id=digest(e)) for e in graph['edges']]
        validate_response(response,graph['nodes'],edges,text,schema)
        for field,value in [('predicate','causes'),('head','new_node'),('quote','模型描述'),('conditions','不存在条件')]:
            changed=copy.deepcopy(response); changed['edges'][0][field]=value
            with self.assertRaises(ValueError): validate_response(changed,graph['nodes'],edges,text,schema)
        changed=copy.deepcopy(response); changed['entities'].append(dict(id='new',include=False,entity_type=None,quote='',reason='new'))
        with self.assertRaises(ValueError): validate_response(changed,graph['nodes'],edges,text,schema)

    def test_alignment_direction_condition_audit_and_conflicts_preserved(self):
        from course_experiments.baselines.alignment import build_projection
        graph,schema,text,response=self.alignment_fixture(); original=copy.deepcopy(graph)
        result=build_projection(graph,[dict(unit_id='u',response=response)],schema,'graphrag_extract')
        self.assertEqual(graph,original)
        relation=result['semantic_view']['relations'][0]
        self.assertEqual(relation['head_entity_id'],'b'); self.assertEqual(relation['conditions'],'运行时')
        self.assertEqual(relation['evidences'],[text]); self.assertEqual(result['method_id'],'graphrag_extract_schema_aligned')
        conflicting=copy.deepcopy(response); conflicting['edges'][0].update(head='a',tail='b')
        rejected=build_projection(graph,[dict(unit_id='u',response=response),dict(unit_id='v',response=conflicting)],schema,'graphrag_extract')
        self.assertEqual(rejected['semantic_view']['relations'],[])
        self.assertEqual(rejected['edge_audit'][0]['reason'],'conflicting_mappings')

    def test_alignment_per_source_checkpoint_reuses_and_only_sends_traceable_native_ids(self):
        from course_experiments.baselines.alignment import align
        graph,schema,text,response=self.alignment_fixture(wire=True)
        units=[dict(unit_id='u',text=text),dict(unit_id='empty',text='其他原文')]
        runtime=type('Runtime',(),{'stats':lambda self:{'native_baseline:response':1}})()
        class FakeChat:
            calls=[]
            def __init__(self,*args,**kwargs): pass
            def generate(self,prompt):
                self.calls.append(prompt)
                self.request=json.loads(prompt.split('输入：\n',1)[1])
                self.assertions=set(n['id'] for n in self.request['nodes'])
                return json.dumps(response)
        with tempfile.TemporaryDirectory() as tmp, patch('course_experiments.baselines.alignment.NativeChat',FakeChat):
            result=align(graph,units,schema,'graphrag_extract',PARAMS,runtime,Path(tmp),'identity')
            again=align(graph,units,schema,'graphrag_extract',PARAMS,runtime,Path(tmp),'identity')
            self.assertEqual(len(FakeChat.calls),1); self.assertEqual(result,again)
            self.assertEqual(result['coverage']['aligned_edges'],1)

    def test_kggen_retry_budget_matches_exact_pinned_dspy_forward_completion(self):
        from types import SimpleNamespace
        from typing import Dict, Any, Optional
        from course_experiments.baselines.native import KGGenDriver
        import logging
        path=ROOT/'.runtime_cache/baselines/dspy-2.6.27/dspy/clients/lm.py'
        if not path.exists(): self.skipTest('Pinned DSPy forwarding source absent')
        tree=ast.parse(path.read_text('utf-8'))
        lm_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LM')
        forward=next(n for n in lm_class.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        completion=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='litellm_completion')
        calls=[]
        def transport(**kwargs):
            calls.append(kwargs); return {'choices':[SimpleNamespace(finish_reason='stop')]}
        scope=dict(Dict=Dict,Any=Any,Optional=Optional,litellm=SimpleNamespace(completion=transport),
                   _get_stream_completion_fn=lambda *args,**kwargs:None,logger=logging.getLogger('dspy_pinned_fixture'),
                   dspy=SimpleNamespace(settings=SimpleNamespace(usage_tracker=None)))
        exec(compile(ast.Module(body=[completion,forward],type_ignores=[]),str(path),'exec'),scope)
        extras=dict(enable_thinking=False,model='wrong',messages=[],temperature=2,max_tokens=1)
        runtime=SimpleNamespace(config={'llm':{'extra_body':extras}})
        for retries in [0,2]:
            lm=SimpleNamespace(kwargs={'temperature':0,'max_tokens':12000},num_retries=99,cache=False,cache_in_memory=False,
                               model='openai/deepseek-ai/DeepSeek-V4-Flash',model_type='chat')
            lm._get_cached_completion_fn=lambda fn,*args:(fn,{})
            module=SimpleNamespace(KGGen=lambda **kwargs:SimpleNamespace(lm=lm))
            with patch('course_experiments.baselines.native.native_import',side_effect=[module,SimpleNamespace(),SimpleNamespace()]), \
                 patch.dict('os.environ',{'SILICONFLOW_API_KEY':'dummy-boundary-key'}):
                driver=KGGenDriver(Path('unused'),runtime,dict(PARAMS,retries=retries))
            result=scope['forward'](driver.kg.lm,prompt='native prompt')
            self.assertEqual(calls[-1]['num_retries'],retries)
            self.assertNotIn('num_retries',driver.kg.lm.kwargs)
            self.assertEqual(calls[-1]['timeout'],240)
            self.assertEqual(calls[-1]['extra_body'],dict(enable_thinking=False))
            self.assertIsNot(driver.kg.lm.kwargs['extra_body'],extras)
            self.assertEqual(calls[-1]['model'],'openai/'+PARAMS['model'])
            self.assertEqual(calls[-1]['messages'],[dict(role='user',content='native prompt')])
            self.assertEqual(calls[-1]['temperature'],0); self.assertEqual(calls[-1]['max_tokens'],12000)
            self.assertEqual(extras['model'],'wrong')

    def kggen_output_fixture(self,finish_reasons):
        from types import SimpleNamespace
        from course_experiments.baselines.native import KGGenDriver
        import logging
        graph=SimpleNamespace(entities={'CPU','缓存'},edges={'part_of'},relations={('缓存','part_of','CPU')},entity_clusters=None,edge_clusters=None)
        history=dict(prompt='native prompt',outputs=['CPU'],usage={'total_tokens':12},cost=0.001,
                     kwargs={'api_key':'secret-not-to-save'},response=SimpleNamespace(
                         choices=[SimpleNamespace(index=i,finish_reason=reason,message=SimpleNamespace(content='CPU',other='SDK-extra-not-to-save'))
                                  for i,reason in enumerate(finish_reasons)],extra='SDK-extra-not-to-save'))
        driver=KGGenDriver.__new__(KGGenDriver); driver.params=PARAMS; driver.runtime=None
        lm=SimpleNamespace(history=[]); driver.calls=0; driver.finish_reasons=finish_reasons
        def generate(*args,**kwargs):
            driver.calls+=1; lm.history.append(history); return graph
        driver.kg=SimpleNamespace(lm=lm,generate=generate,from_dict=lambda raw:graph,aggregate=lambda rows:graph)
        driver.llmdedup=SimpleNamespace(ThreadPoolExecutor=None,LLMDeduplicate=SimpleNamespace(logger=logging.getLogger('kggen_history_fixture')))
        def dedup(*args,**kwargs): lm.history.append(history); return graph
        driver.dedup=SimpleNamespace(run_deduplication=dedup,DeduplicateMethod=SimpleNamespace(LM_BASED='lm_based'))
        return driver,history

    def test_kggen_truncated_extraction_preserves_finish_reason_and_refuses_complete(self):
        driver,history=self.kggen_output_fixture(['stop','length'])
        units=[dict(book_id='book',chunk_id='chunk',text='CPU',text_digest='digest',evidence=[])]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'Unfinished'): process_units(units,driver,root,'identity',1)
            self.assertEqual(driver.calls,2); self.assertEqual(list(root.glob('*/complete.json')),[])
            records=[json.loads(p.read_text('utf-8')) for p in root.glob('*/attempts/*/responses/*.json')]
            self.assertEqual(len(records),2)
            self.assertEqual(records[0]['finish_reasons'],['stop','length'])
            self.assertEqual(records[0]['response_choices'][1]['content'],'CPU')
            self.assertEqual(records[0]['usage'],{'total_tokens':12})
            self.assertNotIn('secret-not-to-save',json.dumps(records)); self.assertNotIn('SDK-extra-not-to-save',json.dumps(records))
            history['response'].choices=history['response'].choices[:1]
            process_units(units,driver,root,'identity',1); saved_calls=driver.calls
            process_units(units,driver,root,'identity',1)
            self.assertEqual(driver.calls,saved_calls); self.assertEqual(len(list(root.glob('*/complete.json'))),1)

    def test_kggen_truncated_dedup_never_commits_aggregate_then_valid_resumes(self):
        driver,history=self.kggen_output_fixture(['length'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError): aggregate_complete(driver,[dict(native={'raw':{}})],root,'identity',0)
            self.assertFalse((root/'aggregate_complete.json').exists())
            records=[json.loads(p.read_text('utf-8')) for p in root.glob('aggregate/*/responses/*.json')]
            self.assertEqual(records[0]['finish_reasons'],['length'])
            history['response'].choices[0].finish_reason='stop'
            result=aggregate_complete(driver,[dict(native={'raw':{}})],root,'identity',0)
            self.assertEqual(result['dedup_status'],'lm_based_complete')
            self.assertEqual(result,aggregate_complete(driver,[],root,'identity',0))

    def test_kggen_saving_truncation_does_not_mask_current_402_failure(self):
        driver,history=self.kggen_output_fixture(['length'])
        def generate(*args,**kwargs):
            driver.kg.lm.history.append(history)
            raise RuntimeError('Model service returned HTTP 402')
        driver.kg.generate=generate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'402'): driver.extract({'text':'CPU'},root)
            record=json.loads(next((root/'responses').glob('*.json')).read_text('utf-8'))
            self.assertEqual(record['finish_reasons'],['length'])

    def test_native_chat_sends_frozen_extra_body_and_mandatory_budget_wins(self):
        class Runtime:
            gate=threading.BoundedSemaphore(1)
            def __init__(self):
                self.config={'llm':{'extra_body':dict(enable_thinking=False,model='wrong',messages=[],temperature=2,max_tokens=1)}}
                self.requests=[]; self.keys=[]
            def _raw_cached(self,key): return None
            def _post(self,endpoint,body,options,key,stage):
                self.requests.append(copy.deepcopy(body)); self.keys.append(key)
                return {'choices':[{'message':{'content':'native text'},'finish_reason':'stop'}]}
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Runtime(); original=copy.deepcopy(runtime.config)
            chat=NativeChat(runtime,PARAMS,Path(tmp))
            chat.generate('source')
            sent=runtime.requests[-1]
            self.assertIs(sent['enable_thinking'],False)
            self.assertEqual(sent['model'],PARAMS['model']); self.assertEqual(sent['temperature'],0); self.assertEqual(sent['max_tokens'],12000)
            self.assertEqual(sent['messages'],[dict(role='user',content='source')]); self.assertEqual(runtime.config,original)
            runtime.config['llm']['extra_body']['enable_thinking']=True
            chat.generate('source')
            self.assertNotEqual(runtime.keys[0],runtime.keys[1])

    def sdk_usage_fixture(self):
        class Details:
            def __init__(self,**values): self.values=values
            def model_dump(self,*,mode='python'): return dict(self.values)
        return dict(completion_tokens=5,prompt_tokens=7,total_tokens=12,
                    completion_tokens_details=Details(reasoning_tokens=2,accepted_prediction_tokens=1,rejected_prediction_tokens=0),
                    prompt_tokens_details=Details(cached_tokens=3,audio_tokens=0,
                                                 nested={'detail':Details(count=4,optional=None)}))

    def test_kggen_nested_sdk_usage_preserves_numbers_and_safe_response_on_success(self):
        driver,history=self.kggen_output_fixture(['stop'])
        history['usage']=self.sdk_usage_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            result=driver.extract({'text':'CPU'},Path(tmp))
            record=json.loads(next((Path(tmp)/'responses').glob('*.json')).read_text('utf-8'))
            self.assertEqual(record['usage']['total_tokens'],12)
            self.assertEqual(record['usage']['completion_tokens_details']['reasoning_tokens'],2)
            self.assertEqual(record['usage']['prompt_tokens_details']['nested']['detail'],{'count':4,'optional':None})
            self.assertEqual(record['finish_reasons'],['stop']); self.assertEqual(record['response_choices'][0]['content'],'CPU')
            self.assertNotIn('secret-not-to-save',json.dumps(record)); self.assertNotIn('SDK-extra-not-to-save',json.dumps(record))
            self.assertEqual(driver.kg.lm.history,[])

    def test_kggen_nested_sdk_usage_does_not_mask_402_or_truncation(self):
        driver,history=self.kggen_output_fixture(['length']); history['usage']=self.sdk_usage_fixture()
        def generate(*args,**kwargs):
            driver.kg.lm.history.append(history)
            raise RuntimeError('Model service returned HTTP 402')
        driver.kg.generate=generate
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'402'): driver.extract({'text':'CPU'},Path(tmp)/'failed')
            driver.kg.lm.history.append(history)
            with self.assertRaisesRegex(ValueError,'length'): driver.history(Path(tmp)/'truncated')
            record=json.loads(next((Path(tmp)/'truncated').glob('*.json')).read_text('utf-8'))
            self.assertEqual(record['usage']['prompt_tokens_details']['cached_tokens'],3)
            self.assertEqual(record['finish_reasons'],['length'])

    def test_kggen_sdk_usage_conversion_retains_credential_guard(self):
        driver,history=self.kggen_output_fixture(['stop']); usage=self.sdk_usage_fixture()
        usage['prompt_tokens_details'].values['provider_detail']='dummy-boundary-secret'
        history['usage']=usage
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'SILICONFLOW_API_KEY':'dummy-boundary-secret'}):
            with self.assertRaisesRegex(RuntimeError,'credential'): driver.extract({'text':'CPU'},Path(tmp))
            self.assertEqual(list(Path(tmp).glob('responses/*.json')),[])


if __name__ == '__main__': unittest.main()
