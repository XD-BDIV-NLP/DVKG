from __future__ import annotations
import copy
import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def reference():
    entities=[dict(entity_id=n,canonical_name=n,scope='cpu',aliases=['寄存器'] if n=='Register' else []) for n in ['Register','CPU','Memory']]
    facts=[dict(fact_id=str(i),head=h,relation=r,tail=t,conditions=c,evidence_ids=['e']) for i,(h,r,t,c) in enumerate([
        ('Register','part_of','CPU',''),('CPU','consists_of','寄存器',''),
        ('Register','part_of','CPU','读周期'),('CPU','compared_with','Memory',''),('Memory','compared_with','CPU','')])]
    return dict(course_id='course',entities=entities,facts=facts,questions=[])

class SplitTests(unittest.TestCase):
    def test_reciprocal_explicit_aliases_collapse_before_split_and_projection(self):
        from course_experiments.splits import freeze_split,project_graph,entity_lookup,resolve
        ref=dict(course_id='c',entities=[dict(entity_id='cpu',canonical_name='CPU',aliases=['中央处理器'],scope='architecture'),dict(entity_id='cn',canonical_name='中央处理器',aliases=['CPU'],scope='architecture'),dict(entity_id='r',canonical_name='寄存器',aliases=[],scope='architecture'),dict(entity_id='cpu_other',canonical_name='CPU',aliases=[],scope='other')],facts=[dict(fact_id='f1',head='r',relation='part_of',tail='cpu',conditions=''),dict(fact_id='f2',head='r',relation='part_of',tail='cn',conditions='')])
        split=freeze_split(ref)
        self.assertEqual(len(split['groups']),1)
        self.assertEqual(next(iter(split['groups'].values()))['fact_ids'],['f1','f2'])
        graph=dict(entities=[dict(entity_id='gc',canonical_name='CPU',scope='architecture'),dict(entity_id='gr',canonical_name='寄存器',scope='architecture')],relations=[])
        projection=project_graph(graph,ref,split)
        self.assertNotIn('gc',projection['audit']['unmatched_entities'])
        self.assertEqual(len(projection['entity_vocabulary']),3)

    def test_assembly_canonicalizes_source_verified_alias_variants(self):
        from course_experiments.evaluation.reference import assemble
        c=[dict(evidence_id='e',quote='CPU又称中央处理器，含有寄存器。')]
        records=[]
        for canonical,alias in [('CPU','中央处理器'),('中央处理器','CPU')]:
            records.append((dict(book_id='B',chunk_id=canonical),dict(entities=[dict(local_id='cpu',canonical_name=canonical,scope='architecture',aliases=[alias],citations=c),dict(local_id='r',canonical_name='寄存器',scope='architecture',aliases=[],citations=c)],facts=[dict(local_id='f',head='r',tail='cpu',relation='part_of',scope='architecture',citations=c)],questions=[])))
        dataset=assemble('c',records,'smoke','i',dict(questions=1,facts=1))
        self.assertEqual(len(dataset['entities']),2)
        self.assertEqual(len({f['tail'] for f in dataset['facts']}),1)

    def test_protocol_modules_exist(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.splits'))

    def test_inverse_alias_symmetric_condition_groups_and_overlap_rejection(self):
        from course_experiments.splits import freeze_split,validate_split
        split=freeze_split(reference())
        self.assertEqual(len(split['groups']),2)
        self.assertEqual(sorted(len(g['fact_ids']) for g in split['groups'].values()),[2,3])
        changed=copy.deepcopy(split);changed['folds']['test']=changed['folds']['train']
        with self.assertRaises(ValueError):validate_split(changed)

    def test_shared_test_and_remove_reverse_alias_training_leaks(self):
        from course_experiments.splits import freeze_split,project_graph
        ref=reference()
        for i in range(40):
            ref['entities'].append(dict(entity_id='X'+str(i),canonical_name='X'+str(i),scope='cpu',aliases=[]))
            ref['facts'].append(dict(fact_id='x'+str(i),head='X'+str(i),relation='part_of',tail='CPU',conditions='',evidence_ids=['e']))
        split=freeze_split(ref)
        entities=[dict(entity_id=e['entity_id'],canonical_name=e['canonical_name'],scope=e['scope'],type='Concept') for e in ref['entities']]
        graph=dict(entities=entities,relations=[dict(head_entity_id=f['head'],tail_entity_id=f['tail'],relation=f['relation']) for f in ref['facts'] if f['head']!='寄存器' and f['tail']!='寄存器'])
        a=project_graph(graph,ref,split); b=project_graph(dict(entities=entities,relations=[]),ref,split)
        self.assertEqual(a['test'],b['test'])
        self.assertFalse(set(map(tuple,a['training'])) & set(map(tuple,a['test']+a['validation'])))
        self.assertTrue(a['audit']['heldout_removed'])

    def test_unscoped_homonym_not_guessed(self):
        from course_experiments.splits import freeze_split,project_graph
        ref=reference();ref['entities'].append(dict(entity_id='other',canonical_name='Register',scope='other',aliases=[]))
        p=project_graph(dict(entities=[dict(entity_id='g',canonical_name='Register',type='Concept')],relations=[]),ref,freeze_split(ref))
        self.assertIn('g',p['audit']['unmatched_entities'])

class EvaluationTests(unittest.TestCase):
    def test_reference_limits_reject_oversized_records_without_padding(self):
        from course_experiments.evaluation.reference import validate_batch
        unit=dict(evidence=[dict(id='e',text='CPU含有寄存器。')]);schema=dict(relation_types={'part_of':''})
        for field,size in [('entities',33),('facts',17),('questions',5)]:
            citations=[dict(evidence_id='e',quote='CPU含有寄存器')]
            value=dict(entities=[dict(local_id='cpu',canonical_name='CPU',scope='architecture',aliases=[],citations=citations),dict(local_id='r',canonical_name='寄存器',scope='architecture',aliases=[],citations=citations)],facts=[],questions=[])
            if field=='entities':value[field]=[dict(local_id=str(i),canonical_name='CPU',scope='architecture',aliases=[],citations=citations) for i in range(size)]
            elif field=='facts':value[field]=[dict(local_id=str(i),head='r',relation='part_of',tail='cpu',conditions='',citations=citations) for i in range(size)]
            else:value[field]=[dict(type='definition',question='CPU是什么'+str(i),answer='CPU',citations=citations,fact_path=[]) for i in range(size)]
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError,'limit'):validate_batch(value,unit,schema)

    def test_reference_feedback_repairs_alias_and_truncation_using_original_source(self):
        from course_experiments.evaluation.reference import generate,PARAMETERS
        from course_experiments.evaluation.protocol import freeze_phase
        from course_experiments.manifest import file_record
        import threading
        source='CPU又称中央处理器。CPU含有寄存器。'
        citation=lambda quote:[dict(evidence_id='e',quote=quote)]
        good=dict(entities=[dict(local_id='cpu',canonical_name='CPU',aliases=[],scope='architecture',citations=citation('CPU')),dict(local_id='r',canonical_name='寄存器',aliases=[],scope='architecture',citations=citation('寄存器'))],facts=[dict(local_id='f',head='r',relation='part_of',tail='cpu',conditions='',scope='architecture',citations=citation('CPU含有寄存器'))],questions=[dict(type='definition',question='CPU是什么',answer='CPU',citations=citation('CPU'),fact_path=[])])
        bad=copy.deepcopy(good);bad['entities'][0]['aliases']=['中央处理器'];bad['questions'][0]['fact_path']=['f']
        class Runtime:
            config=dict(llm=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',extra_body={'enable_thinking':False}),embedding=dict(base_url='https://api.siliconflow.cn/v1',model='BAAI/bge-m3'))
            gate=threading.BoundedSemaphore(1)
            def __init__(self,mode):self.mode=mode;self.requests=[]
            def _raw_cached(self,key):return None
            def _post(self,endpoint,body,*args):
                self.requests.append(json.loads(body['messages'][1]['content']))
                first=len(self.requests)==1;value=bad if first and self.mode=='alias' else good
                return dict(choices=[dict(message=dict(content=json.dumps(value)),finish_reason='length' if first and self.mode=='truncated' else 'stop')],usage={'total_tokens':10})
        for mode in ('alias','truncated'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);corpus=dict(chunks=[dict(id='c',book_id='B',evidence_ids=['e'])],evidence=[dict(id='e',text=source)]);schema={'relation_types':{'part_of':''}}
                for name,value in [('corpus',corpus),('schema',schema),('runtime',Runtime.config)]:
                    (root/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
                construction=dict(courses={'courses':[dict(course_id='c',description='课程说明',books=[dict(book_id='B')])]},files=dict(assets={'schema':file_record(root/'schema.json'),'runtime_config':file_record(root/'runtime.json')},sources={'B':file_record(root/'corpus.json')}))
                cp=root/'construction.json';cp.write_text(json.dumps(construction),encoding='utf-8');phase=root/'phase.json'
                freeze_phase({'construction':cp},dict(PARAMETERS,smoke_chunks=1),phase);runtime=Runtime(mode)
                with patch('course_experiments.evaluation.reference.validate',return_value=construction):dataset,out=generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(len(runtime.requests),2)
                first,second=runtime.requests
                self.assertNotIn('validation_feedback',first);self.assertIn('validation_feedback',second)
                self.assertEqual(first['unit'],second['unit']);self.assertEqual(first['schema'],second['schema'])
                self.assertEqual(first['unit']['text'],source);self.assertEqual(first['unit']['chunk_id'],'c')
                self.assertEqual(second['output_limits'],{'entities':32,'facts':16,'questions':4})
                if mode=='alias':self.assertIn('co-quoted',second['validation_feedback']['error'])
                else:self.assertIn('compact',second['validation_feedback']['correction'])
                self.assertEqual(len(dataset['facts']),1);self.assertEqual(len(dataset['questions']),1);self.assertEqual(dataset['status'],'partial')

    def test_reference_smoke_targets_are_fixed_and_full_targets_unchanged(self):
        from course_experiments.evaluation.reference import reference_targets
        self.assertEqual(reference_targets(None),{'questions':100,'facts':400})
        self.assertEqual(reference_targets(4),{'questions':8,'facts':24})
        self.assertNotEqual(reference_targets(4),reference_targets(None))
        self.assertEqual(reference_targets(16),{'questions':8,'facts':24})
        with self.assertRaises(ValueError):reference_targets(17)

    def test_mandatory_thinking_disabled_rejected_before_cache_or_model(self):
        from course_experiments.evaluation.chat import call_json
        import threading
        class Runtime:
            config={'llm':{'base_url':'https://api.siliconflow.cn/v1','model':'deepseek-ai/DeepSeek-V4-Flash','extra_body':{'enable_thinking':True}}}
            gate=threading.BoundedSemaphore(1)
            def _raw_cached(self,key):raise AssertionError('Mismatch must fail before cache')
            def _post(self,*a):raise AssertionError('Mismatch must fail before API')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):call_json(Runtime(),'p',{},2048,Path(d))

    def test_runtime_options_are_bound_to_frozen_phase_before_reuse(self):
        from course_experiments.evaluation.protocol import bind_runtime,freeze_phase
        config={'llm':{'base_url':'https://api.siliconflow.cn/v1','model':'deepseek-ai/DeepSeek-V4-Flash','extra_body':{'enable_thinking':False}},'embedding':{'base_url':'https://api.siliconflow.cn/v1','model':'BAAI/bge-m3'}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=root/'runtime.json';cfg.write_text(json.dumps(config));phase=freeze_phase({'runtime_config':cfg},dict(seed=42),root/'phase.json')
            bind_runtime(phase,SimpleNamespace(config=config))
            altered=copy.deepcopy(config);altered['llm']['timeout_seconds']=1
            with self.assertRaises(ValueError):bind_runtime(phase,SimpleNamespace(config=altered))
    def test_invalid_quote_and_disconnected_multihop_rejected(self):
        from course_experiments.evaluation.reference import validate_batch
        unit=dict(book_id='B',chunk_id='C',text='CPU含有寄存器。',evidence=[dict(id='e',text='CPU含有寄存器。')])
        payload=dict(entities=[],facts=[],questions=[dict(type='definition',question='CPU是什么',answer='单元',citations=[dict(evidence_id='e',quote='不存在')])])
        with self.assertRaises(ValueError):validate_batch(payload,unit,{'relation_types':{'part_of':'x'}})

    def test_structural_exclusion_existing_metrics(self):
        from course_experiments.evaluation.topology import evaluate
        graph=dict(entities=[dict(entity_id='a',canonical_name='a',type='Concept'),dict(entity_id='b',canonical_name='b',type='Concept'),dict(entity_id='book',canonical_name='book',type='Book')],relations=[dict(head_entity_id='a',tail_entity_id='b',relation='part_of'),dict(head_entity_id='book',tail_entity_id='a',relation='part_of')])
        result=evaluate(graph,ROOT/'.runtime_cache/evaluators_3090_20261014/v2_source/scripts/evaluate_kg_quality.py')
        self.assertEqual(result['entity_count'],2);self.assertEqual(result['metrics']['largest_component_size'],2)
        self.assertEqual(result['metrics']['average_total_degree'],1)
        self.assertEqual(result['utils_dependency']['sha256'],'edc2e128bc32143a52f2ca12cb18428d511e2adae112421ada96578b84c0c691')

    def test_native_undirected_topology_labels_directional_scores_inapplicable(self):
        from course_experiments.evaluation.topology import evaluate
        graph=dict(nodes=[dict(id='a',name='A',type='Concept'),dict(id='b',name='B',type='Concept'),dict(id='skill',name='skill',type='Skill')],edges=[dict(source='a',target='b',description='对比'),dict(source='a',target='skill',description='结构层')],directed=False)
        result=evaluate(graph,ROOT/'.runtime_cache/evaluators_3090_20261014/v2/evaluate_kg_quality.py',native_undirected=True)
        self.assertEqual(result['entity_count'],2);self.assertEqual(result['metrics']['largest_component_size'],2);self.assertIn('not_applicable',result['directional_metrics_applicability'])

    def test_patch_is_guarded_and_optional_old_path_preserved(self):
        from course_experiments.evaluation.kge import patch_evaluators
        source=ROOT/'.runtime_cache/evaluators_3090_20261014/v2'
        with tempfile.TemporaryDirectory() as d:
            result=patch_evaluators(source/'evaluate_kg_quality.py',source/'run_pykeen_epoch_sweep.py',Path(d))
            text=Path(result['evaluator']['path']).read_text('utf-8')
            self.assertIn('factory.split(',text);self.assertIn('external_protocol',text)
            self.assertIn('pipeline(',text);self.assertTrue(Path(result['diff_path']).exists())
            bad=Path(d)/'bad.py';bad.write_text('bad')
            with self.assertRaises(ValueError):patch_evaluators(bad,source/'run_pykeen_epoch_sweep.py',Path(d)/'other')

    def test_rag_shared_index_no_g0_evidence_fabrication_and_identity(self):
        from course_experiments.evaluation.rag import retrieve,result_identity,score_answer
        index=[dict(chunk_id='c',book_id='B',text='CPU含有寄存器。',evidence_ids=['e'],vector=[1.,0.])]
        g0=dict(entities=[dict(entity_id='g0',canonical_name='CPU',type='Concept')],relations=[])
        context=retrieve('CPU',[1.,0.],index,g0,'graph')
        self.assertEqual(context[0]['evidence_ids'],['e']);self.assertNotIn('g0',context[0]['evidence_ids'])
        x=result_identity('q','i','g',['B'],'graph');self.assertNotEqual(x,result_identity('q','i','g',['A'],'graph'))
        self.assertNotEqual(result_identity('q','i','g',['B'],'graph',phase_id='code1'),result_identity('q','i','g',['B'],'graph',phase_id='code2'))
        self.assertEqual(score_answer('寄存器','寄存器')['f1'],1)

    def test_phase_mutation_rejected(self):
        from course_experiments.evaluation.protocol import freeze_phase,validate_phase
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'source.json';f.write_text('{}')
            p=Path(d)/'phase.json';freeze_phase({'source':f},{'seed':42},p)
            f.write_text('{"changed":true}')
            with self.assertRaises(ValueError):validate_phase(p)

    def test_whole_context_byte_ceiling_and_foreign_citation_rejected(self):
        from course_experiments.evaluation.rag import context_rows,citations_score
        rows=[dict(chunk_id='c',book_id='B',text='寄存器'*2000,evidence_ids=['e'],evidence=[dict(id='e',text='寄存器'*2000)])]
        context=context_rows(rows)
        self.assertLessEqual(len(json.dumps(context,ensure_ascii=False).encode('utf-8')),3000)
        self.assertEqual(citations_score([dict(evidence_id='invented',quote='寄存器')],context,rows)['valid_citation_count'],0)

    def test_reference_requires_connected_unique_multihop_path(self):
        from course_experiments.evaluation.reference import validate_batch
        unit=dict(book_id='B',chunk_id='C',text='CPU包含寄存器。内存连接总线。',evidence=[dict(id='e',text='CPU包含寄存器。内存连接总线。')])
        cites=[dict(evidence_id='e',quote=unit['text'])]
        value=dict(entities=[dict(local_id=x,canonical_name=x,scope='cpu',citations=cites) for x in ['CPU','寄存器','内存','总线']],facts=[dict(local_id='f1',head='寄存器',tail='CPU',relation='part_of',citations=cites),dict(local_id='f2',head='内存',tail='总线',relation='affects',citations=cites)],questions=[dict(type='multihop',question='多跳?',answer='答',citations=cites,fact_path=['f1','f2'])])
        with self.assertRaises(ValueError):validate_batch(value,unit,dict(relation_types={'part_of':'','affects':''}))
        value['questions'][0]['fact_path']=['f1','f1']
        with self.assertRaises(ValueError):validate_batch(value,unit,dict(relation_types={'part_of':'','affects':''}))

    def test_reference_fact_requires_quoted_endpoint_names(self):
        from course_experiments.evaluation.reference import validate_batch
        unit=dict(evidence=[dict(id='e',text='CPU含有寄存器。')])
        cites=[dict(evidence_id='e',quote='CPU含有寄存器。')]
        value=dict(entities=[dict(local_id=x,canonical_name=x,scope='cpu',citations=cites) for x in ['CPU','不存在']],facts=[dict(local_id='f',head='不存在',tail='CPU',relation='part_of',citations=cites)],questions=[])
        with self.assertRaises(ValueError):validate_batch(value,unit,dict(relation_types={'part_of':''}))

    def test_rag_complete_checkpoint_shared_budget_and_unanswerable_not_zero(self):
        from course_experiments.evaluation.rag import evaluate,BUDGET
        from course_experiments.evaluation.protocol import freeze_phase
        from course_experiments.manifest import digest
        import threading
        cites=[dict(evidence_id='e',quote='CPU')]
        ref=dict(course_id='c',scope='smoke',status='complete',human_audited=False,entities=[],facts=[],questions=[dict(question_id='q',question='CPU是什么',answer='CPU',citations=cites,evidence_ids=['e']),dict(question_id='other',question='B?',answer='B',citations=[dict(evidence_id='outside',quote='B')],evidence_ids=['outside'])],targets={'questions':2,'facts':0});ref['reference_id']=digest(ref)
        index=dict(source_id='s',embedding_model='BAAI/bge-m3',rows=[dict(chunk_id='chunk',book_id='B',text='CPU',evidence_ids=['e'],evidence=[dict(id='e',text='CPU')],vector=[1.,0.])]);index['index_id']=digest(index)
        graph={'entities':[],'relations':[]}
        class Runtime:
            config=dict(llm=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',extra_body={'enable_thinking':False,'temperature':1,'max_tokens':999}),embedding={'base_url':'https://api.siliconflow.cn/v1','model':'BAAI/bge-m3'})
            gate=threading.BoundedSemaphore(1)
            requests=[]
            def embed(self,texts):return [[1.,0.] for t in texts]
            def _raw_cached(self,key):return None
            def _post(self,endpoint,body,*args):
                self.requests.append(body)
                value={'score':1,'reason':'匹配'} if body['max_tokens']==1024 else {'answer':'CPU','citations':cites}
                return dict(choices=[dict(message=dict(content=json.dumps(value)),finish_reason='stop')],usage={'total_tokens':10})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths={}
            for role,obj in [('reference',ref),('index',index),('graph',graph),('runtime_config',Runtime.config)]:
                path=root/(role+'.json');path.write_text(json.dumps(obj));paths[role]=path
            phase=root/'phase.json';freeze_phase(paths,dict(BUDGET,control='graph',active_books=['B']),phase);runtime=Runtime()
            result=evaluate(ref,index,graph,['B'],'graph',phase,root/'out',runtime)
            again=evaluate(ref,index,graph,['B'],'graph',phase,root/'out',runtime)
            self.assertEqual(result,again);self.assertEqual(len(runtime.requests),2)
            self.assertEqual([b['max_tokens'] for b in runtime.requests],[2048,1024]);self.assertEqual(result['completed_count'],1);self.assertEqual(result['unanswerable_count'],1);self.assertEqual(result['metrics']['f1'],1)

    def test_model_failure_stage_is_recorded_before_propagation(self):
        from course_experiments.evaluation.chat import call_json
        import threading
        class Runtime:
            config=dict(llm=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',extra_body={'enable_thinking':False}))
            gate=threading.BoundedSemaphore(1)
            def _raw_cached(self,key):return None
            def _post(self,*args):raise RuntimeError('Model service returned HTTP 402')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):call_json(Runtime(),'p',{},2048,Path(d))
            records=list(Path(d).glob('*.error.json'))
            self.assertEqual(len(records),1);self.assertEqual(json.loads(records[0].read_text())['error_type'],'RuntimeError')

    def test_reference_duplicate_ids_do_not_create_ambiguous_alias(self):
        from course_experiments.evaluation.reference import assemble
        records=[(dict(book_id='B',chunk_id='C'),dict(entities=[dict(local_id='cpu',canonical_name='CPU',scope='architecture',aliases=[],citations=[dict(evidence_id='e',quote='CPU')])],facts=[],questions=[]))]
        data=assemble('c',records,'smoke','i',dict(questions=1,facts=1))
        self.assertEqual(data['status'],'partial');self.assertFalse(data['human_audited'])

    def test_checkpoint_reuses_raw_answer_without_second_model_request(self):
        from course_experiments.evaluation.chat import call_json
        import threading
        class Runtime:
            config=dict(llm=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',extra_body={'enable_thinking':False,'temperature':1,'max_tokens':999}))
            gate=threading.BoundedSemaphore(1)
            count=0
            def _raw_cached(self,key):return None
            def _post(self,*a):
                self.count+=1
                self.body=a[1]
                return dict(choices=[dict(message=dict(content='{"answer":"CPU"}'),finish_reason='stop')],usage={'total_tokens':4})
        with tempfile.TemporaryDirectory() as d:
            runtime=Runtime();first=call_json(runtime,'prompt',{},2048,Path(d));second=call_json(runtime,'prompt',{},2048,Path(d))
            self.assertEqual(runtime.count,1);self.assertEqual(first,second)
            self.assertFalse(runtime.body['enable_thinking']);self.assertEqual(runtime.body['temperature'],0);self.assertEqual(runtime.body['max_tokens'],2048)

    def test_native_descriptive_graph_navigates_without_invented_predicate(self):
        from course_experiments.evaluation.rag import retrieve
        graph=dict(nodes=[dict(id='cpu',name='CPU',type='Concept'),dict(id='r',name='寄存器',type='Concept')],edges=[dict(source='cpu',target='r',description='包含')],directed=False)
        index=[dict(chunk_id='a',book_id='B',text='无关',evidence_ids=['a'],vector=[1.,0.]),dict(chunk_id='b',book_id='B',text='寄存器',evidence_ids=['b'],vector=[1.,0.])]
        self.assertEqual(retrieve('CPU',[1.,0.],index,graph,'graph')[0]['chunk_id'],'b')

    def test_evidence_recall_requires_quote_in_visible_context(self):
        from course_experiments.evaluation.rag import evidence_recall
        question=dict(citations=[dict(evidence_id='e',quote='仅在末尾')])
        self.assertEqual(evidence_recall(question,[dict(evidence_ids=['e'],text='开头')]),0)

    def test_reference_freeze_rejects_partial_and_changed_identity(self):
        from course_experiments.evaluation.reference import freeze_reference
        with tempfile.TemporaryDirectory() as d:
            value=dict(reference(),status='partial',scope='full',reference_id='invalid')
            with self.assertRaises(ValueError):freeze_reference(value,Path(d)/'ref.json')

    def test_reference_identity_verified_before_rag_scoring(self):
        from course_experiments.evaluation.reference import validate_reference
        value=dict(reference(),status='complete',human_audited=False,scope='full',targets=dict(questions=100,facts=400),reference_id='invalid')
        with self.assertRaises(ValueError):validate_reference(value)

    def test_cli_exposes_required_stages(self):
        from course_experiments.evaluation.__main__ import parser
        args=parser().parse_args(['project-kge','--reference','r','--split','s','--graph','g','--output','o'])
        self.assertEqual(args.stage,'project-kge')

    def test_topology_cli_rejects_different_frozen_utils_dependency(self):
        from course_experiments.evaluation.__main__ import main
        from course_experiments.evaluation.protocol import freeze_phase
        graph=dict(entities=[dict(entity_id='a',canonical_name='CPU',type='Concept')],relations=[])
        evaluator=ROOT/'.runtime_cache/evaluators_3090_20261014/v2_source/scripts/evaluate_kg_quality.py'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'graph.json';p.write_text(json.dumps(graph));phase=root/'phase.json'
            freeze_phase({'graph':p,'evaluator_original':evaluator,'src_utils':ROOT/'src/utils.py'},dict(seed=42),phase)
            with self.assertRaises(ValueError):main(['topology','--graph',str(p),'--evaluator',str(evaluator),'--phase',str(phase),'--output',str(root/'result.json')])

    def test_external_pykeen_uses_fixed_factories_budget_and_graph_guard(self):
        from course_experiments.manifest import digest
        from course_experiments.splits import freeze_split,project_graph
        from course_experiments.evaluation.protocol import freeze_phase
        from course_experiments.evaluation.kge import prepare_protocol,run_external,patch_evaluators
        import numpy as np
        facts=[dict(fact_id=f'{i}-{j}',head=f'N{i}',relation='part_of',tail=f'N{(i+j)%20}',conditions='') for i in range(20) for j in [1,2,3]]
        ref=dict(course_id='c',entities=[dict(entity_id=f'N{i}',canonical_name=f'N{i}',scope='cpu',aliases=[]) for i in range(20)],facts=facts)
        graph=dict(entities=[dict(entity_id=e['entity_id'],canonical_name=e['canonical_name'],scope='cpu',type='Concept') for e in ref['entities']],relations=[dict(head_entity_id=f['head'],tail_entity_id=f['tail'],relation=f['relation']) for f in facts])
        split=freeze_split(ref);projection=project_graph(graph,ref,split);calls=[]
        class Factory:
            @classmethod
            def from_labeled_triples(cls,triples,**kwargs):
                o=cls();o.triples=triples.tolist();o.mapped_triples=triples;o.vocab=kwargs;return o
            def split(self,*a,**k):raise AssertionError('Random split forbidden')
        class Result:
            metric_results=SimpleNamespace(to_dict=lambda:{'mrr':.5})
            model=object()
            def save_to_directory(self,path):Path(path).mkdir()
        def pipeline(**kwargs):calls.append(kwargs);return Result()
        class Rank:
            def __init__(self,**kwargs):self.ranks={('head','realistic'):[np.array([1,2])]}
            def evaluate(self,**kwargs):pass
        modules={name:ModuleType(name) for name in ['torch','pykeen','pykeen.pipeline','pykeen.triples','pykeen.evaluation']}
        modules['torch'].cuda=SimpleNamespace(is_available=lambda:False);modules['pykeen.pipeline'].pipeline=pipeline;modules['pykeen.triples'].TriplesFactory=Factory;modules['pykeen.evaluation'].RankBasedEvaluator=Rank
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);phase_path=root/'phase.json'
            source=ROOT/'.runtime_cache/evaluators_3090_20261014/v2';patched=patch_evaluators(source/'evaluate_kg_quality.py',source/'run_pykeen_epoch_sweep.py',root/'scripts')
            paths={'evaluator_original':source/'evaluate_kg_quality.py','sweep_original':source/'run_pykeen_epoch_sweep.py','evaluator_patched':patched['evaluator']['path'],'sweep_patched':patched['sweep']['path'],'src_utils':ROOT/'src/utils.py'}
            for role,value in [('graph',graph),('projection',projection),('fact_split',split),('construction',{}),('gpu_environment',{}),('dependency_lock',{})]:
                path=root/(role+'.json');path.write_text(json.dumps(value));paths[role]=path
            phase=freeze_phase(paths,dict(seed=42),phase_path);phase['path']=str(phase_path)
            proto=prepare_protocol(projection,split,phase,root/'protocol.json',smoke=True)
            args=SimpleNamespace(external_protocol=root/'protocol.json',seed=42,pykeen_model='TransE',pykeen_epochs=2,pykeen_device='auto')
            with patch.dict('sys.modules',modules):
                result=run_external(SimpleNamespace(concept_kg=graph),args,root/'result')
                self.assertEqual(result['status'],'completed');self.assertEqual(calls[0]['training_kwargs'],{'num_epochs':2,'batch_size':256})
                self.assertEqual(calls[0]['model_kwargs'],{'embedding_dim':128});self.assertEqual(calls[0]['negative_sampler_kwargs'],{'num_negs_per_pos':16})
                self.assertEqual(calls[0]['testing'].triples,[t for t,m in zip(projection['test'],projection['covered_test_mask']) if m])
                with self.assertRaises(ValueError):run_external(SimpleNamespace(concept_kg={'entities':[],'relations':[]}),args,root/'bad')
                def fail_pipeline(**kwargs):
                    self.assertTrue((root/'pipeline_failure/external_kge_identity.json').exists())
                    self.assertTrue((root/'pipeline_failure/external_kge_protocol.json').exists())
                    raise RuntimeError('CUDA out of memory simulated')
                modules['pykeen.pipeline'].pipeline=fail_pipeline
                with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):run_external(SimpleNamespace(concept_kg=graph),args,root/'pipeline_failure')
                failed=json.loads((root/'pipeline_failure/external_kge_result.json').read_text('utf-8'))
                self.assertEqual(failed['status'],'failed');self.assertEqual(failed['failure_stage'],'training');self.assertEqual(failed['error_type'],'RuntimeError')
                modules['pykeen.pipeline'].pipeline=pipeline
                class SaveFailure(Result):
                    def save_to_directory(self,path):
                        Path(path).mkdir();(Path(path)/'partial_model.marker').write_text('retained');raise OSError('Model save simulated failure')
                modules['pykeen.pipeline'].pipeline=lambda **kwargs:SaveFailure()
                with self.assertRaises(OSError):run_external(SimpleNamespace(concept_kg=graph),args,root/'save_failure')
                saved=json.loads((root/'save_failure/external_kge_result.json').read_text('utf-8'));self.assertEqual(saved['failure_stage'],'model_save');self.assertTrue((root/'save_failure/model/partial_model.marker').exists())
                modules['pykeen.pipeline'].pipeline=pipeline
                class RankFailure(Rank):
                    def evaluate(self,**kwargs):raise RuntimeError('Optional ranking simulated failure')
                modules['pykeen.evaluation'].RankBasedEvaluator=RankFailure
                diagnosed=run_external(SimpleNamespace(concept_kg=graph),args,root/'rank_failure')
                self.assertEqual(diagnosed['status'],'completed_with_diagnostic_failure');self.assertEqual(diagnosed['training_status'],'completed');self.assertEqual(diagnosed['diagnostics_status'],'failed')
                self.assertTrue((root/'rank_failure/pykeen_metrics.json').exists());self.assertTrue((root/'rank_failure/model').exists());self.assertTrue(list((root/'rank_failure').glob('failure_*.json')))

    def test_gpu_export_omits_raw_quotes_and_preserves_logical_split(self):
        from course_experiments.splits import freeze_split,project_graph
        from course_experiments.evaluation.protocol import freeze_phase
        from course_experiments.evaluation.kge import export_kge_package,validate_kge_split
        ref=reference();ref['facts'][0]['conditions']='不可传到GPU的原文条件引句'
        graph=dict(entities=[dict(entity_id=e['entity_id'],canonical_name=e['canonical_name'],scope='cpu',type='Concept',definition='不可传到GPU的原文条件引句') for e in ref['entities']],relations=[])
        split=freeze_split(ref);projection=project_graph(graph,ref,split)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths={}
            for role,obj in [('graph',graph),('projection',projection),('fact_split',split)]:
                f=root/(role+'.json');f.write_text(json.dumps(obj));paths[role]=f
            phase=freeze_phase(paths,dict(seed=42),root/'phase.json')
            package=export_kge_package(projection,split,root/'phase.json',root/'gpu')
            sanitized=json.loads((root/'gpu/fact_split.json').read_text())
            validate_kge_split(sanitized)
            self.assertEqual(sanitized['split_id'],split['split_id'])
            self.assertNotIn('不可传到GPU的原文条件引句',''.join(p.read_text('utf-8') for p in (root/'gpu').glob('*.json')))
            self.assertEqual(package['attestation']['graph_id'],projection['graph_id'])

if __name__=='__main__':unittest.main()
