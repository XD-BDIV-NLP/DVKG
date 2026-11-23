from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from course_experiments.dual_view import build_structure
from course_experiments.evaluation.native_provider import AuditedNativeProvider, NativeBudgetExceeded


CONFIG=dict(llm=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',
    extra_body=dict(enable_thinking=False)),embedding=dict(base_url='https://api.siliconflow.cn/v1',model='BAAI/bge-m3'))

def structure(book='book_A'):
    corpus=dict(books=[dict(id=book)],chunks=[],evidence=[])
    for i,text in enumerate(['CPU由运算器组成。运算器执行运算。','中央处理器使用缓存。缓存保存数据。']):
        eid=f'e{i}';cid=f'c{i}'
        corpus['chunks'].append(dict(id=cid,book_id=book,evidence_ids=[eid]))
        corpus['evidence'].append(dict(id=eid,book_id=book,text=text,chunk_id=cid,
            section_path_ids=['ch',f's{i}'],title_path=['组成',f'小节{i}'],section_id=f's{i}',chapter_id='ch',paragraph_order=i))
    return build_structure(corpus)

class FakeProvider(AuditedNativeProvider):
    def __init__(self):
        self.runtime=SimpleNamespace(config=copy.deepcopy(CONFIG));self.calls=[];self.fail_after=None
        self.malformed=False;self.zero_embedding=False;self.fatal=False

    def raise_if_fatal(self):
        if self.fatal:raise NativeBudgetExceeded('test frozen budget exhausted')

    def chat_response(self,messages,**kwargs):
        self.raise_if_fatal()
        payload=json.loads(messages[-1]['content']);stage=payload['operation'];value=payload['input']
        self.calls.append(stage)
        if self.fail_after is not None and len(self.calls)>=self.fail_after:
            raise ConnectionError('temporary fixture transport error')
        if stage in ['summary','context_summary']:
            out=dict(summary='处理器与运算器、缓存。')
        elif stage=='entities':
            out=dict(entities=[dict(name='CPU',aliases=['中央处理器'],description='执行指令'),
                               dict(name='缓存',aliases=[],description='保存数据'),
                               dict(name='存储器',aliases=[],description='存放程序与数据')])
        elif stage in ['entity_relations','section_relations']:
            ids=[n['id'] for n in value['nodes']]
            out=dict(relations=[dict(source=ids[0],target=ids[1],predicate='used_for',description='使用关系')] if len(ids)>1 else [])
        elif stage=='conv':
            out=dict(updates=[dict(id=n['id'],description=n['description']+'；局部结构说明') for n in value['nodes']])
        elif stage=='roles':
            out=dict(roles=[dict(id=n['id'],role='non-core' if n['name']=='缓存' else 'core') for n in value['nodes']])
        elif stage=='dedup':out=dict(equivalent=value['left']['name']==value['right']['name'])
        elif stage=='prediction':out=dict(is_relevant=True,strength=9,predicate='affects',description='可检验的推断关系')
        else:raise AssertionError('Unexpected fixture operation '+stage)
        content='not JSON' if self.malformed else json.dumps(out,ensure_ascii=False)
        return dict(choices=[dict(message=dict(content=content),finish_reason='stop')],
            usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2),_native_cache_hit=False)

    def embed(self,texts,**kwargs):
        self.raise_if_fatal()
        result=np.zeros((len(texts),1024),dtype=np.float32)
        if not self.zero_embedding:
            for i,text in enumerate(texts):
                result[i,0 if '执行' in text or 'CPU' in text else 1]=1
        return result


class TreeKGTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.baselines.tree_kg'),
                             'Tree-KG reconstruction module is not implemented')
        self.tree=importlib.import_module('course_experiments.baselines.tree_kg')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.provider=FakeProvider()

    def engine(self,**kwargs):
        return self.tree.TreeKG(self.provider,self.root,**kwargs)

    def test_complete_flow_preserves_hierarchy_sources_roles_and_all_operator_receipts(self):
        result=self.engine().build_course('computer_organization',{'book_A':structure()})
        graph=result['native_graph'];ids={n['id'] for n in graph['nodes']}
        self.assertEqual(result['status'],'complete')
        self.assertFalse(result['unmodified_native_reproduction'])
        self.assertTrue(all(e['source'] in ids and e['target'] in ids for e in graph['edges']))
        self.assertTrue({'has_subsection','has_entity','has_subordinate'}<=set(e['predicate'] for e in graph['edges']))
        self.assertEqual({e['id'] for e in result['evidence']},{'book_A/evidence/e0','book_A/evidence/e1'})
        self.assertTrue({'summary','entities','entity_relations','section_relations','conv','roles','dedup','prediction'}<=set(self.provider.calls))
        self.assertTrue({'conv','aggr','embed','dedup','pred_stage1','pred_stage2','merge'}<=set(result['executed_operators']))

    def test_resume_uses_completed_checkpoint_without_more_provider_calls(self):
        engine=self.engine();first=engine.build_course('computer_organization',{'book_A':structure()})
        count=len(self.provider.calls)
        second=engine.build_course('computer_organization',{'book_A':structure()})
        self.assertEqual(first,second);self.assertEqual(len(self.provider.calls),count)

    def test_partial_transport_failure_recovers_cached_initial_work(self):
        self.provider.fail_after=4
        with self.assertRaises(ConnectionError):self.engine().build_course('computer_organization',{'book_A':structure()})
        self.assertFalse((self.root/'artifact.json').exists())
        completed_summaries=self.provider.calls.count('summary')
        self.provider.fail_after=None
        self.engine().build_course('computer_organization',{'book_A':structure()})
        self.assertLessEqual(self.provider.calls.count('summary'),completed_summaries+3)

    def test_inherited_budget_halt_is_never_swallowed_or_marked_complete(self):
        self.provider.fatal=True
        with self.assertRaises(NativeBudgetExceeded):self.engine().build_course('computer_organization',{'book_A':structure()})
        self.assertEqual(self.provider.calls,[]);self.assertFalse((self.root/'artifact.json').exists())

    def test_malformed_model_output_is_bounded_and_raw_attempts_are_preserved(self):
        self.provider.malformed=True
        with self.assertRaises(ValueError):self.engine().build_course('computer_organization',{'book_A':structure()})
        self.assertEqual(len(self.provider.calls),2)
        self.assertGreaterEqual(len(list(self.root.glob('requests/*/attempts/*.json'))),2)
        self.assertFalse((self.root/'artifact.json').exists())

    def test_zero_embedding_cannot_create_fake_success(self):
        self.provider.zero_embedding=True
        with self.assertRaisesRegex(ValueError,'vector|embedding'):self.engine().build_course('computer_organization',{'book_A':structure()})
        self.assertFalse((self.root/'artifact.json').exists())

    def test_non_audited_provider_is_rejected(self):
        with self.assertRaises(TypeError):self.tree.TreeKG(object(),self.root)

    def test_foreign_endpoint_config_is_rejected_before_calls(self):
        self.provider.runtime.config['llm']['base_url']='https://other.invalid/v1'
        with self.assertRaises(ValueError):self.engine()
        self.assertEqual(self.provider.calls,[])

    def test_foreign_or_conflicting_structure_is_rejected_before_calls(self):
        bad=structure();bad['evidence']['e0']['book_id']='other'
        with self.assertRaises(ValueError):self.engine().build_course('computer_organization',{'book_A':bad})
        self.assertEqual(self.provider.calls,[])

    def test_modified_config_cannot_overwrite_an_existing_identity(self):
        self.engine().build_course('computer_organization',{'book_A':structure()})
        with self.assertRaisesRegex(ValueError,'Immutable|identity|config'):
            self.engine(config={'strength_threshold':8}).build_course('computer_organization',{'book_A':structure()})

    def test_artifact_tampering_is_detected_without_fresh_model_calls(self):
        engine=self.engine();engine.build_course('computer_organization',{'book_A':structure()})
        path=self.root/'artifact.json';data=json.loads(path.read_text('utf-8'));data['status']='fake';path.write_text(json.dumps(data),'utf-8')
        count=len(self.provider.calls)
        with self.assertRaisesRegex(ValueError,'digest|changed'):engine.build_course('computer_organization',{'book_A':structure()})
        self.assertEqual(count,len(self.provider.calls))

    def test_structured_merge_keeps_all_book_roots_and_namespaced_evidence(self):
        result=self.engine().build_course('computer_organization',{'book_A':structure(),'book_B':structure('book_B')})
        self.assertEqual(result['book_ids'],['book_A','book_B'])
        self.assertEqual(len([n for n in result['native_graph']['nodes'] if n['type']=='Book']),2)
        self.assertEqual(len(result['evidence']),4)
        merged=[n for n in result['native_graph']['nodes'] if n.get('role')=='core']
        self.assertTrue(any(set(n['book_ids'])=={'book_A','book_B'} for n in merged))

    def test_predicted_edges_are_inferred_without_fabricated_source_support(self):
        result=self.engine().build_course('computer_organization',{'book_A':structure()})
        predicted=[e for e in result['native_graph']['edges'] if e.get('origin')=='prediction']
        self.assertTrue(predicted)
        self.assertTrue(all(e['is_inferred'] and not e['evidence_ids'] for e in predicted))
        self.assertTrue(all(r['is_inferred'] for r in result['semantic_view']['relations'] if r.get('origin')=='prediction'))

    def test_neighbor_candidates_respect_roles_threshold_and_top20(self):
        nodes=[dict(id='a',role='core'),dict(id='b',role='non-core'),dict(id='c',role='core')]
        vectors=np.asarray([[1.,0.],[1.,0.],[.999,.01]],dtype=np.float32)
        pairs=self.tree.nearest_pairs(nodes,vectors,neighbors=20,distance_limit=.55,same_role=True)
        self.assertEqual([(a,b) for a,b,_ in pairs],[('a','c')])

    def test_prediction_score_keeps_semantic_AA_and_common_ancestor_terms(self):
        graph=dict(nodes=[dict(id=x,type='Concept') for x in ['u','v','w']]+[dict(id='s',type='Section')],
            edges=[dict(source='s',target=x,predicate='has_entity') for x in ['u','v','w']]+[
                dict(source='u',target='w',predicate='entity_related'),dict(source='v',target='w',predicate='entity_related')],directed=True)
        score=self.tree.pair_score(graph,'u','v',.5,stage=1)
        self.assertAlmostEqual(score['score'],.6*.5+.4*score['common_ancestors'])
        score2=self.tree.pair_score(graph,'u','v',.5,stage=2)
        self.assertGreater(score2['adamic_adar'],0)
        self.assertAlmostEqual(score2['score'],.6*.5+.3*score2['adamic_adar']+.1*score2['common_ancestors'])

    def test_vertical_cycle_is_rejected_before_model_calls(self):
        from course_experiments.manifest import digest
        bad=structure();bad['nodes']['book_A/book']['parent']='book_A/section/s0'
        bad['structure_id']=digest({k:v for k,v in bad.items() if k!='structure_id'})
        with self.assertRaisesRegex(ValueError,'cycle'):self.engine().build_course('computer_organization',{'book_A':bad})
        self.assertEqual(self.provider.calls,[])

    def test_CA_uses_maximum_lineage_intersection_instead_of_union(self):
        graph=dict(nodes=[dict(id=x) for x in ['r','a','b','u','v']],edges=[
            dict(source=a,target=b,predicate='has_entity') for a,b in [('r','a'),('r','b'),('a','u'),('b','u'),('a','v'),('b','v')]])
        result=self.tree.pair_score(graph,'u','v',0.,stage=1)
        self.assertEqual(result['common_ancestors'],2)

    def test_predicted_role_layers_and_repaired_attempt_limit_survive_restart(self):
        self.provider.malformed=True
        with self.assertRaises(ValueError):self.engine().build_course('computer_organization',{'book_A':structure()})
        calls=len(self.provider.calls)
        with self.assertRaisesRegex(ValueError,'repair exhausted'):self.engine().build_course('computer_organization',{'book_A':structure()})
        self.assertEqual(calls,len(self.provider.calls))


if __name__=='__main__':unittest.main()
