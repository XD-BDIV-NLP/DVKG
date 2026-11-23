from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from course_experiments.prior import generate_prior
from course_experiments.textbook import TextbookEngine, scoped_corpus
from course_experiments.contributions import export_contribution, write_immutable
from tests.test_incremental_graph_extraction import ExtractionRuntime

SCHEMA = dict(entity_types={'Structure': '结构'}, relation_types={'part_of': '部分到整体'})
CONFIG = dict(domain='课程描述', workers=8, prior_concepts=60, prior_relations=100,
              llm={'model':'test'}, embedding={'model':'test'})

def task_temp():
    root = Path(__file__).resolve().parents[2] / '.runtime_cache/task2-tests'
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)

def corpus():
    return dict(fingerprint='source', books=[dict(id='b', title='教材')],
        evidence=[dict(id='ev1', book_id='b', section_id='s', text='缓存属于存储系统。')],
        chunks=[dict(id='c1',book_id='b',section_id='s',title_path=['存储'],evidence_ids=['ev1'])])

class ConstructionTests(unittest.TestCase):
    def test_prior_only_and_freezes_once(self):
        runtime=ExtractionRuntime()
        def call(stage,payload,validator):
            self.assertEqual(stage,'prior')
            self.assertEqual(set(payload),{'domain','schema','max_concepts','max_relations'})
            self.assertEqual(payload['domain'],'课程描述')
            runtime.calls.append(stage)
            result=dict(concepts=[dict(id='g1',name='缓存',entity_type='Structure',aliases=[],definition='存储区',scope='',evidence_ids=[])],relations=[])
            validator(result)
            return result
        runtime.call=call
        with task_temp() as tmp:
            result=generate_prior('course', '课程描述', SCHEMA, CONFIG, runtime, Path(tmp), 'identity')
            self.assertEqual(result,generate_prior('course','课程描述',SCHEMA,CONFIG,runtime,Path(tmp),'identity'))
            self.assertEqual(runtime.calls,['prior'])
            self.assertFalse(result['graph']['evidence'])
            self.assertTrue(all(not c['evidence_ids'] for c in result['graph']['claims'].values()))
            with self.assertRaises(ValueError):
                generate_prior('course','课程描述',SCHEMA,CONFIG,runtime,Path(tmp),'changed')

    def test_seedless_real_engine_and_export(self):
        runtime=ExtractionRuntime()
        with task_temp() as tmp:
            engine=TextbookEngine(corpus(),tmp,SCHEMA,CONFIG,runtime)
            try:
                engine.seed()
                self.assertEqual(runtime.calls,[])
                self.assertFalse(engine.store.snapshot()['nodes'])
                engine._commit(engine.analyze(corpus()['chunks'][0],engine._snapshot()))
                result=export_contribution(engine.store.snapshot(),'course','b','full','input')
                nodes=result['graph']['nodes']
                self.assertEqual(len(nodes),2)
                self.assertTrue(all(n.startswith('course/b/node/') for n in nodes))
                for claim in result['graph']['claims'].values():
                    self.assertTrue(claim['evidence_ids'])
                    content=claim['content']
                    for key in ('node_id','head','tail'):
                        if key in content: self.assertIn(content[key],nodes)
                self.assertEqual(len(result['supports']),len(result['graph']['claims']))
                write_immutable(Path(tmp)/'contribution.json',result)
                write_immutable(Path(tmp)/'contribution.json',result)
                with self.assertRaises(ValueError):write_immutable(Path(tmp)/'contribution.json',{'changed':True})
                graph=engine.store.snapshot()
                next(iter(graph['claims'].values()))['evidence_ids']=[]
                with self.assertRaises(ValueError):export_contribution(graph,'course','b','full','input')
            finally: engine.store.close()

    def test_smoke_full_and_source_preservation(self):
        raw=corpus(); original=copy.deepcopy(raw)
        full=scoped_corpus(raw,None); smoke=scoped_corpus(raw,1)
        self.assertNotEqual(full['fingerprint'],smoke['fingerprint'])
        self.assertEqual(raw,original)
        self.assertEqual(smoke['evidence'][0]['id'],'ev1')
        for value in (0,-1,True):
            with self.assertRaises(ValueError):scoped_corpus(raw,value)

class ConstructionBoundaryTests(unittest.TestCase):
    def test_assets_include_actual_runtime_and_all_prompts(self):
        from course_experiments.construction import implementation_assets
        assets=implementation_assets()
        self.assertTrue(any(p.name=='engine.py' for p in assets.values()))
        self.assertTrue(any(p.name=='runtime.py' for p in assets.values()))
        self.assertTrue(any(p.name=='store.py' for p in assets.values()))
        self.assertTrue(any(p.name=='07_同义概念判定_user.txt' for p in assets.values()))

    def test_runtime_values_rejected_before_network(self):
        from course_experiments.construction import check_config
        with self.assertRaises(ValueError):check_config(CONFIG)

class ConstructionExecutionTests(unittest.TestCase):
    def fixture(self, root):
        import json
        from course_experiments.tests.test_manifest_matrix import ManifestMatrixTests
        from course_experiments.manifest import freeze,digest
        from course_experiments.construction import implementation_assets
        courses,experiments,sources,assets=ManifestMatrixTests().fixture(root)
        config=copy.deepcopy(CONFIG)
        config.update(service_retries=6,chunk_chars=4500,evidence_chars=650)
        config['llm']=dict(model='deepseek-ai/DeepSeek-V4-Flash',temperature=0,max_tokens=12000,timeout_seconds=240)
        config['embedding']=dict(model='BAAI/bge-m3',timeout_seconds=180)
        p=root/'runtime.json';p.write_text(json.dumps(config),encoding='utf-8');assets['runtime_config']=str(p)
        p=root/'schema.json';p.write_text(json.dumps(SCHEMA),encoding='utf-8');assets['schema']=str(p)
        assets.update({k:str(v) for k,v in implementation_assets().items()})
        for path in sources.values():
            p=Path(path);raw=json.loads(p.read_text('utf-8'))
            raw['chunks'][0].update(title_path=['存储'],section_id='s')
            raw['fingerprint']=digest({k:v for k,v in raw.items() if k!='fingerprint'})
            p.write_text(json.dumps(raw),encoding='utf-8')
        manifest=root/'frozen.json'
        freeze(courses,experiments,sources,assets,output=manifest,stage='construction')
        return manifest,sources

    def test_construction_resume_smoke_full_and_mutation(self):
        import json
        from unittest.mock import patch
        from course_experiments.construction import construct
        with task_temp() as tmp:
            root=Path(tmp);manifest,sources=self.fixture(root)
            runtime=ExtractionRuntime();runtime.stats=lambda:{};runtime.close=lambda:None
            with patch('course_experiments.construction.ModelRuntime',return_value=runtime):
                smoke=construct(manifest,'one','textbook',root/'out','A',1)
                count=len(runtime.calls)
                self.assertEqual(smoke,construct(manifest,'one','textbook',root/'out','A',1))
                self.assertEqual(len(runtime.calls),count)
                full=construct(manifest,'one','textbook',root/'out','A')
                self.assertNotEqual(smoke['input_id'],full['input_id'])
                p=Path(sources['oneA']);p.write_text(p.read_text('utf-8')+' ',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'digest'):
                    construct(manifest,'one','textbook',root/'out','A')
                self.assertEqual(len(runtime.calls),count*2)
                payload=json.loads(manifest.read_text('utf-8'));payload['stage']='tampered'
                manifest.write_text(json.dumps(payload),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'digest'):
                    construct(manifest,'one','prior',root/'out')

    def test_failed_chunks_do_not_publish_complete_contribution(self):
        from unittest.mock import patch
        from course_experiments.construction import construct
        with task_temp() as tmp:
            root=Path(tmp);manifest,_=self.fixture(root)
            runtime=ExtractionRuntime();runtime.stats=lambda:{};runtime.close=lambda:None
            runtime.call=lambda *args: (_ for _ in ()).throw(ValueError('invalid model response'))
            with patch('course_experiments.construction.ModelRuntime',return_value=runtime):
                with self.assertRaisesRegex(RuntimeError,'Unfinished'):
                    construct(manifest,'one','textbook',root/'out','A',1)
            self.assertFalse(list((root/'out').rglob('contribution.json')))
            self.assertTrue(list((root/'out').rglob('graph.sqlite')))
            self.assertTrue(list((root/'out').rglob('construction_error.json')))


class ExportRecoveryRegressionTests(unittest.TestCase):
    def test_all_current_evidence_and_merge_references_resolve(self):
        runtime=ExtractionRuntime()
        with task_temp() as tmp:
            engine=TextbookEngine(corpus(),tmp,SCHEMA,CONFIG,runtime)
            try:
                engine._commit(engine.analyze(corpus()['chunks'][0],engine._snapshot()))
                original=engine.store.snapshot()
                def edit(result):
                    for node in result['concepts']:
                        node['name']='SQL'+node['name']
                        node['definition']='SQL'+node['definition']
                        node['scope']='SQL'
                runtime.edit=edit
                second=dict(corpus()['chunks'][0],id='second')
                engine._commit(engine.analyze(second,engine._snapshot()))
                graph=engine.store.snapshot()
                original_id=next(n for n,row in original['nodes'].items() if row['name']=='缓存')
                duplicate_id=next(n for n,row in graph['nodes'].items() if row['name']=='SQL缓存')
                engine.store.merge_nodes({duplicate_id:original_id},SCHEMA,engine.store.version)
                native=engine.store.snapshot()
                self.assertTrue(any(c.get('merged_into') for c in native['claims'].values()))
                result=export_contribution(native,'course','b','full','input')
                graph=result['graph']
                for node in graph['nodes'].values():
                    self.assertTrue(set(node['mention_evidence_ids']) <= set(graph['evidence']))
                    for definition in node['definitions']:
                        self.assertTrue(set(definition['evidence_ids']) <= set(graph['evidence']))
                    if node.get('merged_from'):
                        self.assertEqual(node['native_merged_from'],[duplicate_id])
                        self.assertEqual(node['merged_from'],['course/b/node/'+duplicate_id])
                for claim in graph['claims'].values():
                    if claim.get('merged_into'):self.assertIn(claim['merged_into'],graph['claims'])
                self.assertEqual(graph['events'],native['events'])
                bad=copy.deepcopy(native)
                next(iter(bad['nodes'].values()))['mention_evidence_ids']=['unknown']
                with self.assertRaises(ValueError):export_contribution(bad,'course','b','full','input')
            finally:engine.store.close()

    def test_atomic_publication_failure_and_native_prior_resume(self):
        import json
        from unittest.mock import patch
        runtime=ExtractionRuntime()
        def call(stage,payload,validator):
            runtime.calls.append(stage)
            result=dict(concepts=[dict(id='g1',name='缓存',entity_type='Structure',aliases=[],definition='存储区',scope='',evidence_ids=[])],relations=[])
            validator(result);return result
        runtime.call=call
        with task_temp() as tmp:
            root=Path(tmp)
            with patch('course_experiments.contributions.os.link',side_effect=OSError('interrupted publication')):
                with self.assertRaisesRegex(OSError,'interrupted'):
                    generate_prior('course','课程描述',SCHEMA,CONFIG,runtime,root,'input')
            self.assertFalse((root/'contribution.json').exists())
            self.assertTrue((root/'graph.sqlite').exists())
            (root/'contribution.json.orphan.tmp').write_text('{',encoding='utf-8')
            result=generate_prior('course','课程描述',SCHEMA,CONFIG,runtime,root,'input')
            self.assertEqual(runtime.calls,['prior'])
            self.assertEqual(result['input_id'],'input')
            result['graph']['version']+=1
            (root/'contribution.json').write_text(json.dumps(result),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'digest'):
                generate_prior('course','课程描述',SCHEMA,CONFIG,runtime,root,'input')

    def test_immutable_writer_never_overwrites_existing_artifact(self):
        from unittest.mock import patch
        with task_temp() as tmp:
            path=Path(tmp)/'identity.json'
            def competing_publish(source,destination):
                Path(destination).write_text('{"different":true}',encoding='utf-8')
                raise FileExistsError('another writer')
            with patch('course_experiments.contributions.os.link',side_effect=competing_publish):
                with self.assertRaisesRegex(ValueError,'changed'):
                    write_immutable(path,{'input_id':'input'})
            self.assertEqual(path.read_text('utf-8'),'{"different":true}')

    def test_partial_temporary_write_never_exposes_identity(self):
        from unittest.mock import patch
        with task_temp() as tmp:
            root=Path(tmp);path=root/'construction_identity.json'
            with patch('course_experiments.contributions.os.fsync',side_effect=OSError('disk failure')):
                with self.assertRaisesRegex(OSError,'disk failure'):
                    write_immutable(path,{'input_id':'input'})
            self.assertFalse(path.exists())
            self.assertFalse(list(root.glob('*.tmp')))
            self.assertEqual(write_immutable(path,{'input_id':'input'}),{'input_id':'input'})

    def test_cli_resume_rejects_tampered_contribution(self):
        import json
        from unittest.mock import patch
        from course_experiments.construction import construct
        with task_temp() as tmp:
            root=Path(tmp);manifest,_=ConstructionExecutionTests().fixture(root)
            runtime=ExtractionRuntime();runtime.stats=lambda:{};runtime.close=lambda:None
            with patch('course_experiments.construction.ModelRuntime',return_value=runtime):
                result=construct(manifest,'one','textbook',root/'out','A',1)
                count=len(runtime.calls)
                path=next((root/'out').rglob('contribution.json'))
                result['graph']['version']+=1
                path.write_text(json.dumps(result),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'digest'):
                    construct(manifest,'one','textbook',root/'out','A',1)
                self.assertEqual(len(runtime.calls),count)
