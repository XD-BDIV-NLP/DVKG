from __future__ import annotations
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
import numpy as np

from course_experiments.manifest import digest, file_record, read_document
from course_experiments.contributions import write_immutable, export_contribution
from course_experiments.evaluation.protocol import freeze_phase
from course_experiments.tests import test_run as scheduler_tests

CONFIG = dict(llm=dict(base_url='https://api.siliconflow.cn/v1', model='deepseek-ai/DeepSeek-V4-Flash', extra_body={'enable_thinking': False}), embedding=dict(base_url='https://api.siliconflow.cn/v1', model='BAAI/bge-m3'))


class EmbeddingBoundaryTests(unittest.TestCase):
    def test_real_numpy_index_through_rag_answer_and_judge(self):
        from course_experiments.evaluation.rag import build_index, evaluate, BUDGET
        calls=[]
        def post(endpoint,body,*args):
            calls.append(body)
            value={'score':1} if body['max_tokens']==1024 else {'answer':'CPU','citations':[dict(evidence_id='e',quote='CPU')]}
            return dict(choices=[dict(message=dict(content=json.dumps(value)),finish_reason='stop')])
        runtime=SimpleNamespace(config=CONFIG,gate=threading.BoundedSemaphore(1),_raw_cached=lambda key:None,_post=post,embed=lambda texts:np.array([[1,0] for _ in texts],dtype=np.float32))
        corpus=dict(chunks=[dict(id='c',book_id='B',evidence_ids=['e'])],evidence=[dict(id='e',text='CPU')])
        ref=dict(course_id='c',scope='smoke',status='complete',human_audited=False,entities=[],facts=[],questions=[dict(question_id='q',question='CPU?',answer='CPU',evidence_ids=['e'],citations=[dict(evidence_id='e',quote='CPU')])],targets=dict(questions=1,facts=0));ref['reference_id']=digest(ref)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);index=build_index([corpus],runtime,root/'index');graph=dict(entities=[],relations=[]);paths={}
            for role,value in [('reference',ref),('index',index),('graph',graph),('runtime_config',CONFIG)]:
                paths[role]=root/(role+'.json');write_immutable(paths[role],value)
            phase=root/'phase.json';freeze_phase(paths,dict(BUDGET,control='vector_only',active_books=['B']),phase)
            result=evaluate(ref,read_document(paths['index']),graph,['B'],'vector_only',phase,root/'out',runtime)
            self.assertEqual(result['completed_count'],1);self.assertEqual(result['metrics']['f1'],1)
            self.assertEqual(len(calls),2)
            evaluate(ref,index,graph,['B'],'vector_only',phase,root/'out',runtime)
            self.assertEqual(len(calls),2)

    def test_self_hashed_invalid_cached_index_is_not_reused(self):
        from course_experiments.evaluation.rag import build_index
        corpus=dict(chunks=[dict(id='c',book_id='B',evidence_ids=['e'])],evidence=[dict(id='e',text='CPU')])
        with tempfile.TemporaryDirectory() as d:
            runtime=SimpleNamespace(embed=lambda texts:[[1.,0.]])
            index=build_index([corpus],runtime,d);index['rows'][0]['vector']=[]
            index['index_id']=digest({k:v for k,v in index.items() if k!='index_id'})
            next(Path(d).glob('*.json')).write_text(json.dumps(index),encoding='utf-8')
            with self.assertRaises(ValueError):build_index([corpus],runtime,d)

    def test_numpy_corpus_query_roundtrip_and_reuse(self):
        from course_experiments.evaluation.rag import build_index, retrieve
        corpus = dict(chunks=[dict(id='c', book_id='B', evidence_ids=['e'])], evidence=[dict(id='e', text='CPU contains Register')])
        runtime = SimpleNamespace(embed=lambda texts: np.array([[.25, .5, .75]], dtype=np.float32))
        with tempfile.TemporaryDirectory() as d:
            index = build_index([corpus], runtime, d)
            stored = read_document(next(Path(d).glob('*.json')))
            self.assertEqual(stored['rows'][0]['vector'], [.25, .5, .75])
            runtime.embed = lambda texts: self.fail('Frozen index must be reused')
            self.assertEqual(build_index([corpus], runtime, d), index)
            rows = retrieve('CPU', np.array([.25, .5, .75], dtype=np.float32), stored['rows'], {}, 'vector_only')
            self.assertEqual([r['chunk_id'] for r in rows], ['c'])

    def test_embedding_invalid_numeric_dimensions(self):
        from course_experiments.evaluation.rag import cosine, build_index
        for value in [np.array([]), np.array([float('nan'), 1]), np.array([float('inf'), 1]), np.array([[1, 2]]), ['1', '2'], [True, 1]]:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError): cosine(value, [1, 2])
        corpus = dict(chunks=[dict(id=str(i), book_id='B', evidence_ids=[str(i)]) for i in range(2)], evidence=[dict(id=str(i), text='CPU') for i in range(2)])
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            build_index([corpus], SimpleNamespace(embed=lambda texts: [[1, 2], [1, 2, 3]]), d)


class AccountBoundaryTests(unittest.TestCase):
    setUp = scheduler_tests.SchedulerTests.setUp
    plan = scheduler_tests.SchedulerTests.plan

    def second(self):
        item = copy.deepcopy(self.item)
        entry = item['entry']
        phase_path = self.root / 'second-phase.json'
        phase = freeze_phase({k:v['path'] for k,v in read_document(entry['phase'])['files'].items()}, dict(seed=42, attempt=2), phase_path)
        meta = read_document(entry['metadata']['path']); meta['phase_id'] = phase['phase_id']
        metadata = self.root / 'second-metadata.json'; write_immutable(metadata, meta)
        entry.update(phase=str(phase_path), metadata=file_record(metadata))
        from course_experiments.run import validate_entry
        item.update(validate_entry(item, entry))
        return item

    def test_real_runtime_status_stops_new_jobs_and_resume(self):
        from course_experiments.run import run
        from course_experiments.evaluation.chat import call_json
        second = self.second()
        for status in (401, 402, 403):
            with self.subTest(status=status):
                calls = []
                def post(*args):
                    calls.append(status)
                    raise RuntimeError(f'Model service returned HTTP {status}')
                runtime = SimpleNamespace(config=CONFIG, gate=threading.BoundedSemaphore(1), _raw_cached=lambda key: None, _post=post)
                def execute(item, output):
                    call_json(runtime, 'fixture', {}, 10, output)
                plan = self.plan([self.item, second]); root = self.root / str(status)
                records = run(plan, root, execute)
                self.assertEqual([r['status'] for r in records], ['failed', 'blocked'])
                self.assertEqual(records[0]['status_code'], status)
                self.assertTrue(records[0]['account_stop'])
                self.assertEqual(len(calls), 1)
                run(plan, root, execute)
                self.assertEqual(len(calls), 1)

    def test_unrelated_runtime_numbers_do_not_stop_other_jobs(self):
        from course_experiments.run import run
        calls = []
        def fail(item, output):
            calls.append(item['job_id'])
            raise RuntimeError('Unrelated 402, Model service returned HTTP 402 extra')
        records = run(self.plan([self.item, self.second()]), self.root/'ordinary', fail)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(not r['account_stop'] for r in records))

    def test_resume_account_stop_precedes_previously_locked_job(self):
        from course_experiments.run import run
        second=self.second();root=self.root/'resume';directory=root/self.item['job_id'];directory.mkdir(parents=True)
        lock=directory/'job.lock';lock.touch();calls=[]
        def fail(item, output):
            calls.append(item['job_id']);raise RuntimeError('Model service returned HTTP 403')
        path=self.plan([self.item,second]);run(path,root,fail);lock.unlink()
        records=run(path,root,fail)
        self.assertEqual(records[0]['status'],'blocked')
        self.assertEqual(len(calls),1)

    def test_direct_http_account_statuses_escape_rag_retries(self):
        from course_experiments.evaluation.rag import evaluate, BUDGET
        ref=dict(course_id='c',scope='smoke',status='complete',human_audited=False,entities=[],facts=[],questions=[dict(question_id='q',question='CPU?',answer='CPU',citations=[dict(evidence_id='e',quote='CPU')],evidence_ids=['e'])],targets=dict(questions=1,facts=0));ref['reference_id']=digest(ref)
        index=dict(source_id='s',embedding_model='BAAI/bge-m3',rows=[dict(book_id='B',chunk_id='c',evidence_ids=['e'],evidence=[dict(id='e',text='CPU')],text='CPU',vector=[1.,0.])]);index['index_id']=digest(index)
        graph=dict(entities=[],relations=[]);paths={}
        for role,value in [('reference',ref),('index',index),('graph',graph),('runtime_config',CONFIG)]:
            paths[role]=self.root/(role+'.json')
            if role=='graph':paths[role]=self.root/'http-graph.json'
            write_immutable(paths[role],value)
        phase=self.root/'http-phase.json';freeze_phase(paths,dict(BUDGET,control='graph',active_books=['B']),phase)
        for code in (401,402,403):
            calls=[]
            def post(*args):calls.append(code);raise HTTPError('fixture',code,'account',{},None)
            runtime=SimpleNamespace(config=CONFIG,gate=threading.BoundedSemaphore(1),_raw_cached=lambda key:None,_post=post,embed=lambda texts:[[1.,0.]])
            with self.subTest(code=code),self.assertRaises(HTTPError):evaluate(ref,index,graph,['B'],'graph',phase,self.root/('http-'+str(code)),runtime)
            self.assertEqual(calls,[code])


class ScopedExportTests(unittest.TestCase):
    def make(self, book, name, scope, aliases):
        ids = ['e'] if book else []
        nodes = dict(x=dict(id='x', name=name, scope=scope, aliases=aliases, entity_type='Concept', evidence_ids=ids), y=dict(id='y', name='CPU', scope='cpu', aliases=[], entity_type='Concept', evidence_ids=ids))
        claim = dict(id='f', status='active', revision=1, evidence_ids=ids, content=dict(kind='relation', head='x', tail='y', relation='part_of'))
        return export_contribution(dict(nodes=nodes, claims={'f':claim}, books={book:dict(id=book)} if book else {}, evidence={'e':dict(id='e', book_id=book, text=name+' CPU')} if book else {}), 'c', book, 'full', 'fixture')

    def test_fusion_scoped_members_aliases_and_hotplug_reach_projection(self):
        from course_experiments.fusion import build_view
        from course_experiments.splits import freeze_split, project_graph
        g = self.make(None, 'Register', 'cpu', [])
        a = self.make('A', 'Storage cell', 'cpu', ['Latch'])
        ref = dict(course_id='c', entities=[dict(entity_id='r', canonical_name='Latch', aliases=[], scope='cpu'), dict(entity_id='wrong', canonical_name='Register', aliases=[], scope='finance'), dict(entity_id='cpu', canonical_name='CPU', aliases=[], scope='cpu')], facts=[dict(fact_id='f', head='r', tail='cpu', relation='part_of')])
        split = freeze_split(ref)
        alignment = {f'c/A/node/{n}':f'c/g0/node/{n}' for n in ('x', 'y')}
        fused = build_view(g, [a], ['A'], alignment=alignment)
        projection = project_graph(fused, ref, split)
        self.assertNotIn('c/g0/node/x', projection['audit']['unmatched_entities'])
        self.assertEqual(projection['training'], [])
        self.assertTrue(projection['audit']['heldout_removed'])
        x = next(e for e in fused['entities'] if e['entity_id']=='c/g0/node/x')
        self.assertIn('Storage cell', [r['canonical_name'] for r in x['identity_provenance']])
        removed = project_graph(build_view(g, [a], [], alignment=alignment), ref, split)
        self.assertIn('c/g0/node/x', removed['audit']['unmatched_entities'])

    def test_conflicting_scopes_never_become_unscoped_homonym(self):
        from course_experiments.fusion import build_view
        from course_experiments.splits import freeze_split, project_graph
        g = self.make(None, 'Register', 'cpu', [])
        a = self.make('A', 'Register', 'finance', [])
        ref = dict(course_id='c', entities=[dict(entity_id='r', canonical_name='Register', aliases=[], scope='finance')], facts=[])
        view = build_view(g, [a], ['A'], alignment={'c/A/node/x':'c/g0/node/x'})
        projection = project_graph(view, ref, freeze_split(ref))
        self.assertIn('c/g0/node/x', projection['audit']['unmatched_entities'])

    def test_unscoped_member_cannot_override_explicit_scope(self):
        from course_experiments.fusion import build_view
        from course_experiments.splits import freeze_split, project_graph
        g=self.make(None,'Register','',[]);a=self.make('A','CPU register','cpu',[])
        ref=dict(course_id='c',entities=[dict(entity_id='r',canonical_name='Register',aliases=[],scope='finance')],facts=[])
        view=build_view(g,[a],['A'],alignment={'c/A/node/x':'c/g0/node/x'})
        projection=project_graph(view,ref,freeze_split(ref))
        self.assertIn('c/g0/node/x',projection['audit']['unmatched_entities'])


class ReferenceRepairTests(unittest.TestCase):
    def test_object_specific_name_endpoint_feedback_next_request(self):
        from course_experiments.evaluation.reference import generate, PARAMETERS
        source = 'CPU contains Register. CPU is a processor.'
        cite = lambda quote: [dict(evidence_id='e', quote=quote)]
        good = dict(entities=[dict(local_id='cpu', canonical_name='CPU', scope='cpu', aliases=[], citations=cite('CPU')), dict(local_id='r', canonical_name='Register', scope='cpu', aliases=[], citations=cite('Register'))], facts=[dict(local_id='f', head='r', tail='cpu', relation='part_of', citations=cite('CPU contains Register'))], questions=[])
        for mode in ('name', 'endpoint'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d); bad = copy.deepcopy(good)
                if mode=='name': bad['entities'][0]['canonical_name']='Invented CPU'
                else: bad['facts'][0]['citations']=cite('CPU is a processor')
                requests=[]
                def post(endpoint, body, *args):
                    requests.append(json.loads(body['messages'][1]['content']))
                    return dict(choices=[dict(message=dict(content=json.dumps(bad if len(requests)==1 else good)), finish_reason='stop')])
                runtime=SimpleNamespace(config=CONFIG, gate=threading.BoundedSemaphore(1), _raw_cached=lambda key: None, _post=post)
                corpus=dict(chunks=[dict(id='c', book_id='B', evidence_ids=['e'])], evidence=[dict(id='e', text=source)])
                for name,value in [('corpus',corpus),('schema',{'relation_types':{'part_of':''}}),('runtime',CONFIG)]: write_immutable(root/(name+'.json'),value)
                construction=dict(courses={'courses':[dict(course_id='c',description='CPU',books=[dict(book_id='B')])]},files=dict(assets={'schema':file_record(root/'schema.json'),'runtime_config':file_record(root/'runtime.json')},sources={'B':file_record(root/'corpus.json')}))
                cp=root/'construction.json';write_immutable(cp,construction);phase=root/'phase.json'
                freeze_phase({'construction':cp},dict(PARAMETERS,smoke_chunks=1),phase)
                with patch('course_experiments.evaluation.reference.validate',return_value=construction): dataset,out=generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(len(requests),2)
                feedback=requests[1]['validation_feedback']['diagnostic']
                self.assertEqual(feedback['local_id'],'cpu' if mode=='name' else 'f')
                self.assertEqual(feedback['field'],'canonical_name' if mode=='name' else 'head')
                self.assertEqual(feedback['evidence_ids'],['e'])
                self.assertEqual(feedback['rule_id'],'literal_canonical_name' if mode=='name' else 'fact_endpoint_own_citation')
                self.assertEqual(requests[0]['unit'],requests[1]['unit'])
                self.assertEqual(len(dataset['facts']),1)
                self.assertTrue(list(out.glob('errors/*.json')))


class ComparisonBindingTests(unittest.TestCase):
    setUp = scheduler_tests.SchedulerTests.setUp
    gpu_fixture = scheduler_tests.SchedulerTests.gpu_fixture

    def test_bound_changed_row_fails_durably_while_other_row_runs_reports_collects(self):
        from course_experiments.run import prepare, run, collect
        from course_experiments.results import report
        manifest,catalog,_=self.rag_catalog();plan_path=self.root/'isolated-plan.json'
        plan=prepare(manifest,catalog,plan_path);ready=[p for p in plan['positions'] if p['status']=='ready']
        first,second=ready;Path(first['entry']['inputs']['graph']).write_text('{}',encoding='utf-8')
        calls=[]
        def execute(item,output):
            calls.append(item['job_id'])
            result=dict(status='completed',input_digest=read_document(item['entry']['phase'])['phase_id'],graph_id=digest(read_document(item['entry']['inputs']['graph'])))
            path=output/'result.json';write_immutable(path,result);return result,path
        root=self.root/'isolated-jobs';records=run(plan_path,root,execute)
        by_id={r['job_id']:r for r in records if 'job_id' in r}
        self.assertEqual(by_id[first['job_id']]['status'],'failed')
        self.assertEqual(by_id[first['job_id']]['failure_stage'],'input_validation')
        self.assertTrue(list((root/first['job_id']).glob('validation_failure_*.json')))
        self.assertEqual(calls,[second['job_id']])
        summary=report(plan_path,root,self.root/'isolated-report')
        reported={r['state']:r for r in summary['rows'] if r['book_set']=='A' and r['evaluation']=='rag' and r['course_id']=='one'}
        self.assertEqual(reported[first['state']]['status'],'failed')
        self.assertEqual(reported[second['state']]['status'],'completed')
        record=collect(plan_path,second['job_id'],by_id[second['job_id']]['raw_record']['path'],self.root/'isolated-collected')
        self.assertEqual(record['status'],'completed')

    def test_bound_missing_pending_row_reports_failure_independently(self):
        from course_experiments.run import prepare
        from course_experiments.results import report
        manifest,catalog,_=self.rag_catalog();plan_path=self.root/'pending-plan.json'
        plan=prepare(manifest,catalog,plan_path);ready=[p for p in plan['positions'] if p['status']=='ready'];first,second=ready
        Path(first['entry']['inputs']['graph']).unlink()
        root=self.root/'pending-jobs';summary=report(plan_path,root,self.root/'pending-report')
        reported={r['state']:r for r in summary['rows'] if r['book_set']=='A' and r['evaluation']=='rag' and r['course_id']=='one'}
        self.assertEqual(reported[first['state']]['status'],'failed')
        self.assertEqual(reported[second['state']]['status'],'pending')
        self.assertTrue(list((root/first['job_id']).glob('validation_failure_*.json')))

    def test_bound_account_stop_survives_invalid_stopped_row(self):
        from course_experiments.run import prepare, run
        manifest,catalog,_=self.rag_catalog();plan_path=self.root/'account-plan.json'
        plan=prepare(manifest,catalog,plan_path);ready=[p for p in plan['positions'] if p['status']=='ready'];first,second=ready
        root=self.root/'account-jobs';directory=root/first['job_id'];directory.mkdir(parents=True);lock=directory/'job.lock';lock.touch()
        calls=[]
        def fail(item,output):
            calls.append(item['job_id']);raise RuntimeError('Model service returned HTTP 403')
        run(plan_path,root,fail);lock.unlink();Path(second['entry']['inputs']['graph']).unlink()
        records=run(plan_path,root,fail);by_id={r['job_id']:r for r in records if 'job_id' in r}
        self.assertEqual(by_id[first['job_id']]['status'],'blocked')
        self.assertEqual(by_id[second['job_id']]['failure_stage'],'input_validation')
        self.assertEqual(calls,[second['job_id']])
        self.assertTrue(list((root/second['job_id']).glob('validation_failure_*.json')))

    def test_bound_malformed_cache_fails_locally_without_aborting_prescan(self):
        from course_experiments.run import prepare, run
        from course_experiments.results import report
        manifest,catalog,_=self.rag_catalog();plan_path=self.root/'bad-cache-plan.json'
        plan=prepare(manifest,catalog,plan_path);ready=[p for p in plan['positions'] if p['status']=='ready'];first,second=ready
        root=self.root/'bad-cache-jobs';directory=root/first['job_id'];directory.mkdir(parents=True)
        (directory/'record.json').write_text('{invalid json',encoding='utf-8');calls=[]
        def execute(item,output):
            calls.append(item['job_id']);value=dict(status='completed',input_digest=read_document(item['entry']['phase'])['phase_id'])
            path=output/'result.json';write_immutable(path,value);return value,path
        records=run(plan_path,root,execute);by_id={r['job_id']:r for r in records if 'job_id' in r}
        self.assertEqual(by_id[first['job_id']]['failure_stage'],'input_validation')
        self.assertEqual(calls,[second['job_id']])
        self.assertTrue(list(directory.glob('validation_failure_*.json')))
        summary=report(plan_path,root,self.root/'bad-cache-report')
        reported={r['state']:r for r in summary['rows'] if r['book_set']=='A' and r['evaluation']=='rag' and r['course_id']=='one'}
        self.assertEqual(reported[first['state']]['status'],'failed')
        self.assertEqual(reported[second['state']]['status'],'completed')

    def test_protocol_cannot_replace_phase_declared_common_selection(self):
        from course_experiments.evaluation.kge import validate_protocol
        first,_=self.gpu_fixture();protocol=read_document(first['entry']['arguments']['protocol'])
        protocol['common_test_mask']=list(protocol['projection']['covered_test_mask'])
        protocol['protocol_id']=digest({k:v for k,v in protocol.items() if k!='protocol_id'})
        with self.assertRaisesRegex(ValueError,'[Cc]ommon'):validate_protocol(protocol)

    def test_individually_valid_different_splits_cannot_compare(self):
        from course_experiments.run import bind_comparisons, validate_entry
        from course_experiments.evaluation.kge import prepare_protocol
        first,_=self.gpu_fixture();second=copy.deepcopy(first);entry=second['entry']
        paths={k:v['path'] for k,v in read_document(entry['phase'])['files'].items()}
        split=read_document(paths['fact_split']);split['policy']+=' independently frozen variant';split['split_id']=digest({k:v for k,v in split.items() if k!='split_id'})
        projection=read_document(paths['projection']);projection['split_id']=split['split_id'];projection['projection_id']=digest({k:v for k,v in projection.items() if k!='projection_id'})
        mask=read_document(paths['common_mask']);mask.update(split_id=split['split_id'],projection_ids=[projection['projection_id']])
        for role,value in [('fact_split',split),('projection',projection),('common_mask',mask)]:
            paths[role]=self.root/('split-variant-'+role+'.json');write_immutable(paths[role],value);entry['inputs'][role]=str(paths[role])
        phase_path=self.root/'split-variant-phase.json';phase=freeze_phase(paths,dict(seed=42),phase_path);phase['path']=str(phase_path)
        protocol=self.root/'split-variant-protocol.json';prepare_protocol(projection,split,phase,protocol,smoke=True,common_mask=mask['mask'])
        meta=read_document(entry['metadata']['path']);meta['phase_id']=phase['phase_id'];metadata=self.root/'split-variant-meta.json';write_immutable(metadata,meta)
        entry.update(phase=str(phase_path),metadata=file_record(metadata));entry['arguments']['protocol']=str(protocol)
        second.update(validate_entry(second,entry))
        with self.assertRaisesRegex(ValueError,'[Cc]omparison'):bind_comparisons([first,second])

    def test_sanitized_gpu_collection_never_reads_nested_primary_sources(self):
        from course_experiments.run import validate_entry, bind_comparisons, collect
        from course_experiments.comparison import comparison_records
        from course_experiments.evaluation.kge import export_kge_package, prepare_protocol
        first,result=self.gpu_fixture();entry=copy.deepcopy(first['entry'])
        original=read_document(entry['phase']);paths={k:v['path'] for k,v in original['files'].items()}
        projection=read_document(paths['projection']);split=read_document(paths['fact_split'])
        package=export_kge_package(projection,split,entry['phase'],self.root/'gpu-package')
        for role,name in [('graph','graph_attestation.json'),('fact_split','fact_split.json'),('projection','projection.json'),('primary_attestation','primary_attestation.json')]:paths[role]=package['paths'][name]
        absent=dict(path=str(self.root/'primary-only-raw.json'),sha256='0'*64)
        manifest=dict(files=dict(sources={'B':absent},assets={'questions':absent,'reference_facts':absent},configs=[]));manifest['manifest_id']=digest(manifest)
        manifest_path=self.root/'canonical-metadata.json';write_immutable(manifest_path,manifest);paths['construction']=manifest_path
        phase_path=self.root/'sanitized-phase.json';phase=freeze_phase(paths,dict(seed=42),phase_path);phase['path']=str(phase_path)
        protocol_path=self.root/'sanitized-protocol.json';protocol=prepare_protocol(projection,read_document(paths['fact_split']),phase,protocol_path,smoke=True,common_mask=read_document(paths['common_mask'])['mask'])
        meta=read_document(entry['metadata']['path']);meta.update(phase_id=phase['phase_id'],graph=file_record(paths['graph']),source_records=[file_record(manifest_path)]);meta_path=self.root/'sanitized-meta.json';write_immutable(meta_path,meta)
        entry.update(phase=str(phase_path),metadata=file_record(meta_path),inputs={k:str(paths[k]) for k in ('graph','projection','fact_split','common_mask')});entry['arguments'].update(graph=str(paths['graph']),protocol=str(protocol_path))
        item=dict(first,entry=entry);item.update(validate_entry(item,entry))
        binding=dict(manifest_id=manifest['manifest_id'],record=file_record(manifest_path))
        comparison=comparison_records([item],canonical_binding=binding)[0]
        item['comparison']=comparison;item['identity']['comparison_id']=comparison['comparison_id'];item['job_id']=digest(item['identity'])
        plan=dict(positions=[item],comparisons=[comparison],manifest_path=str(manifest_path));plan['plan_id']=digest(plan);plan_path=self.root/'sanitized-plan.json';write_immutable(plan_path,plan)
        result.update(protocol_id=protocol['protocol_id'])
        result['run_id']=digest(dict(protocol_id=protocol['protocol_id'],phase_id=phase['phase_id'],graph_id=projection['graph_id'],split_id=projection['split_id'],epochs=2,requested_device='cpu',options=protocol['budget']))
        result_path=self.root/'sanitized-result.json';write_immutable(result_path,result)
        record=collect(plan_path,item['job_id'],result_path,self.root/'gpu-jobs')
        self.assertEqual(record['status'],'completed_with_diagnostic_failure')
        self.assertEqual(record['identity']['comparison_id'],comparison['comparison_id'])
        self.assertFalse(Path(absent['path']).exists())

    def rag_catalog(self, mismatch=None):
        from course_experiments.tests.test_manifest_matrix import ManifestMatrixTests
        from course_experiments.manifest import freeze
        from course_experiments.matrix import generate
        from course_experiments.evaluation.rag import BUDGET, build_index
        from course_experiments.run import freeze_catalog, validate_entry
        folder=self.root/'canonical';folder.mkdir()
        courses,config,sources,assets=ManifestMatrixTests().fixture(folder)
        config['budgets']['rag']=dict(retrieval_top_k=5,context_token_limit=3000,max_output_tokens=2048,temperature=0)
        reference=dict(course_id='one',scope='smoke',status='complete',human_audited=False,entities=[],facts=[],questions=[dict(question_id='q',question='fixture?',answer='fixture',citations=[dict(evidence_id='e1',quote='fixture')],evidence_ids=['e1'])],targets=dict(questions=1,facts=0))
        reference['reference_id']=digest(reference)
        reference_path=folder/'reference.json';write_immutable(reference_path,reference)
        assets.update(questions=reference_path,reference_facts=reference_path)
        runtime_path=folder/'runtime.json';write_immutable(runtime_path,CONFIG);assets['runtime_config']=runtime_path
        manifest=folder/'manifest.json';frozen=freeze(courses,config,sources,assets,output=manifest)
        rows=[r for r in generate(frozen) if r['course_id']=='one' and r['book_set']=='A' and r['evaluation']=='rag' and r['state'] in ('fixed_g0','fused')]
        corpus=read_document(sources['oneA'])
        index=build_index([corpus],SimpleNamespace(embed=lambda texts:[[1.,0.]]),folder/'index')
        entries={};items=[]
        for i,row in enumerate(rows):
            directory=folder/str(i);directory.mkdir()
            ref=copy.deepcopy(reference);idx=copy.deepcopy(index)
            if i and mismatch=='reference':
                ref['questions'][0]['question_id']='different-question';ref['reference_id']=digest({k:v for k,v in ref.items() if k!='reference_id'})
            if i and mismatch=='index':
                idx['rows'][0]['vector']=[0.,1.];idx['index_id']=digest({k:v for k,v in idx.items() if k!='index_id'})
            paths={'construction':manifest,'runtime_config':runtime_path}
            for role,value in [('reference',ref),('index',idx),('graph',dict(entities=[],relations=[]))]:
                paths[role]=directory/(role+'.json');write_immutable(paths[role],value)
            phase_path=directory/'phase.json';phase=freeze_phase(paths,dict(BUDGET,control='graph',active_books=['oneA']),phase_path)
            metadata=directory/'metadata.json';write_immutable(metadata,dict(course_id='one',method_id=row['method_id'],state=row['state'],scope='smoke',book_ids=['oneA'],status='complete',phase_id=phase['phase_id'],graph=file_record(paths['graph']),source_records=[file_record(manifest)]))
            entry=dict(phase=str(phase_path),metadata=file_record(metadata),scope='smoke',inputs={k:str(paths[k]) for k in ('construction','reference','index','graph')},arguments={k:str(paths[k]) for k in ('construction','reference','index','graph')})
            item=dict(row,entry=entry);item.update(validate_entry(row,entry));items.append(item)
            entries[row['row_id']]=entry
        catalog=folder/'catalog.json';freeze_catalog(entries,catalog)
        return manifest,catalog,items

    def test_prepare_rejects_individually_valid_incompatible_references(self):
        from course_experiments.run import prepare
        manifest,catalog,items=self.rag_catalog('reference')
        self.assertEqual([i['status'] for i in items],['ready','ready'])
        with self.assertRaisesRegex(ValueError,'[Cc]omparison'):prepare(manifest,catalog,self.root/'bad-plan.json')

    def test_prepare_rejects_individually_valid_incompatible_indices(self):
        from course_experiments.run import prepare
        manifest,catalog,items=self.rag_catalog('index')
        self.assertEqual([i['status'] for i in items],['ready','ready'])
        with self.assertRaisesRegex(ValueError,'[Cc]omparison'):prepare(manifest,catalog,self.root/'bad-plan.json')

    def test_compatible_jobs_share_comparison_and_recheck_tampered_binding(self):
        from course_experiments.run import prepare, run, collect
        from course_experiments.results import summarize
        manifest,catalog,items=self.rag_catalog()
        path=self.root/'plan.json';plan=prepare(manifest,catalog,path)
        ready=[r for r in plan['positions'] if r['status']=='ready']
        self.assertEqual(len(ready),2)
        self.assertEqual(ready[0]['identity']['comparison_id'],ready[1]['identity']['comparison_id'])
        plan['comparisons'][0]['reference_digest']='different'
        plan['plan_id']=digest({k:v for k,v in plan.items() if k!='plan_id'})
        bad=self.root/'tampered.json';write_immutable(bad,plan)
        for operation in [lambda:run(bad,self.root/'jobs',lambda *a:self.fail('Must reject before calls')),lambda:collect(bad,ready[0]['job_id'],self.root/'absent.json',self.root/'jobs'),lambda:summarize(bad,self.root/'jobs')]:
            with self.assertRaisesRegex(ValueError,'[Cc]omparison'):operation()

    def test_common_selection_disagreement_rejected_and_own_coverage_labeled(self):
        from course_experiments.run import bind_comparisons, validate_entry
        from course_experiments.evaluation.kge import prepare_protocol
        first,_=self.gpu_fixture();second=copy.deepcopy(first)
        entry=second['entry'];p=read_document(entry['arguments']['protocol']);projection=p['projection']
        mask=list(projection['covered_test_mask'])
        paths={k:v['path'] for k,v in read_document(entry['phase'])['files'].items()}
        mask_path=self.root/'different-mask.json';write_immutable(mask_path,dict(mask=mask,projection_ids=[projection['projection_id']],split_id=projection['split_id']));paths['common_mask']=mask_path
        phase_path=self.root/'different-phase.json';phase=freeze_phase(paths,dict(seed=42),phase_path);phase['path']=str(phase_path)
        protocol_path=self.root/'different-protocol.json';prepare_protocol(projection,read_document(paths['fact_split']),phase,protocol_path,smoke=True,common_mask=mask)
        meta=read_document(entry['metadata']['path']);meta['phase_id']=phase['phase_id'];meta_path=self.root/'different-meta.json';write_immutable(meta_path,meta)
        entry.update(phase=str(phase_path),metadata=file_record(meta_path));entry['inputs']['common_mask']=str(mask_path);entry['arguments']['protocol']=str(protocol_path)
        second.update(validate_entry(second,entry))
        with self.assertRaisesRegex(ValueError,'[Cc]omparison'):bind_comparisons([first,second])
        entry['comparison_role']='own_coverage';second.update(validate_entry(second,entry))
        comparisons=bind_comparisons([first,second])
        self.assertEqual(len(comparisons),1)
        self.assertEqual(second['identity']['comparison_role'],'own_coverage')

