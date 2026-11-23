from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from course_experiments.manifest import digest,file_record,read_document


class CurrentReportingTests(unittest.TestCase):
    def setUp(self):
        from course_experiments.evaluation import current_matrix_reporting
        self.m=current_matrix_reporting
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def save(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8');return file_record(p)

    def fixture(self):
        qs=[dict(question_id='q1',target_book_id='a',type='conditions',question='PRIVATE QUESTION',answer='PRIVATE GOLD'),dict(question_id='q2',target_book_id='a',type='conditions',question='PRIVATE QUESTION2',answer='PRIVATE GOLD2')]
        dataset=self.save('dataset.json',dict(target_book_id='a',questions=qs));contract=dict(datasets={'a':dataset});identity=digest(contract);contract['result_id']=identity
        cr=self.save('contract.json',contract);raw=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(answer='PRIVATE CANDIDATE',citations=[]))))])
        ar=self.save('q1/answer_raw.json',dict(response=raw));jr=self.save('q1/judge_raw.json',dict(response=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(score=.5,grounding=1))))])))
        first=dict(result_id=identity,status='completed',question_id='q1',question_type='conditions',target_book_id='a',candidate=dict(answer='PRIVATE CANDIDATE',citations=[]),scores=dict(f1=.5,em=0),machine_judge=dict(score=.5,grounding=1),answer_receipt=dict(source=ar));first['checkpoint_digest']=digest(first)
        second=dict(result_id=identity,status='failed',question_id='q2',question_type='conditions',target_book_id='a',error_type='ValueError',error_record=self.save('q2/error.json',dict(private='PRIVATE ERROR')));second['checkpoint_digest']=digest(second)
        records=[self.save('q1/record.json',first),self.save('q2/record.json',second)]
        rr=self.save('result.json',dict(result_id=identity,question_count=2,completed_count=1,failed_count=1))
        req=dict(id='qa',category='controlled_graph_RAG',method='tree_kg',cohort='compAB',target_book='a',variant='frozen',split='regression313',question_count=2,required=True)
        obs=dict(adapter='qa',contract=cr,result=rr,dataset=dataset,questions=records,judge_raw={'q1':jr})
        return req,obs

    def test_failure_zero_fixed_denominator_raw_sha_and_no_private_export(self):
        req,obs=self.fixture();row,details=self.m.qa_observation(req,obs)
        self.assertEqual(row['status'],'failed');self.assertEqual(row['counts'],dict(fixed=2,completed=1,failed=1,pending=0));self.assertEqual(row['metrics']['f1'],.25)
        self.assertNotIn('PRIVATE',json.dumps(row));self.assertIn('PRIVATE',json.dumps(details))
        Path(obs['questions'][0]['path']).write_text('{}')
        with self.assertRaises(ValueError):self.m.qa_observation(req,obs)

    def test_fixed_raw_verified_failure_is_complete_coverage_without_claiming_all_success(self):
        req,obs=self.fixture();matrix=dict(requirements=[req]);mr=self.save('matrix.json',matrix)
        terminal=self.save('queue_terminal.json',dict(status='fixed_matrix_queue_terminal_with_failures'))
        value=self.m.assemble(matrix,dict(matrix=mr,queue_terminal=terminal,observations={req['id']:obs}))
        self.assertTrue(value['full_matrix_complete']);self.assertTrue(value['coverage_complete']);self.assertFalse(value['all_questions_successful'])
        proof=value['QA_original_coverage_receipt'];self.assertEqual(proof['failed_questions'],1);self.assertEqual(proof['pending_questions'],0);self.assertEqual(proof['matrix'],mr)
        self.assertEqual(proof['status'],'completed_all_required_QA_coverage_raw_verified');self.assertEqual(value['rows'][0]['status'],'failed')
        obs['questions']=obs['questions'][:1]
        value=self.m.assemble(matrix,dict(matrix=mr,queue_terminal=terminal,observations={req['id']:obs}))
        self.assertFalse(value['coverage_complete']);self.assertFalse(value['full_matrix_complete'])

    def test_pending_question_is_missing_and_never_counted_as_executed_zero(self):
        req,obs=self.fixture();obs['questions']=obs['questions'][:1]
        row,_=self.m.qa_observation(req,obs)
        self.assertEqual(row['counts']['pending'],1);self.assertIsNone(row['metrics']);self.assertEqual(row['status'],'pending')

    def test_primary_paired_scores_are_original_llm_judge_with_f1_auxiliary(self):
        req,obs=self.fixture();value=read_document(obs['questions'][0]['path']);value['scores']['f1']=.9
        value['checkpoint_digest']=digest({k:v for k,v in value.items() if k!='checkpoint_digest'});obs['questions'][0]=self.save('q1/record.json',value)
        row,_=self.m.qa_observation(req,obs)
        self.assertEqual(row['question_scores'],{'q1':.5,'q2':0});self.assertEqual(row['question_f1_scores'],{'q1':.9,'q2':0})
        row.update(cohort='compABCD');tables=self.m.paired_tables([row],{'a':'comp'},replicates=100)
        self.assertEqual(tables['primary_course_macro'][0]['course_macro_llm_judge'],.25)
        self.assertEqual(tables['auxiliary_f1']['primary_course_macro'][0]['course_macro_f1'],.45)

    def test_uncollected_rows_describe_receipt_verification_without_claiming_no_run(self):
        row=self.m.assemble(dict(requirements=[dict(id='t',category='geometry')]),{})['rows'][0]
        self.assertEqual(row['reason'],'receipt_verification_pending')

    def test_new64_unsealed_never_opens_dataset_or_question_records(self):
        req,obs=self.fixture();req['split']='new_blind64'
        with patch.object(self.m,'checked',side_effect=AssertionError('No unsealed question read')):
            row,_=self.m.qa_observation(req,obs)
        self.assertEqual(row['status'],'pending');self.assertEqual(row['reason'],'blind64_seal_and_freeze_required')

    def test_wrong_raw_answer_or_judgment_is_not_an_accepted_success(self):
        req,obs=self.fixture();raw=read_document(obs['judge_raw']['q1']['path']);raw['response']['choices'][0]['message']['content']=json.dumps(dict(score=1,grounding=1));obs['judge_raw']['q1']=self.save('q1/judge_raw.json',raw)
        with self.assertRaises(ValueError):self.m.qa_observation(req,obs)

    def test_qa_revalidates_explicit_original_graph_and_code_dependencies(self):
        req,obs=self.fixture();graph=self.save('original_graph.json',{'entities':[]});code=self.save('frozen_code.py',{'code':'original'})
        c=read_document(obs['contract']['path']);c.update(graph=graph,code={'runner':code});c['result_id']=digest({k:v for k,v in c.items() if k!='result_id'});identity=c['result_id'];obs['contract']=self.save('contract.json',c);obs['result']=self.save('result.json',dict(result_id=identity))
        for i,r in enumerate(obs['questions']):
            q=read_document(r['path']);q['result_id']=identity;q['checkpoint_digest']=digest({k:v for k,v in q.items() if k!='checkpoint_digest'});obs['questions'][i]=self.save('q'+str(i+1)+'/record.json',q)
        self.assertEqual(self.m.qa_observation(req,obs)[0]['status'],'failed')
        for source in [graph,code]:
            original=Path(source['path']).read_bytes();Path(source['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):self.m.qa_observation(req,obs)
            Path(source['path']).write_bytes(original)

    def test_dvkg_requires_actual_frozen_graph_and_exact_semantic_condition(self):
        graph=self.save('g0.json',dict(entities=[]));inv=self.save('states.json',dict(states={'co_G0':dict(cohort='compAB',state='G0',graph=graph)}))
        req=dict(category='DVKG_control',cohort='compAB',variant='G0_prior_frozen_dual_not_ECHA',graph_sha256=graph['sha256'])
        deps=self.m.verify_QA_dependencies(req,dict(inventory=inv),{},{});self.assertIn(graph,deps)
        Path(graph['path']).write_text('{}')
        with self.assertRaises(ValueError):self.m.verify_QA_dependencies(req,dict(inventory=inv),{},{})

    def test_native_answer_requires_original_provider_response_and_track_binding(self):
        req,obs=self.fixture();req.update(category='native_end_to_end_RAG',method='graphrag',variant='Local_only',cohort=None)
        c=read_document(obs['contract']['path']);c.update(method='graphrag',book='a',tracks=['local']);c['result_id']=digest({k:v for k,v in c.items() if k!='result_id'});identity=c['result_id'];obs['contract']=self.save('contract.json',c)
        obs['result']=self.save('result.json',dict(result_id=identity));obs['track']='local';obs['native_qa']={};obs['provider_receipts']={}
        for i,record in enumerate(obs['questions']):
            q=read_document(record['path']);q['result_id']=identity;q['track']='local'
            if i==0:
                q.pop('answer_receipt');q['answer']='PRIVATE NATIVE';q.pop('candidate');native=dict(job_identity=identity,answer=q['answer']);native['checkpoint_digest']=digest(native);obs['native_qa']['q1']=self.save('q1/native_qa.json',native)
                body=dict(messages=[dict(role='user',content='PRIVATE QUESTION')]);config={};raw=dict(choices=[dict(finish_reason='stop',message=dict(content='PRIVATE NATIVE'))]);p=dict(endpoint='/chat/completions',request=body,runtime_config=config,response=raw);p['request_id']=digest(dict(endpoint=p['endpoint'],body=body,config=config));obs['provider_receipts']['q1']=[self.save('q1/provider.json',p)]
            q['checkpoint_digest']=digest({k:v for k,v in q.items() if k!='checkpoint_digest'});obs['questions'][i]=self.save('q'+str(i+1)+'/record.json',q)
        row,_=self.m.qa_observation(req,obs);self.assertEqual(row['counts']['completed'],1)
        obs['track']='global'
        with self.assertRaises(ValueError):self.m.qa_observation(req,obs)
        obs['track']='local';obs['provider_receipts']={}
        with self.assertRaises(ValueError):self.m.qa_observation(req,obs)

    def test_hippo_answer_uses_pinned_official_first_answer_segment_and_no_other_normalization(self):
        parser=self.save('HippoRAG/src/hipporag/HippoRAG.py',dict(code='frozen official fixture'));contract=dict(source=dict(path=str(self.root/'HippoRAG')))
        with patch.object(self.m,'HIPPORAG_PARSER_SHA',parser['sha256']):
            self.assertEqual(self.m.native_response_answer('hipporag','reason Answer: first Answer: last',contract)[0],'first')
            self.assertEqual(self.m.native_response_answer('hipporag','  no marker  ',contract)[0],'  no marker  ')
        with self.assertRaises(ValueError):self.m.native_response_answer('hipporag','Answer: first',contract)

    def test_light_query_cache_links_exact_question_params_and_bounded_original_receipt(self):
        source=self.save('LightRAG/lightrag/operate.py',{'official':'fixture'});graph=self.save('index/graph.graphml',{})
        question=dict(question_id='qid',target_book_id='a',question='PRIVATE QUESTION');native=dict(answer='PRIVATE ANSWER',native_result=dict(query=question['question'],target_book_id='a'))
        config=dict(llm=dict(model='flash'));body=dict(model='flash',messages=[dict(role='user',content=question['question'])]);raw=dict(endpoint='/chat/completions',request=body,runtime_config=config,response=dict(choices=[dict(finish_reason='stop',message=dict(content=native['answer']))]));key=digest(dict(endpoint=raw['endpoint'],body=body,config=config));raw['request_id']=key;receipt=self.save('formal_provider/receipts/raw.json',raw)
        dbpath=self.root/'formal_provider/audit.sqlite'
        with closing(sqlite3.connect(dbpath)) as db:
            db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,created REAL,key TEXT,stage TEXT,status TEXT)');db.execute('CREATE TABLE receipts(key TEXT,record TEXT)');db.execute('INSERT INTO events VALUES(1,100,?,?,?)',(key,'lightrag.a.query','request_started'));db.execute('INSERT INTO receipts VALUES(?,?)',(key,json.dumps(receipt)));db.commit()
        params=dict(answer_cache_version='query-answer-cache-v2',mode='hybrid',response_type='Single Paragraph',top_k=40,chunk_top_k=20,max_entity_tokens=6000,max_relation_tokens=8000,max_total_tokens=30000,enable_rerank=False,enable_content_headings=True,related_chunk_number=5,kg_chunk_pick_method='VECTOR',hl_keywords=[],ll_keywords=[],user_prompt=None)
        self.save('index/kv_store_llm_response_cache.json',{'hybrid:query:exact':dict(cache_type='query',original_prompt=question['question'],return_value='unused',**{'return':native['answer']},queryparam=params,create_time=100)})
        contract=dict(source=dict(path=str(self.root/'LightRAG')),runtime_config=config)
        with patch.object(self.m,'LIGHT_OPERATE_SHA',source['sha256']):
            proof=self.m.light_cached_answer_origin(dict(native_graphs=[graph]),contract,question,native,self.root,self.root/'private')
            self.assertEqual(proof['receipt'],receipt);self.assertEqual(proof['event_ids'],[1]);self.assertNotIn('PRIVATE',json.dumps({k:v for k,v in proof.items() if k!='entry_snapshot'}))
            question['question']='ANOTHER QUESTION'
            with self.assertRaises(ValueError):self.m.light_cached_answer_origin(dict(native_graphs=[graph]),contract,question,native,self.root,self.root/'private')

    def test_geometry_requires_same_graph_and_frozen_helper(self):
        graph=self.save('graph.json',dict(entities=[]));helper=self.save('helper.json',dict(version='geometry_stats_20261018_v1'))
        result=self.save('geometry.json',dict(status='completed_new_geometry',graph=graph,helper=helper,metrics=dict(version='geometry_stats_20261018_v1',status='completed',nodes=2)))
        req=dict(id='tree',category='geometry',method='tree_kg',graph_sha256=graph['sha256']);obs=dict(result=result,graph=graph,helper=helper)
        with patch.object(self.m,'GEOMETRY_HELPER_SHA',helper['sha256']):
            row=self.m.geometry_observation(req,obs);self.assertEqual(row['status'],'completed');self.assertEqual(row['metrics']['nodes'],2)
            req['graph_sha256']='0'*64
            with self.assertRaises(ValueError):self.m.geometry_observation(req,obs)

    def test_all_required_tree_rows_remain_pending_even_with_exit_zero(self):
        matrix=dict(requirements=[dict(id='tree',category='geometry',method='tree_kg',required=True),dict(id='kge',category='kge_training',method='own',required=True)],scope={})
        value=self.m.assemble(matrix,dict(observations={},queue_terminal={'exit_code':0,'status':'running'}))
        self.assertFalse(value['full_matrix_complete']);self.assertEqual(len(value['rows']),2);self.assertEqual(value['tables']['geometry'][0]['method'],'tree_kg')

    def test_schema_missing_is_not_automatically_na(self):
        req=dict(id='tree',category='geometry',method='tree_kg',required=True)
        value=self.m.assemble(dict(requirements=[req],scope={}),dict(observations={'tree':dict(status='not_applicable')}))
        self.assertFalse(value['full_matrix_complete']);self.assertNotEqual(value['rows'][0]['status'],'not_applicable')

    def test_book_pairing_equal_courses_and_ab_not_double_counted(self):
        courses={'a':'comp','b':'comp','c':'comp','d':'comp','e':'os','f':'os','g':'net','h':'net'};rows=[]
        for method in ['left','right']:
            for book,course in courses.items():
                cohort='compABCD' if course=='comp' else course+'AB'
                score=1 if method=='left' and course=='comp' else 0
                rows.append(dict(id=method+book,category='controlled_graph_RAG',method=method,variant='main',split='regression313',cohort=cohort,target_book=book,status='completed',counts=dict(fixed=1),metrics=dict(f1=score),question_scores={book+'q':score}))
        rows.append(dict(rows[0],id='duplicateAB',cohort='compAB',question_scores={'aq':0},metrics=dict(f1=0)))
        tables=self.m.paired_tables(rows,courses,replicates=100)
        primary=next(r for r in tables['paired_course_uncertainty'] if r['left'].startswith('controlled_graph_RAG:left'))
        self.assertAlmostEqual(primary['mean_difference'],1/3);self.assertEqual(primary['paired_questions'],8)
        self.assertEqual(tables['AB_ABCD'][0]['paired_questions'],1)

    def test_http_cost_counts_actual_events_not_response_or_cache(self):
        p=self.root/'audit.sqlite'
        with closing(sqlite3.connect(p)) as db:
            db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,created REAL,key TEXT,stage TEXT,status TEXT)')
            for status in ['request_started','response','cache_hit','request_started']:db.execute('INSERT INTO events(created,key,stage,status) VALUES(1,?,?,?)',('same','identity_single_v2.source.primary',status))
            db.commit()
        value=self.m.ledger_snapshot(p)
        self.assertEqual(value['actual_HTTP'],2);self.assertEqual(value['successful_responses'],1);self.assertEqual(value['cache_hits'],1);self.assertEqual(value['groups']['identity'],2)

    def test_model_receipt_needs_actual_1000_losses_and_complete_artifact_receipts(self):
        from course_experiments.evaluation.identity_kge_execution import training_artifacts
        root=self.root/'model';root.mkdir();job=dict(training_job_id='j',model='TransE',seed=42,epochs=1000)
        for name in ['model/trained_model.pkl','external_kge_identity.json','external_kge_protocol.json','pykeen_metrics.json','model/training_triples/entity_to_id.tsv.gz']:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        self.save('model/model/results.json',dict(losses=[1.0]*1000));artifacts=training_artifacts(root,1000)
        receipt=dict(plan_id='p',job=job,smoke=False,actual_epochs=1000,artifacts=artifacts);receipt['result_id']=digest(receipt);record=self.save('model/verified_result.json',receipt)
        verified=self.m.verify_model_receipt(record);self.assertEqual(verified['actual_epochs'],1000)
        self.save('model/model/results.json',dict(losses=[1.0]*999))
        with self.assertRaises(ValueError):self.m.verify_model_receipt(record)

    def test_render_is_aggregate_only_and_includes_three_tables(self):
        value=self.m.assemble(dict(requirements=[dict(id='t',category='geometry',method='tree_kg')],scope={}),dict(observations={}))
        self.m.render(value,self.root/'report')
        page=(self.root/'report/index.html').read_text('utf-8');self.assertIn('geometry',page);self.assertIn('KGE',page);self.assertIn('RAG',page)
        self.assertTrue((self.root/'report/summary.json').exists())

    def test_finite_collector_reuses_unchanged_file_sha_but_rehashes_changed_bytes(self):
        path=Path(self.save('immutable_graph.bin',{'large_graph':1})['path'])
        with patch.object(self.m,'_hash_file_record',wraps=file_record) as hashing:
            first=self.m.file_record(path);self.assertEqual(self.m.file_record(path),first);self.assertEqual(hashing.call_count,1)
            path.write_text('changed graph')
            self.assertNotEqual(self.m.file_record(path),first);self.assertEqual(hashing.call_count,2)

    def test_current_required_append_keeps_original_tree_and_registered_global_states(self):
        matrix=dict(requirements=[dict(id='tree',category='geometry',method='tree_kg')],scope=dict(cohorts=dict(compAB=['a','b'],compABCD=['a','b','c','d'],osAB=['e','f'],netAB=['g','h'])),datasets={s:dict(by_book={b:8 for b in 'abcdefgh'}) for s in ['regression313','existing_independent123','new_blind64']})
        result=self.m.current_requirements(matrix)
        self.assertEqual(result['requirements'][0],matrix['requirements'][0]);self.assertEqual(len(matrix['requirements']),1)
        self.assertEqual(sum(r['category']=='native_end_to_end_RAG' and r['variant']=='Global' for r in result['requirements']),24)
        self.assertEqual(sum(r['category']=='DVKG_control' for r in result['requirements']),60)

    def test_pinned_collector_never_reads_new64_or_guesses_a_result_version(self):
        matrix=dict(requirements=[dict(id='blind',category='controlled_graph_RAG',split='new_blind64',evidence=[dict(path='DO_NOT_READ',sha256='x')])],scope={})
        with patch.object(self.m,'checked',side_effect=AssertionError('Unsealed question source read')):
            value=self.m.collect_pinned_evidence(matrix,dict(path='matrix',sha256='x'))
        self.assertEqual(value['observations'],{});self.assertFalse(value['new64_question_sources_read'])

    def test_original_native_index_requires_exact_index_contract_and_binary_sha(self):
        ic=dict(method='hipporag',book='a',scope='full');idx=digest(ic);ic['index_identity']=idx
        receipt=dict(job_identity=idx,native_index=dict(status='completed'));receipt['checkpoint_digest']=digest(receipt)
        c=dict(method='hipporag',book='a',index_identity=idx);c['job_identity']=digest(c)
        art=dict(method='hipporag',book='a',contract=self.save('native/contract.json',c),complete=self.save('native/complete.json',dict(status='completed',job_identity=c['job_identity'])),index_contract=self.save('index/contract.json',ic),index_receipt=self.save('index/receipt.json',receipt),native_graphs=[self.save('index/native_graph.bin',{'opaque_fixture':1})])
        req=dict(id='idx',category='native_index',method='hipporag',target_book='a')
        row=self.m.native_index_observation(req,art);self.assertEqual(row['status'],'completed');self.assertFalse(row['common_schema_semantic_qualification_claimed'])
        Path(art['native_graphs'][0]['path']).write_text('tampered')
        with self.assertRaises(ValueError):self.m.native_index_observation(req,art)

    def test_failed_zero_stays_in_course_paired_comparison(self):
        rows=[dict(category='controlled_graph_RAG',method='left',variant='main',split='regression313',target_book='a',cohort='compABCD',status='failed',counts=dict(fixed=2,pending=0),question_scores={'q1':.5,'q2':0}),dict(category='controlled_graph_RAG',method='right',variant='main',split='regression313',target_book='a',cohort='compABCD',status='completed',counts=dict(fixed=2,pending=0),question_scores={'q1':0,'q2':0})]
        value=self.m.paired_tables(rows,{'a':'comp'},replicates=100)
        self.assertEqual(value['paired_course_uncertainty'][0]['paired_questions'],2);self.assertEqual(value['paired_course_uncertainty'][0]['mean_difference'],.25)


if __name__=='__main__':unittest.main()

