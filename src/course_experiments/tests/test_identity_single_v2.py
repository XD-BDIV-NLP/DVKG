import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from course_experiments.manifest import digest,file_record
from course_experiments.tests.test_scope_identity_batch import packet,origins
from course_experiments.tests import test_identity_kge as oldtests


class Provider:
    def __init__(self,mutate=None):self.calls=[];self.mutate=mutate
    def raise_if_fatal(self):pass
    def request_executor(self,endpoint,body,stage):
        self.calls.append(copy.deepcopy(body));wire=json.loads(body['messages'][-1]['content'])
        pair=wire['pairs'][0] if 'pairs' in wire else wire
        value=dict(candidate_id=pair['candidate_id'],decision='same_sense',reason='synthetic')
        prior=wire.get('kind')=='prior_definition_identity'
        value.update({k:True for k in (['necessary_scope_consistent','definition_identity_sufficient','homonym_excluded'] if prior else ['scope_consistent','source_definition_sufficient','homonym_excluded'])})
        for side in ['left','right']:
            quotes=pair[side].get('source_quotes',pair[side].get('evidence_quotes'))
            value[side+'_citations']=[dict(evidence_id=quotes[0]['evidence_id'],quote=quotes[0]['quote'])]
        result=dict(batch_id=wire['batch_id'],results=[value]) if 'pairs' in wire else value
        if self.mutate:self.mutate(result,len(self.calls))
        return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(result)))]),False


class SingleV2Tests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_single_v2
        return identity_single_v2

    def test_frozen_original_single_messages_no_short_labels_or_multipair(self):
        api=self.api()
        from course_experiments.evaluation import scope_identity as source,scope_identity_packing as packing
        p=packet(1);body=api.request_body(p,'source')
        self.assertEqual(body['messages'],[dict(role='system',content=source.JUDGE),dict(role='user',content=json.dumps(source.anonymous_packet(p),ensure_ascii=False))])
        self.assertEqual(body['max_tokens'],1024)
        self.assertEqual(api.request_body(packet(1,is_prior=True),'prior')['messages'],packing.messages_for([packet(1,is_prior=True)]))
        self.assertEqual(api.request_body(p,'reference')['messages'],packing.messages_for([p]))
        self.assertNotIn('P0',json.dumps(body))

    def test_positive_requires_strict_primary_and_reverse_and_resume_is_free(self):
        api=self.api();provider=Provider();work=[packet(1)]
        with tempfile.TemporaryDirectory() as tmp:
            a=api.run_population(work,'source',provider,Path(tmp));b=api.run_population(work,'source',provider,Path(tmp))
        self.assertEqual(len(provider.calls),2);self.assertEqual(a,b)
        self.assertEqual(a['decisions'][work[0]['candidate_id']]['classification'],'accepted')
        self.assertTrue(a['all_candidates_processed']);self.assertFalse(a['old_batch_gate_passed'])

    def test_bad_quote_unknown_and_next_candidate_still_runs(self):
        api=self.api();work=[packet(1),packet(2)]
        def bad(value,count):
            if value.get('candidate_id')==work[0]['candidate_id']:value['right_citations'][0]['quote']='invented'
        provider=Provider(bad)
        with tempfile.TemporaryDirectory() as tmp:result=api.run_population(work,'source',provider,Path(tmp))
        self.assertEqual(len(provider.calls),3)
        self.assertEqual(result['decisions'][work[0]['candidate_id']]['classification'],'UNKNOWN')
        self.assertEqual(result['candidate_count'],2);self.assertTrue(result['all_candidates_processed'])
        self.assertEqual(result['unknown_count'],1);self.assertEqual(result['accepted_count'],1)

    def test_negative_insufficient_and_reverse_disagreement_stay_distinct(self):
        api=self.api()
        for label,expected in [('different_sense','confirmed_negative'),('insufficient','insufficient')]:
            def change(value,count):value['decision']=label
            with tempfile.TemporaryDirectory() as tmp:r=api.run_population([packet(1)],'source',Provider(change),Path(tmp))
            self.assertEqual(next(iter(r['decisions'].values()))['classification'],expected)
        def reverse(value,count):
            if count==2:value['decision']='different_sense'
        with tempfile.TemporaryDirectory() as tmp:r=api.run_population([packet(1)],'source',Provider(reverse),Path(tmp))
        self.assertEqual(next(iter(r['decisions'].values()))['classification'],'UNKNOWN')
        self.assertEqual(r['accepted_count'],0)

    def test_exhausted_original_case_sealed_unknown_without_retry_or_false_negative(self):
        api=self.api();p=packet(1)
        from course_experiments.evaluation.run_native_full import _checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);failed=root/'failed.json';attempts=root/'attempts.json'
            _checkpoint(failed,dict(candidate_id=p['candidate_id'],status='failed',eligible_identity_edge=False),'old')
            attempts.write_text(json.dumps(dict(candidate_id=p['candidate_id'],attempts={'1':dict(status='failed'),'2':dict(status='failed')})))
            sealed=api.sealed_unknown(file_record(failed),file_record(attempts));provider=Provider()
            r=api.run_population([p],'source',provider,root/'new',sealed={p['candidate_id']:sealed})
            self.assertEqual(provider.calls,[]);self.assertEqual(r['unknown_count'],1);self.assertEqual(r['negative_count'],0)
            self.assertFalse(next(iter(r['decisions'].values()))['eligible_identity_edge'])

    def test_conflicting_targets_all_vetoed_and_unknown_never_becomes_negative(self):
        api=self.api();work=[packet(1),packet(2)];work[1]['right']['entity_id']=work[0]['right']['entity_id'];work[1]['candidate_id']=digest({k:v for k,v in work[1].items() if k!='candidate_id'})
        with tempfile.TemporaryDirectory() as tmp:r=api.run_population(work,'source',Provider(),Path(tmp))
        projection=api.mapping_projection(origins(work),r)
        self.assertEqual(projection['mapping'],[]);self.assertEqual(len(projection['rejections']),2)
        broken=copy.deepcopy(r);broken['decisions'].pop(work[0]['candidate_id'])
        with self.assertRaises(ValueError):api.mapping_projection(origins(work),broken)

    def test_reference_unknown_internal_pair_rejects_entire_positive_component(self):
        api=self.api();orig=[];dec={}
        for cid,a,b,good in [('ab','a','b',True),('bc','b','c',True),('ac','a','c',False)]:
            orig.append(dict(candidate_id=cid,course_id='course',left_reference_entity_id=a,right_reference_entity_id=b))
            dec[cid]=dict(processed=True,classification='accepted' if good else 'UNKNOWN',eligible_identity_edge=good,citation_valid=good,reversed_positive_confirmed=good)
        r=dict(protocol=api.POLICY['version'],candidate_count=3,decisions=dec,all_candidates_processed=True)
        result=api.reference_components(orig,r)
        self.assertEqual(result['components'],{});self.assertEqual(len(result['rejected_components']),1)
        self.assertEqual(result['fixed_candidate_count'],3)

    def test_freeze_keeps_all_reference_facts_and_equivalent_inverse_conditions_coheld(self):
        api=self.api();reference=oldtests.IdentityKGETests().reference();before=copy.deepcopy(reference)
        qualification=dict(protocol=api.POLICY['version'],components={'course':[]},all_candidates_processed=True,original_reference_digests={'course':digest(reference)})
        qualification['qualification_id']=digest(qualification)
        frozen,split=api.freeze_reference(reference,qualification)
        self.assertEqual(reference,before);self.assertEqual(sorted(f['fact_id'] for f in frozen['facts']),sorted(f['fact_id'] for f in reference['facts']))
        group=next(g for g in split['groups'].values() if 'f0' in g['fact_ids'])
        self.assertEqual(set(group['fact_ids']),{'f0','inverse'})
        self.assertEqual(frozen['identity_kge_protocol']['protocol'],api.POLICY['version'])
        self.assertFalse(frozen['identity_kge_protocol']['old_batch_gates_passed'])
        changed=copy.deepcopy(reference);changed['facts'][0]['conditions']='changed'
        with self.assertRaises(ValueError):api.freeze_reference(changed,qualification)

    def test_actual_http_attempt_cap_counts_existing_cumulative_attempts(self):
        api=self.api()
        class Audited:
            def __init__(self):
                self.db=sqlite3.connect(':memory:');self.db.execute('CREATE TABLE events(key TEXT,status TEXT)');self.db.executemany('INSERT INTO events VALUES(?,?)',[('key','request_started')]*2)
            def _write_transaction(self,fn,*,operation,request_id=None):return fn(self.db)
        provider=Audited()
        with api.actual_attempt_cap(provider,2):
            with self.assertRaises(api.RequestAttemptsExhausted):provider._write_transaction(lambda db:db.execute("INSERT INTO events VALUES('key','request_started')"),operation='event_request_started',request_id='key')
        self.assertEqual(provider.db.execute('SELECT COUNT(*) FROM events').fetchone()[0],2)
        provider.db.close()

    def test_tampered_candidate_identity_is_rejected_before_calls(self):
        api=self.api();p=packet(1);p['right']['scope']='changed';provider=Provider()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):api.run_population([p],'source',provider,Path(tmp))
        self.assertEqual(provider.calls,[])

    def test_prior_and_reference_single_records_revalidate_from_original_raw(self):
        api=self.api()
        for branch,p in [('prior',packet(1,is_prior=True)),('reference',packet(2))]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);r=api.run_population([p],branch,Provider(),root)
                self.assertEqual(api.validate_saved_population([p],branch,root),r)
                self.assertEqual(r['accepted_count'],1)

    def test_changed_raw_checkpoint_aborts_validation_instead_of_becoming_unknown(self):
        api=self.api();p=packet(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);api.run_population([p],'source',Provider(),root)
            saved=root/'decisions'/p['candidate_id']/'primary.json';value=json.loads(saved.read_text());value['raw_response']={};saved.write_text(json.dumps(value))
            with self.assertRaises(ValueError):api.validate_saved_population([p],'source',root)

    def test_unknown_without_a_processed_response_or_exhaustion_proof_is_rejected(self):
        api=self.api();p=packet(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);api.run_population([p],'source',Provider(),root);folder=root/'decisions'/p['candidate_id']
            decision=json.loads((folder/'decision.json').read_text());decision.update(classification='UNKNOWN',eligible_identity_edge=False,reversed_positive_confirmed=False)
            for key in ['primary','primary_record','reverse']:decision.pop(key,None)
            decision['checkpoint_digest']=digest({k:v for k,v in decision.items() if k!='checkpoint_digest'});(folder/'decision.json').write_text(json.dumps(decision))
            result=json.loads((root/'result.json').read_text());result['decisions'][p['candidate_id']]=decision;result.update(accepted_count=0,unknown_count=1)
            result['checkpoint_digest']=digest({k:v for k,v in result.items() if k!='checkpoint_digest'});(root/'result.json').write_text(json.dumps(result))
            (folder/'primary.json').unlink();(folder/'reverse.json').unlink()
            with self.assertRaises(ValueError):api.validate_saved_population([p],'source',root)

    def test_shared_budget_exhaustion_stops_without_fake_complete_unknown(self):
        api=self.api()
        from course_experiments.evaluation.native_provider import NativeBudgetExceeded
        class Halted(Provider):
            def request_executor(self,*args):raise NativeBudgetExceeded('global exhausted')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(NativeBudgetExceeded):api.run_population([packet(1)],'source',Halted(),root)
            self.assertFalse((root/'result.json').exists())

    def test_existing_exporter_accepts_new_protocol_without_shrinking_uncovered_test(self):
        api=self.api();fixture=oldtests.IdentityKGETests();original=fixture.reference();graph=fixture.graph();before=copy.deepcopy(graph)
        qualification=dict(protocol=api.POLICY['version'],components={'course':[]},all_candidates_processed=True,original_reference_digests={'course':digest(original)},usable_for_KGE=True)
        qualification['qualification_id']=digest(qualification);reference,split=api.freeze_reference(original,qualification)
        from course_experiments.evaluation.identity_kge import project_identity_graph
        from course_experiments.evaluation.run_identity_kge import export_projection
        from course_experiments.manifest import read_document
        projection=project_identity_graph(graph,reference,split,fixture.mappings()[:2],graph_sha='graph-sha')
        with tempfile.TemporaryDirectory() as tmp:
            paths=export_projection(Path(tmp),graph,reference,split,projection,qualification,dict(path='/original/graph',sha256='graph-sha'))
            sanitized=read_document(paths['projection.json'])
        self.assertEqual(graph,before);self.assertEqual(len(sanitized['test']),len(split['folds']['test']))
        self.assertEqual(sanitized['covered_test_mask'],[False]*len(sanitized['test']))

    def test_full_original_manifests_qualify_unknown_without_claiming_old_gates(self):
        api=self.api();source=packet(1);prior=packet(2,is_prior=True);reference=packet(3)
        from course_experiments.contributions import write_immutable
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp=root/'mapping_preparation';rp=root/'reference_preparation';output=root/'processing';graph=root/'graph.json';ref=root/'reference.json'
            write_immutable(graph,dict(entities=[dict(entity_id=p['right']['entity_id'],canonical_name=p['right']['canonical_name'],scope=p['right']['scope'],entity_type='Concept') for p in [source,prior]],relations=[]))
            reference_ids={source['left']['entity_id'],prior['left']['entity_id'],reference['left']['entity_id'],reference['right']['entity_id']}
            write_immutable(ref,dict(course_id='os',entities=[dict(entity_id=eid,canonical_name=eid,scope='course',aliases=[]) for eid in sorted(reference_ids)],facts=[]));grec=file_record(graph);rrec=file_record(ref)
            for branch,p,folder,stem in [('source',source,mp,'source_full'),('prior',prior,mp,'prior_full'),('reference',reference,rp,'full')]:
                origin=dict(candidate_id=p['candidate_id'],course_id=p['course_id'],graph_record=rrec if branch=='reference' else grec,
                    graph_entity_id=p['right']['entity_id'],reference_entity_id=p['left']['entity_id'])
                if branch=='reference':origin.update(left_reference_entity_id=p['left']['entity_id'],right_reference_entity_id=p['right']['entity_id'])
                for name,value in [(stem+'_packets_server_only.json',[p]),(stem+'_origins_server_only.json',[origin])]:write_immutable(folder/name,value)
                def bad(value,count):value['right_citations'][0]['quote']='invented'
                api.run_population([p],branch,Provider(bad if branch=='source' else None),output/branch)
            write_immutable(mp/'input_records.json',{str(graph):grec,str(ref):rrec})
            for folder,extra in [(mp,dict(requests={'source-full':dict(actual_pairs=1),'prior-full':dict(actual_pairs=1)})),(rp,dict(full_count=1,references={'os':rrec},source_records={}))]:
                manifest=dict(files={p.name:file_record(p) for p in folder.glob('*.json')},**extra);manifest['preparation_id']=digest(manifest);write_immutable(folder/'manifest.json',manifest)
            qualified=api.qualify_all(mp,rp,output)
            self.assertTrue(qualified['usable_for_KGE']);self.assertEqual(qualified['branch_counts']['source']['unknown_count'],1)
            self.assertFalse(qualified['old_batch_gates_passed']);self.assertFalse(qualified['old_source32_calibration_passed'])
            self.assertEqual(len(qualified['mapping']),1);self.assertTrue(qualified['all_reference_facts_preserved'])

    def test_disjoint_workers_only_publish_full_result_after_all_fixed_candidates(self):
        api=self.api();work=[packet(i) for i in range(4)];a=Provider();b=Provider()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);partial=api.run_population(work,'source',a,root,worker_index=0,worker_count=2)
            self.assertFalse(partial['all_candidates_processed']);self.assertFalse((root/'result.json').exists())
            api.run_population(work,'source',b,root,worker_index=1,worker_count=2)
            self.assertFalse((root/'result.json').exists())
            complete=api.finalize_population(work,'source',root)
            self.assertTrue(complete['all_candidates_processed']);self.assertEqual(complete['candidate_count'],4)
            self.assertEqual(api.validate_saved_population(work,'source',root),complete)
        first={json.loads(body['messages'][-1]['content'])['candidate_id'] for body in a.calls}
        second={json.loads(body['messages'][-1]['content'])['candidate_id'] for body in b.calls}
        self.assertFalse(first&second);self.assertEqual(len(a.calls)+len(b.calls),8)

    def test_new_protocol_total_actual_http_cap_stops_without_hiding_as_unknown(self):
        api=self.api()
        class Audited:
            def __init__(self):
                self.db=sqlite3.connect(':memory:');self.db.execute('CREATE TABLE events(key TEXT,status TEXT,stage TEXT)')
                self.db.executemany('INSERT INTO events VALUES(?,?,?)',[('old','request_started','identity_single_v2.source.primary')]*2)
            def _write_transaction(self,fn,*,operation,request_id=None):return fn(self.db)
        provider=Audited()
        with api.actual_attempt_cap(provider,2,total_http_cap=2):
            with self.assertRaises(api.StageHTTPBudgetExhausted):provider._write_transaction(lambda db:None,operation='event_request_started',request_id='fresh')
        self.assertEqual(provider.db.execute('SELECT COUNT(*) FROM events').fetchone()[0],2);provider.db.close()


if __name__=='__main__':unittest.main()
