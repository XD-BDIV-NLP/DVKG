from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from course_experiments.evaluation import run_scope_identity_batch as runner
from course_experiments.tests.test_scope_identity_batch import Provider,packet,origins


class BatchRunnerTests(unittest.TestCase):
    def test_prior24_is_course_hash_selection_without_decisions_and_no_backfill(self):
        work=[packet(i,course,is_prior=True) for course in ['os','net','comp'] for i in range(12)]
        first=runner.prior24_plan(work);second=runner.prior24_plan(list(reversed(work)))
        self.assertEqual(first,second)
        self.assertEqual(first['actual_pairs'],24)
        self.assertEqual(set(first['selected_per_course'].values()),{8})
        self.assertEqual(first['minimum_exact_decision_agreement'],23)
        short=runner.prior24_plan(work[:2])
        self.assertEqual(short['actual_pairs'],2)
        self.assertFalse(short['sufficient_for_declared24'])

    def test_checkpoint_resumes_successes_and_saves_native_response_server_only(self):
        work=[packet(1),packet(2)];original=copy.deepcopy(work)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);provider=Provider()
            result=runner.execute_batches(work,origins(work),provider,root,'test_job')
            self.assertEqual(len(provider.calls),2)
            self.assertEqual(result['candidate_count'],2)
            other=Provider();again=runner.execute_batches(work,origins(work),other,root,'test_job')
            self.assertEqual(other.calls,[])
            self.assertEqual(again['decisions'],result['decisions'])
            self.assertEqual(len(list((root/'batches').glob('*/result.json'))),2)
        self.assertEqual(work,original)

    def test_missing_batch_reply_keeps_failed_pairs_and_cannot_be_resumed_as_success(self):
        def missing(wire,result,count):result['results'].pop()
        work=[packet(1),packet(2)]
        with tempfile.TemporaryDirectory() as folder:
            result=runner.execute_batches(work,origins(work),Provider(missing),Path(folder),'test_job')
            self.assertEqual(len(result['decisions']),2)
            self.assertTrue(all(r['status']=='failed' for r in result['decisions'].values()))
            other=Provider();again=runner.execute_batches(work,origins(work),other,Path(folder),'test_job')
            self.assertEqual(other.calls,[])
            self.assertFalse(any(r['eligible_identity_edge'] for r in again['decisions'].values()))

    def test_single_pair_prior_gate_reverses_each_positive_individually(self):
        work=[packet(i,is_prior=True) for i in range(3)];provider=Provider()
        with tempfile.TemporaryDirectory() as folder:
            result=runner.execute_batches(work,origins(work),provider,Path(folder),'single',single=True)
        self.assertEqual(len(provider.calls),6)
        self.assertTrue(all(len(call[0]['pairs'])==1 for call in provider.calls))
        self.assertTrue(all(r['eligible_identity_edge'] for r in result['decisions'].values()))

    def test_local_http_cap_is_frozen_shared_across_outputs_and_not_reset(self):
        class Runtime:
            def _record(self,*args):pass
        class Stub:
            runtime=Runtime()
        provider=Stub()
        with tempfile.TemporaryDirectory() as folder:
            for turn in range(2):
                with runner.local_http_cap(provider,Path(folder),1):
                    if turn==0:provider.runtime._record('id','stage','request_started',{})
                    else:
                        with self.assertRaises(runner.LocalHTTPAllowanceExceeded):
                            provider.runtime._record('id2','stage','request_started',{})

    def test_prior24_comparison_rejects_negative_expansion_and_missing_citations(self):
        work=[packet(i,course,is_prior=True) for course in ['os','net','comp'] for i in range(8)]
        plan=runner.prior24_plan(work)
        rows={cid:dict(status='completed',primary=dict(decision='insufficient'),citation_valid=True,
            eligible_identity_edge=False) for cid in plan['candidate_ids']}
        self.assertTrue(runner.compare_prior24(plan,rows,rows)['passes_declared_gate'])
        changed=copy.deepcopy(rows);changed[plan['candidate_ids'][0]]['eligible_identity_edge']=True
        self.assertFalse(runner.compare_prior24(plan,rows,changed)['passes_declared_gate'])
        changed=copy.deepcopy(rows);changed[plan['candidate_ids'][0]]['citation_valid']=False
        self.assertFalse(runner.compare_prior24(plan,rows,changed)['passes_declared_gate'])

    def test_old_negative_veto_uses_original_graph_and_endpoints_not_method_names(self):
        work=[packet(1)];rows=origins(work)
        decisions={work[0]['candidate_id']:dict(eligible_identity_edge=True)}
        old=dict(rows[0],candidate_id='old_id',member='hidden_method')
        result=runner.apply_negative_veto(rows,decisions,[old],{'old_id':dict(eligible_identity_edge=False)})
        self.assertFalse(result[work[0]['candidate_id']]['eligible_identity_edge'])
        self.assertEqual(result[work[0]['candidate_id']]['veto'],'preserved_frozen_single_pair_negative')

    def test_stage_caps_and_source_prior_boundary_are_explicit(self):
        self.assertEqual(runner.STAGES['source-full']['http_cap'],2200)
        self.assertEqual(runner.STAGES['prior24-single']['http_cap'],64)
        with self.assertRaises(ValueError):runner.validate_stage_packets('source-full',[packet(1,is_prior=True)])
        with self.assertRaises(ValueError):runner.validate_stage_packets('prior-full',[packet(1)])

    def test_population_preparation_uses_native_sources_and_keeps_prior_separate(self):
        from course_experiments.manifest import digest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def save(name,value):
                path=root/name;path.write_text(json.dumps(value),encoding='utf-8');return str(path)
            reference=dict(course_id='os',entities=[dict(entity_id='r',canonical_name='Alpha',scope='domain',
                citations=[dict(evidence_id='s',quote='Alpha is a method.')])],facts=[])
            ref=save('reference.json',reference);evidence=save('evidence.json',{'s':dict(text='Alpha is a method.')})
            entity=dict(entity_id='g',canonical_name='Alpha',scope='',type='Concept',definition='A method.')
            edge=dict(head_entity_id='g',tail_entity_id='g',relation='is_a')
            graph=save('graph.json',dict(entities=[dict(entity,evidence_ids=['s'])],relations=[edge]))
            second=save('second.json',dict(entities=[dict(entity,entity_id='g2',evidence_ids=['s'])],
                relations=[dict(edge,head_entity_id='g2',tail_entity_id='g2')]))
            prior_graph=save('prior.json',dict(entities=[entity],relations=[edge]))
            inventory=dict(graphs=[dict(method='typed',course_id='os',path=graph,reference=dict(path=ref),evidence_path=evidence),
                dict(method='G0',course_id='os',path=prior_graph,reference=dict(path=ref))])
            source,prior,source_origins,prior_origins=runner.prepare_population(inventory)
            self.assertEqual(len(source),1);self.assertEqual(len(prior),1)
            self.assertTrue(source[0]['both_sides_have_source'])
            self.assertEqual(prior[0]['right']['definition_quotes'][0]['provenance'],'prior_generated_definition')
            self.assertEqual(source_origins[0]['graph_record']['path'],graph)
            self.assertNotIn('relations',prior[0])
            inventory['graphs'].append(dict(method='typed',course_id='os',path=second,reference=dict(path=ref),evidence_path=evidence))
            self.assertEqual(runner.prepare_population(inventory),runner.prepare_population(dict(graphs=list(reversed(inventory['graphs'])))))

    def test_frozen_input_receipt_rejects_changed_content(self):
        from course_experiments.manifest import file_record
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'frozen.json';path.write_text('{"x":1}',encoding='utf-8');record=file_record(path)
            self.assertEqual(runner.verified_read(record),{'x':1})
            path.write_text('{"x":2}',encoding='utf-8')
            with self.assertRaises(ValueError):runner.verified_read(record)

    def test_input_bound_covers_reverse_regrouping_of_largest_positive_pairs(self):
        from course_experiments.evaluation import scope_identity_batch as core
        work=[packet(i) for i in range(16)]
        order=[p for rows in core.plan_batches(work) for p in rows]
        for number,p in enumerate(order):
            p['right']['source_quotes'][0]['quote']='large original source '+('x'*2000 if number%2==0 else 'x')
        maximum=runner._input_max(work)
        largest=order[::2]
        wire=dict(version=core.POLICY['version'],kind=core._kind(largest[0]),course_id='os',
            pairs=[core._anonymous(p) for p in largest]);wire['batch_id']=core.digest(wire)
        actual=len(json.dumps(wire,ensure_ascii=False).encode('utf-8'))
        self.assertGreaterEqual(maximum['maximum_user_json_utf8_bytes'],actual)

    def test_execution_uses_token_packing_and_preserves_primary_membership_for_reverse(self):
        from course_experiments.tests.test_scope_identity_packing import Counter
        work=[packet(i) for i in range(5)]
        for p in work:p['right']['source_quotes'][0]['quote']='x'*40
        provider=Provider()
        with tempfile.TemporaryDirectory() as folder:
            result=runner.execute_batches(work,origins(work),provider,Path(folder),'tokens',counter=Counter())
        self.assertEqual(len(provider.calls),6)
        self.assertTrue(all(len(call[0]['pairs'])<=2 for call in provider.calls))
        self.assertTrue(all(r['eligible_identity_edge'] for r in result['decisions'].values()))


if __name__=='__main__':unittest.main()
