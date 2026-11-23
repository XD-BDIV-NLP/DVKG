import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from course_experiments.evaluation import identity_single_v2 as identity
from course_experiments.manifest import digest


class ExecutionTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_kge_execution
        return identity_kge_execution

    def package(self):
        from course_experiments.evaluation.run_identity_kge import training_jobs,TRAINING_MATRIX
        row=dict(entry_id='g0',method='G0',graph=dict(sha256='a'*64),split_id='split',projection_id='projection',trainable_by_existing_adapter=True,
            usable_for_KGE=True,cohorts=['fixed'],training_triples=10,fixed_test_count=10,covered_test_count=2,covered_validation_count=1)
        value=dict(protocol=identity.POLICY['version'],full_matrix_qualified=False,status='G0_only_inputs_prepared_for_review',training_started=False,
            qualified_graph_count=1,graphs=[row],existing_training_matrix=dict(policy=TRAINING_MATRIX,jobs=training_jobs([row]),total_jobs=9))
        value['package_id']=digest(value);return value

    def test_exact_g0_matrix_is_only_nine_jobs_per_trainable_graph_and_no_500(self):
        api=self.api();package=self.package();jobs=api.frozen_jobs(package)
        self.assertEqual(len(jobs),9);self.assertEqual({j['epochs'] for j in jobs},{1000})
        for changed in ['epochs','seed','model']:
            other=copy.deepcopy(package);other['existing_training_matrix']['jobs'][0][changed]={'epochs':500,'seed':99,'model':'ConvE'}[changed]
            other['package_id']=digest({k:v for k,v in other.items() if k!='package_id'})
            with self.assertRaises(ValueError):api.frozen_jobs(other)

    def test_source_graph_cannot_enter_g0_execution_even_if_marked_trainable(self):
        api=self.api();package=self.package();package['graphs'][0]['method']='fused';package['package_id']=digest({k:v for k,v in package.items() if k!='package_id'})
        with self.assertRaises(ValueError):api.frozen_jobs(package)

    def test_separate_full_inventory_scope_requires_complete_source_proof_and_new_original_clock(self):
        api=self.api();package=self.package()
        from course_experiments.evaluation.identity_source_full_execution import START,AUTHORIZATION_SHA,ALLOCATION_SHA
        package.update(status='complete_frozen_inventory_identity_inputs_prepared',current_inventory_source_full_qualified=True,
            future_graphs_qualified=False,user_full_matrix_complete=False,original_population_counts={'source':7519,'prior':200,'reference':485},
            reference_qualification_id='5fe6c8630a70897424f72ffc95484bb40170d24ae02f3fe8d443d99e245dbf06',
            source_actual_HTTP_proof=dict(candidate_count=7519,zero_HTTP_without_raw=0,all_original_raw_revalidated_at_finalization=True),
            execution_authorization=dict(original_started_unix=START,deadline_unix=START+345600,authorized_total_seconds=345600,
                authorized_cumulative_HTTP=200000,authorization_sha256=AUTHORIZATION_SHA,allocation_sha256=ALLOCATION_SHA))
        package['graphs'][0]['method']='fused';package['package_id']=digest({k:v for k,v in package.items() if k!='package_id'})
        self.assertEqual(len(api.frozen_jobs(package)),9)
        self.assertEqual(api.execution_scope(package)['original_deadline_unix'],START+345600)
        self.assertFalse(api.execution_scope(package)['source_graphs_forbidden'])
        for field,value in [('current_inventory_source_full_qualified',False),('source_actual_HTTP_proof',{'candidate_count':7518}),('execution_authorization',{})]:
            changed=copy.deepcopy(package);changed[field]=value;changed['package_id']=digest({k:v for k,v in changed.items() if k!='package_id'})
            with self.assertRaises(ValueError):api.frozen_jobs(changed)

    def test_zero_uncovered_scores_retain_full_denominator_and_do_not_invent_failed_scores(self):
        api=self.api();metrics={'inverse_harmonic_mean_rank':.5,'hits_at_1':.25,'hits_at_3':.75,'hits_at_10':1.0,'arithmetic_mean_rank':4}
        value=api.fixed_denominator_metrics(metrics,covered=2,fixed=10)
        self.assertEqual(value['full_fixed_mrr'],.1);self.assertEqual(value['full_fixed_hits_at_10'],.2)
        self.assertEqual(value['coverage'],.2);self.assertNotIn('full_fixed_mean_rank',value)
        with self.assertRaises(ValueError):api.fixed_denominator_metrics(metrics,covered=11,fixed=10)
        with self.assertRaises(ValueError):api.fixed_denominator_metrics(None,covered=0,fixed=10)

    def test_extended_inventory_requires_verified_inheritance_not_only_a_new_population_count(self):
        api=self.api();package=self.package()
        from course_experiments.evaluation.identity_source_full_execution import START,AUTHORIZATION_SHA,ALLOCATION_SHA
        proof=dict(parent_package_id='a'*64,original_reference_splits_unchanged=True,all_original_classifications_and_raw_unchanged=True,
            original_HTTP_not_reasked=True,previous_population_counts={'source':7519,'prior':200,'reference':485},
            current_population_counts={'source':7520,'prior':201,'reference':485},adopted_candidates={'source':7519,'prior':200})
        proof['proof_id']=digest(proof)
        package.update(status='complete_extended_inventory_identity_inputs_prepared',current_inventory_source_full_qualified=True,future_graphs_qualified=False,user_full_matrix_complete=False,
            original_population_counts=proof['current_population_counts'],inventory_append_execution=proof,
            reference_qualification_id='5fe6c8630a70897424f72ffc95484bb40170d24ae02f3fe8d443d99e245dbf06',
            source_actual_HTTP_proof=dict(candidate_count=7520,zero_HTTP_without_raw=0,all_original_raw_revalidated_at_finalization=True),
            execution_authorization=dict(original_started_unix=START,deadline_unix=START+345600,authorized_total_seconds=345600,authorized_cumulative_HTTP=200000,
                authorization_sha256=AUTHORIZATION_SHA,allocation_sha256=ALLOCATION_SHA))
        package['package_id']=digest({k:v for k,v in package.items() if k!='package_id'});self.assertEqual(len(api.frozen_jobs(package)),9)
        package['inventory_append_execution']['original_HTTP_not_reasked']=False;package['package_id']=digest({k:v for k,v in package.items() if k!='package_id'})
        with self.assertRaises(ValueError):api.frozen_jobs(package)

    def test_untrainable_population_yields_empty_matrix_without_fake_training(self):
        api=self.api();package=self.package();package['graphs'][0]['trainable_by_existing_adapter']=False
        package['existing_training_matrix'].update(jobs=[],total_jobs=0);package['package_id']=digest({k:v for k,v in package.items() if k!='package_id'})
        self.assertEqual(api.frozen_jobs(package),[])

    def saved_receipt(self,api,root):
        from course_experiments.contributions import write_immutable
        for name,value in [('external_kge_identity.json',{}),('external_kge_protocol.json',{}),('pykeen_metrics.json',{}),('model/results.json',dict(losses=[1.0,0.5]))]:
            write_immutable(root/name,value)
        for name in ['model/trained_model.pkl','model/training_triples/entity_to_id.tsv.gz','model/training_triples/numeric_triples.tsv.gz']:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original-frozen-bytes')
        receipt=dict(plan_id='plan',job={'job':'one'},smoke=True,actual_epochs=2,artifacts=api.training_artifacts(root,2))
        receipt['result_id']=digest(receipt);return receipt

    def test_reuse_rejects_changed_saved_mapping_or_actual_training_binary(self):
        api=self.api()
        for name in ['entity_to_id.tsv.gz','numeric_triples.tsv.gz']:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);receipt=self.saved_receipt(api,root)
                self.assertEqual(api.validate_saved_receipt(receipt,root,{'job':'one'},'plan',True),receipt)
                (root/'model/training_triples'/name).write_bytes(b'tampered')
                with self.assertRaises(ValueError):api.validate_saved_receipt(receipt,root,{'job':'one'},'plan',True)

    def test_actual_training_history_must_prove_every_epoch_and_is_bound_on_reuse(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipt=self.saved_receipt(api,root)
            history=root/'model/results.json';history.write_text(json.dumps(dict(losses=[.5])))
            with self.assertRaises(ValueError):api.training_artifacts(root,2)
            with self.assertRaises(ValueError):api.validate_saved_receipt(receipt,root,{'job':'one'},'plan',True)
            history.write_text(json.dumps(dict(losses=[.5]*999)))
            with self.assertRaises(ValueError):api.training_artifacts(root,1000)

    def test_inherited_actual_training_is_reused_only_with_matching_input_and_all_original_SHA(self):
        api=self.api()
        from course_experiments.contributions import write_immutable
        from course_experiments.manifest import file_record
        projection={key:[] for key in ['training','validation','test','test_ids','validation_ids','entity_vocabulary','relation_vocabulary','covered_test_mask','covered_validation_mask']}
        projection.update(reference_id='fixed-ref',split_id='fixed-split',condition_projection={})
        job={'training_signature':'same-data-and-model','model':'TransE','seed':42}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';receipt=self.saved_receipt(api,old)
            (old/'external_kge_protocol.json').write_text(json.dumps({'projection':projection,'phase_path':'old-phase'}))
            receipt.update(job=job,software={},native_status='completed',artifacts=api.training_artifacts(old,2));receipt['result_id']=digest({k:v for k,v in receipt.items() if k!='result_id'})
            path=old/'verified_result.json';write_immutable(path,receipt)
            collection={'smoke_results':[{'result':file_record(path)}]}
            with patch('course_experiments.evaluation.kge.validate_protocol',side_effect=lambda value:value),patch('course_experiments.evaluation.protocol.validate_phase',return_value={'files':{'actual_kge_adapter':{'sha256':'same-code'}}}):
                reused=api.inherit_actual_training(collection,root/'new',job,projection,{},'same-code',smoke=True)
                self.assertTrue(reused['training_reused']);self.assertEqual((root/'new/model/results.json').read_bytes(),(old/'model/results.json').read_bytes())
                other=copy.deepcopy(projection);other['test']=[['changed','predicate','target']]
                with self.assertRaises(ValueError):api.inherit_actual_training(collection,root/'other',job,other,{},'same-code',smoke=True)
                (old/'model/training_triples/entity_to_id.tsv.gz').write_bytes(b'tampered')
                with self.assertRaises(ValueError):api.inherit_actual_training(collection,root/'tampered',job,projection,{},'same-code',smoke=True)

    def test_reuse_of_inherited_receipt_also_checks_original_receipt_lineage_SHA(self):
        api=self.api()
        from course_experiments.contributions import write_immutable
        from course_experiments.manifest import file_record
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';old=self.saved_receipt(api,source);path=source/'verified_result.json';write_immutable(path,old)
            current=self.saved_receipt(api,root/'current');current.update(training_reused=True,inherited_actual_model_receipt=file_record(path));current['result_id']=digest({k:v for k,v in current.items() if k!='result_id'})
            api.validate_saved_receipt(current,root/'current',{'job':'one'},'plan',True)
            path.write_text('{}')
            with self.assertRaises(ValueError):api.validate_saved_receipt(current,root/'current',{'job':'one'},'plan',True)

    def test_reuse_rejects_unreceipted_saved_training_mapping_files(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipt=self.saved_receipt(api,root)
            (root/'model/training_triples/unreceipted.bin').write_bytes(b'new')
            with self.assertRaises(ValueError):api.validate_saved_receipt(receipt,root,{'job':'one'},'plan',True)

    def test_busy_gpu_is_finitely_waited_for_without_killing_other_jobs(self):
        api=self.api();notifications=[];closed=[];calls=[]
        class Guard:
            def close(self):closed.append(True)
        guard=Guard();busy=iter([True,False]);clock=iter([0,1]);sleeps=[]
        result=api.await_resources(lambda:guard,lambda:next(busy),10,notifications.append,pause=sleeps.append,now=lambda:next(clock))
        self.assertIs(result,guard);self.assertEqual(closed,[True]);self.assertEqual(sleeps,[30])
        self.assertEqual(notifications,['waiting_existing_GPU_execution'])
        with self.assertRaises(RuntimeError):api.await_resources(lambda:None,lambda:True,10,notifications.append,pause=sleeps.append,now=lambda:11)

    def test_smoke_diagnostic_failure_cannot_unlock_formal_matrix(self):
        api=self.api();good=dict(status='completed',training_status='completed',diagnostics_status='completed')
        api.validate_native_completion(good,True)
        for status in ['failed','unavailable']:
            with self.assertRaises(RuntimeError):api.validate_native_completion(dict(good,diagnostics_status=status),True)
        with self.assertRaises(RuntimeError):api.validate_native_completion(dict(good,status='completed_with_diagnostic_failure',diagnostics_status='failed'),True)
        with self.assertRaises(RuntimeError):api.validate_native_completion(dict(good,training_status='not_started'),False)


if __name__=='__main__':unittest.main()
