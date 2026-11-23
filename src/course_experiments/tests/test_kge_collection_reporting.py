import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from course_experiments.manifest import digest,file_record,read_document
from course_experiments.evaluation.identity_single_v2 import POLICY
from course_experiments.evaluation.run_identity_kge import TRAINING_MATRIX,training_jobs
from course_experiments.evaluation.scope_mapping_preflight import qualification
from course_experiments.splits import canonical


class KGECollectionReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.evaluation.kge_collection_reporting'),'The independent read-only collection adapter is required')
        from course_experiments.evaluation import kge_collection_reporting
        return kge_collection_reporting

    def save(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');return file_record(p)

    def package(self,ready=False,untyped=False):
        graph=dict(entities=[dict(entity_id='e'+str(i),entity_type='native' if untyped else 'Concept') for i in range(13)],relations=[dict(head_entity_id='e0',relation='is_a',tail_entity_id='e1')])
        gr=self.save('graph.json',graph);entry=dict(method='kggen_lm_dedup' if untyped else 'G0',course_id='computer_organization',cohorts=['compAB','compABCD'],book_set=['A','B'],graph=gr,qualification=qualification(graph))
        entry['entry_id']=digest(dict(method=entry['method'],course=entry['course_id'],graph_sha256=gr['sha256'],book_set=entry['book_set']))
        if untyped:entry.update(status=entry['qualification']['status'],usable_for_KGE=False,no_schema_types_invented=True)
        else:
            triples=[['e'+str(i),'is_a','e'+str(i+1)] for i in range(12)];groups={digest(canonical(*t)):dict(triple=list(canonical(*t)),fact_ids=['f'+str(i)]) for i,t in enumerate(triples)};ids=list(groups)
            split=dict(format='sanitized_kge_split',split_id='split-fixed',reference_id='reference-fixed',original_split_digest='old-independent-split',groups=groups,folds=dict(train=ids[:10],validation=ids[10:11],test=ids[11:]),seed=42,policy={});split['export_id']=digest(split)
            projection=dict(graph_id=digest(graph),split_id=split['split_id'],reference_id=split['reference_id'],training=triples[:10] if ready else [],validation=[groups[x]['triple'] for x in split['folds']['validation']],test=[groups[x]['triple'] for x in split['folds']['test']],validation_ids=split['folds']['validation'],test_ids=split['folds']['test'],covered_test_mask=[ready],covered_validation_mask=[ready],entity_vocabulary=['e'+str(i) for i in range(13)],relation_vocabulary=['is_a'],condition_projection={},audit=dict(original_graph_file_sha256=gr['sha256']))
            projection['projection_id']=digest(projection)
            pp=self.save('inputs/projection.json',projection);sp=self.save('inputs/fact_split.json',split)
            entry.update(status='prepared_for_review_training_not_authorized' if ready else 'frozen_insufficient_training_coverage',usable_for_KGE=True,trainable_by_existing_adapter=ready,projection_id=projection['projection_id'],graph_id=projection['graph_id'],split_id=split['split_id'],training_triples=len(projection['training']),covered_test_count=int(ready),fixed_test_count=1,covered_validation_count=int(ready),fixed_validation_count=1,paths={'projection.json':pp['path'],'fact_split.json':sp['path']})
        jobs=training_jobs([entry]);package=dict(protocol=POLICY['version'],status='G0_only_inputs_prepared_for_review',graphs=[entry],reference_qualification_id='reference-qualification',reference_courses=dict(computer_organization=dict(reference_id='reference-fixed',split_id='split-fixed',folds=dict(train=10,validation=1,test=1))),existing_training_matrix=dict(policy=TRAINING_MATRIX,jobs=jobs,total_jobs=len(jobs)))
        package['package_id']=digest(package);record=self.save('package.json',package)
        req=[dict(id='q',category='kge_qualification',method='kggen_lm_dedup' if untyped else 'own_concept_graph',variant=None if untyped else 'G0',cohort='compAB',graph_sha256=gr['sha256']),dict(id='t',category='kge_training',method='kggen_lm_dedup' if untyped else 'own_concept_graph',variant=None if untyped else 'G0',cohort='compAB',graph_sha256=gr['sha256'])]
        return record,package,req

    def collection(self,package,losses=1000):
        jobs=[dict(j,private=str(self.root/'GPU/private'/j['training_job_id']),input_folder=str(self.root/'inputs'),training_signature='same-'+j['training_job_id']) for j in package['existing_training_matrix']['jobs']]
        plan=dict(package_id=package['package_id'],jobs=jobs,raw_text_received=False);plan['plan_id']=digest(plan);plan_record=self.save('GPU/prepared.json',plan);results=[]
        for job in jobs:
            root=self.root/'GPU/training'/job['training_job_id'];artifacts={}
            for name,value in [('model/results.json',dict(losses=[.25]*losses)),('model/trained_model.pkl',dict(opaque='model')),('external_kge_identity.json',{}),('external_kge_protocol.json',{}),('pykeen_metrics.json',{}),('model/training_triples/entity_to_id.tsv.gz',{})]:artifacts[name]=self.save(str((root/name).relative_to(self.root)),value)
            value=dict(status='completed_actual_model_checkpoint_verified',job=job,plan_id=plan['plan_id'],smoke=False,actual_epochs=1000,native_status='completed',software={'fixture':'frozen'},artifacts=artifacts,scores=dict(covered_test_count=1,fixed_test_count=1,coverage=1,uncovered_contribution=0,covered_mrr=.5,full_fixed_mrr=.5,covered_hits_at_1=.5,full_fixed_hits_at_1=.5,covered_hits_at_3=.5,full_fixed_hits_at_3=.5,covered_hits_at_10=.5,full_fixed_hits_at_10=.5),training_reused=False,new_training_run=True)
            value['result_id']=digest(value);result=self.save(str((root/'verified_result.json').relative_to(self.root)),value)
            results.append(dict(result=result,training_job_id=job['training_job_id'],model=job['model'],seed=job['seed'],epochs=1000,scores=value['scores'],status=value['status']))
        collection=dict(status='all_actual_full_inventory_models_collected_and_checkpoint_SHA_reverified',package_id=package['package_id'],results=results,source_text_transferred=False,raw_text_exported=False)
        return self.save('collection.json',collection),collection,plan_record

    def test_insufficient_projection_is_proven_terminal_disposition_not_a_training_success(self):
        api=self.api();p,package,req=self.package();out=api.collect_observations(req,p)
        self.assertEqual(out['observations']['q']['status'],'completed');row=out['observations']['t'];self.assertEqual(row['status'],'not_applicable');self.assertEqual(row['reason'],'frozen_insufficient_training_coverage');self.assertFalse(row['training_completed']);self.assertIsNone(row['metrics'])

    def test_unknown_or_failed_training_never_becomes_not_applicable(self):
        api=self.api();p,package,req=self.package(ready=True);cr,collection,plan=self.collection(package)
        out=api.collect_observations(req,p,cr);self.assertEqual(out['observations']['t']['status'],'pending');self.assertIn('remote',out['observations']['t']['reason'])

    def test_exact_remote_proof_accepts_all_nine_jobs_without_opening_remote_paths_on_primary(self):
        api=self.api();p,package,req=self.package(ready=True);cr,collection,plan=self.collection(package);proof=api.capture_remote_evidence(collection,plan['path']);pr=self.save('remote_proof.json',proof)
        # 捕获以后只有primary证明副本可读；GPU路径不能在primary被当作本地工件打开。
        import shutil
        shutil.rmtree(self.root/'GPU')
        out=api.collect_observations(req,p,cr,pr);row=out['observations']['t'];self.assertEqual(row['status'],'completed');self.assertEqual(len(row['models']),9);self.assertTrue(all(r['actual_loss_count']==1000 for r in row['models']));self.assertTrue(row['remote_model_artifacts_verified'])

    def test_claimed_epochs_without_actual_loss_history_are_rejected(self):
        api=self.api();p,package,req=self.package(ready=True);cr,collection,plan=self.collection(package,losses=999)
        with self.assertRaises(ValueError):api.capture_remote_evidence(collection,plan['path'])

    def test_wrong_collection_package_or_duplicate_job_is_rejected(self):
        api=self.api();p,package,req=self.package(ready=True);cr,collection,plan=self.collection(package);proof=api.capture_remote_evidence(collection,plan['path']);pr=self.save('remote_proof.json',proof)
        bad=copy.deepcopy(collection);bad['package_id']='different';out=api.collect_observations(req,p,self.save('bad.json',bad),pr);self.assertNotEqual(out['observations']['t']['status'],'completed')

    def test_original_untyped_schema_can_be_na_but_changed_graph_cannot(self):
        api=self.api();p,package,req=self.package(untyped=True);out=api.collect_observations(req,p);self.assertEqual(out['observations']['t']['status'],'not_applicable')
        Path(package['graphs'][0]['graph']['path']).write_text('{}');out=api.collect_observations(req,p);self.assertEqual(out['observations']['t']['status'],'invalid')

    def test_receipt_reuse_requires_the_original_chain_and_identical_training_signature(self):
        api=self.api();p,package,req=self.package(ready=True);cr,collection,plan=self.collection(package);r=collection['results'][0]['result'];value=read_document(r['path']);value.update(training_reused=True,new_training_run=False,inherited_actual_model_receipt=dict(path='/GPU/missing-original.json',sha256='0'*64));value['result_id']=digest({k:v for k,v in value.items() if k!='result_id'});new=self.save(str(Path(r['path']).relative_to(self.root)),value);collection['results'][0]['result']=new
        with self.assertRaises((OSError,ValueError)):api.capture_remote_evidence(collection,plan['path'])


if __name__=='__main__':unittest.main()
