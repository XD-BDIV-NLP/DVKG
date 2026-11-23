import copy
import tempfile
import unittest
from pathlib import Path

from course_experiments.contributions import write_immutable
from course_experiments.manifest import digest,file_record
from course_experiments.tests import test_identity_kge as fixtures


class RunIdentityKGETests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import run_identity_kge
        return run_identity_kge

    def test_check_only_missing_full_dependencies_never_freezes_or_trains(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as root:
            result=api.inspect_dependencies(Path(root))
            self.assertFalse(result['ready']);self.assertEqual(result['model_calls'],0)
            self.assertFalse(result['training_started'])
            self.assertEqual(set(result['missing_stages']),set(api.STAGES))

    def test_mapping_preparation_accepts_only_reviewed_provider_identity_revision(self):
        api=self.api()
        from unittest.mock import patch
        from course_experiments.evaluation import scope_identity_batch as core
        with tempfile.TemporaryDirectory() as root:
            value=dict(files={},implementation={'native_provider.py':'5e8661663e476183f401a201971636a0402dfa6ba2a275c6b2ae3fd4ad80e859',
                'scope_identity_batch.py':file_record(core.__file__)['sha256']},token_assets_root='synthetic')
            value['preparation_id']=digest(value);write_immutable(Path(root)/'manifest.json',value)
            with patch('course_experiments.evaluation.scope_identity_packing.OfficialFlashCounter'):
                self.assertEqual(api.validate_mapping_preparation(Path(root)),value)

    def test_checkpoint_tampering_is_not_treated_as_waiting(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'source32/result.json';write_immutable(path,dict(status='completed',job_identity='fake'))
            with self.assertRaises(ValueError):api.inspect_dependencies(Path(root))

    def test_inventory_loads_original_schema_semantic_view_and_rejects_incomplete(self):
        api=self.api();fixture=fixtures.IdentityKGETests();graph=fixture.graph()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'artifact.json'
            write_immutable(path,dict(status='complete',semantic_view=dict(graph,capabilities=dict(kge=True))))
            row=dict(path=str(path),artifact=True)
            result,cap=api.load_graph(row)
            self.assertEqual(result,graph);self.assertTrue(cap['kge'])
            bad=Path(root)/'bad.json';write_immutable(bad,dict(status='partial',semantic_view=graph))
            with self.assertRaises(ValueError):api.load_graph(dict(path=str(bad),artifact=True))

    def test_frozen_package_uses_existing_exporter_and_retains_entire_fixed_test(self):
        api=self.api();fixture=fixtures.IdentityKGETests()
        from course_experiments.evaluation.identity_kge import freeze_reference,project_identity_graph
        from course_experiments.evaluation.kge import validate_kge_split
        graph=fixture.graph();reference,split=freeze_reference(fixture.reference(),'qualified')
        projection=project_identity_graph(graph,reference,split,fixture.mappings(),graph_sha='graph-sha')
        with tempfile.TemporaryDirectory() as root:
            paths=api.export_projection(Path(root),graph,reference,split,projection,
                dict(qualification_id='qualified',usable_for_KGE=True),dict(path='/original/graph',sha256='graph-sha'))
            from course_experiments.manifest import read_document
            sanitized=read_document(paths['fact_split.json'])
            validate_kge_split(sanitized)
            self.assertEqual(sanitized['folds'],split['folds'])
            self.assertEqual(sanitized['groups'].keys(),split['groups'].keys())
            self.assertTrue(all(set(g)=={'triple','fact_ids'} for g in sanitized['groups'].values()))
            self.assertEqual(read_document(paths['graph_attestation.json'])['graph_id'],digest(graph))
            again=api.export_projection(Path(root),graph,reference,split,projection,
                dict(qualification_id='qualified',usable_for_KGE=True),dict(path='/original/graph',sha256='graph-sha'))
            self.assertEqual(paths,again)

    def test_export_refuses_unqualified_identity_projection(self):
        api=self.api();fixture=fixtures.IdentityKGETests()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'qualification'):
                api.export_projection(Path(root),fixture.graph(),fixture.reference(),{}, {},
                    dict(usable_for_KGE=False),dict(path='/original/graph',sha256='graph-sha'))

    def test_export_refuses_legacy_reference_split_even_if_mapping_qualified(self):
        api=self.api();fixture=fixtures.IdentityKGETests();reference=fixture.reference()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'reference qualification'):
                api.export_projection(Path(root),fixture.graph(),reference,{}, {},
                    dict(qualification_id='qualified',usable_for_KGE=True),dict(path='/original/graph',sha256='graph-sha'))

    def test_fixed_training_matrix_deduplicates_shared_graph_and_skips_untrainable(self):
        api=self.api()
        row=dict(entry_id='e1',method='G0',graph=dict(sha256='same'),split_id='split',projection_id='projection',
            trainable_by_existing_adapter=True,cohorts=['AB','ABCD'])
        duplicate=dict(row,entry_id='e2',cohorts=['another'])
        bad=dict(row,entry_id='bad',graph=dict(sha256='bad'),trainable_by_existing_adapter=False)
        jobs=api.training_jobs([row,duplicate,bad])
        self.assertEqual(len(jobs),9)
        self.assertEqual({j['epochs'] for j in jobs},{1000})
        self.assertEqual({j['seed'] for j in jobs if j['model']=='TransE'},{42,43,44})
        self.assertEqual({j['seed'] for j in jobs if j['model']=='TransR'},{42,43,44})
        self.assertTrue(all(j['seed']==42 for j in jobs if j['model'] in {'DistMult','ComplEx','RotatE'}))
        self.assertTrue(all(j['entry_ids']==['e1','e2'] for j in jobs))
        self.assertTrue(all(j['cohorts']==['AB','ABCD','another'] for j in jobs))
        self.assertTrue(all('bad' not in j['entry_ids'] for j in jobs))
        self.assertEqual(len({j['training_job_id'] for j in jobs}),9)

    def test_shared_graph_cannot_have_two_projection_contents_for_same_training_key(self):
        api=self.api()
        row=dict(entry_id='e1',method='G0',graph=dict(sha256='same'),split_id='split',projection_id='p1',
            trainable_by_existing_adapter=True,cohorts=['AB'])
        with self.assertRaisesRegex(ValueError,'projection'):
            api.training_jobs([row,dict(row,entry_id='e2',projection_id='p2')])

    def test_completed_decision_revalidates_exact_primary_and_reverse_quotes(self):
        api=self.api()
        from course_experiments.evaluation import run_scope_identity_batch as batch
        from course_experiments.tests.test_scope_identity_batch import Provider,packet,origins
        work=[packet(1)]
        with tempfile.TemporaryDirectory() as root:
            decision=batch.execute_batches(work,origins(work),Provider(),Path(root),'offline')['decisions'][work[0]['candidate_id']]
        api.validate_completed_decision(work[0],decision)
        broken=copy.deepcopy(decision);broken['reverse']['judgment']['left_citations'][0]['quote']='not in source'
        with self.assertRaises(ValueError):api.validate_completed_decision(work[0],broken)
        broken=copy.deepcopy(decision);broken['eligible_identity_edge']=True;broken['reverse']['eligible_identity_edge']=False
        with self.assertRaises(ValueError):api.validate_completed_decision(work[0],broken)

    def test_frozen_negative_remains_negative_despite_raw_positive(self):
        api=self.api()
        from course_experiments.evaluation import run_scope_identity_batch as batch
        from course_experiments.tests.test_scope_identity_batch import Provider,packet,origins
        work=[packet(1)]
        with tempfile.TemporaryDirectory() as root:
            decision=batch.execute_batches(work,origins(work),Provider(),Path(root),'offline')['decisions'][work[0]['candidate_id']]
        decision.update(eligible_identity_edge=False,veto='preserved_frozen_single_pair_negative')
        api.validate_completed_decision(work[0],decision)
        self.assertFalse(decision['eligible_identity_edge'])


if __name__=='__main__':unittest.main()
