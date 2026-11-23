import copy
from pathlib import Path
import tempfile
import unittest

from course_experiments.contributions import write_immutable
from course_experiments.manifest import digest,file_record,read_document
from course_experiments.tests.test_identity_single_v2 import Provider
from course_experiments.tests.test_scope_identity_batch import packet
from course_experiments.tests import test_identity_kge as oldfixtures
from course_experiments.evaluation import identity_single_v2 as v2


class G0V2Tests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_single_v2_g0
        return identity_single_v2_g0

    def fixture(self,root):
        mp=root/'mapping';rp=root/'reference';processed=root/'processed';f=oldfixtures.IdentityKGETests()
        graph=root/'g0.json';other=root/'sourcegraph.json';ref=root/'ref.json'
        write_immutable(graph,f.graph());write_immutable(other,dict(f.graph(),unrelated=True));write_immutable(ref,f.reference())
        grec=file_record(graph);orec=file_record(other);rrec=file_record(ref)
        populations={}
        for branch,p,folder,stem,record in [('prior',packet(1,is_prior=True),mp,'prior_full',grec),('reference',packet(2),rp,'full',rrec),('source',packet(3),mp,'source_full',orec)]:
            p['course_id']='course';p['left']['entity_id']='r0';p['right']['entity_id']='r1' if branch=='reference' else 'g0';p['candidate_id']=digest({k:v for k,v in p.items() if k!='candidate_id'})
            origin=dict(candidate_id=p['candidate_id'],course_id='course',graph_record=record,graph_entity_id=p['right']['entity_id'],reference_entity_id='r0',method='G0' if branch=='prior' else 'other',book_set=[])
            if branch=='reference':origin.update(left_reference_entity_id='r0',right_reference_entity_id='r1')
            write_immutable(folder/(stem+'_packets_server_only.json'),[p]);write_immutable(folder/(stem+'_origins_server_only.json'),[origin]);populations[branch]=[p]
            if branch!='source':v2.run_population([p],branch,Provider(),processed/branch)
        write_immutable(mp/'input_records.json',{r['path']:r for r in [grec,orec,rrec]})
        write_immutable(mp/'input_inventory.json',dict(graphs=[dict(method=method,course_id='course',path=record['path'],reference=rrec,cohorts=['fixed'],book_set=[])
            for method,record in [('G0',grec),('other',orec)]]))
        for folder,extra in [(mp,dict(requests={'source-full':dict(actual_pairs=1),'prior-full':dict(actual_pairs=1)})),(rp,dict(full_count=1,references={'course':rrec},source_records={}))]:
            manifest=dict(files={p.name:file_record(p) for p in folder.glob('*.json')},**extra);manifest['preparation_id']=digest(manifest);write_immutable(folder/'manifest.json',manifest)
        return mp,rp,processed,populations

    def test_only_g0_qualifies_with_entire_prior_and_reference_and_source_stays_incomplete(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,_=self.fixture(root)
            result=api.prepare_g0(mp,rp,processed,root/'package')
            self.assertEqual(result['inventory_graph_count'],2);self.assertEqual(result['qualified_graph_count'],1)
            self.assertFalse(result['full_matrix_qualified']);self.assertFalse(result['training_started'])
            self.assertEqual(result['graphs'][1]['status'],'incomplete_source_identity_qualification')
            self.assertEqual(result['reference_courses']['course']['original_fact_count'],14)
            g0=result['graphs'][0];projection=read_document(g0['paths']['projection.json'])
            self.assertEqual(len(projection['test']),g0['fixed_test_count']);self.assertEqual(projection['covered_test_mask'],[False]*g0['fixed_test_count'])
            self.assertEqual(projection['identity_disclosure']['policy']['version'],v2.POLICY['version'])
            self.assertFalse(result['old_batch_gates_passed'])

    def test_missing_full_prior_result_cannot_become_partial_g0_qualification(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,_=self.fixture(root);(processed/'prior/result.json').unlink()
            with self.assertRaises(FileNotFoundError):api.prepare_g0(mp,rp,processed,root/'package')
            self.assertFalse((root/'package/package.json').exists())

    def test_reference_split_is_reference_only_and_does_not_depend_on_graph_mapping(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,populations=self.fixture(root)
            first=api.prepare_g0(mp,rp,processed,root/'one')
            # A different independently processed prior body result changes mapping qualification,
            # while the already frozen independent reference response and input stay the same.
            alt=root/'alt';p=populations['prior'][0]
            def insufficient(value,count):value['decision']='insufficient'
            v2.run_population([p],'prior',Provider(insufficient),alt/'prior')
            import shutil
            shutil.copytree(processed/'reference',alt/'reference')
            second=api.prepare_g0(mp,rp,alt,root/'two')
            self.assertNotEqual(first['qualification_id'],second['qualification_id'])
            self.assertEqual(first['reference_qualification_id'],second['reference_qualification_id'])
            self.assertEqual(first['reference_courses'],second['reference_courses'])

    def test_g0_cannot_hide_required_source_candidates(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,_=self.fixture(root)
            rows=read_document(mp/'source_full_origins_server_only.json');rows[0]['graph_record']=file_record(root/'g0.json')
            (mp/'source_full_origins_server_only.json').write_text(__import__('json').dumps(rows))
            manifest=read_document(mp/'manifest.json');manifest['files']['source_full_origins_server_only.json']=file_record(mp/'source_full_origins_server_only.json');manifest['preparation_id']=digest({k:v for k,v in manifest.items() if k!='preparation_id'});(mp/'manifest.json').write_text(__import__('json').dumps(manifest))
            with self.assertRaisesRegex(ValueError,'source'):api.prepare_g0(mp,rp,processed,root/'package')

    def test_new_inventory_request_retains_previous_rows_and_exact_reference_split_identity(self):
        api=self.api();old=dict(reference_qualification_id='independent-reference',graphs=[dict(method='G0',course_id='course',graph=dict(path='/old/graph',sha256='old'),book_set=[])])
        old['package_id']=digest(old);new=old['graphs']+[dict(method='new_source_graph',course_id='course',graph=dict(path='/new/graph',sha256='new'),book_set=[])]
        request=api.inventory_append_request(old,new,'independent-reference')
        self.assertEqual(request['additional_graph_count'],1);self.assertFalse(request['new_graphs_qualified']);self.assertEqual(request['model_calls'],0)
        with self.assertRaises(ValueError):api.inventory_append_request(old,new,'changed-reference')
        with self.assertRaises(ValueError):api.inventory_append_request(old,new[1:],'independent-reference')

    def test_append_output_checks_actual_split_not_only_reference_qualification(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,_=self.fixture(root);first=api.prepare_g0(mp,rp,processed,root/'first')
            changed=copy.deepcopy(first);changed['reference_courses']['course']['split_id']='changed-by-another-split-implementation'
            changed['package_id']=digest({k:v for k,v in changed.items() if k!='package_id'});parent=root/'parent.json';parent.write_text(__import__('json').dumps(changed))
            with self.assertRaisesRegex(ValueError,'split'):api.prepare_g0(mp,rp,processed,root/'append',parent_package=parent)
            self.assertFalse((root/'append/package.json').exists())


if __name__=='__main__':unittest.main()
