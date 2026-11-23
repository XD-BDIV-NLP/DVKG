import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from course_experiments.contributions import write_immutable
from course_experiments.manifest import digest,file_record,read_document
from course_experiments.tests import test_identity_single_v2_g0 as fixtures


class AppendInventoryTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_inventory_append
        return identity_inventory_append

    def artifact(self,root,*,complete=True,typed=True,semantic=True):
        entity=dict(entity_id='x',type='Concept' if typed else None)
        if not typed:entity['type_status']='native_untyped_candidate'
        edge=dict(source='x',target='x',predicate='is_a')
        value=dict(status='complete' if complete else 'partial',scope='full',method_id='k12_kgraph',course_id='course',
            native_graph=dict(nodes=[dict(id='x',type=entity['type'])],edges=[edge],directed=True))
        if semantic:value['semantic_view']=dict(entities=[entity],relations=[dict(head_entity_id='x',tail_entity_id='x',relation='is_a',relation_id=digest(edge))],capabilities={'kge':typed})
        value['artifact_id']=digest(value);path=root/'artifact.json';write_immutable(path,value);return file_record(path)

    def test_partial_and_schema_missing_are_pending_and_only_proven_native_untyped_is_NA(self):
        api=self.api()
        for complete,typed,semantic,expected in [(False,True,True,'pending_incomplete_native_artifact'),(True,True,False,'pending_common_schema_semantic_qualification'),(True,False,True,'not_applicable_proven_native_untyped')]:
            with tempfile.TemporaryDirectory() as tmp:self.assertEqual(api.classify_complete(self.artifact(Path(tmp),complete=complete,typed=typed,semantic=semantic))['status'],expected)

    def test_semantic_projection_cannot_invent_native_type_or_predicate(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=self.artifact(root);self.assertEqual(api.classify_complete(record)['status'],'eligible_complete_original_semantic_projection')
            value=read_document(record['path']);value['semantic_view']['relations'][0]['relation']='causes';value['artifact_id']=digest({k:v for k,v in value.items() if k!='artifact_id'})
            changed=root/'changed.json';write_immutable(changed,value)
            self.assertEqual(api.classify_complete(file_record(changed))['status'],'pending_common_schema_semantic_qualification')

    def test_original_undirected_edges_without_predicates_are_proven_relation_NA(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=self.artifact(root);value=read_document(record['path']);value['native_graph']['directed']=False
            value['native_graph']['edges']=[{'source':'x','target':'x','description':'native navigation'}]
            value['semantic_view']['relations']=[];value['semantic_view']['capabilities']['kge']=None
            value['artifact_id']=digest({k:v for k,v in value.items() if k!='artifact_id'});path=root/'undirected.json';write_immutable(path,value)
            self.assertEqual(self.api().classify_complete(file_record(path))['status'],'not_applicable_proven_native_undirected_relations')

    def test_appending_preserves_original_packets_including_unknown_and_all_original_graph_records(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,populations=fixtures.G0V2Tests().fixture(root)
            inv=read_document(mp/'input_inventory.json');new_graph=root/'new_graph.json';write_immutable(new_graph,dict(entities=[],relations=[]))
            row=dict(method='new_method',course_id='course',path=str(new_graph),reference=inv['graphs'][0]['reference'],book_set=[],cohorts=['new'])
            old_source=read_document(mp/'source_full_packets_server_only.json');old_prior=read_document(mp/'prior_full_packets_server_only.json')
            origins=[read_document(mp/(b+'_full_origins_server_only.json')) for b in ['source','prior']]
            with patch.object(api,'prepare_population',return_value=(old_source,old_prior,*origins)):
                value=api.append_preparation(mp,[row],root/'next',evidence={'inventory':'public-proof'},token_assets_root='/official')
            self.assertEqual(value['additional_source_candidates'],0);self.assertEqual(value['additional_prior_candidates'],0)
            self.assertFalse(value['qualification_created']);self.assertEqual(value['model_calls'],0)
            self.assertEqual(read_document(root/'next/input_inventory.json')['graphs'][:len(inv['graphs'])],inv['graphs'])
            self.assertEqual(read_document(root/'next/source_full_packets_server_only.json'),old_source)
            self.assertEqual(value['old_decisions_policy'],'inherit_every_original_classification_including_UNKNOWN_without_new_HTTP')

    def test_append_rejects_a_regenerated_old_candidate_changed_in_place(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,_,_,_=fixtures.G0V2Tests().fixture(root)
            source=read_document(mp/'source_full_packets_server_only.json');source[0]['course_id']='changed'
            prior=read_document(mp/'prior_full_packets_server_only.json');origins=[read_document(mp/(b+'_full_origins_server_only.json')) for b in ['source','prior']]
            with patch.object(api,'prepare_population',return_value=(source,prior,*origins)):
                with self.assertRaises(ValueError):api.append_preparation(mp,[],root/'next',evidence={},token_assets_root='/official')
            self.assertFalse((root/'next/manifest.json').exists())

    def corrected_fixture(self,root):
        api=self.api();graphs=[];cohorts={};states={};pins={}
        ref=root/'ref.json';write_immutable(ref,{'entities':[],'relations':[]});evidence=root/'evidence.json';write_immutable(evidence,{})
        for co,course in [('compAB','comp'),('compABCD','comp'),('osAB','os'),('netAB','net')]:
            graph=root/(co+'.json');write_immutable(graph,dict(entities=[dict(entity_id=co,type='Concept')],relations=[dict(head_entity_id=co,tail_entity_id=co,relation='is_a')]))
            record=file_record(graph);pins[co]=record['sha256'];states[co]=dict(cohort=co,state='F_corrected',active_books=[co+'book'],graph=record)
            cohorts[co]=dict(active_books=[co+'book'],reference=file_record(ref),graph=record);graphs.append(dict(method='T',course_id=course,cohorts=[co],book_set=[co+'book'],path=str(graph),reference=file_record(ref),evidence_path=str(evidence)))
        assets=root/'assets.json';write_immutable(assets,dict(cohorts=cohorts));inventory=root/'corrected_inventory.json';write_immutable(inventory,dict(version='echa_asset_inventory_v1',assets=file_record(assets),states=states))
        return dict(graphs=graphs),file_record(inventory),pins

    def test_frozen_corrected_four_are_exact_sha_new_rows_with_old_reference_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,record,pins=self.corrected_fixture(root);rows,audits=self.api().corrected_rows(old,record,pins)
            self.assertEqual(len(rows),4);self.assertEqual({r['method'] for r in rows},{'F_corrected'})
            self.assertTrue(all(r['reference']==old['graphs'][0]['reference'] for r in rows));self.assertTrue(all(r['evidence_path']==old['graphs'][0]['evidence_path'] for r in rows))
            self.assertTrue(all(r['status']=='eligible_frozen_corrected_semantic_graph' for r in audits))
            self.assertTrue(all('artifact' not in r for r in rows));self.assertEqual(len(self.api().corrected_rows(dict(graphs=old['graphs']+rows),record,pins)[0]),0)

    def test_corrected_f_uses_pinned_fused_source_evidence_when_textbook_evidence_also_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,record,pins=self.corrected_fixture(root);other=root/'textbook_evidence.json';write_immutable(other,{})
            old['graphs'].append(dict(old['graphs'][0],evidence_path=str(other),method='textbook_only'))
            rows,_=self.api().corrected_rows(old,record,pins)
            self.assertEqual(rows[0]['evidence_path'],str(root/'evidence.json'))

    def test_corrected_selector_rejects_missing_cohort_changed_sha_or_foreign_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,record,pins=self.corrected_fixture(root);pins['compAB']='0'*64
            with self.assertRaises(ValueError):self.api().corrected_rows(old,record,pins)
            old,record,pins=self.corrected_fixture(root/'second')
            value=read_document(record['path']);del value['states']['netAB'];changed=root/'missing.json';write_immutable(changed,value)
            with self.assertRaises(ValueError):self.api().corrected_rows(old,file_record(changed),pins)

    def test_corrected_rag_assets_cannot_replace_the_original_kge_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,record,pins=self.corrected_fixture(root)
            inventory=read_document(record['path']);assets=read_document(inventory['assets']['path']);foreign=root/'foreign_ref.json';write_immutable(foreign,{'entities':[],'relations':[],'foreign':True})
            assets['cohorts']['osAB']['reference']=file_record(foreign);ar=root/'foreign_assets.json';write_immutable(ar,assets);inventory['assets']=file_record(ar);ir=root/'foreign_inventory.json';write_immutable(ir,inventory)
            with self.assertRaises(ValueError):self.api().corrected_rows(old,file_record(ir),pins)

    def test_native_cd_append_inherits_cd_original_evidence_and_aliases_instead_of_first_course_ab(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=root/'ref.json';write_immutable(reference,{})
            inventory=dict(graphs=[]);assets=dict(cohorts={})
            for co,books in [('compAB',['A','B']),('compABCD',['A','B','C','D'])]:
                graph=root/co/'graph.json';write_immutable(graph,{});evidence=graph.parent/'evidence.json';write_immutable(evidence,{b:b for b in books})
                row=dict(course_id='comp',method='fused',cohorts=[co],path=str(graph),evidence_path=str(evidence),book_set=books,reference=file_record(reference));inventory['graphs'].append(row);assets['cohorts'][co]=dict(graph=file_record(graph),reference=file_record(reference),active_books=['book_'+b for b in books])
            row=self.api().original_cohort_source(inventory,assets,'compABCD','comp')
            self.assertEqual(row['evidence_path'],str(root/'compABCD/evidence.json'));self.assertEqual(row['book_set'],['A','B','C','D'])


if __name__=='__main__':unittest.main()
