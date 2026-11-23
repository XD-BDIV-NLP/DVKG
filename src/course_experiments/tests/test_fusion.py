import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from course_experiments.contributions import export_contribution
from course_experiments.fusion import build_view
from course_experiments.views import export_view

def contribution(book=None, conditions='', status='active', scope='full'):
    evidence = {} if book is None else {'e':dict(id='e',book_id=book,text='Independent source')}
    ids = list(evidence)
    nodes = {i:dict(id=i,name=i,entity_type='Concept',definition=None,evidence_ids=ids) for i in ('x','y')}
    claim = dict(id='r',status=status,revision=1,evidence_ids=ids,content=dict(kind='relation',head='x',relation='requires',tail='y',conditions=conditions))
    return export_contribution(dict(version=1,nodes=nodes,claims={'r':claim},evidence=evidence,books={} if book is None else {book:dict(id=book)},events=[]),'course',book,scope,'input')

def mapping(*books):
    return {f'course/{b}/node/{n}':f'course/g0/node/{n}' for b in books for n in ('x','y')}


def audited_contribution(book=None, fact_count=1, event_count=3, payload_chars=128):
    evidence={} if book is None else {'e':dict(id='e',book_id=book,text='Independent source')}
    ids=list(evidence)
    nodes={i:dict(id=i,name=i,entity_type='Concept',definition=None,evidence_ids=ids) for i in ('x','y')}
    claims={f'r{i}':dict(id=f'r{i}',status='active',revision=2,evidence_ids=ids,
        content=dict(kind='relation',head='x',relation='requires',tail='y',conditions=f'condition {i}'))
        for i in range(fact_count)}
    claims['retired']=dict(id='retired',status='retired',revision=1,evidence_ids=ids,
        content=dict(kind='relation',head='y',relation='requires',tail='x',conditions='retired'))
    events=[dict(event_id=f'ledger-only-{book or "g0"}-{i}',claim_id='r0',
        operation='revise',before=dict(claim_id='retired',status='active'),
        after=dict(claim_id='r0',revision=2),raw_response='原始审计'+('x'*payload_chars))
        for i in range(event_count)]
    return export_contribution(dict(version=1,nodes=nodes,claims=claims,evidence=evidence,
        books={} if book is None else {book:dict(id=book)},events=events),'course',book,'full','input')


class AuditMemoryTests(unittest.TestCase):
    def test_serialized_exports_keep_one_ledger_and_bounded_fact_growth(self):
        g=contribution()
        a=audited_contribution('A',fact_count=40,event_count=32,payload_chars=4096)
        original=copy.deepcopy([g,a])
        with tempfile.TemporaryDirectory() as folder:
            output=export_view(build_view(g,[a],['A'],alignment=mapping('A')),folder)
            fused_text=(output/'fused.json').read_text('utf-8')
            ledger_bytes=len(json.dumps(a['graph']['events'],ensure_ascii=False).encode('utf-8'))
            total_bytes=sum(p.stat().st_size for p in output.glob('*.json'))
            self.assertLess(total_bytes,ledger_bytes*2+40*20000)
            self.assertEqual(fused_text.count('ledger-only-A-0'),1)
            for name in ('sources.json','relations.json','concept_kg.json'):
                self.assertNotIn('ledger-only-A-0',(output/name).read_text('utf-8'))
            loaded=json.loads(fused_text)
            self.assertEqual(loaded['native_contributions']['A'],a)
            self.assertEqual(len(loaded['facts']),41)
        self.assertEqual([g,a],original)

    def test_exported_audit_refs_resolve_exact_history_and_native_identity(self):
        g,a=audited_contribution(),audited_contribution('A',fact_count=4)
        old=copy.deepcopy([g,a])
        view=build_view(g,[a],['A'],alignment=mapping('A'))
        with tempfile.TemporaryDirectory() as folder:
            output=export_view(view,folder)
            fused=json.loads((output/'fused.json').read_text('utf-8'))
            sources=json.loads((output/'sources.json').read_text('utf-8'))
            self.assertEqual(sources['facts'],fused['facts'])
            self.assertEqual(sources['input_hashes'],fused['input_hashes'])
            for fact in sources['facts']:
                for support in fact['supports']:
                    self.assertNotIn('audit_events',support)
                    self.assertIn('audit_ref',support)
                    ref=support['audit_ref']
                    self.assertLess(len(json.dumps(ref)),512)
                    self.assertEqual(ref['artifact'],'fused.json')
                    key=support['book_id'] or 'g0'
                    self.assertEqual(ref['contribution_key'],key)
                    resolved=json.loads((output/ref['artifact']).read_text('utf-8'))['native_contributions'][ref['contribution_key']]
                    expected=g if key=='g0' else a
                    self.assertEqual(resolved,expected)
                    self.assertEqual(ref['artifact_id'],expected['artifact_id'])
                    self.assertEqual(ref['artifact_id'],sources['input_hashes'][key])
                    events=resolved['graph']['events']
                    encoded=json.dumps(events,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
                    self.assertEqual(ref['events_digest'],hashlib.sha256(encoded).hexdigest())
                    self.assertEqual(ref['event_count'],3)
                    claim=resolved['graph']['claims'][support['claim_id']]
                    self.assertEqual(support['evidence_ids'],claim['evidence_ids'])
                    self.assertEqual(support['revision'],2)
                    self.assertEqual(support['native_source_state'],claim['source_state'])
                    self.assertEqual(resolved['native_id_map']['claims'][claim['native_id']],support['claim_id'])
            self.assertEqual(fused['view_id'],view['view_id'])
        view['native_contributions']['A']['graph']['events'][0]['after']['revision']=99
        self.assertEqual([g,a],old)

    def test_hotplug_exports_remove_only_removed_book_audit_references(self):
        g,a,b=audited_contribution(),audited_contribution('A'),audited_contribution('B')
        old=copy.deepcopy([g,a,b])
        with tempfile.TemporaryDirectory() as folder:
            both=export_view(build_view(g,[a,b],['A','B'],alignment=mapping('A','B')),folder)
            remaining=build_view(g,[a,b],['B'],alignment=mapping('A','B'))
            removed=export_view(remaining,folder)
            self.assertEqual(remaining,build_view(g,[b],['B'],alignment=mapping('B')))
            fused=json.loads((removed/'fused.json').read_text('utf-8'))
            self.assertEqual(set(fused['native_contributions']),{'g0','B'})
            for name in ('fused.json','sources.json','relations.json','concept_kg.json'):
                text=(removed/name).read_text('utf-8')
                self.assertNotIn('ledger-only-A-',text)
                exported=json.loads(text)
                rows=exported if isinstance(exported,list) else exported.get('facts',exported.get('relations',[]))
                for row in rows:
                    self.assertEqual({s['book_id'] for s in row['supports']},{None,'B'})
                    for support in row['supports']:
                        self.assertIn('audit_ref',support)
                        ref=support['audit_ref']
                        resolved=fused['native_contributions'][ref['contribution_key']]
                        self.assertEqual(ref['artifact_id'],resolved['artifact_id'])
                        self.assertEqual(ref['event_count'],len(resolved['graph']['events']))
            self.assertIn('ledger-only-A-0',(both/'fused.json').read_text('utf-8'))
            self.assertEqual(fused['native_contributions']['g0'],g)
            self.assertEqual(fused['native_contributions']['B'],b)
        self.assertEqual([g,a,b],old)

class FusionTests(unittest.TestCase):
    def test_hotplug_shared_support_and_immutable(self):
        g,a,b=contribution(),contribution('A'),contribution('B')
        old=copy.deepcopy([g,a,b])
        both=build_view(g,[a,b],['B','A'],alignment=mapping('A','B'))
        self.assertEqual(len(both['facts'][0]['supports']),3)
        self.assertEqual(both['facts'][0]['source_state'],'prior_textbook_supported')
        b1=build_view(g,[a,b],['B'],alignment=mapping('A','B'))
        b2=build_view(g,[b],['B'],alignment=mapping('B'))
        self.assertEqual(b1,b2)
        self.assertEqual([g,a,b],old)
        self.assertEqual(len(b1['relations'][0]['evidence']),1)
    def test_unresolved_homonyms_conditions_inverse_inactive(self):
        g,a,b=contribution(),contribution('A','conditional'),contribution('B',status='retired')
        view=build_view(g,[a,b],['A','B'])
        self.assertEqual(len(view['entities']),6)
        self.assertEqual(len(view['relations']),2)
        self.assertEqual(view['facts'][0]['source_state'],'prior_only_unverified')
        view=build_view(g,[a,b],['A','B'],alignment=mapping('A','B'))
        self.assertEqual(len(view['relations']),2)
        self.assertEqual({r['conditions'] for r in view['relations']},{'','conditional'})
        self.assertTrue(view['native_contributions']['B']['graph']['claims']['course/B/claim/r']['status']=='retired')
    def test_digest_course_scope_and_namespace_boundaries(self):
        g,a=contribution(),contribution('A')
        bad=copy.deepcopy(a);bad['graph']['nodes'].clear()
        with self.assertRaises(ValueError):build_view(g,[bad],['A'])
        with self.assertRaises(ValueError):build_view(g,[contribution('A',scope='smoke'),contribution('B')],['A','B'])
        with self.assertRaises(ValueError):build_view(g,[a],['missing'])
        with self.assertRaises(ValueError):build_view(g,[a],['A'],alignment={'course/A/node/x':'missing'})

if __name__=='__main__':unittest.main()

class IntegrationTests(unittest.TestCase):
    def test_recomputed_foreign_evidence_namespace_is_rejected(self):
        from course_experiments.manifest import digest
        g,a=contribution(),contribution('A')
        a['graph']['nodes']['course/A/node/x']['evidence_ids']=['foreign']
        a['artifact_id']=digest({k:v for k,v in a.items() if k!='artifact_id'})
        with self.assertRaises(ValueError):build_view(g,[a],['A'])
    def test_real_merger_runtime_keeps_prior_identity(self):
        from tests.test_incremental_semantic_merge import Vectors
        g,a=contribution(),contribution('A')
        for v in (g,a):
            for node in v['graph']['nodes'].values():
                node['definition']='same meaning '+node['name']
            from course_experiments.manifest import digest
            v['artifact_id']=digest({k:r for k,r in v.items() if k!='artifact_id'})
        view=build_view(g,[a],['A'],runtime=Vectors('merge'))
        self.assertEqual(view['alignment_mapping']['course/A/node/x'],'course/g0/node/x')
        self.assertTrue(view['alignment_audit'])
    def test_inverse_distinct_and_prior_alternative_never_retired(self):
        from course_experiments.manifest import digest
        g,a=contribution(),contribution('A')
        c=a['graph']['claims']['course/A/claim/r']['content']
        c['head'],c['tail']=c['tail'],c['head']
        a['artifact_id']=digest({k:v for k,v in a.items() if k!='artifact_id'})
        view=build_view(g,[a],['A'],alignment=mapping('A'))
        self.assertEqual(len(view['relations']),2)
        self.assertEqual({r['source_state'] for r in view['relations']},{'prior_only_unverified','textbook_only_addition'})
    def test_export_can_be_loaded_by_existing_evaluator(self):
        import tempfile
        from pathlib import Path
        from argparse import Namespace
        from course_experiments.views import export_view
        from scripts.evaluate_kg_quality import load_inputs
        with tempfile.TemporaryDirectory() as folder:
            view=build_view(contribution(),[contribution('A')],['A'],alignment=mapping('A'))
            output=export_view(view,folder)
            inputs=load_inputs(Namespace(kg_output_dir=output,schema=Path('config/schema.yaml')))
            self.assertEqual(inputs.concept_kg['relations'][0]['head_entity_id'],'course/g0/node/x')
            self.assertIn('course/A/evidence/e',inputs.doc_nodes)

class SourceStateTests(unittest.TestCase):
    def test_definition_alternatives_preserve_evidence_without_revision(self):
        from course_experiments.manifest import digest
        g,a=contribution(),contribution('A')
        for value,text in ((g,'prior definition'),(a,'alternative definition')):
            value['graph']['claims'][f"course/{value['book_id'] or 'g0'}/claim/r"]['content']=dict(kind='node_fact',node_id=f"course/{value['book_id'] or 'g0'}/node/x",field='definition',value=text,conditions='')
            value['artifact_id']=digest({k:v for k,v in value.items() if k!='artifact_id'})
        view=build_view(g,[a],['A'],alignment=mapping('A'))
        self.assertEqual(len(view['potential_revisions']),1)
        self.assertFalse(view['potential_revisions'][0]['retires_prior'])
        self.assertEqual(view['potential_revisions'][0]['evidence_ids'],['course/A/evidence/e'])
        self.assertEqual(len(view['facts']),2)
    def test_differing_relations_annotated_only_as_potential_alternatives(self):
        from course_experiments.manifest import digest
        g,a=contribution(),contribution('A')
        a['graph']['claims']['course/A/claim/r']['content']['relation']='part_of'
        a['artifact_id']=digest({k:v for k,v in a.items() if k!='artifact_id'})
        view=build_view(g,[a],['A'],alignment=mapping('A'))
        self.assertEqual(len(view['potential_revisions']),1)
        self.assertFalse(view['potential_revisions'][0]['retires_prior'])
        self.assertEqual(len(view['relations']),2)
    def test_namespaced_forged_map_and_foreign_course_rejected(self):
        from course_experiments.manifest import digest
        g,a=contribution(),contribution('A')
        for mutation in ('map','course'):
            bad=copy.deepcopy(a)
            if mutation=='map':bad['native_id_map']['nodes']['x']='foreign'
            else:bad['course_id']='other'
            bad['artifact_id']=digest({k:v for k,v in bad.items() if k!='artifact_id'})
            with self.assertRaises(ValueError):build_view(g,[bad],['A'])

class LedgerBoundaryTests(unittest.TestCase):
    def test_rehashed_support_ledger_mismatch_rejected(self):
        from course_experiments.manifest import digest
        g,a=contribution(),contribution('A')
        a['supports'][0]['evidence_ids']=[]
        a['artifact_id']=digest({k:v for k,v in a.items() if k!='artifact_id'})
        with self.assertRaises(ValueError):build_view(g,[a],['A'])
    def test_structural_nodes_excluded_from_semantics(self):
        from course_experiments.manifest import digest
        g,a=contribution(),contribution('A')
        a['graph']['nodes']['course/A/node/y']['entity_type']='Book'
        a['artifact_id']=digest({k:v for k,v in a.items() if k!='artifact_id'})
        view=build_view(g,[a],['A'])
        self.assertEqual(len(view['entities']),3)
        self.assertEqual(len(view['relations']),1)
        self.assertIn('course/A/node/y',view['alignment_mapping'])


class SmokeScopeTests(unittest.TestCase):
    def test_fixed_full_prior_with_homogeneous_smoke_books(self):
        g,a,b=contribution(),contribution('A',scope='smoke'),contribution('B',scope='smoke')
        old=copy.deepcopy(g)
        view=build_view(g,[a,b],['A','B'],alignment=mapping('A','B'))
        self.assertEqual(view['scope'],'smoke')
        self.assertEqual(view['native_contributions']['g0']['scope'],'full')
        self.assertEqual(g,old)
        self.assertEqual(len(view['relations'][0]['evidence']),2)
    def test_selected_books_determine_scope_and_empty_is_full(self):
        g,a,b=contribution(),contribution('A',scope='smoke'),contribution('B')
        self.assertEqual(build_view(g,[a,b],['A'])['scope'],'smoke')
        self.assertEqual(build_view(g,[a,b],[])['scope'],'full')
        with self.assertRaises(ValueError):build_view(g,[a,b],['A','B'])

class ReviewRegressionTests(unittest.TestCase):
    def test_evaluator_consumes_merged_definitions_provenance_and_text_context(self):
        import json
        import tempfile
        from pathlib import Path
        from argparse import Namespace
        from course_experiments.manifest import digest
        from course_experiments.views import export_view
        from scripts.evaluate_kg_quality import load_inputs, evaluate_entities, build_entity_context, build_relation_context
        g,a=contribution(),contribution('A')
        for value in (g,a):
            for node in value['graph']['nodes'].values():
                node['definition']=('prior ' if value['book_id'] is None else 'book ')+node['name']
                node['definitions']=[dict(value='conditional detail '+node['name'],conditions='condition preserved',evidence_ids=node['evidence_ids'])]
            value['artifact_id']=digest({k:v for k,v in value.items() if k!='artifact_id'})
        original=copy.deepcopy(a)
        view=build_view(g,[a],['A'],alignment=mapping('A'))
        with tempfile.TemporaryDirectory() as folder:
            output=export_view(view,folder)
            inputs=load_inputs(Namespace(kg_output_dir=output,schema=Path('config/schema.yaml')))
            metrics,_=evaluate_entities(inputs)
            self.assertEqual(metrics['missing_definition_count'],0)
            self.assertEqual(metrics['synthetic_or_untraced_entity_count'],0)
            self.assertEqual(metrics['missing_source_reference_count'],0)
            self.assertEqual(metrics['source_reference_count'],2)
            entity=next(e for e in inputs.concept_kg['entities'] if e['entity_id']=='course/g0/node/x')
            self.assertEqual(set(entity['merged_definitions']),{'prior x','book x','conditional detail x'})
            self.assertEqual(set(entity['merged_raw_entity_ids']),{'course/g0/node/x','course/A/node/x'})
            self.assertIn('Independent source',build_entity_context(inputs,entity,1000))
            self.assertIn('Independent source',build_relation_context(inputs,inputs.concept_kg['relations'][0],1000))
            self.assertEqual(inputs.doc_nodes['course/A/evidence/e']['text'],'Independent source')
            self.assertEqual(json.loads((output/'fused.json').read_text('utf-8'))['native_contributions']['A'],original)
    def test_runtime_generic_specific_type_keeps_prior_identity_and_specific_type(self):
        from course_experiments.manifest import digest
        from tests.test_incremental_semantic_merge import Vectors
        g,a=contribution(),contribution('A')
        a['graph']['nodes']['course/A/node/x']['entity_type']='Algorithm'
        for value in (g,a):
            for node in value['graph']['nodes'].values():
                node['name']='algorithm1' if node['native_id']=='x' else 'algorithm2'
                node['definition']='matching definition '+node['name']
            value['artifact_id']=digest({k:v for k,v in value.items() if k!='artifact_id'})
        view=build_view(g,[a],['A'],runtime=Vectors('merge'))
        self.assertEqual(view['alignment_mapping']['course/A/node/x'],'course/g0/node/x')
        entity=next(e for e in view['entities'] if e['entity_id']=='course/g0/node/x')
        self.assertEqual(entity['type'],'Algorithm')
        self.assertEqual({r['type'] for r in entity['type_provenance']},{'Concept','Algorithm'})
        self.assertEqual(view['native_contributions']['g0']['graph']['nodes']['course/g0/node/x']['entity_type'],'Concept')
    def test_generic_bridge_cannot_flatten_incompatible_specific_types(self):
        from course_experiments.manifest import digest
        g,a,b=contribution(),contribution('A'),contribution('B')
        for value,kind in ((a,'Algorithm'),(b,'DataStructure')):
            value['graph']['nodes'][f"course/{value['book_id']}/node/x"]['entity_type']=kind
            value['artifact_id']=digest({k:v for k,v in value.items() if k!='artifact_id'})
        with self.assertRaises(ValueError):build_view(g,[a,b],['A','B'],alignment=mapping('A','B'))
