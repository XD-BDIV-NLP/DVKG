import copy
import unittest

from course_experiments.manifest import digest
from course_experiments.splits import canonical, freeze_split


class IdentityKGETests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_kge
        return identity_kge

    def reference(self):
        entities=[dict(entity_id='r'+str(i),canonical_name='reference '+str(i),scope='course',aliases=[]) for i in range(14)]
        facts=[dict(fact_id='f'+str(i),head='r'+str(i),relation='part_of',tail='r'+str(i+1)) for i in range(13)]
        entities.append(dict(entity_id='zalias',canonical_name='R alias',scope='course',aliases=['reference 0']))
        facts += [dict(fact_id='inverse',head='r1',relation='consists_of',tail='zalias',conditions='different condition')]
        return dict(course_id='course',entities=entities,facts=facts)

    def graph(self):
        return dict(entities=[dict(entity_id='g'+str(i),canonical_name='native '+str(i),scope='different free text',entity_type='Concept') for i in range(14)],
            relations=[dict(relation_id='e'+str(i),head_entity_id='g'+str(i),relation='part_of',tail_entity_id='g'+str(i+1)) for i in range(13)])

    def mappings(self, graph_sha='graph-sha'):
        return [dict(candidate_id='c'+str(i),course_id='course',method='G0',book_set=[],graph_record=dict(path='/original/graph.json',sha256=graph_sha),
            graph_entity_id='g'+str(i),reference_entity_id='r'+str(i),identity_basis='prior_generated_definition') for i in range(14)]

    def test_new_reference_keeps_all_facts_and_alias_inverse_conditions_together(self):
        api=self.api();reference=self.reference();before=copy.deepcopy(reference)
        frozen,split=api.freeze_reference(reference,'qualified-mapping')
        self.assertEqual(reference,before)
        self.assertEqual(len(frozen['facts']),len(reference['facts']))
        group=next(g for g in split['groups'].values() if 'f0' in g['fact_ids'])
        self.assertEqual(set(group['fact_ids']),{'f0','inverse'})
        self.assertNotEqual(split['split_id'],freeze_split(reference)['split_id'])
        self.assertEqual(split['folds'],freeze_split(reference)['folds'])

    def test_unresolved_cross_scope_reference_name_candidates_block_new_split(self):
        api=self.api();reference=self.reference()
        reference['entities'].append(dict(entity_id='ambiguous',canonical_name='reference 0',scope='another sense scope',aliases=[]))
        with self.assertRaisesRegex(ValueError,'cross.scope reference'):
            api.freeze_reference(reference,'qualified-graph-mapping-only')

    def test_projection_reuses_all_edges_removes_all_equivalent_heldout(self):
        api=self.api();reference,split=api.freeze_reference(self.reference(),'qualified')
        graph=self.graph();before=copy.deepcopy(graph)
        graph['relations'].append(dict(relation_id='reverse',head_entity_id='g1',relation='consists_of',tail_entity_id='g0'))
        before=copy.deepcopy(graph)
        projection=api.project_identity_graph(graph,reference,split,self.mappings(),graph_sha='graph-sha')
        self.assertEqual(graph,before)
        self.assertEqual(projection['graph_id'],digest(graph))
        held={canonical(*split['groups'][g]['triple']) for k in ('validation','test') for g in split['folds'][k]}
        self.assertFalse(held & {canonical(*t) for t in projection['training']})
        self.assertEqual(len(projection['test']),len(split['folds']['test']))
        self.assertEqual(len(projection['training']),len(split['folds']['train']))
        self.assertEqual(projection['identity_disclosure']['prior_identity_is_textbook_support'],False)
        self.assertTrue(all(row['original_relation_id'] in {r['relation_id'] for r in graph['relations']} for row in projection['audit']['identity_edge_sources']))

    def test_conflicting_native_entity_targets_are_rejected_not_merged(self):
        api=self.api();reference,split=api.freeze_reference(self.reference(),'qualified')
        rows=self.mappings()+[dict(self.mappings()[0],candidate_id='conflict',reference_entity_id='r2')]
        with self.assertRaisesRegex(ValueError,'conflict'):
            api.project_identity_graph(self.graph(),reference,split,rows,graph_sha='graph-sha')

    def test_changed_corrected_graph_sha_cannot_borrow_original_maps(self):
        api=self.api();reference,split=api.freeze_reference(self.reference(),'qualified')
        with self.assertRaisesRegex(ValueError,'SHA'):
            api.project_identity_graph(self.graph(),reference,split,self.mappings(),graph_sha='corrected')

    def test_unknown_reference_and_native_endpoints_rejected(self):
        api=self.api();reference,split=api.freeze_reference(self.reference(),'qualified')
        for field,value in [('reference_entity_id','absent'),('graph_entity_id','absent')]:
            rows=self.mappings();rows[0][field]=value
            with self.assertRaises(ValueError):api.project_identity_graph(self.graph(),reference,split,rows,graph_sha='graph-sha')

    def test_endpoint_relation_coverage_comes_from_training_not_graph_presence(self):
        api=self.api();reference,split=api.freeze_reference(self.reference(),'qualified')
        projection=api.project_identity_graph(self.graph(),reference,split,self.mappings(),graph_sha='graph-sha')
        trained={x for h,_,t in projection['training'] for x in (h,t)};relations={r for _,r,_ in projection['training']}
        for key in ('test','validation'):
            self.assertEqual(projection['covered_'+key+'_mask'],[h in trained and t in trained and r in relations for h,r,t in projection[key]])
        audit=projection['coverage_audit']
        self.assertEqual(audit['fixed_test_count'],len(projection['test']))
        self.assertEqual(audit['eligible_test_count'],sum(projection['covered_test_mask']))

    def test_untyped_graph_never_gains_types_or_relations(self):
        api=self.api();reference,split=api.freeze_reference(self.reference(),'qualified');graph=self.graph()
        graph['entities'][0]['type_status']='native_untyped_candidate'
        with self.assertRaisesRegex(ValueError,'untyped'):
            api.project_identity_graph(graph,reference,split,self.mappings(),graph_sha='graph-sha')

    def stages(self):
        from course_experiments.evaluation.run_scope_identity_gate import identity_projection
        source_origins=[self.mappings()[0]];prior_origins=[self.mappings()[1]]
        def stage(name,origins):
            decisions={r['candidate_id']:dict(status='completed',citation_valid=True,eligible_identity_edge=True,reversed_positive_confirmed=True) for r in origins}
            return dict(stage=name,status='completed',preparation_id='prep',candidate_count=len(decisions),failed_pairs=0,decisions=decisions,identity_projection=identity_projection(origins,decisions))
        return {name:stage(name,source_origins if name.startswith('source') else prior_origins) for name in ['source32','source-full','prior24-single','prior24-batch','prior-full']},source_origins,prior_origins

    def test_usable_false_only_qualifies_after_complete_stages_and_both_gates(self):
        api=self.api();stages,source,prior=self.stages()
        for name in ['source32','prior24-batch']:stages[name]['comparison']=dict(passes_declared_gate=True)
        result=api.qualify_mappings(stages,source,prior,'prep')
        self.assertTrue(result['usable_for_KGE'])
        self.assertEqual(len(result['mapping']),2)
        for name in stages:
            broken=copy.deepcopy(stages);broken[name]['status']='partial'
            with self.assertRaises(ValueError):api.qualify_mappings(broken,source,prior,'prep')

    def test_failed_or_unreversed_positive_and_gate_failure_refused(self):
        api=self.api();stages,source,prior=self.stages()
        for name in ['source32','prior24-batch']:stages[name]['comparison']=dict(passes_declared_gate=True)
        for field,value in [('status','failed'),('citation_valid',False),('reversed_positive_confirmed',False)]:
            broken=copy.deepcopy(stages);broken['prior-full']['decisions']['c1'][field]=value
            with self.assertRaises(ValueError):api.qualify_mappings(broken,source,prior,'prep')
        stages['source32']['comparison']['passes_declared_gate']=False
        with self.assertRaises(ValueError):api.qualify_mappings(stages,source,prior,'prep')

    def test_qualification_does_not_infer_reference_equivalence_from_conflict(self):
        api=self.api();stages,source,prior=self.stages()
        for name in ['source32','prior24-batch']:stages[name]['comparison']=dict(passes_declared_gate=True)
        prior[0].update(graph_entity_id='g0')
        from course_experiments.evaluation.run_scope_identity_gate import identity_projection
        stages['prior-full']['identity_projection']=identity_projection(prior,stages['prior-full']['decisions'])
        result=api.qualify_mappings(stages,source,prior,'prep')
        self.assertEqual(result['mapping'],[])
        self.assertEqual(len(result['rejections']),2)
        self.assertFalse(result['reference_components_inferred_from_graphs'])

    def test_independent_reference_component_normalizes_raw_positive_targets_without_old_decision_changes(self):
        api=self.api();stages,source,prior=self.stages()
        for name in ['source32','prior24-batch']:stages[name]['comparison']=dict(passes_declared_gate=True)
        prior[0].update(graph_entity_id='g0')
        for name,cid in [('source-full','c0'),('prior-full','c1')]:
            stages[name]['decisions'][cid].update(primary_record=dict(eligible_identity_edge=True),reverse=dict(eligible_identity_edge=True))
        from course_experiments.evaluation.run_scope_identity_gate import identity_projection
        stages['prior-full']['identity_projection']=identity_projection(prior,stages['prior-full']['decisions'])
        before=copy.deepcopy(stages)
        independent=dict(status='completed_reference_identity_qualification',qualification_id='ref-qualified',
            components={'course':[['r0','r1']]})
        result=api.qualify_mappings(stages,source,prior,'prep',reference_qualification=independent)
        self.assertEqual(stages,before)
        self.assertEqual({r['reference_entity_id'] for r in result['mapping']},{'r0'})
        self.assertEqual(len(result['mapping']),2)
        stages['prior-full']['decisions']['c1'].update(eligible_identity_edge=False,veto='preserved_frozen_single_pair_negative')
        stages['prior-full']['identity_projection']=identity_projection(prior,stages['prior-full']['decisions'])
        result=api.qualify_mappings(stages,source,prior,'prep',reference_qualification=independent)
        self.assertEqual(len(result['mapping']),1)

    def test_reference_components_rewrite_endpoints_before_freezing_without_dropping_facts(self):
        api=self.api();reference=self.reference()
        reference['entities'].append(dict(entity_id='other_scope',canonical_name='reference 0',scope='other scope',aliases=[]))
        reference['facts'].append(dict(fact_id='equiv_new',head='other_scope',relation='part_of',tail='r1'))
        independent=dict(status='completed_reference_identity_qualification',qualification_id='ref-qualified',
            components={'course':[['r0','other_scope']]},original_reference_digests={'course':digest(reference)})
        frozen,split=api.freeze_reference(reference,'qualified',reference_qualification=independent)
        self.assertEqual(len(frozen['facts']),len(reference['facts']))
        group=next(g for g in split['groups'].values() if 'f0' in g['fact_ids'])
        self.assertEqual(set(group['fact_ids']),{'f0','inverse','equiv_new'})

    def test_uncertain_or_different_sense_name_risks_share_fold_without_identity_merge(self):
        api=self.api();reference=self.reference()
        reference['entities'].append(dict(entity_id='uncertain',canonical_name='reference 0',scope='unknown different scope',aliases=[]))
        reference['facts'].append(dict(fact_id='uncertain_fact',head='uncertain',relation='part_of',tail='r1'))
        independent=dict(status='completed_reference_identity_qualification',qualification_id='ref-qualified',
            components={},original_reference_digests={'course':digest(reference)})
        frozen,split=api.freeze_reference(reference,'qualified',reference_qualification=independent)
        self.assertIn('uncertain',{r['entity_id'] for r in frozen['entities']})
        groups={fid:key for key,row in split['groups'].items() for fid in row['fact_ids']}
        self.assertNotEqual(groups['f0'],groups['uncertain_fact'])
        folds={gid:key for key,ids in split['folds'].items() for gid in ids}
        self.assertEqual(folds[groups['f0']],folds[groups['uncertain_fact']])
        self.assertEqual(sum(len(ids) for ids in split['folds'].values()),len(split['groups']))
        self.assertEqual(split['identity_holdout_block_policy']['different_sense_handling'],'conservatively_coheld_same_as_insufficient')
        self.assertEqual(len(frozen['facts']),len(reference['facts']))


if __name__=='__main__':unittest.main()
