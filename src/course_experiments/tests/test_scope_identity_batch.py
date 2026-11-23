from __future__ import annotations

import copy
import json
import unittest

from course_experiments.evaluation import scope_identity_batch as batch
from course_experiments.evaluation import scope_identity as source
from course_experiments.evaluation import prior_definition_identity as prior
from course_experiments.tests.test_scope_identity import fixture


def packet(number=0,course='os',is_prior=False):
    left,right,evidence=fixture()
    left['entity_id']='reference_'+str(number);right['entity_id']='graph_'+str(number)
    evidence={str(number):dict(text='Alpha is a scheduling method. '+str(number))}
    left['citations']=[dict(evidence_id=str(number),quote=evidence[str(number)]['text'])]
    right['source_node_ids']=[] if is_prior else [str(number)]
    value=source.prepare_pair(course,left,right,evidence,evidence)
    return prior.prepare_pair(value) if is_prior else value


def origins(packets):
    return [dict(candidate_id=p['candidate_id'],graph_record=dict(sha256='a'*64),
        graph_entity_id=p['right']['entity_id'],reference_entity_id=p['left']['entity_id']) for p in packets]


class Provider:
    def __init__(self,mutate=None):self.calls=[];self.mutate=mutate
    def raise_if_fatal(self):pass
    def chat_response(self,messages,**kwargs):
        wire=json.loads(messages[1]['content']);self.calls.append((wire,kwargs))
        results=[]
        for pair in wire['pairs']:
            value=dict(candidate_id=pair['candidate_id'],decision='same_sense',reason='synthetic',
                left_citations=[{k:v for k,v in pair['left']['evidence_quotes'][0].items() if k!='provenance'}],
                right_citations=[{k:v for k,v in pair['right']['evidence_quotes'][0].items() if k!='provenance'}])
            flags=['scope_consistent','source_definition_sufficient','homonym_excluded'] if wire['kind']=='two_sided_source' else [
                'necessary_scope_consistent','definition_identity_sufficient','homonym_excluded']
            value.update({key:True for key in flags});results.append(value)
        result=dict(batch_id=wire['batch_id'],results=results)
        if self.mutate:self.mutate(wire,result,len(self.calls))
        return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(result)))])


class ScopeIdentityBatchTests(unittest.TestCase):
    def test_fixed_batches_are_same_course_same_provenance_and_at_most_eight(self):
        work=[packet(i,'os' if i<10 else 'net') for i in range(17)]+[packet(50,is_prior=True)]
        planned=batch.plan_batches(work)
        self.assertEqual(sum(map(len,planned)),18)
        self.assertTrue(all(1<=len(rows)<=8 for rows in planned))
        for rows in planned:
            self.assertEqual(len({p['course_id'] for p in rows}),1)
            self.assertEqual(len({p.get('kind','two_sided_source') for p in rows}),1)
        self.assertEqual(batch.plan_batches(list(reversed(work))),planned)
        with self.assertRaises(ValueError):batch.plan_batches([work[0],work[0]])

    def test_development_selection_is_fixed_32_without_decisions_or_scores(self):
        work=[packet(i) for i in range(119)]
        first=batch.development_plan(work)
        second=batch.development_plan(list(reversed(work)))
        self.assertEqual(first,second)
        self.assertEqual(len(first['candidate_ids']),32)
        self.assertEqual(first['thresholds']['minimum_exact_decision_agreement'],31)
        self.assertEqual(first['thresholds']['maximum_single_negative_to_batch_positive'],0)
        self.assertEqual(first['maximum_primary_and_reverse_logical_requests'],8)
        self.assertFalse(first['enabled_for_full_mapping'])

    def test_anonymous_pairs_cannot_borrow_citations_from_another_pair(self):
        work=[packet(1),packet(2)]
        def corrupt(wire,result,count):
            result['results'][0]['right_citations']=result['results'][1]['right_citations']
        provider=Provider(corrupt);result=batch.judge_batch(provider,work)
        self.assertEqual(sum(row['status']=='validated' for row in result['decisions'].values()),1)
        self.assertEqual(len(result['decisions']),2)
        wire=json.dumps(provider.calls[0][0])
        for forbidden in ['reference_1','graph_1','"method":','"rank":','"fold":','graph_record']:
            self.assertNotIn(forbidden,wire)
        self.assertEqual(provider.calls[0][1]['max_tokens'],8192)

    def test_every_positive_is_reversed_and_conflicting_targets_all_rejected(self):
        work=[packet(1),packet(2)]
        work[1]['right']['entity_id']=work[0]['right']['entity_id']
        work[1]['candidate_id']=source.digest({k:v for k,v in work[1].items() if k!='candidate_id'})
        original=copy.deepcopy(work);rows=origins(work)
        provider=Provider();result=batch.run_batches(work,rows,provider)
        self.assertEqual(len(provider.calls),2)
        self.assertTrue(all(r['reverse']['status']=='validated' for r in result['decisions'].values()))
        self.assertEqual(result['identity_projection']['mapping'],[])
        self.assertEqual(len(result['identity_projection']['rejections']),2)
        self.assertEqual(work,original)
        self.assertFalse(result['usable_for_KGE'])

    def test_reverse_disagreement_and_incomplete_batch_keep_fixed_denominator(self):
        work=[packet(1),packet(2)]
        def reverse_reject(wire,result,count):
            if count==2:result['results'][0]['scope_consistent']=False
        result=batch.run_batches(work,origins(work),Provider(reverse_reject))
        self.assertEqual(result['candidate_count'],2)
        self.assertEqual(sum(r['eligible_identity_edge'] for r in result['decisions'].values()),1)
        def missing(wire,result,count):result['results'].pop()
        failed=batch.run_batches(work,origins(work),Provider(missing))
        self.assertEqual(len(failed['decisions']),2)
        self.assertFalse(any(r['eligible_identity_edge'] for r in failed['decisions'].values()))
        self.assertEqual(failed['identity_projection']['mapping'],[])

    def test_prior_definition_quotes_stay_prior_after_reverse_and_scope_failure_rejects(self):
        work=[packet(1,is_prior=True)];provider=Provider()
        result=batch.run_batches(work,origins(work),provider)
        self.assertTrue(next(iter(result['decisions'].values()))['eligible_identity_edge'])
        first,reverse=[call[0]['pairs'][0] for call in provider.calls]
        self.assertEqual(first['right']['evidence_quotes'][0]['provenance'],'prior_generated_definition')
        self.assertEqual(reverse['left']['evidence_quotes'][0]['provenance'],'prior_generated_definition')
        self.assertFalse(result['textbook_relation_evidence_supported'])
        def scope_failure(wire,result,count):result['results'][0]['necessary_scope_consistent']=False
        failed=batch.run_batches(work,origins(work),Provider(scope_failure))
        self.assertFalse(next(iter(failed['decisions'].values()))['eligible_identity_edge'])
        with self.assertRaises(ValueError):batch.judge_batch(Provider(),work+[packet(2)])

    def test_thresholds_refuse_negative_expansion_and_missing_single_results(self):
        work=[packet(i) for i in range(119)];plan=batch.development_plan(work)
        decisions={p['candidate_id']:dict(status='completed',eligible_identity_edge=False,
            primary=dict(decision='insufficient')) for p in work}
        batched={cid:copy.deepcopy(decisions[cid]) for cid in plan['candidate_ids']}
        for row in batched.values():row['citation_valid']=True
        self.assertTrue(batch.compare_development(plan,decisions,batched)['passes_declared_gate'])
        batched[plan['candidate_ids'][0]]['eligible_identity_edge']=True
        self.assertFalse(batch.compare_development(plan,decisions,batched)['passes_declared_gate'])

    def test_incomplete_single_population_blocks_gate_even_if_selected32_are_complete(self):
        work=[packet(i) for i in range(119)];plan=batch.development_plan(work)
        decisions={cid:dict(status='completed',eligible_identity_edge=False,
            primary=dict(decision='insufficient'),citation_valid=True) for cid in plan['candidate_ids']}
        self.assertFalse(batch.compare_development(plan,decisions,decisions)['passes_declared_gate'])
        provider=Provider();selected=[p for p in work if p['candidate_id'] in plan['candidate_ids']]
        with self.assertRaises(ValueError):
            batch.run_development(plan,selected,origins(selected),provider,decisions)
        self.assertEqual(provider.calls,[])
        batched=copy.deepcopy(decisions);decisions.pop(plan['candidate_ids'][0])
        self.assertFalse(batch.compare_development(plan,decisions,batched)['passes_declared_gate'])

    def test_negative_batch_decision_still_requires_two_sided_exact_quotes(self):
        def missing_quotes(wire,result,count):
            row=result['results'][0];row['decision']='insufficient';row['right_citations']=[]
        result=batch.judge_batch(Provider(missing_quotes),[packet(1)])
        self.assertEqual(next(iter(result['decisions'].values()))['status'],'failed')


if __name__=='__main__':unittest.main()
