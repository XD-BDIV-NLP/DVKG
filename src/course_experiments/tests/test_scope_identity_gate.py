from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from course_experiments.evaluation import scope_identity as scope
from course_experiments.evaluation import run_scope_identity_gate as gate
from course_experiments.tests.test_scope_identity import fixture
from course_experiments.splits import freeze_split,project_graph


class JudgeProvider:
    def __init__(self,reverse_reject=False):
        self.reverse_reject=reverse_reject;self.calls=[];self.closed=False
    def chat_response(self,messages,**kwargs):
        packet=json.loads(messages[1]['content']);self.calls.append(packet)
        bad=self.reverse_reject and len(self.calls)%2==0
        value=dict(candidate_id=packet['candidate_id'],decision='different_sense' if bad else 'same_sense',
            scope_consistent=not bad,source_definition_sufficient=True,homonym_excluded=True,
            left_citations=packet['left']['source_quotes'][:1],right_citations=packet['right']['source_quotes'][:1],reason='synthetic')
        return dict(choices=[dict(message=dict(content=json.dumps(value)),finish_reason='stop')])
    def raise_if_fatal(self):pass
    def close(self):self.closed=True


def packets():
    left,right,evidence=fixture()
    return [scope.prepare_pair('os',left,right,evidence,evidence)]


def origins(packet):
    return [dict(candidate_id=packet['candidate_id'],cohort='osAB',member='method_secret',
        graph_record=dict(path='/server/original_graph.json',sha256='a'*64),
        graph_entity_id='graph_alpha',reference_entity_id='reference_alpha')]


class ScopeIdentityGateTests(unittest.TestCase):
    def test_positive_requires_reverse_check_anonymous_payload_and_adds_no_facts(self):
        work=packets();original=copy.deepcopy(work);provider=JudgeProvider()
        with tempfile.TemporaryDirectory() as folder:
            result=gate.run_gate(work,origins(work[0]),provider,Path(folder))
            self.assertEqual(result['status'],'completed_development_gate')
            self.assertEqual(result['accepted_candidates'],1)
            self.assertEqual(len(result['identity_projection']['mapping']),1)
            self.assertEqual(len(provider.calls),2)
            wire=json.dumps(provider.calls)
            for forbidden in ['graph_alpha','reference_alpha','method_secret','graph_record','fold','rank']:
                self.assertNotIn(forbidden,wire)
            self.assertNotIn('relations',result['identity_projection'])
            self.assertNotIn('training',result['identity_projection'])
            other=JudgeProvider()
            gate.run_gate(work,origins(work[0]),other,Path(folder))
            self.assertEqual(other.calls,[])
        self.assertEqual(work,original)
        self.assertTrue(provider.closed)

    def test_reversal_disagreement_rejects_identity_and_preserves_fixed_denominator(self):
        work=packets()
        with tempfile.TemporaryDirectory() as folder:
            result=gate.run_gate(work,origins(work[0]),JudgeProvider(reverse_reject=True),Path(folder))
        self.assertEqual(result['candidate_count'],1)
        self.assertEqual(result['accepted_candidates'],0)
        self.assertEqual(result['rejected_candidates'],1)
        self.assertEqual(result['identity_projection']['mapping'],[])

    def test_conflicting_reference_targets_for_one_graph_identity_are_all_rejected(self):
        rows=origins(packets()[0]);rows.append(dict(rows[0],candidate_id='other',reference_entity_id='homonym'))
        decisions={rows[0]['candidate_id']:dict(eligible_identity_edge=True),
            'other':dict(eligible_identity_edge=True)}
        projected=gate.identity_projection(rows,decisions)
        self.assertEqual(projected['mapping'],[])
        self.assertEqual(len(projected['rejections']),2)

    def test_missing_source_or_necessary_scope_uncertainty_cannot_be_accepted(self):
        work=packets();work[0]['right']['source_quotes']=[];work[0]['both_sides_have_source']=False
        work[0]['candidate_id']=scope.digest({k:v for k,v in work[0].items() if k!='candidate_id'})
        with tempfile.TemporaryDirectory() as folder:
            result=gate.run_gate(work,origins(work[0]),JudgeProvider(),Path(folder))
        self.assertEqual(result['accepted_candidates'],0)
        self.assertEqual(result['identity_projection']['mapping'],[])

    def test_reference_equivalence_precedes_split_and_holds_inverse_conditional_variants_together(self):
        reference=dict(course_id='os',entities=[dict(entity_id=x,canonical_name=x,aliases=[],scope='s'+x,citations=[])
            for x in ['whole','whole_alias','part']],facts=[
                dict(fact_id='f1',head='whole',relation='consists_of',tail='part',conditions=''),
                dict(fact_id='f2',head='part',relation='part_of',tail='whole_alias',conditions='when ready')],questions=[])
        untouched=copy.deepcopy(reference)
        merged,split=gate.freeze_identity_split(reference,[['whole','whole_alias']])
        self.assertEqual(len(split['groups']),1)
        self.assertEqual(set(next(iter(split['groups'].values()))['fact_ids']),{'f1','f2'})
        self.assertEqual(reference,untouched)
        self.assertNotEqual(split['split_id'],freeze_split(reference)['split_id'])
        self.assertEqual(len(merged['entities']),2)
        graph=dict(entities=[dict(entity_id=x,canonical_name=x,aliases=[],scope='')
            for x in ['whole','whole_alias','part']],relations=[
                dict(relation_id='r1',head_entity_id='whole',relation='consists_of',tail_entity_id='part'),
                dict(relation_id='r2',head_entity_id='part',relation='part_of',tail_entity_id='whole_alias')])
        projection=project_graph(graph,merged,split)
        self.assertEqual(projection['training'],[])
        self.assertEqual(len(projection['audit']['heldout_removed']),2)

    def test_necessary_scope_disagreement_rejects_even_same_sense_label(self):
        packet=packets()[0]
        value=dict(candidate_id=packet['candidate_id'],decision='same_sense',scope_consistent=False,
            source_definition_sufficient=True,homonym_excluded=True,reason='necessary scope differs',
            left_citations=packet['left']['source_quotes'],right_citations=packet['right']['source_quotes'])
        self.assertFalse(scope.validate_judgment(packet,value)['eligible_identity_edge'])

    def test_overlapping_identity_components_and_wrong_origin_endpoints_are_rejected(self):
        reference=dict(course_id='os',entities=[dict(entity_id=x,canonical_name=x,scope='s') for x in 'abc'],facts=[],questions=[])
        with self.assertRaises(ValueError):gate.freeze_identity_split(reference,[['a','b'],['b','c']])
        packet=packets()[0];origin=origins(packet);origin[0]['reference_entity_id']='different'
        with self.assertRaises(ValueError):gate.contract_for([packet],origin)

    def test_local_http_cap_counts_transport_attempts_without_resetting_global_provider(self):
        class Runtime:
            def _record(self,*args):pass
        class Provider:
            runtime=Runtime()
        provider=Provider();called=[];provider.runtime._record=lambda *args:called.append(args)
        with tempfile.TemporaryDirectory() as folder:
            with gate.local_http_cap(provider,Path(folder),1):
                provider.runtime._record('one','judge','request_started',{})
                with self.assertRaises(gate.LocalHTTPAllowanceExceeded):
                    provider.runtime._record('two','judge','request_started',{})
            with gate.local_http_cap(provider,Path(folder),1):
                with self.assertRaises(gate.LocalHTTPAllowanceExceeded):
                    provider.runtime._record('three','judge','request_started',{})
        self.assertEqual(len(called),1)

    def test_actual_formal_provider_sees_only_allowed_attempt_and_preserves_halt_checkpoint(self):
        from course_experiments.evaluation.native_provider import AuditedNativeProvider
        from course_experiments.tests.test_native_provider import FakeRuntime
        class Runtime(FakeRuntime):
            def _post(self,endpoint,body,options,key,stage):
                self._record(key,stage,'request_started',{})
                packet=json.loads(body['messages'][1]['content'])
                judgment=dict(candidate_id=packet['candidate_id'],decision='same_sense',
                    scope_consistent=True,source_definition_sufficient=True,homonym_excluded=True,
                    left_citations=packet['left']['source_quotes'][:1],right_citations=packet['right']['source_quotes'][:1],reason='synthetic')
                raw=dict(choices=[dict(message=dict(content=json.dumps(judgment)),finish_reason='stop')],usage={})
                self._record(key,stage,'response',raw)
                return raw
        work=packets()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);provider=AuditedNativeProvider(Runtime(),root/'formal',
                max_http_attempts=20,max_seconds=3600,max_concurrency=2)
            with self.assertRaises(gate.LocalHTTPAllowanceExceeded):
                gate.run_gate(work,origins(work[0]),provider,root/'gate',max_http_attempts=1)
            self.assertEqual(provider.statistics()['http_attempts'],1)
            self.assertEqual(provider.statistics()['limits']['max_http_attempts'],20)
            state=json.loads((root/'gate/state.json').read_text())
            self.assertEqual(state['status'],'halted_preserved_existing_budget')
            self.assertEqual(state['candidate_count'],1)


if __name__=='__main__':unittest.main()
