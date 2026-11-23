import copy
import json
from pathlib import Path
import tempfile
import unittest

from course_experiments.tests.test_scope_identity_batch import packet
from course_experiments.evaluation import identity_short_single as single


def raw_answer(body):
    wire=json.loads(body['messages'][-1]['content'])
    value=dict(candidate_id='P0',decision='same_sense',reason='synthetic',scope_consistent=True,
        source_definition_sufficient=True,homonym_excluded=True,
        left_citations=[copy.deepcopy(wire['left']['source_quotes'][0])],
        right_citations=[copy.deepcopy(wire['right']['source_quotes'][0])])
    return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))])


class ShortSourceGateTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import run_short_source_gate
        return run_short_source_gate

    def test_failure_target_selection_is_fixed_hash_and_only_failed(self):
        api=self.api();rows={'a':dict(status='completed'),'b':dict(status='failed'),'c':dict(status='failed')}
        self.assertEqual(api.choose_target(rows),api.choose_target(dict(reversed(list(rows.items())))))
        self.assertIn(api.choose_target(rows),{'b','c'})
        with self.assertRaises(ValueError):api.choose_target({'a':dict(status='completed')})

    def test_valid_short_response_restores_strict_ids_and_preserves_semantic_lock(self):
        api=self.api();p=packet(1);locks=dict(decision='same_sense',scope_consistent=True,
            source_definition_sufficient=True,homonym_excluded=True)
        body,table=api.direction_body(p,False,locks);before=copy.deepcopy(body)
        row=api.validate_response(p,raw_answer(body),table,False,locks)
        self.assertTrue(row['eligible_identity_edge']);self.assertEqual(body,before)
        self.assertEqual(row['candidate_id'],p['candidate_id'])

    def test_changed_semantic_guard_or_decision_still_rejected(self):
        api=self.api();p=packet(1);locks=dict(decision='same_sense',scope_consistent=True)
        body,table=api.direction_body(p,False,locks)
        for key,value in [('decision','insufficient'),('scope_consistent',False)]:
            raw=raw_answer(body);parsed=json.loads(raw['choices'][0]['message']['content']);parsed[key]=value
            raw['choices'][0]['message']['content']=json.dumps(parsed)
            with self.assertRaises(ValueError):api.validate_response(p,raw,table,False,locks)

    def test_wrong_side_quote_id_and_fabricated_quote_not_repaired(self):
        api=self.api();p=packet(1);body,table=api.direction_body(p,False,{})
        for field,value in [('evidence_id','L0'),('quote','fabricated')]:
            raw=raw_answer(body);parsed=json.loads(raw['choices'][0]['message']['content']);parsed['right_citations'][0][field]=value
            raw['choices'][0]['message']['content']=json.dumps(parsed)
            with self.assertRaises(ValueError):api.validate_response(p,raw,table,False,{})

    def test_completion_length_is_failed_without_parsing_as_success(self):
        api=self.api();p=packet(1);body,table=api.direction_body(p,False,{})
        raw=raw_answer(body);raw['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):api.validate_response(p,raw,table,False,{})

    def test_each_direction_is_at_most_one_logical_request_and_exact_cache_resume(self):
        api=self.api();p=packet(1);body,_=api.direction_body(p,False,{})
        class Provider:
            def __init__(self,root):self.root=root;self.runtime=type('Runtime',(),dict(config={'frozen':True}))();self.calls=0
            def request_executor(self,endpoint,value,stage):
                self.calls+=1;raw=raw_answer(value)
                from course_experiments.manifest import digest
                key=digest(dict(endpoint=endpoint,body=value,config=self.runtime.config));path=self.root/'receipts'/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(dict(request_id=key,endpoint=endpoint,request=value,response=raw,runtime_config=self.runtime.config)),encoding='utf-8')
                return raw,False
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);provider=Provider(root/'formal')
            from unittest.mock import patch
            from contextlib import nullcontext
            with patch.object(api.gate,'local_http_cap',side_effect=lambda *a,**k:nullcontext()):
                first=api.execute_direction(provider,body,root/'direction','job',root/'source300')
                again=api.execute_direction(provider,body,root/'direction','job',root/'source300')
            self.assertEqual(first,again);self.assertEqual(provider.calls,1)

    def test_reserved_logical_without_response_never_issues_second_request(self):
        api=self.api();p=packet(1);body,_=api.direction_body(p,False,{})
        from course_experiments.evaluation.run_native_full import _checkpoint
        from course_experiments.manifest import digest
        class Provider:
            runtime=type('Runtime',(),dict(config={'frozen':True}))()
            def request_executor(self,*a):raise AssertionError('Ambiguous request cannot relaunch')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);provider=Provider();provider.root=root/'formal'
            key=digest(dict(endpoint='/chat/completions',body=body,config=provider.runtime.config))
            _checkpoint(root/'direction/logical_request.json',dict(request_id=key,body=body), 'job')
            with self.assertRaises(RuntimeError):api.execute_direction(provider,body,root/'direction','job',root/'source300')


if __name__=='__main__':unittest.main()
