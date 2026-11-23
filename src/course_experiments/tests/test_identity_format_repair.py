import copy
import json
import tempfile
import unittest
from pathlib import Path


class IdentityFormatRepairTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_format_repair
        return identity_format_repair

    def fixture(self):
        from course_experiments.tests.test_scope_identity_batch import packet
        from course_experiments.evaluation.scope_identity import anonymous_packet
        p=packet(1);wire=anonymous_packet(p)
        value=dict(candidate_id=p['candidate_id'],decision='insufficient',scope_consistent=False,
            source_definition_sufficient=False,homonym_excluded=True,reason='offline',
            left_citations=[wire['left']['source_quotes'][0]],right_citations=[wire['right']['source_quotes'][0]])
        return p,wire,value

    def test_repair_keeps_decision_and_boolean_guards_and_exact_original_sides(self):
        api=self.api();packet,wire,value=self.fixture();bad=dict(value,candidate_id='bad_id')
        raw=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))])
        result=api.validate_repaired(wire,bad,raw)
        self.assertEqual(result['decision'],bad['decision'])
        changed=copy.deepcopy(value);changed['decision']='same_sense';raw['choices'][0]['message']['content']=json.dumps(changed)
        with self.assertRaisesRegex(ValueError,'semantic'):api.validate_repaired(wire,bad,raw)
        changed=copy.deepcopy(value);changed['left_citations'][0]['quote']='not source';raw['choices'][0]['message']['content']=json.dumps(changed)
        with self.assertRaises(ValueError):api.validate_repaired(wire,bad,raw)

    def test_attempts_are_new_deterministic_requests_and_never_exceed_two_per_pair(self):
        api=self.api();packet,wire,value=self.fixture()
        class Provider:
            def __init__(self):self.calls=[]
            def chat_response(self,messages,**kwargs):
                self.calls.append((messages,kwargs))
                bad=dict(value,candidate_id='bad_id')
                return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(bad)))])
            def raise_if_fatal(self):pass
        from course_experiments.evaluation.scope_identity import JUDGE
        messages=[dict(role='system',content=JUDGE),dict(role='user',content=json.dumps(wire))]
        with tempfile.TemporaryDirectory() as root:
            provider=Provider();wrapper=api.FormatRepairProvider(provider,Path(root))
            for _ in range(2):
                with self.assertRaises(ValueError):wrapper.chat_response(messages,max_tokens=1024,stage='scope_identity_development.judge_primary')
            repair=[m for m,k in provider.calls if k['stage'].startswith('scope_identity_format_repair')]
            self.assertEqual(len(repair),2)
            self.assertNotEqual(repair[0],repair[1])
            for request in repair:
                data=json.loads(request[-1]['content'])
                self.assertEqual(data['original_pair'],wire)
                self.assertEqual(data['previous_response']['decision'],value['decision'])

    def test_successful_initial_response_has_no_repair_call(self):
        api=self.api();packet,wire,value=self.fixture()
        class Provider:
            calls=0
            def chat_response(self,*args,**kwargs):
                self.calls+=1;return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))])
            def raise_if_fatal(self):pass
        with tempfile.TemporaryDirectory() as root:
            provider=Provider();wrapped=api.FormatRepairProvider(provider,Path(root))
            wrapped.chat_response([dict(role='user',content=json.dumps(wire))],max_tokens=1024,stage='scope_identity_development.judge_primary')
            self.assertEqual(provider.calls,1)


if __name__=='__main__':unittest.main()
