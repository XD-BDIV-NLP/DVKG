import json
from pathlib import Path
import tempfile
import unittest

from course_experiments.tests.test_identity_single_v2 import Provider
from course_experiments.tests.test_scope_identity_batch import packet
from course_experiments.manifest import digest,file_record


class PreflightTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import run_identity_single_v2
        return run_identity_single_v2

    def test_missing_bodies_are_unprocessed_not_false_unknown_and_no_api(self):
        api=self.api();work={'source':[packet(1)],'prior':[packet(2,is_prior=True)],'reference':[packet(3)]}
        result=api.scan_exact_cache(work,{'llm':{'model':'deepseek-ai/DeepSeek-V4-Flash'}},{})
        self.assertEqual(result['all_positive_exact_body_count'],6)
        self.assertEqual(result['missing_exact_body_count'],6)
        self.assertEqual(result['first_attempt_upper'],6);self.assertEqual(result['actual_HTTP_cap2_upper'],12)
        self.assertEqual(result['branches']['source']['primary_missing'],1)
        self.assertEqual(result['model_calls'],0)

    def test_receipt_exists_is_not_strict_valid_identity(self):
        api=self.api();p=packet(1);config={'llm':{'model':'deepseek-ai/DeepSeek-V4-Flash'}};records={}
        from course_experiments.evaluation.identity_single_v2 import request_body
        with tempfile.TemporaryDirectory() as tmp:
            for reverse in [False,True]:
                body=request_body(p,'source',reverse=reverse);raw,_=Provider().request_executor('/chat/completions',body,'test')
                if reverse:
                    value=json.loads(raw['choices'][0]['message']['content']);value['right_citations'][0]['quote']='invented';raw['choices'][0]['message']['content']=json.dumps(value)
                key=digest(dict(endpoint='/chat/completions',body=body,config=config));path=Path(tmp)/(key+'.json');path.write_text(json.dumps(dict(request=body,runtime_config=config,response=raw)));records[key]=file_record(path)
            result=api.scan_exact_cache({'source':[p]},config,records)
        self.assertEqual(result['existing_exact_receipts'],2)
        self.assertEqual(result['branches']['source']['primary_strict_valid'],1)
        self.assertEqual(result['branches']['source']['reverse_invalid_UNKNOWN'],1)
        self.assertEqual(result['branches']['source']['accepted_before_conflict'],0)
        self.assertEqual(result['branches']['source']['cached_UNKNOWN'],1)


if __name__=='__main__':unittest.main()
