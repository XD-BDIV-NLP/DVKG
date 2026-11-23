import copy
import tempfile
import unittest
from pathlib import Path


class IdentityRecoveryTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import run_identity_recovery
        return run_identity_recovery

    def test_success_checkpoint_rebind_preserves_judgment_and_original_bytes(self):
        api=self.api()
        from course_experiments.evaluation.run_native_full import _checkpoint,_read_checkpoint
        from course_experiments.manifest import file_record
        with tempfile.TemporaryDirectory() as root:
            old=Path(root)/'old.json';new=Path(root)/'new.json'
            _checkpoint(old,dict(status='completed',judgment=dict(decision='different_sense',reason='synthetic')), 'old')
            before=file_record(old);api.rebind_checkpoint(old,new,'new')
            self.assertEqual(file_record(old),before)
            migrated=_read_checkpoint(new,'new')
            self.assertEqual(migrated['judgment'],_read_checkpoint(old,'old')['judgment'])
            self.assertEqual(migrated['recovery_from'],before)

    def test_offline_raw_reparse_recovers_nested_guards_without_model_calls(self):
        api=self.api()
        from course_experiments.evaluation import scope_identity_batch as core
        from course_experiments.tests.test_scope_identity_batch import Provider,packet
        work=[packet(1,is_prior=True)]
        def nest(wire,result,count):
            for row in result['results']:
                row['flags']={key:row.pop(key) for key in ['necessary_scope_consistent','definition_identity_sufficient','homonym_excluded']}
        old=core.judge_batch(Provider(nest),work);before=copy.deepcopy(old)
        recovered=api.reparse_batch(old,work)
        self.assertEqual(old,before)
        self.assertTrue(next(iter(recovered['decisions'].values()))['eligible_identity_edge'])
        self.assertEqual(recovered['raw_response'],old['raw_response'])

    def test_changed_batch_or_candidate_ids_are_not_repaired_offline(self):
        api=self.api()
        from course_experiments.evaluation import scope_identity_batch as core
        from course_experiments.tests.test_scope_identity_batch import Provider,packet
        import json
        work=[packet(1,is_prior=True)];old=core.judge_batch(Provider(),work)
        raw=json.loads(old['raw_response']['choices'][0]['message']['content']);raw['results'][0]['candidate_id']='wrong'
        old['raw_response']['choices'][0]['message']['content']=json.dumps(raw)
        with self.assertRaises(ValueError):api.reparse_batch(old,work)

    def test_original_preparation_allows_only_reviewed_provider_substitution(self):
        api=self.api()
        from unittest.mock import patch
        from course_experiments.manifest import digest,file_record
        from course_experiments.contributions import write_immutable
        from course_experiments.evaluation import scope_identity_batch as core
        with tempfile.TemporaryDirectory() as root:
            value=dict(files={},implementation={'native_provider.py':'5e8661663e476183f401a201971636a0402dfa6ba2a275c6b2ae3fd4ad80e859',
                'scope_identity_batch.py':file_record(core.__file__)['sha256']},token_assets_root='synthetic')
            value['preparation_id']=digest(value);write_immutable(Path(root)/'manifest.json',value)
            with patch.object(api.packing,'OfficialFlashCounter'):
                self.assertEqual(api.validate_original_preparation(Path(root)),value)
            other=Path(root)/'other';other.mkdir();value['implementation']['scope_identity_batch.py']='unknown'
            value['preparation_id']=digest({k:v for k,v in value.items() if k!='preparation_id'})
            write_immutable(other/'manifest.json',value)
            with patch.object(api.packing,'OfficialFlashCounter'):
                with self.assertRaises(ValueError):api.validate_original_preparation(other)


if __name__=='__main__':unittest.main()
