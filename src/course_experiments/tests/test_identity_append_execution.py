from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from course_experiments.manifest import digest,read_document
from course_experiments.tests.test_identity_single_v2 import Provider
from course_experiments.tests.test_scope_identity_batch import packet
from course_experiments.evaluation import identity_single_v2 as identity


class AppendExecutionTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_append_execution
        return identity_append_execution

    def test_old_unknown_and_known_decisions_are_inherited_without_reasking_and_only_delta_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=[packet(1),packet(2)];new=old+[packet(3)]
            def invalid(value,count):
                if value['candidate_id']==old[0]['candidate_id']:value['left_citations'][0]['quote']='fabricated'
            identity.run_population(old,'source',Provider(invalid),root/'old')
            provider=Provider();provider._identity_single_attempt_cap=2;provider._identity_single_total_cap=33000
            receipt=self.api().adopt_population(old,new,'source',root/'old',root/'new',provider,None)
            for index in range(4):identity.run_population(new,'source',provider,root/'new',worker_index=index,worker_count=4)
            result=identity.finalize_population(new,'source',root/'new')
            self.assertEqual(len(provider.calls),2);self.assertEqual(receipt['old_candidate_count'],2)
            self.assertEqual(result['decisions'][old[0]['candidate_id']]['classification'],'UNKNOWN')
            self.assertEqual(result['accepted_count'],2);self.assertEqual(result['candidate_count'],3)

    def test_rebound_candidate_body_changes_are_rejected_before_any_HTTP(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=[packet(1)];identity.run_population(old,'source',Provider(),root/'old');new=[dict(old[0],course_id='changed')];provider=Provider()
            with self.assertRaises(ValueError):self.api().adopt_population(old,new,'source',root/'old',root/'new',provider,None)
            self.assertEqual(provider.calls,[])

    def test_adoption_receipt_rejects_later_reclassification_of_original_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=[packet(1)]
            def invalid(value,count):value['left_citations'][0]['quote']='invented'
            identity.run_population(old,'source',Provider(invalid),root/'old');provider=Provider();provider._identity_single_attempt_cap=2;provider._identity_single_total_cap=33000
            self.api().adopt_population(old,old,'source',root/'old',root/'new',provider,None)
            self.api().validate_adoption(old,'source',root/'new')
            path=root/'new/decisions'/old[0]['candidate_id']/'decision.json';row=read_document(path);row['classification']='confirmed_negative'
            from course_experiments.evaluation.run_native_full import _checkpoint
            path.unlink();_checkpoint(path,{k:v for k,v in row.items() if k not in {'job_identity','checkpoint_digest'}},row['job_identity'])
            with self.assertRaises(ValueError):self.api().validate_adoption(old,'source',root/'new')

    def test_prior_missing_raw_zero_actual_is_not_a_complete_appended_population(self):
        from course_experiments.evaluation.identity_source_full_execution import validate_actual_proofs
        p=packet(1,is_prior=True)
        class Unresolved(Provider):
            def request_executor(self,*args):raise TimeoutError('synthetic first HTTP not issued')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity.run_population([p],'prior',Unresolved(),root/'prior')
            database=root/'audit.sqlite'
            with closing(sqlite3.connect(database)) as db,db:db.execute('CREATE TABLE events(key TEXT,status TEXT)')
            with self.assertRaises(ValueError):validate_actual_proofs([p],root/'prior',{},database,read_document(root/'prior/contract.json'),branch='prior')


if __name__=='__main__':unittest.main()
