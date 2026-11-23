import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from course_experiments.evaluation import run_scope_identity_gate as gate
from course_experiments.evaluation import run_identity_recovery as recovery
from course_experiments.evaluation.run_native_full import _checkpoint, _read_checkpoint
from course_experiments.manifest import digest, file_record
from course_experiments.tests.test_scope_identity_batch import packet, origins


class ClosedOnlyProvider:
    def __init__(self): self.closed=False
    def raise_if_fatal(self): pass
    def close(self): self.closed=True
    def chat_response(self,*a,**k): raise AssertionError('Successful checkpoint must not call API')


class SourceResumeTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import resume_identity_source
        return resume_identity_source

    def test_real_gate_full_contract_reproduces_duplicate_then_resume_keeps_job(self):
        api=self.api();work=[packet(8)];rows=origins(work)
        contract=dict(gate.contract_for(work,rows),local_http_cap=300)
        migration=dict(source_contract=contract,source_job=digest(contract))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);provider=ClosedOnlyProvider()
            _checkpoint(root/'decisions'/work[0]['candidate_id']/'decision.json',dict(
                status='completed',candidate_id=work[0]['candidate_id'],eligible_identity_edge=False),digest(contract))
            with self.assertRaisesRegex(TypeError,'local_http_cap'):
                gate.run_gate(work,rows,provider,root,contract=contract)
            result=api.invoke_source_gate(gate,migration,work,rows,provider,root,root/'allowance')
            self.assertEqual(result['job_identity'],migration['source_job'])
            self.assertEqual(result['status'],'completed_development_gate')
            self.assertTrue(provider.closed)
            self.assertEqual(contract,migration['source_contract'])

    def test_changed_source_job_is_rejected_before_provider_call(self):
        api=self.api();work=[packet(8)];rows=origins(work)
        contract=dict(gate.contract_for(work,rows),local_http_cap=300)
        with self.assertRaises(ValueError):
            api.invoke_source_gate(gate,dict(source_contract=contract,source_job='changed'),work,rows,
                ClosedOnlyProvider(),Path('unused'),Path('unused'))

    def test_preserved_completed_and_pending_primary_checked_without_rebinding(self):
        api=self.api();work=[packet(1),packet(2)]
        with tempfile.TemporaryDirectory() as folder:
            old=Path(folder)/'old';new=Path(folder)/'new';job='same'
            for i,p in enumerate(work):
                base=old/'decisions'/p['candidate_id'];dest=new/'decisions'/p['candidate_id']
                _checkpoint(base/'decision.json',dict(status='completed' if i==0 else 'failed',candidate_id=p['candidate_id']), 'old')
                _checkpoint(base/'primary.json',dict(judgment=dict(decision='same_sense')), 'old')
                recovery.rebind_checkpoint(base/'primary.json',dest/'primary.json',job)
                if i==0: recovery.rebind_checkpoint(base/'decision.json',dest/'decision.json',job)
            migration=dict(source_job=job,source_original_counts=dict(completed=1,failed=1),source_prior_success_primary_count=1)
            before=file_record(new/'decisions'/work[1]['candidate_id']/'primary.json')
            result=api.verify_preserved(recovery,migration,work,old,new)
            self.assertEqual(result['completed_decisions_preserved'],1)
            self.assertEqual(result['pending_primary_preserved'],1)
            self.assertEqual(before,file_record(before['path']))
            _checkpoint(new/'decisions'/work[1]['candidate_id']/'primary.json',dict(judgment=dict(decision='different_sense')),job,immutable=False)
            with self.assertRaises(ValueError):api.verify_preserved(recovery,migration,work,old,new)

    def test_continuation_order_does_not_repeat_prior_development(self):
        self.assertEqual(self.api().CONTINUATION,('source32','source-full','prior-full'))

    def test_completed_stage_reads_exact_job_without_constructing_provider(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);contract=dict(synthetic=True);job=digest(contract)
            _checkpoint(root/'source32'/'result.json',dict(status='completed'),job)
            result=api.completed_stage(root,'source32',contract)
            self.assertEqual(result['status'],'completed')
            with self.assertRaises(ValueError):api.completed_stage(root,'source32',dict(changed=True))

    def test_incomplete_downstream_result_is_retained_and_not_relaunched(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);contract=dict(synthetic=True);path=root/'source32'/'result.json'
            _checkpoint(path,dict(status='failed_frozen_gate'),digest(contract));before=file_record(path)
            with self.assertRaises(ValueError):api.completed_stage(root,'source32',contract)
            self.assertEqual(before,file_record(path))

    def test_local_cap_cannot_change_during_call_adaptation(self):
        api=self.api();work=[packet(8)];rows=origins(work)
        contract=dict(gate.contract_for(work,rows),local_http_cap=301)
        with self.assertRaises(ValueError):api.source_contract_arguments(gate,
            dict(source_contract=contract,source_job=digest(contract)),work,rows)

    def test_process_identity_rejects_missing_or_different_command(self):
        api=self.api()
        self.assertIsNone(api.process_identity(-1))
        import os
        if Path('/proc').exists():
            current=api.process_identity(os.getpid())
            self.assertEqual(current['pid'],os.getpid())
            self.assertIsNone(api.process_identity(os.getpid(),('not-this-process-1939',)))


if __name__=='__main__':unittest.main()
