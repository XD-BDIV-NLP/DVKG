import tempfile
from pathlib import Path
import unittest

from course_experiments.evaluation import run_scope_identity_controller as controller
from course_experiments.evaluation.run_native_full import _checkpoint
from course_experiments.evaluation.run_repaired_full import atomic


class ControllerTests(unittest.TestCase):
    def test_source_gate_waits_for_complete_fixed_population_and_process_exit_is_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ids=['a','b']
            self.assertEqual(controller.source_readiness(root,ids,alive=True),'waiting')
            self.assertEqual(controller.source_readiness(root,ids,alive=False),'failed')
            _checkpoint(root/'result.json',dict(status='completed_development_gate',candidate_count=2,failed_candidates=0),'job')
            _checkpoint(root/'decisions/a/decision.json',dict(status='completed',candidate_id='a'),'job')
            self.assertEqual(controller.source_readiness(root,ids,alive=True),'failed')
            _checkpoint(root/'decisions/b/decision.json',dict(status='completed',candidate_id='b'),'job')
            self.assertEqual(controller.source_readiness(root,ids,alive=False),'ready')

    def test_partial_or_failed_single_population_is_never_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            _checkpoint(root/'result.json',dict(status='partial_development_gate',candidate_count=2,failed_candidates=1),'job')
            self.assertEqual(controller.source_readiness(root,['a','b'],alive=True),'failed')

    def test_independent_prior_gate_survives_source_failure_and_prior_full_stays_blocked(self):
        stages={}
        self.assertEqual(controller.next_stage(stages,'failed'),'prior24-single')
        stages['prior24-single']=dict(exit_code=0)
        self.assertEqual(controller.next_stage(stages,'failed'),'prior24-batch')
        stages['prior24-batch']=dict(exit_code=0)
        self.assertIsNone(controller.next_stage(stages,'failed'))
        self.assertEqual(controller.next_stage(stages,'ready'),'source32')
        stages['source32']=dict(exit_code=2)
        self.assertIsNone(controller.next_stage(stages,'ready'))

    def test_completed_stages_are_not_launched_again_and_full_requires_both_gates(self):
        stages={name:dict(exit_code=0) for name in ['prior24-single','prior24-batch','source32']}
        self.assertEqual(controller.next_stage(stages,'ready'),'source-full')
        stages['source-full']=dict(exit_code=0)
        self.assertEqual(controller.next_stage(stages,'ready'),'prior-full')
        stages['prior-full']=dict(exit_code=0)
        self.assertIsNone(controller.next_stage(stages,'ready'))

    def test_deadline_is_original_formal_start_not_controller_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic(root/'limits.json',dict(started_unix=100,limits=dict(max_seconds=172800)))
            self.assertEqual(controller.formal_deadline(root),172900)

    def test_launch_is_reserved_before_process_spawn_and_never_reserved_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'queue.json';state=dict(stages={})
            controller.reserve_launch(state,'source32',path)
            self.assertEqual(state['stages']['source32']['status'],'launch_reserved')
            self.assertTrue(path.exists())
            with self.assertRaises(ValueError):controller.reserve_launch(state,'source32',path)

    def test_even_internal_worker_mode_requires_explicit_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=[]
            for name in ['run-root','config','preparation-root','output-root','single119-root','queue-root']:
                args+=['--'+name,str(root)]
            self.assertEqual(controller.main(args+['--worker-stage','prior24-single']),0)
            self.assertFalse((root/'stages').exists())

    def test_old_worker_receipt_cannot_be_attached_to_new_frozen_population(self):
        receipt=dict(stage='source32',preparation_id='old',controller_code='code')
        with self.assertRaises(ValueError):controller.verify_worker_receipt(receipt,'source32','new','code')
        controller.verify_worker_receipt(receipt,'source32','old','code')

    def test_verified_terminal_source_snapshot_avoids_reloading119_decision_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state={}
            _checkpoint(root/'result.json',dict(status='completed_development_gate',candidate_count=1,failed_candidates=0),'job')
            path=root/'decisions/a/decision.json';_checkpoint(path,dict(status='completed',candidate_id='a'),'job')
            self.assertEqual(controller.source_snapshot(state,root,['a'],alive=False),'ready')
            path.unlink()
            self.assertEqual(controller.source_snapshot(state,root,['a'],alive=False),'ready')


if __name__=='__main__':unittest.main()
