from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest

from course_experiments.manifest import digest,read_document
from course_experiments.contributions import write_immutable
from course_experiments.tests import test_identity_single_v2_g0 as fixtures
from course_experiments.tests.test_identity_single_v2 import Provider
from course_experiments.evaluation import identity_single_v2 as identity


class FullPackageTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_single_v2_full
        return identity_single_v2_full

    def fixture(self,root):
        mp,rp,processed,populations=fixtures.G0V2Tests().fixture(root)
        database=root/'audit.sqlite'
        with closing(sqlite3.connect(database)) as db,db:db.execute('CREATE TABLE events(key TEXT,status TEXT)')
        from course_experiments.evaluation.identity_single_v2_g0 import prepare_g0
        prepare_g0(mp,rp,processed,root/'G0')
        from course_experiments.evaluation.identity_source_full_execution import START,AUTHORIZATION_SHA,ALLOCATION_SHA
        authorization=dict(allocation=dict(stage_prefix='identity_single_v2.source.',watermark_event_id=102848,maximum_additional_actual_HTTP=33000,
            authorized_cumulative_HTTP=200000,authorized_total_seconds=345600,original_started_unix=START,deadline_unix=START+345600,
            authorization_sha256=AUTHORIZATION_SHA,allocation_sha256=ALLOCATION_SHA),source_candidate_count=len(populations['source']),source_packets_digest=digest(populations['source']),worker_count=4)
        authorization['execution_identity']=digest(authorization);write_immutable(processed/'source_execution_authorization.json',authorization)
        return mp,rp,processed,populations,database

    def test_full_export_requires_complete_source_not_only_complete_reference_and_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,populations,database=self.fixture(root)
            with self.assertRaises(FileNotFoundError):self.api().prepare_full(mp,rp,processed,root/'full',database=database,config={},parent_package=root/'G0/package.json')
            self.assertFalse((root/'full/package.json').exists())

    def test_source_execution_authorization_cannot_be_invented_from_complete_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,populations,database=self.fixture(root)
            identity.run_population(populations['source'],'source',Provider(),processed/'source')
            (processed/'source_execution_authorization.json').unlink()
            with self.assertRaises(FileNotFoundError):self.api().prepare_full(mp,rp,processed,root/'full',database=database,config={},parent_package=root/'G0/package.json')
            self.assertFalse((root/'full/package.json').exists())

    def test_full_mapping_uses_unchanged_reference_only_split_and_preserves_all_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,rp,processed,populations,database=self.fixture(root)
            identity.run_population(populations['source'],'source',Provider(),processed/'source')
            package=self.api().prepare_full(mp,rp,processed,root/'full',database=database,config={},parent_package=root/'G0/package.json')
            previous=read_document(root/'G0/package.json')
            self.assertEqual(package['reference_courses'],previous['reference_courses'])
            self.assertEqual(package['reference_qualification_id'],previous['reference_qualification_id'])
            self.assertEqual(len(package['graphs']),2)
            self.assertEqual(package['qualified_graph_count'],2)
            self.assertTrue(package['current_inventory_source_full_qualified'])
            self.assertFalse(package['future_graphs_qualified'])
            self.assertFalse(package['user_full_matrix_complete'])
            self.assertFalse(package['old_batch_gates_passed'])
            for row in package['graphs']:
                projection=read_document(row['paths']['projection.json'])
                self.assertEqual(len(projection['test']),row['fixed_test_count'])
                self.assertEqual(projection['split_id'],previous['reference_courses']['course']['split_id'])


if __name__=='__main__':unittest.main()
