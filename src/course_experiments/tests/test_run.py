from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from course_experiments.manifest import digest, file_record, read_document
from course_experiments.contributions import write_immutable
from course_experiments.evaluation.protocol import freeze_phase
from course_experiments.run import validate_entry, run, textbook_only, collect, prepare, freeze_catalog
from course_experiments.results import report


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        graph = self.root / 'graph.json'
        write_immutable(graph, dict(entities=[], relations=[]))
        evaluator = self.root / 'evaluator.py'
        evaluator.write_text('fixture', 'utf-8')
        source = self.root / 'source.json'
        write_immutable(source, dict(course='c'))
        phase_path = self.root / 'phase.json'
        phase = freeze_phase(dict(graph=graph, evaluator_original=evaluator, construction=source), dict(seed=42), phase_path)
        metadata = self.root / 'metadata.json'
        write_immutable(metadata, dict(course_id='c', method_id='course_g0_textbook_fusion', state='fixed_g0', scope='full', status='complete', phase_id=phase['phase_id'], graph=file_record(graph), source_records=[file_record(source)]))
        self.item = dict(row_id='r', course_id='c', method_id='course_g0_textbook_fusion', state='fixed_g0', evaluation='topology', book_set='A', book_ids=['a'])
        entry = dict(phase=str(phase_path), metadata=file_record(metadata), scope='full', inputs=dict(graph=str(graph)), arguments=dict(graph=str(graph), evaluator=str(evaluator)))
        self.item.update(validate_entry(self.item, entry), entry=entry)
        self.calls = 0

    def plan(self, items):
        value = dict(positions=items)
        if any(i['evaluation'] in {'rag', 'kge'} and i['status']=='ready' for i in items):
            from course_experiments.run import bind_comparisons
            value['comparisons'] = bind_comparisons(items)
        value['plan_id'] = digest(value)
        path = self.root / ('plan-' + value['plan_id'] + '.json')
        write_immutable(path, value)
        return path

    def executor(self, item, output):
        self.calls += 1
        value = dict(status='completed_with_diagnostic_failure', graph_id=digest(dict(entities=[], relations=[])), metrics=None, input_digest=read_document(item['entry']['phase'])['phase_id'])
        write_immutable(output, value)
        return value, output

    def test_cache_deduplicates_positions_and_preserves_diagnostic(self):
        second = dict(self.item, book_set='B', book_ids=['b'])
        path = self.plan([self.item, second])
        records = run(path, self.root / 'jobs', self.executor)
        self.assertEqual(self.calls, 1)
        self.assertEqual(records[1]['status'], 'completed_with_diagnostic_failure')
        run(path, self.root / 'jobs', self.executor)
        self.assertEqual(self.calls, 1)

    def test_changed_hash_rejected_before_cache(self):
        path = self.plan([self.item])
        run(path, self.root / 'jobs', self.executor)
        Path(self.item['entry']['inputs']['graph']).write_text('{}')
        records = run(path, self.root / 'jobs', self.executor)
        self.assertEqual(records[0]['status'], 'failed')
        self.assertIn('Phase asset changed', records[0]['error_message'])
        self.assertTrue(list((self.root / 'jobs').rglob('validation_failure_*.json')))

    def test_wrong_course_scope_source(self):
        for key, value in [('course_id', 'other'), ('state', 'fused')]:
            item = dict(self.item, **{key: value})
            with self.assertRaises(ValueError): validate_entry(item, self.item['entry'])
        entry = dict(self.item['entry'], scope='smoke')
        with self.assertRaises(ValueError): validate_entry(self.item, entry)
        Path(self.root / 'source.json').write_text('{}')
        with self.assertRaises(ValueError): validate_entry(self.item, self.item['entry'])

    def test_independent_errors_and_account_stop(self):
        def fail(item, output): raise HTTPError('local', 402, 'payment', {}, None)
        second = dict(self.item, job_id='different', identity={})
        with patch('course_experiments.run.validate_entry', side_effect=lambda item, entry: dict(job_id=item['job_id'])):
            records = run(self.plan([self.item, second]), self.root / 'jobs', fail)
        self.assertEqual(records[0]['status'], 'failed')
        self.assertEqual(records[1]['status'], 'blocked')
        self.assertIn('payment', records[0]['error_message'])

    def test_locked_job_not_deleted(self):
        directory = self.root / 'jobs' / self.item['job_id']
        directory.mkdir(parents=True)
        (directory / 'job.lock').touch()
        result = run(self.plan([self.item]), self.root / 'jobs', self.executor)
        self.assertEqual(result[0]['status'], 'locked')
        self.assertTrue((directory / 'job.lock').exists())

    def test_missing_baselines_null_and_escaped_view(self):
        missing = dict(self.item, status='missing', method_id='K12', reason='<script>alert(1)</script>')
        missing.pop('entry')
        missing.pop('job_id')
        value = report(self.plan([missing]), self.root / 'jobs', self.root / 'report')
        self.assertFalse(value['core_comparison_complete'])
        self.assertIsNone(value['rows'][0]['score'])
        self.assertIn('&lt;script&gt;', (self.root / 'report/index.html').read_text())

    def test_textbook_carrier_empty(self):
        view = textbook_only('c', 'schema', [], [], alignment={})
        self.assertEqual(view['entities'], [])
        self.assertEqual(view['relations'], [])
        self.assertEqual(view['active_books'], [])

    def test_phase_budget_and_common_mask_invalidate_identity(self):
        entry = copy.deepcopy(self.item['entry'])
        old_phase = read_document(entry['phase'])
        mask_path = self.root / 'mask.json'
        write_immutable(mask_path, dict(mask=[True]))
        for attempt, parameters in enumerate([dict(seed=42, budget='new'), dict(seed=42, budget='new', mask_version=2)]):
            phase_path = self.root / ('new-phase-' + str(attempt) + '.json')
            paths = {k: v['path'] for k, v in old_phase['files'].items()}
            paths['common_mask'] = str(mask_path)
            phase = freeze_phase(paths, parameters, phase_path)
            metadata = read_document(entry['metadata']['path'])
            metadata['phase_id'] = phase['phase_id']
            metadata_path = self.root / ('new-metadata-' + str(attempt) + '.json')
            write_immutable(metadata_path, metadata)
            entry.update(phase=str(phase_path), metadata=file_record(metadata_path))
            entry['inputs']['common_mask'] = str(mask_path)
            self.assertNotEqual(validate_entry(self.item, entry)['job_id'], self.item['job_id'])

    def test_native_kge_not_applicable_is_null(self):
        item = dict(self.item, evaluation='kge')
        entry = dict(self.item['entry'], native_semantic_kge=False)
        with self.assertRaises(ValueError): validate_entry(item, entry)

    def gpu_fixture(self):
        from course_experiments.splits import freeze_split, project_graph
        from course_experiments.evaluation.kge import prepare_protocol, patch_evaluators
        project = Path(__file__).resolve().parents[2]
        originals = project / '.runtime_cache/evaluators_3090_20261014/v2'
        patched = patch_evaluators(originals / 'evaluate_kg_quality.py', originals / 'run_pykeen_epoch_sweep.py', self.root / 'patched')
        facts = [dict(fact_id=f'{i}-{j}', head=f'N{i}', relation='part_of', tail=f'N{(i+j)%20}', conditions='') for i in range(20) for j in [1, 2, 3]]
        ref = dict(course_id='c', entities=[dict(entity_id=f'N{i}', canonical_name=f'N{i}', scope='cpu', aliases=[]) for i in range(20)], facts=facts)
        graph = dict(entities=[dict(entity_id=e['entity_id'], canonical_name=e['canonical_name'], scope='cpu', type='Concept') for e in ref['entities']], relations=[dict(head_entity_id=f['head'], tail_entity_id=f['tail'], relation=f['relation']) for f in facts])
        split = freeze_split(ref)
        projection = project_graph(graph, ref, split)
        mask = list(projection['covered_test_mask'])
        if any(mask): mask[mask.index(True)] = False
        paths = dict(evaluator_original=originals/'evaluate_kg_quality.py', sweep_original=originals/'run_pykeen_epoch_sweep.py', evaluator_patched=patched['evaluator']['path'], sweep_patched=patched['sweep']['path'], src_utils=project/'src/utils.py')
        for role, value in [('graph', graph), ('projection', projection), ('fact_split', split), ('construction', {}), ('gpu_environment', {}), ('dependency_lock', {}), ('common_mask', dict(mask=mask, projection_ids=[projection['projection_id']], split_id=split['split_id']))]:
            path = self.root / ('gpu-' + role + '.json')
            write_immutable(path, value)
            paths[role] = path
        phase_path = self.root / 'gpu-phase.json'
        phase = freeze_phase(paths, dict(seed=42), phase_path)
        phase['path'] = str(phase_path)
        protocol_path = self.root / 'protocol.json'
        protocol = prepare_protocol(projection, split, phase, protocol_path, smoke=True, common_mask=mask)
        metadata = self.root / 'gpu-metadata.json'
        write_immutable(metadata, dict(course_id='c', method_id=self.item['method_id'], state='fixed_g0', scope='smoke', status='complete', phase_id=phase['phase_id'], graph=file_record(paths['graph']), source_records=[file_record(paths['construction'])]))
        entry = dict(phase=str(phase_path), metadata=file_record(metadata), scope='smoke', inputs={k:str(paths[k]) for k in ('graph', 'projection', 'fact_split', 'common_mask')}, arguments=dict(graph=str(paths['graph']), protocol=str(protocol_path), patched_evaluator=patched['evaluator']['path']), epochs=2, device='cpu')
        item = dict(self.item, evaluation='kge', entry=entry)
        item.update(validate_entry(item, entry))
        identity = dict(protocol_id=protocol['protocol_id'], phase_id=phase['phase_id'], graph_id=projection['graph_id'], split_id=split['split_id'], epochs=2, requested_device='cpu', options=protocol['budget'])
        metrics = self.root / 'pykeen_metrics.json'
        write_immutable(metrics, dict(both=dict(realistic=dict(inverse_harmonic_mean_rank=.5))))
        result = dict(status='completed_with_diagnostic_failure', run_id=digest(identity), protocol_id=protocol['protocol_id'], split_id=split['split_id'], graph_id=projection['graph_id'], scope='smoke', options=protocol['budget'], epochs=2, test_ids=[t for t,m in zip(projection['test_ids'],mask) if m], fixed_test_count=len(projection['test']), covered_test_count=sum(projection['covered_test_mask']), common_test_count=sum(mask), train_triples=len(projection['training']), validation_triples=sum(projection['covered_validation_mask']), test_triples=sum(mask), metrics_file=str(metrics), reason='rank diagnostic failed', raw_paths=['diagnostic.json'])
        return item, result

    def test_gpu_collection_rejects_wrong_ids_counts_epochs(self):
        item, result = self.gpu_fixture()
        for key, value in [('test_ids', ['wrong']), ('fixed_test_count', 999), ('covered_test_count', 999), ('common_test_count', 999), ('epochs', 500), ('train_triples', 999), ('validation_triples', 999), ('test_triples', 999)]:
            path = self.root / ('wrong-' + key + '.json')
            write_immutable(path, dict(result, **{key:value}))
            with self.assertRaises(ValueError, msg=key): collect(self.plan([item]), item['job_id'], path, self.root/'jobs')

    def test_gpu_metrics_digest_and_view(self):
        item, result = self.gpu_fixture()
        result_path = self.root/'external_kge_result.json'
        write_immutable(result_path, result)
        plan_path = self.plan([item])
        record = collect(plan_path, item['job_id'], result_path, self.root/'jobs')
        self.assertEqual(record['metrics_record'], file_record(result['metrics_file']))
        view = report(plan_path, self.root/'jobs', self.root/'viewer')['rows'][0]
        self.assertEqual(view['metrics']['both']['realistic']['inverse_harmonic_mean_rank'], .5)
        self.assertEqual(view['status'], 'completed_with_diagnostic_failure')
        self.assertEqual(view['reason'], 'rank diagnostic failed')
        self.assertIn('diagnostic.json', view['raw_paths'])
        Path(result['metrics_file']).write_text('{}')
        changed = report(plan_path, self.root/'jobs', self.root/'viewer')['rows'][0]
        self.assertEqual(changed['status'], 'failed')
        self.assertIsNone(changed['metrics'])
        self.assertIn('metrics artifact changed', changed['reason'])

    def native_fixture(self):
        method = dict(method_id='kggen_lm_dedup', commit='a'*40, status='verified', repository='fixture', capability_scope=['rag', 'topology', 'kge'])
        manifest_path = self.root/'native-manifest.json'
        write_immutable(manifest_path, dict(experiments=dict(methods=[method])))
        graph = dict(entities=[], relations=[])
        artifact = dict(course_id='c', method_id='kggen_lm_dedup', native_commit='a'*40, status='complete', scope='full', native_graph=dict(directed=False, nodes=[], edges=[]), semantic_view=dict(graph, capabilities=dict(kge=None), capability_reasons=dict(kge='Native undirected descriptions have no directed typed facts')))
        artifact['artifact_id'] = digest(artifact)
        native_path = self.root/'native-artifact.json'
        write_immutable(native_path, artifact)
        phase_path = self.root/'native-phase.json'
        phase = freeze_phase(dict(graph=self.item['entry']['inputs']['graph'], native_artifact=native_path, construction=self.root/'source.json', evaluator_original=self.root/'evaluator.py'), dict(seed=42), phase_path)
        metadata_path = self.root/'native-metadata.json'
        write_immutable(metadata_path, dict(course_id='c', method_id='kggen_lm_dedup', state='native', book_ids=['a'], scope='full', status='complete', phase_id=phase['phase_id'], graph=file_record(self.item['entry']['inputs']['graph']), native_commit='a'*40, source_records=[file_record(native_path)]))
        item = dict(self.item, method_id='kggen_lm_dedup', state='native', evaluation='kge', canonical_method=method, manifest_record=file_record(manifest_path))
        entry = dict(self.item['entry'], phase=str(phase_path), metadata=file_record(metadata_path), native_semantic_kge=False)
        return item, entry

    def test_native_na_requires_frozen_capabilities_and_commit(self):
        item, entry = self.native_fixture()
        self.assertEqual(validate_entry(item, entry)['status'], 'not_applicable')
        meta = read_document(entry['metadata']['path'])
        meta['native_commit'] = 'b'*40
        bad_path = self.root/'wrong-commit.json'
        write_immutable(bad_path, meta)
        wrong = dict(entry, metadata=file_record(bad_path))
        with self.assertRaises(ValueError): validate_entry(item, wrong)
        artifact = read_document(self.root/'native-artifact.json')
        artifact['semantic_view']['capabilities']['kge'] = True
        Path(self.root/'native-artifact.json').write_text(__import__('json').dumps(artifact))
        with self.assertRaises(ValueError): validate_entry(item, entry)

    def test_invalid_input_continues_independent_job(self):
        second = dict(self.item, job_id='valid-independent', identity={})
        def validation(item, entry):
            if item['job_id'] == self.item['job_id']: raise ValueError('changed source')
            return dict(job_id=item['job_id'])
        with patch('course_experiments.run.validate_entry', side_effect=validation):
            records = run(self.plan([self.item, second]), self.root/'jobs', self.executor)
        self.assertEqual([r['status'] for r in records], ['failed', 'completed_with_diagnostic_failure'])
        self.assertEqual(self.calls, 1)
        self.assertTrue(list((self.root/'jobs'/self.item['job_id']).glob('validation_failure_*.json')))

    def test_prepare_uses_actual_manifest_commit_and_canonical_rows(self):
        from course_experiments.tests.test_manifest_matrix import ManifestMatrixTests
        from course_experiments.manifest import freeze
        from course_experiments.matrix import generate
        fixture_root = self.root/'manifest'
        fixture_root.mkdir()
        courses, config, sources, assets = ManifestMatrixTests().fixture(fixture_root)
        config['methods'][1] = dict(method_id='kggen_lm_dedup', commit='a'*40, status='verified', repository='fixture', capability_scope=['rag', 'topology', 'kge'])
        manifest_path = self.root/'evaluation-manifest.json'
        frozen = freeze(courses, config, sources, assets, output=manifest_path)
        canonical = next(r for r in generate(frozen) if r['course_id']=='one' and r['book_set']=='A' and r['state']=='native' and r['evaluation']=='topology')
        metadata = read_document(self.item['entry']['metadata']['path'])
        metadata.update(course_id='one', method_id='kggen_lm_dedup', state='native', native_commit='b'*40, book_ids=['oneA'])
        metadata_path = self.root/'prepare-metadata.json'
        write_immutable(metadata_path, metadata)
        entry = dict(self.item['entry'], metadata=file_record(metadata_path))
        catalog_path = self.root/'catalog.json'
        freeze_catalog({canonical['row_id']:entry}, catalog_path)
        plan = prepare(manifest_path, catalog_path, self.root/'prepared.json')
        row = next(p for p in plan['positions'] if p['row_id']==canonical['row_id'])
        self.assertEqual(row['status'], 'blocked')
        self.assertIn('commit/method mismatch', row['reason'])
        self.assertEqual(row['canonical_method']['commit'], 'a'*40)
        self.assertEqual(len(plan['positions']), len(generate(frozen)))
        self.assertTrue(any(p['status']=='missing' for p in plan['positions']))

    def test_aligned_graph_cannot_suppress_semantic_kge(self):
        item, entry = self.native_fixture()
        item.update(method_id='kggen_lm_dedup_schema_aligned', state='schema_aligned')
        metadata = read_document(entry['metadata']['path'])
        metadata.update(method_id=item['method_id'], state=item['state'])
        path = self.root/'aligned-metadata.json'
        write_immutable(path, metadata)
        entry['metadata'] = file_record(path)
        with self.assertRaisesRegex(ValueError, 'N/A restricted'): validate_entry(item, entry)

    def test_stage_accepts_no_freeform_argument(self):
        entry = copy.deepcopy(self.item['entry'])
        entry['arguments']['command'] = 'anything'
        with self.assertRaises(ValueError): validate_entry(self.item, entry)


if __name__ == '__main__': unittest.main()
