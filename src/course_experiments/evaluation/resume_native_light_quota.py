"""仅恢复已核实的原生LightRAG配额中断；默认准备，不访问模型。

以独立脚本运行：--config CONFIG [--run]。配置固定source_contract、
source_index_contract、source_index_receipt、authorization、provider_factory的
path/sha256引用，以及run_root、output_root和actual_http_cap（至多100）。
旧结果、已就绪索引、模型请求内容及所有历史全局/原生用量均保持不变。
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback


VERSION = 'native_light_os_quota_resume_v1'
BOOK = 'book_d96f472c8179'
ORIGINAL_ID = 'dee77a80d4ccc2ac8e76a1d162003a7359643cd383b931e74c00293565518f20'
INDEX_ID = '81d94c612ea656103ef41d288b2d6564629e4679cf0921cc55ab7ecc285702d5'
QUOTA_ERROR = 'Native query failed: Query failed: Inherited owned actualHTTP cap exhausted'
PREFIX = 'lightrag.book_d96f472c8179.supplement_quota_resume_20261028/'
CAP = 100


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    path = Path(path)
    if path.suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        return yaml.safe_load(path.read_text('utf-8-sig'))
    return json.loads(path.read_text('utf-8-sig'))


def rec(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(reference):
    if rec(reference['path']) != reference:
        raise ValueError('Pinned native quota-resume dependency changed')
    return read(reference['path'])


def load(reference, name):
    if rec(reference['path']) != reference:
        raise ValueError('Pinned native quota-resume code changed')
    spec = importlib.util.spec_from_file_location(name, reference['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_frozen_runtime(reference):
    """即使原生app内有同内容的同名模块，仍选合同钉住的完整路径。"""
    parent = importlib.import_module('domain_ckg_workspace.ckg_incremental')
    name = 'domain_ckg_workspace.ckg_incremental.runtime'
    module = load(reference, name)
    parent.runtime = module
    if rec(module.__file__) != reference or sys.modules.get(name) is not module:
        raise ValueError('Frozen original transport exact module identity differs')
    return module


def freeze(path, value):
    from course_experiments.contributions import write_immutable
    write_immutable(Path(path), value)
    return value


def state(root, status, **extra):
    from course_experiments.evaluation.run_repaired_full import atomic
    atomic(Path(root)/'state.json', dict(status=status, pid=os.getpid(),
        observed_unix=time.time(), **extra))


def quota_retry_ids(questions, records, error_reader):
    selected = []
    for question in questions:
        if question['target_book_id'] != BOOK:
            raise ValueError('Quota resume is restricted to the verified OS textbook')
        qid = question['question_id']
        row = records.get(qid, {})
        if row.get('status') == 'failed' and QUOTA_ERROR in error_reader(row['error_record']):
            selected.append(qid)
    return selected


def migrate_checkpoints(runner, old_root, new_root, old_identity, new_identity,
                        selected_ids, archive_root):
    """重新封装校验身份，不覆盖旧记录；不复用被截断的QA检查点。"""
    old_root, new_root, archive_root = map(Path, (old_root, new_root, archive_root))
    counts = dict(completed_records_reused=0, retry_failures_archived=0,
                  stage_checkpoints_reused=0, incomplete_retry_stages_archived=0)
    names = {'record.json', 'native_qa.json', 'retrieval.json', 'judge.json', 'judge_raw.json'}
    for split in ('development', 'confirmation'):
        splitroot = old_root/split
        if not splitroot.exists():
            continue
        for folder in sorted(p for p in splitroot.iterdir() if p.is_dir()):
            record_path = folder/'record.json'
            row = runner._read_checkpoint(record_path, old_identity) if record_path.exists() else None
            selected = bool(row and row.get('question_id') in selected_ids and split == 'confirmation')
            if row and row.get('status') == 'completed':
                counts['completed_records_reused'] += 1
            if selected:
                freeze(archive_root/split/folder.name/'prior_failed_record.json',
                    dict(original=rec(record_path), original_value=row))
                counts['retry_failures_archived'] += 1
            for source in sorted(p for p in folder.iterdir() if p.name in names):
                value = runner._read_checkpoint(source, old_identity)
                if selected and source.name == 'native_qa.json' and not value.get('qa_complete'):
                    freeze(archive_root/split/folder.name/source.name,
                        dict(original=rec(source), original_value=value))
                    counts['incomplete_retry_stages_archived'] += 1
                    continue
                destination = new_root/split/folder.name/source.name
                body = {k:v for k,v in value.items() if k not in {'job_identity', 'checkpoint_digest'}}
                body['reused_checkpoint'] = rec(source)
                expected = dict(body, job_identity=new_identity)
                expected['checkpoint_digest'] = digest(expected)
                if destination.exists():
                    observed = runner._read_checkpoint(destination, new_identity)
                    # 已恢复的目标题不能因再次准备而被旧失败覆盖。
                    if selected and source.name == 'record.json' and (
                            observed.get('status') == 'completed' or (
                                observed.get('supplement_retry_attempted')
                                and observed.get('prior_failed_checkpoint') == rec(source))):
                        continue
                    if observed != expected:
                        raise ValueError('Migrated checkpoint differs from pinned ancestor')
                else:
                    runner._checkpoint(destination, body, new_identity)
                counts['stage_checkpoints_reused'] += 1
    return counts


class QuotaBudgetExhausted(RuntimeError):
    pass


class QuotaHTTPBudget:
    """同一stage前缀累积计数所有实际HTTP及重试，不重置旧总账。"""
    def __init__(self, provider, root, cap=CAP):
        from filelock import FileLock
        if type(cap) is not int or not 0 < cap <= CAP:
            raise ValueError('LightRAG quota-resume actual-HTTP cap must be at most 100')
        self.provider, self.root, self.cap = provider, Path(root), cap
        self.lock = FileLock(str(provider.root/'supplement_light_OSB_quota_20261028.lock'), timeout=600)
        self.exhausted = False

    def count(self):
        with closing(sqlite3.connect(self.provider.database, timeout=30)) as db:
            return db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,?)=?",
                (len(PREFIX), PREFIX)).fetchone()[0]

    def __enter__(self):
        freeze(self.root/'local_cap.json', dict(version=VERSION, prefix=PREFIX, cap=self.cap,
            ledger=str(self.provider.database), inherited_spend=True, budget_reset=False))
        self.original = self.provider.runtime._record
        def record(key, stage, status, value):
            if status == 'request_started':
                if not stage.startswith(PREFIX):
                    raise ValueError('LightRAG quota resume attempted an unscoped request')
                with self.lock:
                    if self.count() >= self.cap:
                        self.exhausted = True
                        raise QuotaBudgetExhausted('LightRAG quota-resume local actual-HTTP cap exhausted')
                    return self.original(key, stage, status, value)
            return self.original(key, stage, status, value)
        self.provider.runtime._record = record
        return self

    def __exit__(self, *args):
        self.provider.runtime._record = self.original


def _frozen_runner(contract):
    """先选冻结包及原传输层，避免不慎导入当前工作区版本。"""
    for reference in contract['code'].values():
        if isinstance(reference, dict) and set(reference) == {'path', 'sha256'}:
            if rec(reference['path']) != reference:
                raise ValueError('Original native implementation source changed')
    app = Path(contract['code']['run_native_full.py']['path']).parents[2]
    runtime_app = Path(contract['code']['model_runtime.py']['path']).parents[2]
    sys.path.insert(0, str(runtime_app)); sys.path.insert(0, str(app))
    package = importlib.import_module('course_experiments')
    if Path(package.__file__).resolve().parent != app/'course_experiments':
        raise ValueError('Run quota-resume as a standalone script to select the frozen package')
    original_runtime = load_frozen_runtime(contract['code']['model_runtime.py'])
    runner = load(contract['code']['run_native_full.py'],
        'course_experiments.evaluation.quota_resume_original_runner')
    if rec(original_runtime.__file__) != contract['code']['model_runtime.py']:
        raise ValueError('Frozen original transport was not selected')
    return runner


def _error_text(reference):
    if rec(reference['path']) != reference:
        raise ValueError('Original native failure log changed')
    return Path(reference['path']).read_text('utf-8')


def _semantic_assets(index):
    """查询缓存可写；图、向量及原始文本索引不可改写。"""
    exact = {'native_adapter_manifest.json', 'native_index_complete.json',
        'kv_store_doc_status.json', 'kv_store_text_chunks.json', 'kv_store_full_docs.json',
        'kv_store_entity_chunks.json', 'kv_store_relation_chunks.json',
        'kv_store_full_entities.json', 'kv_store_full_relations.json'}
    return {str(p.relative_to(index)):rec(p)['sha256'] for p in sorted((index/'storage').rglob('*'))
        if p.is_file() and (p.name in exact or p.suffix == '.graphml' or p.name.startswith('vdb_'))}


def prepare(config):
    if config.get('version') != VERSION:
        raise ValueError('Explicit native quota-resume configuration required')
    original = checked(config['source_contract'])
    old_id = original['job_identity']
    ancestor = {k:v for k,v in original.items() if k != 'job_identity'}
    if (old_id != ORIGINAL_ID or digest(ancestor) != old_id or original['book'] != BOOK
            or original['method'] != 'lightrag' or original['index_identity'] != INDEX_ID):
        raise ValueError('Only the verified ten OS-B quota failures may be resumed')
    runner = _frozen_runner(original)
    run_root, output = Path(config['run_root']), Path(config['output_root'])
    old_root = run_root/'native_full'/'lightrag'/BOOK/old_id[:16]
    if Path(config['source_contract']['path']).resolve() != (old_root/'contract.json').resolve():
        raise ValueError('Original result-root ancestry differs')
    index = run_root/'native_indices'/'lightrag'/BOOK/INDEX_ID[:16]
    if any(Path(config[key]['path']).resolve() != (index/filename).resolve() for key,filename in (
            ('source_index_contract','contract.json'), ('source_index_receipt','index_receipt.json'))):
        raise ValueError('Existing ready native index must be reused in place')
    index_contract = checked(config['source_index_contract'])
    body = {k:v for k,v in index_contract.items() if k != 'index_identity'}
    if index_contract.get('index_identity') != INDEX_ID or digest(body) != INDEX_ID:
        raise ValueError('Current ready LightRAG index identity differs')
    checked(config['source_index_receipt'])
    index_receipt = runner._read_checkpoint(index/'index_receipt.json', INDEX_ID)['native_index']
    manifests = list((index/'storage').glob('*/native_adapter_manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Expected one ready native OS-B source manifest')
    manifest = read(manifests[0]); complete = read(manifests[0].parent/'native_index_complete.json')
    if (manifest['target_book_id'] != BOOK or manifest['source_map'] != index_receipt['source_map']
            or complete.get('manifest_digest') != digest(manifest) or index_receipt.get('status') != 'processed'):
        raise ValueError('Existing native index is not completely processed or provenance differs')
    units, structure = checked(original['source_units']), checked(original['structure'])
    passages, audit = runner.native_passages(units['rows'], structure, BOOK)
    if audit != original['corpus_audit']:
        raise ValueError('Native corpus restoration differs from original')
    if runner.source_identity(run_root, 'lightrag') != original['source']:
        raise ValueError('Official LightRAG commit/source identity changed')
    datasets = {split:checked(original['datasets'][split]) for split in ('development','confirmation')}
    if len(datasets['confirmation']['questions']) != 37 or len(datasets['development']['questions']) != 4:
        raise ValueError('Original full frozen question denominators changed')
    records = {split:{} for split in datasets}
    for split, dataset in datasets.items():
        for q in dataset['questions']:
            if q['target_book_id'] != BOOK or not set(q['evidence_ids']) <= set(structure['evidence']):
                raise ValueError('Frozen native question/source scope differs')
            path = old_root/split/digest(q['question_id'])/'record.json'
            if path.exists():
                row = runner._read_checkpoint(path, old_id)
                if row['question_id'] != q['question_id'] or row['question_type'] != q['type']:
                    raise ValueError('Original question checkpoint identity differs')
                records[split][q['question_id']] = row
    selected = quota_retry_ids(datasets['confirmation']['questions'], records['confirmation'], _error_text)
    if len(selected) != 10 or sum(r.get('status') == 'completed' for r in records['confirmation'].values()) != 27:
        raise ValueError('Expected the exact original ten quota failures and 27 completed confirmations')
    if any(records['confirmation'][q]['status'] != 'failed' for q in selected):
        raise ValueError('Only prior failed confirmation records may be selected')
    approval = checked(config['authorization'])
    if rec(config['provider_factory']['path']) != config['provider_factory']:
        raise ValueError('Pinned supplement provider factory changed')
    provider_ref = approval['provider']
    if rec(provider_ref['path']) != provider_ref or PREFIX not in approval['allowed_stage_prefixes']:
        raise ValueError('New supplement authorization must pin provider and exact LightRAG scope')
    cap = config.get('actual_http_cap', CAP)
    if type(cap) is not int or not 0 < cap <= CAP:
        raise ValueError('Native LightRAG supplementary local cap must be <=100 actual HTTP starts')
    contract = copy.deepcopy(ancestor)
    contract['supplement_quota_resume'] = dict(version=VERSION, original_contract=config['source_contract'],
        original_index_contract=config['source_index_contract'], original_index_receipt=config['source_index_receipt'],
        authorization=config['authorization'], provider_factory=config['provider_factory'], provider=provider_ref,
        worker=rec(__file__), selected_question_ids=selected, selected_reason=QUOTA_ERROR,
        stage_prefix=PREFIX, local_actual_http_cap=cap, historical_spend_retained=True,
        no_reindex=True, model_prompt_algorithm_unchanged=True, no_other_questions_retried=True)
    identity = digest(contract); root = run_root/'native_full'/'lightrag'/BOOK/identity[:16]
    freeze(root/'contract.json', dict(contract, job_identity=identity))
    migration = migrate_checkpoints(runner, old_root, root, old_id, identity, set(selected), output/'preserved_failures')
    construction = checked(original['construction'])
    runtime_config = checked(original['runtime_config_record'])
    if runtime_config != original['runtime_config']:
        raise ValueError('Original runtime configuration changed')
    plan = dict(method='lightrag', book=BOOK, passages=passages, passage_audit=audit, structure=structure,
        datasets=datasets, source=Path(original['source']['path']), contract=contract, index_contract=body,
        cohort_memberships=original['cohort_memberships'], allowed_books=sorted(original['global_question_counts']),
        construction=construction)
    prepared = dict(version=VERSION, status='prepared_NO_API', original_root=str(old_root), result_root=str(root),
        job_identity=identity, index_identity=INDEX_ID, index_root=str(index), selected_question_ids=selected,
        selected_count=len(selected), confirmation_denominator=37, model_calls=0,
        local_actual_http_cap=cap, original_source_window_failure_other_book_untouched=True,
        migration=migration, semantic_index_assets=_semantic_assets(index))
    freeze(output/'preflight.json', prepared)
    return prepared, plan, runner, index_receipt


def _read_records(runner, plan, root, identity):
    result = {split:{} for split in plan['datasets']}
    for split, dataset in plan['datasets'].items():
        for q in dataset['questions']:
            path = root/split/digest(q['question_id'])/'record.json'
            if path.exists():
                result[split][q['question_id']] = runner._read_checkpoint(path, identity)
    return result


async def execute_selected(config, prepared, plan, runner, index_receipt, provider, budget):
    """直接打开完成的索引；从不调用run_book、_index或index_book。"""
    from filelock import FileLock
    root, index = Path(prepared['result_root']), Path(prepared['index_root'])
    identity = prepared['job_identity']; selected = set(prepared['selected_question_ids'])
    tracker = runner.ReceiptTracker(provider); adapter = None
    original_executor = tracker.original
    def scoped_executor(endpoint, body, stage):
        return original_executor(endpoint, body, PREFIX+stage)
    tracker.original = scoped_executor
    try:
        with FileLock(str(index/'operation.lock'), timeout=0):
            adapter = await runner.open_adapter(plan, index, provider)
            for question in plan['datasets']['confirmation']['questions']:
                qid = question['question_id']
                if qid not in selected:
                    continue
                qroot = root/'confirmation'/digest(qid); path = qroot/'record.json'
                previous = runner._read_checkpoint(path, identity)
                if previous['status'] == 'completed':
                    continue
                # 除首次明确核实的配额中断外，后续真实方法失败不自动反复重试。
                if previous.get('supplement_retry_attempted'):
                    continue
                provider.raise_if_fatal()
                state(root, 'running_selected_quota_retry', question_id=qid)
                try:
                    with tracker.scope(qroot/'requests.jsonl'):
                        native = await runner.native_question(adapter, plan, question, qroot, identity, index_receipt)
                        provider.raise_if_fatal()
                        judgment = runner.blind_judge(provider, question, native, qroot, identity)
                    scores = dict(runner.score_answer(native['answer'], question['answer']),
                        evidence_recall=runner.evidence_recall(question, native['provided_context']),
                        citation_count=None, citation_correctness=None)
                    row = dict(status='completed', question_id=qid, question_type=question['type'], split='confirmation',
                        target_book_id=BOOK, answer=native['answer'], provided_context=native['provided_context'],
                        native_result=native['native_result'], finish_metadata=native['finish_metadata'],
                        machine_judge=judgment, scores=scores, human_audited=False,
                        receipts_path=str(qroot/'requests.jsonl'), completed_unix=time.time(),
                        supplement_retry_attempted=True, prior_failed_checkpoint=previous['reused_checkpoint'])
                except Exception as exc:
                    if budget.exhausted:
                        # 配额停止不是方法失败；原失败仍在，下一轮可以显式续跑。
                        raise QuotaBudgetExhausted('LightRAG local quota exhausted; original failed record retained') from exc
                    provider.raise_if_fatal()
                    if any(marker in str(exc) for marker in ('allocation exhausted',
                            'budget exhausted', 'deadline exhausted', 'account_or_permission',
                            'New supplement 24-hour deadline')):
                        raise RuntimeError('Inherited provider stop; original failed record retained') from exc
                    row = dict(status='failed', question_id=qid, question_type=question['type'], split='confirmation',
                        target_book_id=BOOK, **runner._error(qroot, exc), human_audited=False,
                        supplement_retry_attempted=True, prior_failed_checkpoint=previous['reused_checkpoint'])
                runner._checkpoint(path, row, identity, immutable=False)
            provider.raise_if_fatal()
    finally:
        try:
            if adapter is not None:
                await runner._await(adapter.close())
        finally:
            tracker.close()
    if _semantic_assets(index) != prepared['semantic_index_assets']:
        raise ValueError('Native semantic index assets changed during query-only resume')
    records = _read_records(runner, plan, root, identity)
    result = runner.summary_for(plan, records, root, identity, provider.statistics())
    runner.atomic(root/'result.json', result)
    if result['status'] == 'completed':
        freeze(root/'complete.json', result)
    state(root, result['status'], selected_retry_count=len(selected), local_actual_http=budget.count(),
        fixed_confirmation_denominator=37)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(); config = read(args.config)
    prepared, plan, runner, receipt = prepare(config)
    if not args.run:
        print(json.dumps(prepared, ensure_ascii=False)); return 0
    from filelock import FileLock
    from course_experiments.evaluation.protocol import runtime_for
    authorization = checked(config['authorization'])
    provider_module = load(authorization['provider'], 'course_experiments.evaluation.supplement_quota_native_provider')
    factory = load(config['provider_factory'], 'supplement_quota_provider_factory')
    Provider = factory.provider_type(provider_module, config['authorization'])
    if hasattr(factory, 'retry_writer'):
        original_write = Provider._write_transaction
        def write(provider, operation_fn, *, operation, request_id=None):
            return factory.retry_writer(provider,
                lambda fn, **kw: original_write(provider, fn, **kw), operation_fn,
                operation=operation, request_id=request_id)
        Provider._write_transaction = write
    runner.credentials()
    runtime = runtime_for(plan['construction'],
        Path(plan['contract']['transaction_writer_recovery']['original_runtime_cache']))
    provider = Provider(runtime, Path(config['run_root'])/'formal_provider', **runner.LIMITS)
    try:
        # 沿用原查询回调；本次不抽取或写入索引，不需要旧提取长度恢复回调。
        if runtime.config != plan['contract']['runtime_config']:
            raise ValueError('Quota resume would change provider request/cache identity')
        with FileLock(str(Path(config['output_root'])/'execution.lock'), timeout=0):
            with QuotaHTTPBudget(provider, Path(config['output_root'])/'budget', config.get('actual_http_cap', CAP)) as budget:
                result = asyncio.run(execute_selected(config, prepared, plan, runner, receipt, provider, budget))
        print(json.dumps(dict(result_root=prepared['result_root'], status=result['status'],
            confirmation=result['splits']['confirmation']), ensure_ascii=False))
        return 0 if result['status'] == 'completed' else 2
    except Exception as exc:
        output = Path(config['output_root']); output.mkdir(parents=True, exist_ok=True)
        (output/'resume_error.log').write_text(traceback.format_exc(), encoding='utf-8')
        state(Path(prepared['result_root']), 'halted_preserved', error_type=type(exc).__name__)
        raise
    finally:
        provider.close()


if __name__ == '__main__':
    raise SystemExit(main())
