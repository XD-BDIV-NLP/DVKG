"""Tree-KG论文受控重建的教材构图入口；原文/位置/收据仅留原服务器。

gate只做有限原文构图接入，full覆盖指定教材组合；二者都不宣称RAG/KGE已完成。
本模块不读取问题或答案，也不初始化新的预算账本。
"""
from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
import copy
import json
import math
from pathlib import Path
import sqlite3
import time
import traceback

from filelock import FileLock

from ..baselines.tree_kg import TreeKG, VERSION
from ..contributions import read_contribution, write_immutable
from ..manifest import digest, file_record, read_document
from .native_corpus import native_passages
from .native_provider import AuditedNativeProvider
from .protocol import check_runtime, runtime_for
from .run_native_full import LIMITS, ReceiptTracker
from .run_native_rag import PARENT, checked
from .run_repaired_full import atomic, credentials


AUTHORIZED_BOOKS = {
    'book_49bb8d31ddb9':'computer_organization', 'book_25f718562685':'computer_organization',
    'book_efaf8522b95d':'computer_organization', 'book_bef9d155e6d8':'computer_organization',
    'book_0b8d87cb0667':'operating_systems', 'book_d96f472c8179':'operating_systems',
    'book_dd8183b4cb60':'computer_networks', 'book_00849acee112':'computer_networks'}
COHORT_BOOKS = {
    'compAB':['book_49bb8d31ddb9','book_25f718562685'],
    'compABCD':['book_49bb8d31ddb9','book_25f718562685','book_efaf8522b95d','book_bef9d155e6d8'],
    'osAB':['book_0b8d87cb0667','book_d96f472c8179'],
    'netAB':['book_dd8183b4cb60','book_00849acee112']}
POLICY = dict(version='tree_kg_faithful_construction_source_chunk_v1',
    reconstruction=VERSION, unmodified_author_code=False,
    input='losslessly_restored_original_source_chunks_and_original_TOC',
    model_input='complete selected source paragraphs grouped under original TOC for bottom-up summaries',
    answers_or_question_support_used=False, gate_sampling='stable_source_only_digest_first_k_original_chunks',
    source_positions='exact_unicode_character_offsets_in_restored_original_chunk_plus_unchanged_original_source_fields',
    provider='existing_shared_formal_provider_only_no_budget_or_start_reset',
    kge='NA_until_independently_frozen_concept_type_and_relation_alignment',
    server_only=True, full_benchmark_completed=False)


def _signed(value):
    value=copy.deepcopy(value);value['artifact_id']=digest(value);return value


def require_formal_provider(run_root):
    """只读要求现有账本完整，绝不替缺失账本创建新起点。"""
    root=Path(run_root)/'formal_provider'
    if not (root/'limits.json').is_file() or not (root/'audit.sqlite').is_file():
        raise ValueError('Existing shared formal_provider limits and audit ledger required; no initialization/reset')
    value=read_document(root/'limits.json')
    started=value.get('started_unix')
    if value.get('limits')!=LIMITS or type(started) not in (int,float) or not math.isfinite(started):
        raise ValueError('Frozen formal provider30000HTTP/48h/4slots and original start required')
    try:
        with closing(sqlite3.connect('file:'+ (root/'audit.sqlite').resolve().as_posix()+'?mode=ro',uri=True,timeout=30)) as db:
            db.execute('SELECT count(*) FROM events').fetchone()
            db.execute('SELECT count(*) FROM receipts').fetchone()
    except sqlite3.Error as error:
        raise ValueError('Existing shared formal audit ledger is invalid; never reconstruct/reset it') from error
    return value


def _positioned_structure(original, passages, *, full):
    selected={eid for passage in passages for eid in passage['evidence_ids']}
    nodes=original['nodes'];book=original['book_id'];keep=set()
    for eid in selected:
        nid=book+'/evidence/'+eid
        if nid not in nodes:
            raise ValueError('Original evidence node missing')
        while nid:
            if nid in keep:
                break
            keep.add(nid);nid=nodes[nid].get('parent')
    value=copy.deepcopy(original)
    value['nodes']={nid:n for nid,n in value['nodes'].items() if nid in keep}
    value['edges']=[edge for edge in value['edges'] if edge[0] in keep and edge[1] in keep]
    value['evidence']={eid:row for eid,row in value['evidence'].items() if eid in selected}
    for index,passage in enumerate(passages):
        for span in passage['evidence_spans']:
            row=value['evidence'][span['evidence_id']]
            if passage['text'][span['start']:span['end']]!=row['text']:
                raise ValueError('Restored quote span does not equal original evidence')
            row['original_source_position']=dict(span,book_id=book,chunk_id=passage['chunk_id'],
                passage_index=index,position_unit='unicode_character',passage_digest=digest(passage))
    if full and selected!=set(original['evidence']):
        raise ValueError('Full Tree-KG structure lost original evidence')
    value['original_structure_id']=original['structure_id']
    value['input_scope']='full' if full else 'gate_partial_source_only'
    value['source_chunks_digest']=digest(passages)
    value['structure_id']=digest({k:v for k,v in value.items() if k!='structure_id'})
    return value


def prepare_plan(run_root, *, scope, cohort='all', book=None, sample_size=12, parent=PARENT):
    if scope not in ('gate','full') or cohort not in ('all',*COHORT_BOOKS):
        raise ValueError('Explicit authorized scope and cohort required')
    if scope=='full' and book is not None:
        raise ValueError('Full construction is a complete cohort, not one selected book')
    if type(sample_size) is not int or not 1<=sample_size<=12:
        raise ValueError('Gate sample must be between1 and12 original source chunks')
    parent=Path(parent);asset_record=file_record(parent/'assets/result.json');assets=checked(asset_record)
    if set(assets['unit_assets'])!=set(AUTHORIZED_BOOKS) or set(assets['structures'])!=set(AUTHORIZED_BOOKS):
        raise ValueError('Exact authorized eight-book frozen corpus required')
    for key,expected in COHORT_BOOKS.items():
        if key not in assets['cohorts'] or set(assets['cohorts'][key]['active_books'])!=set(expected):
            raise ValueError('Frozen textbook cohort differs from authorized AB/ABCD/OS/NET scope')
    cohorts=list(COHORT_BOOKS) if cohort=='all' else [cohort]
    if scope=='gate':
        book=book or COHORT_BOOKS[cohorts[0]][0]
        choices=[c for c in cohorts if book in COHORT_BOOKS[c]]
        if book not in AUTHORIZED_BOOKS or not choices:
            raise ValueError('Gate book outside requested authorized cohort')
        cohorts=choices[:1]
        selections={cohorts[0]:[book]}
    else:
        selections={key:COHORT_BOOKS[key] for key in cohorts}
    inputs={};configs={};construction={}
    for key in cohorts:
        record=assets['cohorts'][key]['construction'];construction[key]=checked(record)
        cfg_record=construction[key]['files']['assets']['runtime_config'];configs[key]=checked(cfg_record)
        check_runtime(configs[key])
    if len({digest(config) for config in configs.values()})!=1:
        raise ValueError('All selected courses must use the same frozen official runtime')
    for candidate in sorted({b for values in selections.values() for b in values}):
        units=checked(assets['unit_assets'][candidate]);structure=checked(assets['structures'][candidate])
        TreeKG._validate_structures({candidate:structure})
        ids=[eid for row in units['rows'] for eid in row['evidence_ids']]
        if set(ids)!=set(structure['evidence']) or len(ids)!=len(set(ids)):
            raise ValueError('Complete source evidence coverage is required before Tree-KG sampling')
        passages,audit=native_passages(units['rows'],structure,candidate)
        original_count=len(passages)
        if scope=='gate':
            passages=sorted(passages,key=lambda row:digest([candidate,row['chunk_id'],row['evidence_ids'],row['text']]))[:sample_size]
        audit=dict(audit,all_original_book_evidence_covered={eid for p in passages for eid in p['evidence_ids']}==set(structure['evidence']),
            original_passage_count=original_count,selected_passage_count=len(passages),
            selected_evidence_count=sum(len(p['evidence_ids']) for p in passages),
            selected_passages_digest=digest(passages),selection_uses_answers=False)
        inputs[candidate]=dict(passages=passages,audit=audit,
            structure=_positioned_structure(structure,passages,full=scope=='full'))
    code_paths=[Path(__file__),Path(__file__).parents[1]/'baselines/tree_kg.py',
        Path(__file__).with_name('native_corpus.py'),Path(__file__).with_name('native_provider.py'),
        Path(__file__).with_name('protocol.py'),Path(__file__).with_name('run_native_full.py'),
        Path(__file__).with_name('run_native_rag.py'),Path(__file__).with_name('run_repaired_full.py'),
        Path(__file__).parents[1]/'manifest.py',Path(__file__).parents[1]/'contributions.py']
    from domain_ckg_workspace.ckg_incremental import runtime as original_runtime
    code_paths.append(Path(original_runtime.__file__))
    contract=dict(version=1,scope=scope,requested_gate_source_chunk_limit=sample_size if scope=='gate' else None,
        cohorts=selections,unique_book_count=len(inputs),
        uses_questions_or_answers=False,asset_manifest=asset_record,policy=POLICY,code=[file_record(p) for p in code_paths],
        shared_provider_limits=LIMITS,shared_provider_root=str(Path(run_root)/'formal_provider'),
        construction_records={c:assets['cohorts'][c]['construction'] for c in cohorts},runtime_config=next(iter(configs.values())),
        books={b:dict(source_units=assets['unit_assets'][b],source_structure=assets['structures'][b],
            prepared_structure_id=v['structure']['structure_id'],audit=v['audit']) for b,v in inputs.items()})
    return dict(contract=contract,book_inputs=inputs,
        jobs={key:dict(course_id=AUTHORIZED_BOOKS[values[0]],book_ids=values,construction=construction[key]) for key,values in selections.items()})


def require_matching_gate(plan, run_root):
    """真实构图门禁的元数据匹配；fixture只在测试临时目录生成。"""
    for path in sorted((Path(run_root)/'tree_kg/gate').glob('*/complete.json')):
        gate=read_contribution(path);contract=read_contribution(path.with_name('contract.json'))
        if gate.get('status')!='complete' or gate.get('scope')!='gate' or gate.get('version')!=VERSION:
            continue
        if gate.get('identity')!=digest({k:v for k,v in contract.items() if k!='artifact_id'}):
            raise ValueError('Tree gate identity changed')
        if any(contract.get(k)!=plan['contract'].get(k) for k in ('code','policy','runtime_config','asset_manifest')):
            continue
        if not contract['books'] or not set(contract['books'])<=set(plan['contract']['books']):
            continue
        if any(any(record[k]!=plan['contract']['books'][book][k] for k in ('source_units','source_structure'))
               for book,record in contract['books'].items()):
            continue
        for row in gate['jobs'].values():
            if file_record(row['artifact']['path'])!=row['artifact']:
                raise ValueError('Tree gate artifact changed')
            read_contribution(row['artifact']['path'])
        return file_record(path)
    raise ValueError('A completed same-code/source/model Tree-KG gate is required before full construction')


def run_plan(plan, run_root, provider, *, tracker=None):
    """注入调用；provider由调用者持有，关闭责任不属于此函数。"""
    identity=digest(plan['contract']);scope=plan['contract']['scope']
    root=Path(run_root)/'tree_kg'/scope/identity[:16];root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'operation.lock'),timeout=172800):
        provider.raise_if_fatal()
        if provider.runtime.config!=plan['contract']['runtime_config']:
            raise ValueError('Injected provider differs from frozen Tree-KG runtime')
        if scope=='full':
            write_immutable(root/'gate_prerequisite.json',require_matching_gate(plan,run_root))
        write_immutable(root/'contract.json',_signed(plan['contract']))
        if (root/'complete.json').exists():
            summary=read_contribution(root/'complete.json')
            for row in summary['jobs'].values():
                if file_record(row['artifact']['path'])!=row['artifact']:
                    raise ValueError('Completed Tree-KG source artifact changed')
                read_contribution(row['artifact']['path'])
            if file_record(summary['source_corpus_record']['path'])!=summary['source_corpus_record']:
                raise ValueError('Completed Tree-KG source corpus changed')
            return summary
        # 仅在执行服务器保存完整原文和quote位置；stdout仅提供元数据。
        write_immutable(root/'source_corpus.json',_signed(dict(books={b:dict(passages=v['passages'],audit=v['audit'])
            for b,v in plan['book_inputs'].items()},server_only=True)))
        jobs={}
        try:
            for cohort,job in plan['jobs'].items():
                provider.raise_if_fatal()
                job_root=root/cohort;job_root.mkdir(exist_ok=True)
                atomic(root/'state.json',dict(status='running',phase='tree_kg_construction',cohort=cohort,identity=identity))
                before=provider.statistics()
                with tracker.scope(job_root/'requests.jsonl') if tracker else nullcontext():
                    result=TreeKG(provider,job_root/'builder').build_course(job['course_id'],
                        {b:plan['book_inputs'][b]['structure'] for b in job['book_ids']})
                provider.raise_if_fatal()
                artifact=file_record(job_root/'builder/artifact.json')
                expected={b+'/evidence/'+eid for b in job['book_ids'] for eid in plan['book_inputs'][b]['structure']['evidence']}
                if {row['id'] for row in result['evidence']}!=expected:
                    raise ValueError('Completed reconstruction lost selected textbook evidence')
                jobs[cohort]=dict(status='complete',course_id=job['course_id'],book_ids=job['book_ids'],artifact=artifact,
                    node_count=len(result['native_graph']['nodes']),edge_count=len(result['native_graph']['edges']),
                    evidence_count=len(expected),all_selected_source_evidence_covered=True,
                    capabilities=result['semantic_view']['capabilities'],capability_reasons=result['semantic_view']['capability_reasons'],
                    unmodified_native_reproduction=False,provider_before=before,provider_after=provider.statistics(),
                    statistics_scope='shared_formal_provider_global_not_tree_cost',requests_path=str(job_root/'requests.jsonl'))
            provider.raise_if_fatal()
            summary=dict(status='complete',scope=scope,identity=identity,method='tree_kg',version=VERSION,
                faithful_paper_reconstruction=True,unmodified_native_reproduction=False,jobs=jobs,
                unique_book_count=plan['contract']['unique_book_count'],
                full_eight_book_construction_completed=scope=='full' and plan['contract']['unique_book_count']==8,
                full_benchmark_completed=False,rag_completed=False,kge_completed=False,
                gate_is_not_quality_result=scope=='gate',raw_text_exported=False,uses_questions_or_answers=False,
                source_corpus_record=file_record(root/'source_corpus.json'),completed_unix=time.time())
            summary=write_immutable(root/'complete.json',_signed(summary))
            atomic(root/'state.json',dict(status='complete',identity=identity,full_benchmark_completed=False))
            return summary
        except Exception as error:
            error_path=root/('error_'+str(time.time_ns())+'.log');error_path.write_text(traceback.format_exc(),'utf-8')
            atomic(root/'state.json',dict(status='failed',error_type=type(error).__name__,identity=identity,error_record=file_record(error_path)))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--scope',choices=['gate','full'],required=True)
    parser.add_argument('--cohort',choices=['all',*COHORT_BOOKS],default='all')
    parser.add_argument('--book')
    parser.add_argument('--sample-size',type=int,default=12)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    plan=prepare_plan(args.run_root,scope=args.scope,cohort=args.cohort,book=args.book,
        sample_size=args.sample_size,parent=args.parent)
    if args.check_only:
        print(json.dumps(dict(status='prepared_only_no_model_calls',identity=digest(plan['contract']),scope=args.scope,
            cohorts=list(plan['jobs']),book_count=len(plan['book_inputs']),
            original_source_chunks=sum(len(v['passages']) for v in plan['book_inputs'].values()),
            raw_text_exported=False,uses_questions_or_answers=False),ensure_ascii=False))
        return 0
    require_formal_provider(args.run_root)
    if args.scope=='full':require_matching_gate(plan,args.run_root)
    credentials()
    first=next(iter(plan['jobs'].values()))
    transport=args.run_root/'tree_kg'/'transport'/digest(plan['contract'])
    provider=AuditedNativeProvider(runtime_for(first['construction'],transport),args.run_root/'formal_provider',**LIMITS)
    tracker=None
    try:
        provider.raise_if_fatal();tracker=ReceiptTracker(provider)
        result=run_plan(plan,args.run_root,provider,tracker=tracker)
        provider.raise_if_fatal()
        print(json.dumps(result,ensure_ascii=False))
        return 0
    finally:
        if tracker is not None:tracker.close()
        provider.close()


if __name__=='__main__':
    raise SystemExit(main())
