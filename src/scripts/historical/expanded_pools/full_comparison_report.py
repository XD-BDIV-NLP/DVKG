"""只读审计原353题全量比较；不完整或失败无原始依据时不发布新版F1。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import time


QA=Path(__file__).resolve().parent
REPO=QA.parents[1]
CANONICAL_SHA='43cb0cb57a2288f1b39ebde8d77575c76b4b847764d279392f0e4b02922e1fcb'
RUNTIME_DIGEST='5b6f4bc24861a4a303c48552d514678a3e0763d4b8e74dc4dd65b716557ff67e'
EXPECTED_COUNTS={'book_25f718562685':100,'book_49bb8d31ddb9':100,'book_0b8d87cb0667':40,
    'book_d96f472c8179':37,'book_00849acee112':38,'book_dd8183b4cb60':38}
COURSES={b:course for books,course in [(['book_25f718562685','book_49bb8d31ddb9'],'computer_organization'),
    (['book_0b8d87cb0667','book_d96f472c8179'],'operating_systems'),
    (['book_00849acee112','book_dd8183b4cb60'],'computer_networks')]for b in books}
SCOPES=('same_subject_pool','mixed_six_book_pool')
BASELINES=('K12-KGraph','GraphRAG','LightRAG','HippoRAG')


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def record(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    text=value if isinstance(value,str)else json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    tmp.write_text(text,encoding='utf-8');tmp.replace(path)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def frozen_module(reference,fallback,name):
    for candidate in (Path(reference['path']),Path(fallback)):
        if candidate.is_file()and record(candidate)['sha256']==reference['sha256']:
            if 'course_experiments'in candidate.parts:sys.path.insert(0,str(candidate.parents[2]))
            return load(candidate,name)
    raise ValueError('Frozen pure evaluation module is unavailable or changed: '+name)


def dependencies(contract):
    # 只导入纯函数，绝不调用任何初始化器或教材上下文加载器。
    core=frozen_module(contract['development_answer_core'],QA/'development_answer_worker.py','full_report_pure_core')
    parser=frozen_module(contract['dependencies']['joint_answer_module'],REPO/'course_experiments/evaluation/run_joint_pool_answers.py',
        'course_experiments.evaluation.full_report_pure_parser')
    scorer=frozen_module(contract['dependencies']['fixed_f1'],REPO/'course_experiments/evaluation/rag.py',
        'course_experiments.evaluation.full_report_pure_scorer')
    return dict(core=core,parser=parser,score_answer=scorer.score_answer,runtime_digest=RUNTIME_DIGEST,
        code_refs=dict(core=contract['development_answer_core'],parser=contract['dependencies']['joint_answer_module'],scorer=contract['dependencies']['fixed_f1']))


def reference_path(reference,root,declared_root):
    direct=Path(reference['path'])
    if direct.is_file():path=direct
    else:
        original=PurePosixPath(reference['path'].replace('\\','/'));base=PurePosixPath(declared_root.replace('\\','/'))
        try:relative=original.relative_to(base)
        except ValueError:raise FileNotFoundError('External original reference not collected: '+str(original))from None
        path=Path(root).joinpath(*relative.parts)
    if record(path)['sha256']!=reference['sha256']:raise ValueError('Raw source reference SHA changed')
    return path


def canonical_index(canonical,core):
    if (canonical.get('results_publishable')is not True or canonical.get('question_count')!=353
            or canonical.get('observations')!=3530 or len(canonical.get('rows',[]))!=3530):
        raise ValueError('Complete audited five-method canonical required')
    index={};bridge={};table=[]
    for method in (*BASELINES,'DVKG'):
        entry=dict(method=method)
        for scope in SCOPES:
            rows=[r for r in canonical['rows']if r['method']==method and r['scope']==scope]
            if len(rows)!=353 or any(r['status']!='completed'for r in rows):raise ValueError('Canonical fixed denominator is incomplete')
            counts={b:sum(r['book_id']==b for r in rows)for b in EXPECTED_COUNTS}
            if counts!=EXPECTED_COUNTS:raise ValueError('Canonical textbook counts changed')
            tasks=[];f1={};recall={}
            for row in rows:
                key=method,scope,row['canonical_question_key']
                if key in index:raise ValueError('Duplicate canonical question identity')
                index[key]=row;tid=row['canonical_question_key'];tasks.append(dict(task_key=tid,book_id=row['book_id'],course_id=row['course_id']))
                f1[tid]=dict(status='completed',scores=dict(f1=finite_score(row['f1'])))
                recall[tid]=dict(status='completed',scores=dict(f1=finite_score(row['paragraph_recall_at5'])))
                if method=='DVKG':bridge[scope,row['question_id']]=row
            original=canonical['metrics'][method]['by_scope'][scope]
            if (abs(core.aggregate(tasks,f1)['equal_book_macro_f1']-original['equal_book_macro_f1'])>1e-12
                    or abs(core.aggregate(tasks,recall)['equal_book_macro_f1']-original['equal_book_macro_paragraph_recall_at5'])>1e-12):
                raise ValueError('Canonical macro values differ from frozen rows')
            entry[scope]=dict(f1=original['equal_book_macro_f1'],paragraph_recall_at5=original['equal_book_macro_paragraph_recall_at5'],question_count=353)
        if method in BASELINES:table.append(entry)
    return index,bridge,table


def finite_score(value):
    if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or not 0<=value<=1:
        raise ValueError('Finite evaluation value in [0,1] required')
    return float(value)


def verified_terminal(path,task,contract,deps,root,declared_root):
    core=deps['core'];value=core.read_checkpoint(path)
    if (value.get('result_identity')!=contract['result_identity']or value.get('task_contract_digest')!=digest(task)
            or value.get('request_id')!=task['request_id']or value.get('context_identity')!=task['context_identity']
            or value.get('status')not in {'completed','failed'}):raise ValueError('Terminal task identity or status differs')
    raw_ref=value.get('raw_checkpoint')or value.get('answer_receipt',{}).get('raw_checkpoint')
    if not raw_ref:raise ValueError('Terminal failure or answer lacks an auditable raw response checkpoint')
    raw_path=reference_path(raw_ref,root,declared_root);raw=core.read_checkpoint(raw_path)
    if raw.get('request_id')!=task['request_id']or raw.get('input_identity')!=task['input_identity']:
        raise ValueError('Raw response input identity differs')
    receipt_ref=raw['provider_receipt'];receipt=read(reference_path(receipt_ref,root,declared_root));runtime=receipt['runtime_config']
    response={k:v for k,v in raw['response'].items()if k!='_native_cache_hit'}
    if (digest(runtime)!=contract['runtime_digest']or receipt.get('request_id')!=task['request_id']
            or receipt.get('endpoint')!='/chat/completions'or receipt.get('request')!=task['body']or receipt.get('response')!=response
            or task['request_id']!=digest(dict(endpoint='/chat/completions',body=task['body'],config=runtime))
            or task['input_identity']!=digest(dict(body=task['body'],runtime=runtime))):
        raise ValueError('Provider receipt differs from exact frozen body/runtime/response')
    messages=deps['parser'].answer_messages(dict(question=task['question'],context=task['context']))
    expected_body=dict(model=runtime['llm']['model'],messages=messages,temperature=0,seed=42,enable_thinking=False,max_tokens=2048,response_format=dict(type='json_object'))
    context=json.loads(messages[1]['content'])['context']
    if task['body']!=expected_body or digest(context)!=task['native_context_identity']:
        raise ValueError('Frozen native source prompt and answer controls differ')
    parser_error=None;candidate=audit=None
    try:candidate,audit=deps['parser'].parse_candidate(raw['response'],context)
    except (ValueError,KeyError,TypeError)as exc:parser_error=type(exc).__name__
    if value['status']=='completed':
        if parser_error or candidate!=value.get('candidate'):raise ValueError('Saved answer does not replay from raw response')
        scores=deps['score_answer'](candidate['answer'],task['reference_answer'])
        if (abs(finite_score(value['scores']['f1'])-scores['f1'])>1e-12 or value['scores']['em']!=scores['em']
                or value.get('answer_receipt',{}).get('citation_audit')!=audit):raise ValueError('Saved score or citation audit differs from frozen replay')
    else:
        if parser_error is None:raise ValueError('Failed task is not a reproducible raw model-output parsing/truncation failure')
        if value.get('error_type')!=parser_error:raise ValueError('Failed task type differs from raw parser failure')
        reference_path(value['error_record'],root,declared_root)
        if finite_score(value['scores']['f1'])!=0:raise ValueError('Verified terminal failure must retain fixed-denominator zero')
        scores=dict(f1=0.,em=0)
    return dict(status=value['status'],scores=scores),dict(raw_checkpoint=raw_ref,provider_receipt=receipt_ref,
        error_record=value.get('error_record'),failure_type=parser_error,exact_body_runtime_and_raw_verified=True,
        response_source=receipt.get('response_source'),raw_finish_reason=(response.get('choices')or[{}])[0].get('finish_reason'))


def recall_at5(source,task,anchor):
    scoring=source['scoring_ids']
    if (scoring['question_id']!=task['question_id']or scoring['target_book_id']!=task['book_id']
            or scoring['canonical_question_key']!=anchor['canonical_question_key']):raise ValueError('Recall scoring identities differ')
    def pairs(values):
        if not isinstance(values,list)or any(not isinstance(v,list)or len(v)!=2 or any(not isinstance(x,str)or not x for x in v)for v in values):
            raise ValueError('Exact textbook/original-paragraph pairs required')
        return {tuple(v)for v in values}
    gold=pairs(scoring['gold_paragraphs']);selected=pairs(source['selected_paragraphs'])
    if not gold or any(b!=task['book_id']for b,_ in gold)or len(selected)>5:raise ValueError('Original gold paragraphs and top-five selection required')
    score=len(gold&selected)/len(gold)
    if abs(finite_score(source['metrics']['paragraph_recall_at5'])-score)>1e-12:raise ValueError('Recall@5 differs from exact original paragraph overlap')
    return score


def public_groups(groups,ready):
    result=deepcopy(groups)
    if not ready:
        for group in result.values():
            group.update(equal_book_macro_f1=None,question_micro_f1=None,publication_pending=True)
            for row in group['by_book'].values():row['f1']=None
            for row in group['by_course'].values():row['equal_book_macro_f1']=None
    return result


def build_report(root,canonical_reference,provided_dependencies=None):
    root=Path(root);contract_path=root/'full_answer_contract.json';summary_path=root/'full_answer_summary.json'
    before=dict(contract=record(contract_path),summary=record(summary_path));contract=read(contract_path);saved_summary=read(summary_path)
    if (contract.get('result_identity')!=digest({k:v for k,v in contract.items()if k!='result_identity'})
            or contract.get('question_count')!=353 or contract.get('task_count')!=706 or contract.get('expected_question_counts')!=EXPECTED_COUNTS):
        raise ValueError('Full-answer contract identity or original denominator changed')
    canonical_path=Path(canonical_reference['path'])
    if record(canonical_path)!=canonical_reference:raise ValueError('Canonical result SHA differs')
    canonical=read(canonical_path);deps=provided_dependencies or dependencies(contract);core=deps['core']
    if contract.get('runtime_digest')!=deps['runtime_digest']:raise ValueError('Full-answer generator is not the frozen runtime')
    index,bridge,table=canonical_index(canonical,core)
    tasks=contract['tasks'];seen=set();seen_questions=set();scope_counts={s:defaultdict(int)for s in SCOPES};issues=[];outcomes={};row_audits=[]
    declared_root=str(PurePosixPath(contract['source_refs']['retrieval']['path'].replace('\\','/')).parent)
    retrieval=read(reference_path(contract['source_refs']['retrieval'],root,declared_root))
    source_rows={r['task_id']:r for r in retrieval['records']}
    if len(tasks)!=706 or len(source_rows)!=706:raise ValueError('All706 unique original question-pool tasks required')
    recall_outcomes={};missing=completed_declared=failed_declared=0
    for task in tasks:
        if (task['task_key']in seen or task['scope']not in SCOPES or task.get('variant')!='rerank'
                or task['course_id']!=COURSES.get(task['book_id'])):raise ValueError('Task identity, course or rerank scope differs')
        question_key=task['scope'],task['question_id']
        if question_key in seen_questions:raise ValueError('Repeated original question within one pool')
        seen_questions.add(question_key);seen.add(task['task_key']);scope_counts[task['scope']][task['book_id']]+=1
        anchor=bridge.get((task['scope'],task['question_id']));source=source_rows.get(task['task_id'],{})
        if (anchor is None or anchor['book_id']!=task['book_id']or anchor['question_text_sha256']!=sha(task['question'])
                or any(source.get(k)!=task[k]for k in ('question_id','question','reference_answer','book_id','course_id','scope','context'))
                or source.get('source')!='original353_frozen_expanded_pool_questions'):
            raise ValueError('New task differs from original canonical question or frozen retrieval')
        recall_outcomes[task['task_key']]=dict(status='completed',scores=dict(f1=recall_at5(source,task,anchor)))
        path=core.task_folder(root,task)/'record.json';detail=dict(task_id=task['task_id'],question_id=task['question_id'],book_id=task['book_id'],course_id=task['course_id'],scope=task['scope'])
        if not path.is_file():missing+=1;detail['status']='missing'
        else:
            declared=read(path).get('status');completed_declared+=declared=='completed';failed_declared+=declared=='failed'
            try:
                outcome,proof=verified_terminal(path,task,contract,deps,root,declared_root)
                outcomes[task['task_key']]=outcome;detail.update(status=outcome['status'],f1=outcome['scores']['f1'],proof=proof)
            except (ValueError,KeyError,TypeError,OSError)as exc:
                detail.update(status='verification_pending',declared_status=declared,reason=type(exc).__name__+': '+str(exc)[:180]);issues.append(detail)
        row_audits.append(detail)
    if any(dict(scope_counts[s])!=EXPECTED_COUNTS for s in SCOPES)or seen_questions!=set(bridge):raise ValueError('Original canonical question set or per-book fixed denominator changed')
    if (saved_summary.get('completed')!=completed_declared or saved_summary.get('failed')!=failed_declared
            or saved_summary.get('pending')!=706-completed_declared-failed_declared):issues.append(dict(status='summary_snapshot_count_mismatch'))
    groups={s:core.aggregate([t for t in tasks if t['scope']==s],outcomes)for s in SCOPES}
    for scope in SCOPES:
        original=saved_summary.get('groups',{}).get(scope,{})
        if groups[scope]['final']and(original.get('equal_book_macro_f1')is None or abs(original['equal_book_macro_f1']-groups[scope]['equal_book_macro_f1'])>1e-12):
            issues.append(dict(status='summary_macro_mismatch',scope=scope))
    if record(contract_path)!=before['contract']or record(summary_path)!=before['summary']:issues.append(dict(status='concurrent_input_update'))
    ready=len(outcomes)==706 and not issues
    visible_groups=public_groups(groups,ready)
    new=dict(method='DVKG+cross-encoder')
    for scope in SCOPES:
        scoped=[t for t in tasks if t['scope']==scope];r=core.aggregate(scoped,recall_outcomes)
        new[scope]=dict(f1=groups[scope]['equal_book_macro_f1']if ready else None,
            paragraph_recall_at5=r['equal_book_macro_f1'],question_count=353,by_book=visible_groups[scope]['by_book'],by_course=visible_groups[scope]['by_course'],
            paragraph_recall_at5_by_book={b:v['f1']for b,v in r['by_book'].items()},
            paragraph_recall_at5_by_course={c:v['equal_book_macro_f1']for c,v in r['by_course'].items()},
            answer_status='complete_verified'if ready else'pending_or_verification_pending')
    table.append(new);comparisons={}
    for scope in SCOPES:
        comparisons[scope]={};scoped=sorted((t for t in tasks if t['scope']==scope),key=lambda t:t['question_id'])
        for metric in ('f1','paragraph_recall_at5'):
            best=sorted((r for r in table if r['method']in BASELINES),key=lambda r:(-r[scope][metric],r['method']))[0]
            value=new[scope][metric];base=best[scope][metric];comparison=dict(baseline=best['method'],baseline_value=base,new_value=value,
                delta=None if value is None else value-base,relative_change_percent=None if value is None or not base else(value/base-1)*100,
                paired_bootstrap=None,baseline_selection='highest_frozen_equal_book_macro_for_this_metric;selection_not_multiplicity_adjusted')
            if value is not None:
                left=[outcomes[t['task_key']]['scores']['f1']if metric=='f1'else recall_outcomes[t['task_key']]['scores']['f1']for t in scoped]
                right=[index[best['method'],scope,bridge[scope,t['question_id']]['canonical_question_key']][metric]for t in scoped]
                comparison['paired_bootstrap']=deps['parser'].paired_book_ci(left,right,[t['book_id']for t in scoped],iterations=10000,seed=42)
            comparisons[scope][metric]=comparison
    return dict(version='original353_full_comparison_report_v1',status='ready_exploratory_comparison'if ready else'pending_incomplete_or_unverified',new_F1_publishable=ready,
        question_count=353,task_count=706,book_count=6,course_count=3,expected_question_counts=EXPECTED_COUNTS,
        counts=dict(completed_declared=completed_declared,failed_declared=failed_declared,completed_verified=sum(v['status']=='completed'for v in outcomes.values()),
            failed_verified=sum(v['status']=='failed'for v in outcomes.values()),missing=missing,verification_pending=706-missing-len(outcomes)),
        main_table=table,new_metrics=visible_groups,best_baseline_comparisons=comparisons,issues=issues,per_task_audits=row_audits,
        fixed_failed_denominator=True,pending_scored_zero=False,observed_regression=True,independent_unseen_test=False,
        extra_crossencoder=True,common_reranker_control_pending=True,SOTA_claim=False,
        comparison='exploratory_extra_compute_system_row;original_four_canonical_baselines_unchanged',
        source_refs=dict(**before,canonical=canonical_reference,retrieval=contract['source_refs']['retrieval']),
        pure_evaluation_code_refs=deps.get('code_refs',{}),model_API_calls=0,old_inputs_modified=False)


def markdown(result):
    def number(value):return 'pending'if value is None else f'{value:.3f}'
    text=['# 原353题双池比较','',
        '计组、操作系统、计算机网络，共六本教材；先书内平均，再对六书等权。属于 observed regression，非独立未见盲测。',
        'DVKG新增交叉编码器，common-reranker control pending；本表不能用于宣称构图 SOTA。','',
        f"状态：{result['status']}；核实完成 {result['counts']['completed_verified']}，核实失败 {result['counts']['failed_verified']}，缺失 {result['counts']['missing']}，待核验 {result['counts']['verification_pending']}。",'',
        '| 方法 | 同课程 F1 | 同课程 R@5 | 混合 F1 | 混合 R@5 |','|---|---:|---:|---:|---:|']
    for row in result['main_table']:
        same=row['same_subject_pool'];mixed=row['mixed_six_book_pool']
        text.append(f"| {row['method']} | {number(same['f1'])} | {number(same['paragraph_recall_at5'])} | {number(mixed['f1'])} | {number(mixed['paragraph_recall_at5'])} |")
    text+=['','相对每项原最强基线的配对差值（固定教材内重采样10,000次，seed42；差值及区间为百分点）：','']
    for scope,label in [('same_subject_pool','同课程'),('mixed_six_book_pool','混合')]:
        for metric in ('f1','paragraph_recall_at5'):
            value=result['best_baseline_comparisons'][scope][metric];ci=value['paired_bootstrap']
            detail='pending'if ci is None else f"{100*value['delta']:+.2f}，95% CI [{100*ci['ci95'][0]:+.2f}, {100*ci['ci95'][1]:+.2f}]"
            text.append(f"- {label} {metric} vs {value['baseline']}：{detail}。")
    if result['counts']['failed_declared']:text+=['',f"原始记录声明失败 {result['counts']['failed_declared']} 条；全部保留。缺少可重放失败来源时，新版 F1 不发布。"]
    if not result['new_F1_publishable']:text+=['','新版回答或来源核验尚未齐全：F1 保持 pending，未完成任务没有按零分补齐。']
    return '\n'.join(text)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--canonical',required=True,type=Path);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();reference=record(args.canonical)
    if reference['sha256']!=CANONICAL_SHA:raise ValueError('Only the frozen canonical43cb source is allowed')
    result=build_report(args.root,reference);output=args.output_dir or args.root/'full_comparison_report'
    atomic(output/'full_comparison_report.json',result);atomic(output/'full_comparison_report.md',markdown(result))
    print(json.dumps(dict(status=result['status'],new_F1_publishable=result['new_F1_publishable'],counts=result['counts'],output=str(output),model_API_calls=0)))


if __name__=='__main__':main()
