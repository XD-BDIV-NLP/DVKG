"""零API完成各方法原候选的共享BGE精排检索；缺整包时保持pending。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


METHODS=('GraphRAG','LightRAG','HippoRAG','DVKG','K12-KGraph')
SCOPES=('same_subject_pool','mixed_six_book_pool')
EXPECTED_COUNTS={'book_25f718562685':100,'book_49bb8d31ddb9':100,'book_0b8d87cb0667':40,
    'book_d96f472c8179':37,'book_00849acee112':38,'book_dd8183b4cb60':38}
COURSES={b:c for books,c in [(['book_25f718562685','book_49bb8d31ddb9'],'computer_organization'),
    (['book_0b8d87cb0667','book_d96f472c8179'],'operating_systems'),
    (['book_00849acee112','book_dd8183b4cb60'],'computer_networks')]for b in books}
CANONICAL_SHA='43cb0cb57a2288f1b39ebde8d77575c76b4b847764d279392f0e4b02922e1fcb'
PREPARER_SHA='17c65fc34a78f232e5c508a514d619e109ba973ec5c93bb6d674e4a9daacb6ac'
FULL_FINISH_SHA='5a3a86bd4cb341da238dc9e9776ca8f533446ae52a7c1fb86dd909c3c7894338'
GPU_WORKER_SHA='9156f3356f68c4b609fbc1142895c8a737b50033c087af876898fc58d4a4d5fb'
MODEL='BAAI/bge-reranker-v2-m3'
REVISION='e099d4b9cdbd291b1569d416f19aaa6523570bc3'
MODEL_FILES={
    'config.json':'13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a',
    'configuration.json':'90dccac936e44dcb4567017a5dbb807b583604032fd53bfe2372a15881ed47e1',
    'model.safetensors':'d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286',
    'sentencepiece.bpe.model':'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865',
    'special_tokens_map.json':'8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835',
    'tokenizer.json':'69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15',
    'tokenizer_config.json':'7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4'}


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def record(path):
    path=Path(path).resolve(strict=True);h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return dict(path=str(path),sha256=h.hexdigest())
def checked(reference):
    if record(reference['path'])!=reference:raise ValueError('Frozen shared-control source SHA changed')
    return read(reference['path'])
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Refusing to change a completed shared-control artifact')
    else:atomic(path,value)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def initialize(full_root,owner):
    if os.name!='posix':raise RuntimeError('Real original source assets remain on PRIMARY')
    if owner['worker']['sha256']!=PREPARER_SHA or record(owner['worker']['path'])!=owner['worker']:
        raise ValueError('Frozen shared candidate preparer differs')
    preparer=load(owner['worker']['path'],'shared_finish_original_preparer')
    finish_path=Path(owner['worker']['path']).with_name('full_rerank_finish.py')
    if record(finish_path)['sha256']!=FULL_FINISH_SHA:raise ValueError('Frozen asset-only finish helper differs')
    helper=load(finish_path,'shared_finish_original_asset_helper')
    with preparer.zero_api_guard():
        contract=read(Path(full_root)/'input_contract.json');deps=helper.initialize_dependencies(contract)
        questions={q['question_id']:dict(q,evaluation_course_id=s['course_id'])for s in deps['cfg']['sources']for q in deps['datasets'][s['book_id']]}
        deps.update(finish=helper,preparer=preparer,questions=questions,canonical=helper.canonical_rows(deps,questions))
    return deps


def validate_questions(deps):
    questions=deps['questions']
    counts=Counter(q['target_book_id']for q in questions.values())
    if len(questions)!=353 or dict(counts)!=EXPECTED_COUNTS:raise ValueError('Exactly the original353 questions required')
    for qid,q in questions.items():
        if q['question_id']!=qid or q['evaluation_course_id']!=COURSES[q['target_book_id']]:raise ValueError('Original course/question identity differs')
    if deps['canonical_ref']['sha256']!=CANONICAL_SHA:raise ValueError('Canonical original five-method anchor changed')


def original_keys(method,task,anchor,deps,full_root):
    saved=checked(anchor['source_retrieval']);trace=task['candidate_trace']
    if method in {'GraphRAG','LightRAG','HippoRAG','DVKG'}:
        internal={'GraphRAG':'graphrag','LightRAG':'lightrag','HippoRAG':'hipporag','DVKG':'dvkg'}[method]
        if (saved.get('question_id')!=task['question_id']or saved.get('question')!=task['question']or saved.get('scope')!=task['scope']
                or saved.get('method')!=internal or saved.get('checkpoint_digest')!=digest({k:v for k,v in saved.items()if k!='checkpoint_digest'})):
            raise ValueError('Original method source checkpoint differs')
        if method!='DVKG':
            if saved.get('status')!='ready':raise ValueError('Original native source is not ready')
            keys=deps['preparer'].native_candidate_keys(saved)
        else:
            if '_original_prepared'in deps:prepared=deps['_original_prepared']
            else:
                full=read(Path(full_root)/'input_contract.json');private=read(Path(full_root)/'private_prepared_tasks.json')
                if private['contract_digest']!=digest(full):raise ValueError('Original DVKG private task freeze differs')
                prepared={(t['scope'],t['question_id']):t for t in private['tasks']}
                if len(prepared)!=706:raise ValueError('Original DVKG requires all706 prepared tasks')
                deps['_original_prepared']=prepared
            source=prepared[task['scope'],task['question_id']]
            if source['question']!=task['question']or source['book_id']!=task['book_id']:raise ValueError('Original DVKG question differs')
            raw=source['candidate_trace']['original_candidate_evidence_ids'];keys=set()
            for uid in raw:
                pair=json.loads(uid)
                if not isinstance(pair,list)or len(pair)!=2 or uid!=json.dumps(pair,ensure_ascii=False,separators=(',',':')):raise ValueError('Original DVKG native identity differs')
                keys.add(tuple(pair))
            if len(keys)!=len(raw)or len(keys)!=source['candidate_trace']['original_candidate_count']or len(keys)!=saved['trace']['candidate_count']:
                raise ValueError('Original DVKG own candidate count differs')
            routed={tuple(p)for p in saved['trace']['routing']['evidence_ids']}
            if not routed<=keys or(not saved['trace']['fallback']and keys!=routed):raise ValueError('Original DVKG route/fallback source differs')
        if len(keys)!=trace['input_candidate_count']or digest(sorted(keys))!=trace['input_candidate_digest']:
            raise ValueError('Own full pre-common candidate keys differ')
        return keys,'original_method_native_candidate_keys'
    proof=task.get('K12_replay');record_value=saved.get('record',{})
    if (not isinstance(proof,dict)or proof.get('exact_replay')is not True
            or proof.get('candidate_count')!=trace['input_candidate_count']or proof.get('candidate_count')!=record_value.get('candidate_count')
            or proof.get('selected_digest')!=record_value.get('selected_digest')
            or proof.get('context_sha256')!=record_value.get('context_sha256')or proof.get('old_answer_body_digest')!=digest(saved['body'])
            or record_value.get('question_id')!=task['question_id']or record_value.get('book')!=task['book_id']or record_value.get('scope')!=task['scope']):
        raise ValueError('K12 complete own-candidate CPU replay evidence is pending')
    return None,'K12_frozen_CPU_replay_count_selected_context_body_proof'


def validate_task(method,q,scope,task,owner,deps,full_root):
    qid=q['question_id'];book=q['target_book_id'];anchor=deps['canonical'].get((method,scope,qid))
    if anchor is None:raise ValueError('Original canonical method observation missing')
    if (task.get('task_id')!=digest(dict(method=method,scope=scope,question=qid))or task.get('method')!=method
            or task.get('question_id')!=qid or task.get('question')!=q['question']or task.get('scope')!=scope
            or task.get('book_id')!=book or task.get('course_id')!=COURSES[book]
            or task.get('canonical_question_key')!=anchor['canonical_question_key']
            or task['canonical_question_key']!=digest([qid,book,q['question']])
            or task.get('source_retrieval')!=anchor['source_retrieval']or task.get('policy')!=owner['policy']
            or task.get('original353_observed_questions')is not True or task.get('independent_test')is not False
            or task.get('model_API_calls')!=0 or any(k in task for k in ('reference_answer','gold_paragraphs','answer'))):
        raise ValueError('Shared method task identity, original source or scope differs')
    rows=task['candidates'];allowed=set(EXPECTED_COUNTS)if scope==SCOPES[1]else{b for b,c in COURSES.items()if c==COURSES[book]}
    if not 1<=len(rows)<=64 or task.get('candidate_count')!=len(rows):raise ValueError('Own one-to64 candidate paragraphs required')
    keys=[deps['finish'].validate_candidate(row,allowed,deps)for row in rows]
    if len(set(keys))!=len(keys):raise ValueError('Repeated original candidate paragraph')
    trace=task['candidate_trace']
    if (trace.get('global_fill')is not False or trace.get('route_boost')is not False or trace.get('rrf_constant')!=60 or trace.get('per_channel_limit')!=64
            or trace.get('output_paragraph_count')!=len(rows)or trace.get('output_paragraph_digest')!=digest(sorted(keys))):
        raise ValueError('Common preselection controls or paragraph identities differ')
    native,proof=original_keys(method,task,anchor,deps,full_root)
    if native is not None and any((row['book_id'],row['source_evidence_id'])not in native for row in rows):raise ValueError('Candidate is outside original method own native candidates')
    return proof


def verified_scores(packet,receipt,deps,packet_ref):
    if (packet.get('model')!=MODEL or packet.get('revision')!=REVISION or packet.get('max_length')!=512 or packet.get('answers_included')is not False
            or receipt.get('actual_inference')is not True or receipt.get('score_kind')!='raw_logit'or receipt.get('normalization')!='none'
            or receipt.get('eval_mode')is not True or receipt.get('inference_mode')is not True or receipt.get('network_disabled')is not True
            or receipt.get('local_files_only')is not True or receipt.get('trust_remote_code')is not False
            or not str(receipt.get('device','')).startswith('cuda:')):raise ValueError('Real frozen offline BGE inference receipt required')
    files=receipt.get('model_files',[])
    if len(files)!=len(MODEL_FILES)or {r['name']:r['sha256']for r in files}!=MODEL_FILES:raise ValueError('Frozen model/tokenizer asset hashes differ')
    if receipt.get('worker_sha256')!=GPU_WORKER_SHA:raise ValueError('Frozen offline GPU worker SHA differs')
    if receipt.get('input_file_sha256')!=packet_ref['sha256']:raise ValueError('GPU input file SHA differs')
    sources={item['task_id']:{tuple(c['key']):c for c in item['candidates']}for item in packet['packets']}
    for item in receipt.get('items',[]):
        for row in item.get('scores',[]):
            original=sources.get(item.get('task_id'),{}).get(tuple(row.get('key',[])))
            if original is None or any(row.get(k)!=original[k]for k in ('input_sha256','source_sha256','char_start','char_end')):
                raise ValueError('GPU score window/source hashes or exact span differ')
    return deps['verify_score_receipt'](packet,receipt)


def method_result(method,full_root,control_root,owner,deps,output):
    expected=[(q,scope,control_root/'tasks'/method/scope/(digest(qid)+'.json'))for qid,q in deps['questions'].items()for scope in SCOPES]
    present=sum(p.is_file()for _,_,p in expected);pending=dict(status='pending_candidates',expected_tasks=706,ready_task_files=present,R5={s:None for s in SCOPES})
    if present!=706:return pending
    packet_path=control_root/'score_packets'/(method+'.json');score_path=control_root/'GPU_scores'/(method+'.json')
    if not packet_path.is_file():return dict(pending,status='pending_score_packet')
    if not score_path.is_file():return dict(pending,status='pending_GPU_scores')
    tasks=[];task_refs=[];proofs=[]
    for q,scope,path in expected:
        task=read(path);proofs.append(validate_task(method,q,scope,task,owner,deps,full_root));tasks.append(task);task_refs.append(record(path))
    packet=read(packet_path)
    if packet!=deps['build_score_packet'](tasks):raise ValueError('Shared score packet does not exactly reproduce the frozen original candidate windows')
    packet_ref=record(packet_path);score_ref=record(score_path);scores=verified_scores(packet,read(score_path),deps,packet_ref)
    records=[]
    for task in tasks:
        q=deps['questions'][task['question_id']]
        selected,ranking=deps['rerank_evidence'](task['question'],task['candidates'],route_scores={},pair_scores=scores[task['task_id']],limit=5)
        context=deps['pack_context'](selected,task['question']);book=task['book_id']
        gold=sorted({(book,deps['structures'][book]['evidence'][eid]['source']['source_id'])for eid in q['evidence_ids']})
        metrics=deps['score_stages'](gold,task['candidates'],selected,context)
        records.append(dict(task_id=task['task_id'],question_id=task['question_id'],canonical_question_key=task['canonical_question_key'],method=method,
            scope=task['scope'],book_id=book,course_id=task['course_id'],question=task['question'],context=context,
            selected_paragraphs=[list(deps['paragraph_key'](r))for r in selected],metrics=metrics,ranking=ranking,
            source_retrieval=task['source_retrieval'],score_packet=packet_ref,GPU_scores=score_ref,
            route_scores_used={},final_packer='frozen_original_graph_rag_v2.pack_context',model_API_calls=0,answer_generation=False))
    if len(records)!=706:raise ValueError('No partial method aggregation is allowed')
    summary=deps['finish'].summarize(records)
    frozen=dict(version='original353_shared_rerank_method_finish_v1',method=method,owner=record(control_root/'owner.json'),
        worker=record(__file__),tasks=task_refs,score_packet=packet_ref,GPU_scores=score_ref,canonical=deps['canonical_ref'],
        task_count=706,shared_model=MODEL,revision=REVISION,max_length=512,route_prior=False,global_fill=False,answer_generation=False,model_API_calls=0)
    immutable(output/'method_contracts'/(method+'.json'),frozen)
    immutable(output/'private_contexts'/(method+'.json'),dict(contract_digest=digest(frozen),records=records))
    return dict(status='complete706_shared_rerank_retrieval',expected_tasks=706,ready_task_files=706,
        R5={s:summary[s]['equal_book_macro']['paragraph_recall_at5']for s in SCOPES},summary=summary,
        candidate_provenance=sorted(set(proofs)),method_contract=record(output/'method_contracts'/(method+'.json')),
        private_contexts=record(output/'private_contexts'/(method+'.json')))


def finish(full_root,control_root,output,deps=None):
    full_root=Path(full_root).resolve();control_root=Path(control_root).resolve();output=Path(output).resolve()
    if output in (full_root,control_root)or output in full_root.parents or output in control_root.parents:raise ValueError('Use a separate new output directory')
    owner=read(control_root/'owner.json')
    if owner.get('full_sources')!=[record(full_root/'input_contract.json'),record(full_root/'private_prepared_tasks.json')]:raise ValueError('Shared control belongs to another original full input run')
    deps=deps or initialize(full_root,owner);validate_questions(deps)
    if owner.get('policy')!=deps['preparer'].POLICY:raise ValueError('Shared preselection policy changed')
    results={}
    with deps['preparer'].zero_api_guard():
        for method in METHODS:
            try:results[method]=method_result(method,full_root,control_root,owner,deps,output)
            except (ValueError,KeyError,TypeError,OSError)as exc:
                results[method]=dict(status='verification_pending',expected_tasks=706,R5={s:None for s in SCOPES},error_type=type(exc).__name__,reason=str(exc)[:200])
    complete=[name for name,value in results.items()if value['status']=='complete706_shared_rerank_retrieval']
    report=dict(version='original353_own_candidate_shared_BGE_retrieval_control_v1',status='all_methods_complete'if len(complete)==len(METHODS)else'partial_methods_explicit_pending',
        question_count=353,expected_tasks_per_method=706,book_count=6,course_count=3,expected_question_counts=EXPECTED_COUNTS,
        methods=results,complete_methods=complete,canonical=deps['canonical_ref'],shared_model=MODEL,revision=REVISION,max_length=512,
        equal_book_macro=True,course_two_equal_books=True,observed_regression=True,independent_unseen_test=False,
        no_global_candidate_fill=True,no_old_top5_substitution=True,route_scores_used={},answer_generation=False,model_API_calls=0,
        QA_control_complete=False,SOTA_claim=False,old_inputs_modified=False)
    atomic(output/'shared_control_retrieval_summary.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full-root',required=True,type=Path)
    parser.add_argument('--control-root',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();sys.dont_write_bytecode=True
    result=finish(args.full_root,args.control_root,args.output_dir)
    print(json.dumps(dict(status=result['status'],complete_methods=result['complete_methods'],methods={k:dict(status=v['status'],R5=v['R5'])for k,v in result['methods'].items()},model_API_calls=0,answer_generation=False)))


if __name__=='__main__':main()
