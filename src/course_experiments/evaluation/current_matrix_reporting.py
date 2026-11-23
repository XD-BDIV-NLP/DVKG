"""当前矩阵只读汇总；原文仅在primary的私有错误表中保留。"""
from __future__ import annotations

import argparse
import copy
from collections import Counter,defaultdict
from contextlib import closing
import html
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time

from ..manifest import digest,file_record as _hash_file_record,read_document
from .followup_reporting import paired,paired_course_macro

GEOMETRY_HELPER_SHA='0229c00505e36b0a27cbe68b86b885d5de3f3225f659b3dad17867c0668c7a15'
HIPPORAG_PARSER_SHA='3935854368bb11e960323c31fb3cc67bee9e393079ce522711586d5e643bdb46'
LIGHT_OPERATE_SHA='176585e3f686574ab98657b666982302ce47d58229baacb529ef601bd6799326'
QA_CATEGORIES={'controlled_graph_RAG','native_end_to_end_RAG','ECHA_RAG','DVKG_control'}
IDENTITY_FIELDS=('id','category','method','cohort','target_book','variant','split','required')
_file_verification_cache={}


def file_record(path):
    path=Path(path);s=path.stat();key=str(path.resolve());fingerprint=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    previous=_file_verification_cache.get(key)
    if previous and previous[0]==fingerprint:return dict(previous[1])
    value=_hash_file_record(path);after=path.stat()
    if fingerprint!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Report source changed during SHA verification')
    _file_verification_cache[key]=(fingerprint,value);return dict(value)


def checked(record):
    if file_record(record['path'])!=record:raise ValueError('Report source SHA changed')
    return read_document(record['path'])


def checkpoint(value):
    if value.get('checkpoint_digest')!=digest({k:v for k,v in value.items() if k!='checkpoint_digest'}):raise ValueError('Question checkpoint digest differs')
    return value


def records(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:return [value]
        return [r for v in value.values() for r in records(v)]
    if isinstance(value,list):return [r for v in value for r in records(v)]
    return []


def numbers(value):
    if type(value) in (int,float) and math.isfinite(value):return value
    if value is None or type(value) is bool:return value
    if isinstance(value,dict):return {k:numbers(v) for k,v in value.items() if isinstance(v,(dict,list)) or v is None or type(v) in (int,float,bool)}
    if isinstance(value,list):return [numbers(v) for v in value if isinstance(v,(dict,list)) or v is None or type(v) in (int,float,bool)]
    return None


def _raw(record):
    value=checked(record)
    if 'checkpoint_digest' in value:checkpoint(value)
    raw=value.get('response',value)
    choice=raw['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Successful QA requires complete original model output')
    try:return json.loads(choice['message']['content'])
    except json.JSONDecodeError:return choice['message']['content']


def _blind_ready(obs):
    if not obs.get('seal') or not obs.get('algorithm_freeze'):return False
    from .blind64_pipeline import require_freeze
    _,freeze=require_freeze(obs['algorithm_freeze']['path'],obs['algorithm_freeze']['sha256'])
    seal=checked(obs['seal'])
    if (seal.get('status')!='sealed_complete' or seal.get('accepted_count')!=64 or seal.get('target_count')!=64 or
            seal.get('shortage')!=0 or seal.get('released_for_quality_evaluation') is not True):return False
    if checked(seal['contract']).get('algorithm_freeze')!=freeze:raise ValueError('Question seal belongs to another algorithm freeze')
    index=checked(seal['dataset_index'])
    if obs['dataset'] not in index['books'].values():raise ValueError('New64 dataset is outside the single frozen seal')
    return True


def verify_QA_dependencies(req,contract,obs,verified):
    dependencies=[]
    for key in ['graph','graphs','artifact','code']:dependencies+=records(contract.get(key,{}))
    if contract.get('plan'):
        plan=checked(contract['plan'])
        for key in ['graph','graphs','artifact','code']:dependencies+=records(plan.get(key,{}))
    if req['category']=='DVKG_control':
        population=contract.get('inventories') or {'states':contract['inventory']}
        state=req['variant'].split('_prior')[0] if req['variant'].startswith(('G0_','T_')) else 'F_corrected'
        matches=[]
        for record in population.values():
            key=('inventory',record['path'],record['sha256'])
            if key not in verified:verified[key]=checked(record)
            inventory=verified[key];dependencies.append(record)
            matches += [r['graph'] for r in inventory['states'].values() if r.get('cohort')==req.get('cohort') and r.get('state')==state]
        if len(matches)!=1 or (req.get('graph_sha256') and matches[0]['sha256']!=req['graph_sha256']):raise ValueError('Actual DVKG graph differs from the frozen required semantic state')
        dependencies+=matches
    if obs.get('context_contract'):
        context=checked(obs['context_contract']);dependencies+=records(context.get('code',{}))
        if contract.get('plan') and digest(context)!=checked(contract['plan'])['context_identity']:raise ValueError('ECHA original context binding differs')
        for r in context.get('source_records',[]):
            if Path(r['path']).name=='concept_kg.json':dependencies.append(r)
    dependencies+=obs.get('native_index_sources',[])
    dependencies+=obs.get('request_attributions',[])
    dependencies+=records(obs.get('native_cache_origins',{}))
    unique={}
    for record in dependencies:
        if record['path'] in unique and unique[record['path']]!=record:raise ValueError('Conflicting original QA dependency SHA')
        unique[record['path']]=record
    for record in unique.values():
        key=('file',record['path'],record['sha256'])
        if key not in verified:
            if file_record(record['path'])!=record:raise ValueError('Original QA graph or code dependency SHA changed')
            verified[key]=True
    return list(unique.values())


def native_response_answer(method,content,contract):
    if method!='hipporag':return content,[]
    path=Path(contract['source']['path'])/'src/hipporag/HippoRAG.py';record=file_record(path)
    if record['sha256']!=HIPPORAG_PARSER_SHA:raise ValueError('Original official HippoRAG answer parser SHA differs')
    # 原官方HippoRAG.py1167–1172，保留split[1]及except原样返回的精确语义。
    try:answer=content.split('Answer:')[1].strip()
    except Exception:answer=content
    return answer,[record]


def light_cached_answer_origin(artifact,contract,question,native,run_root,private_root):
    """只沿官方query cache的精确原题entry和时间窄窗查原receipt，零API。"""
    from ..contributions import write_immutable
    source=file_record(Path(contract['source']['path'])/'lightrag/operate.py')
    if source['sha256']!=LIGHT_OPERATE_SHA:raise ValueError('Original official LightRAG cache implementation SHA differs')
    result=native['native_result']
    if result.get('query')!=question['question'] or result.get('target_book_id')!=question['target_book_id']:raise ValueError('Original native cache query and book differ')
    paths={str(Path(r['path']).parent/'kv_store_llm_response_cache.json') for r in artifact['native_graphs'] if Path(r['path']).suffix=='.graphml'}
    if len(paths)!=1:raise ValueError('One exact native LightRAG index cache required')
    cache_path=Path(next(iter(paths)));cache=read_document(cache_path)
    expected=dict(answer_cache_version='query-answer-cache-v2',mode='hybrid',response_type='Single Paragraph',top_k=40,chunk_top_k=20,max_entity_tokens=6000,max_relation_tokens=8000,max_total_tokens=30000,enable_rerank=False,enable_content_headings=True,related_chunk_number=5,kg_chunk_pick_method='VECTOR')
    matches=[(k,e) for k,e in cache.items() if isinstance(e,dict) and k.startswith('hybrid:query:') and e.get('cache_type')=='query' and e.get('original_prompt')==question['question'] and e.get('return')==native['answer']]
    if len(matches)!=1:raise ValueError('One exact original query and answer cache entry required')
    cache_key,entry=matches[0];params=entry.get('queryparam',{})
    if set(params)!=set(expected)|{'hl_keywords','ll_keywords','user_prompt'} or any(params.get(k)!=v for k,v in expected.items()):raise ValueError('Original native query cache parameters differ')
    created=entry['create_time'];found={};events={}
    with closing(sqlite3.connect('file:'+str(Path(run_root)/'formal_provider/audit.sqlite')+'?mode=ro',uri=True)) as db:
        candidates=db.execute("SELECT e.id,e.key,r.record FROM events e JOIN receipts r ON r.key=e.key WHERE e.created BETWEEN ? AND ? AND e.status='request_started' AND e.stage=?",(created-30,created+30,'lightrag.'+question['target_book_id']+'.query')).fetchall()
    for event_id,key,encoded in candidates:
        record=json.loads(encoded);raw=checked(record);choices=raw.get('response',{}).get('choices',[]);body=raw['request']
        if not choices or choices[0].get('finish_reason')!='stop' or choices[0].get('message',{}).get('content')!=native['answer']:continue
        if (raw.get('runtime_config')!=contract['runtime_config'] or body.get('model')!=contract['runtime_config']['llm']['model'] or
                not any(m.get('role')=='user' and m.get('content')==entry['original_prompt'] for m in body.get('messages',[]))):continue
        if raw['request_id']!=key or key!=digest(dict(endpoint=raw['endpoint'],body=body,config=raw['runtime_config'])):raise ValueError('Original cached query request identity differs')
        found[key]=record;events.setdefault(key,[]).append(event_id)
    if len(found)!=1:raise ValueError('One exact original cached query provider receipt required')
    key=next(iter(found));path=Path(private_root)/'native_cache_entries'/(digest(entry)+'.json');write_immutable(path,dict(cache_path=str(cache_path),cache_key=cache_key,entry=entry,official_source=source,original_receipt=found[key]))
    return dict(receipt=found[key],event_ids=events[key],cache_path=str(cache_path),cache_key=cache_key,entry_digest=digest(entry),official_source=source,entry_snapshot=file_record(path),queryparam_digest=digest(params))


def qa_observation(req,obs,*,_verified=None):
    base={k:req.get(k) for k in IDENTITY_FIELDS};base.update(status='pending',metrics=None,counts=dict(fixed=req.get('question_count'),completed=0,failed=0,pending=req.get('question_count')),question_scores={})
    if req.get('split')=='new_blind64' and not _blind_ready(obs):return dict(base,reason='blind64_seal_and_freeze_required'),[]
    contract=checked(obs['contract']);result=checked(obs['result']);identity=contract.get('result_id',contract.get('job_identity'))
    dependencies=verify_QA_dependencies(req,contract,obs,_verified if _verified is not None else {})
    if contract.get('plan'):
        plan=checked(contract['plan'])
        if plan.get('plan_identity')!=digest({k:v for k,v in plan.items() if k!='plan_identity'}) or identity!=plan['plan_identity']:raise ValueError('Deferred QA plan identity differs')
    elif identity!=digest({k:v for k,v in contract.items() if k not in {'result_id','job_identity','effective_execution_authorization'}}):raise ValueError('Result contract identity differs')
    if result.get('result_id',result.get('job_identity'))!=identity:raise ValueError('QA result belongs to another execution')
    if req['category']=='DVKG_control':
        state=req['variant'].split('_prior')[0] if req['variant'].startswith(('G0_','T_')) else 'F_corrected';job=result['job']
        if job.get('cohort')!=req['cohort'] or job.get('state')!=state or job.get('book')!=req['target_book']:raise ValueError('DVKG result belongs to another semantic condition')
    if contract.get('method') and contract['method']!=req['method']:raise ValueError('QA method differs from the required condition')
    if contract.get('cohort') and contract['cohort']!=req.get('cohort'):raise ValueError('QA cohort differs from the required condition')
    if req['category']=='native_end_to_end_RAG':
        if contract.get('method')!=req['method'] or contract.get('book')!=req['target_book']:raise ValueError('Original native method and book binding required')
        if req['method']=='graphrag':
            track='global' if req['variant']=='Global' else 'local'
            if obs.get('track')!=track or track not in contract.get('tracks',[]):raise ValueError('Native Local and Global require their own original track')
    allowed=records(contract)
    if contract.get('plan'):allowed+=records(checked(contract['plan']))
    if obs['dataset'] not in allowed:raise ValueError('Dataset is not bound to the actual QA contract')
    dataset=checked(obs['dataset']);questions=dataset['questions'];book=req['target_book'];expected={q['question_id']:q for q in questions}
    if (len(expected)!=len(questions) or len(expected)!=req['question_count'] or dataset.get('target_book_id',book)!=book or
            any(q.get('target_book_id')!=book for q in questions)):raise ValueError('Full fixed question denominator or target book differs')
    seen={};details=[];sources=[obs['contract'],obs['result'],obs['dataset'],*dependencies];scores={};f1_scores={};metric_values=defaultdict(list);errors=[]
    for record in obs['questions']:
        value=checkpoint(checked(record));qid=value['question_id']
        if qid not in expected or qid in seen or value.get('result_id',value.get('job_identity'))!=identity or value.get('target_book_id')!=book or value.get('question_type')!=expected[qid]['type']:
            raise ValueError('Question checkpoint is duplicate, foreign or from another execution')
        if obs.get('track') and value.get('track')!=obs['track']:raise ValueError('Native question belongs to a different track')
        if value['status'] not in {'completed','failed'}:raise ValueError('Question execution state is not terminal')
        seen[qid]=value;sources.append(record)
        if value['status']=='completed':
            if value.get('answer_receipt'):
                answer_record=value['answer_receipt']['source'];answer=_raw(answer_record);sources.append(answer_record)
                if answer!=value['candidate']:raise ValueError('Candidate answer differs from its original raw response')
            else:
                native=checkpoint(checked(obs['native_qa'][qid]));sources.append(obs['native_qa'][qid])
                if native.get('job_identity')!=identity or native['answer']!=value['answer']:raise ValueError('Original native QA differs from scored answer')
                matched=False
                for proof in obs.get('provider_receipts',{}).get(qid,[]):
                    raw=checked(proof)
                    choices=raw.get('response',{}).get('choices',[])
                    if choices and choices[0].get('finish_reason')=='stop':
                        answer,parser=native_response_answer(req['method'],choices[0].get('message',{}).get('content'),contract)
                        if answer==native['answer']:matched=True;sources.extend(parser)
                if not matched:raise ValueError('Native answer requires its exact original successful provider response')
            judgment=_raw(obs['judge_raw'][qid]);sources.append(obs['judge_raw'][qid])
            if not isinstance(judgment,dict) or any(judgment.get(k)!=value['machine_judge'].get(k) for k in ['score','grounding'] if value['machine_judge'].get(k) is not None):raise ValueError('Scored judgment differs from original raw judgment')
            for proof in obs.get('provider_receipts',{}).get(qid,[]):
                raw=checked(proof)
                if raw['request_id']!=digest(dict(endpoint=raw['endpoint'],body=raw['request'],config=raw['runtime_config'])):raise ValueError('Original provider request identity differs')
                sources.append(proof)
            measures=dict(value['scores'],machine_judge_mean=value['machine_judge']['score'])
            if value.get('source_grounding_evaluable') is not False and value['machine_judge'].get('grounding') is not None:measures['machine_grounding_mean']=value['machine_judge']['grounding']
            for name,v in measures.items():
                if type(v) in (int,float) and math.isfinite(v):metric_values[name].append(v)
            score=measures['machine_judge_mean'];f1=measures['f1']
            if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in [score,f1]):raise ValueError('Finite normalized LLM judgment and auxiliary F1 required')
            scores[qid]=score;f1_scores[qid]=f1
        else:
            error=value.get('error_record')
            if not error:raise ValueError('Failed question requires the preserved original failure record')
            if file_record(error['path'])!=error:raise ValueError('Preserved failure evidence changed')
            sources.append(error);scores[qid]=0.0;f1_scores[qid]=0.0
        if value['status']=='failed' or scores[qid]<1:
            category=value.get('error_type','answer_mismatch') if value['status']=='failed' else ('no_reference_evidence' if value['scores'].get('evidence_recall')==0 else 'answer_mismatch')
            errors.append(dict(question_id=qid,target_book_id=book,category=category,status=value['status'],llm_judge=scores[qid],f1=f1_scores[qid]))
            details.append(dict(requirement_id=req['id'],question=expected[qid],record=value,source=record,error_category=category))
    n=len(expected);completed=sum(v['status']=='completed' for v in seen.values());failed=len(seen)-completed;pending=n-len(seen)
    metric={k:(sum(v)/n if n and len(v)==completed else None) for k,v in metric_values.items()} if not pending else None
    base.update(status='pending' if pending else 'failed' if failed else 'completed',counts=dict(fixed=n,completed=completed,failed=failed,pending=pending),metrics=metric,
        success_rate=completed/n if n else None,question_scores=scores,question_f1_scores=f1_scores,primary_metric='original_llm_judge_score',question_errors=errors,sources=list({digest(r):r for r in sources}.values()),
        denominator='all_fixed_questions;failed_zero;pending_not_executed',original_raw_revalidated=True)
    return base,details


def geometry_observation(req,obs):
    value=checked(obs['result']);graph=obs['graph'];helper=obs['helper']
    if file_record(graph['path'])!=graph or file_record(helper['path'])!=helper or helper['sha256']!=GEOMETRY_HELPER_SHA:raise ValueError('Frozen geometry graph or helper differs')
    if graph['sha256']!=req.get('graph_sha256',graph['sha256']) or value.get('graph')!=graph or value.get('helper')!=helper:raise ValueError('Geometry result belongs to a different source graph or helper')
    if value.get('status') not in {'completed_new_geometry','completed_exact_graph_SHA_reuse'} or value['metrics'].get('status')!='completed':raise ValueError('Geometry has no completed original measurement')
    if value['metrics'].get('version')!='geometry_stats_20261018_v1':raise ValueError('Geometry metric convention differs')
    if value.get('historical_result'):
        old=checked(value['historical_result'])
        if old['input']['graph']['sha256']!=graph['sha256'] or old['metrics']!=value['metrics']:raise ValueError('Historical geometry exact-SHA reuse differs')
    return dict(status='completed',metrics=numbers(value['metrics']),graph_sha256=graph['sha256'],sources=[obs['result'],graph,helper],exact_graph_SHA_verified=True)


def verify_model_receipt(record):
    from .identity_kge_execution import validate_saved_receipt
    value=checked(record);validate_saved_receipt(value,Path(record['path']).parent,value['job'],value['plan_id'],False)
    if value['actual_epochs']!=1000:raise ValueError('Formal model has not actually completed1000 epochs')
    return value


def native_index_observation(req,artifact):
    contract=checked(artifact['contract']);result=checked(artifact['complete']);index=checked(artifact['index_contract']);receipt=checkpoint(checked(artifact['index_receipt']))
    identity=contract['job_identity'];idx=contract['index_identity']
    if (identity!=digest({k:v for k,v in contract.items() if k!='job_identity'}) or result.get('job_identity')!=identity or
            result.get('status')!='completed' or contract.get('method')!=req['method'] or contract.get('book')!=req['target_book']):raise ValueError('Original native completion binding differs')
    if index.get('index_identity')!=idx or idx!=digest({k:v for k,v in index.items() if k!='index_identity'}) or receipt.get('job_identity')!=idx:raise ValueError('Original full native index receipt differs')
    sources=records(artifact)+records(contract.get('code',{}))
    for record in sources:
        if file_record(record['path'])!=record:raise ValueError('Original native index artifact or implementation SHA changed')
    return dict(status='completed',index_identity=idx,original_native_index_SHA_verified=True,common_schema_semantic_qualification_claimed=False,metrics=None,sources=sources)


def collect_native_inventory(matrix,inventory_record,*,run_root=None,private_root=None):
    """读取发布的实际原始完成索引；不重建索引，不读取任何新64题。"""
    inventory=checked(inventory_record);observations={};errors=[]
    for artifact in inventory['artifacts']:
        for req in matrix['requirements']:
            if req['method']!=artifact['method'] or req.get('target_book')!=artifact['book']:continue
            if req['category'] not in {'native_index','native_end_to_end_RAG'} or (req['category']=='native_end_to_end_RAG' and req.get('split')!='regression313'):continue
            try:
                proof=native_index_observation(req,artifact)
                if req['category']=='native_index':observations[req['id']]=dict(adapter='native_index',artifact=artifact,inventory=inventory_record);continue
                contract=checked(artifact['contract']);root=Path(artifact['complete']['path']).parent;track=None
                if req['method']=='graphrag':
                    track='global' if req.get('variant')=='Global' else 'local'
                    if track not in contract.get('tracks',[]):continue
                    root=root/track
                folder=root/'confirmation';questions=[file_record(p) for p in sorted(folder.glob('*/record.json'))];judges={};native={};receipts={};attributions=[];origins={};dataset=checked(contract['datasets']['confirmation']);byq={q['question_id']:q for q in dataset['questions']}
                for record in questions:
                    q=checked(record);qid=q['question_id'];qr=Path(record['path']).parent
                    for name,target in [('judge_raw.json',judges),('native_qa.json',native)]:
                        if (qr/name).exists():target[qid]=file_record(qr/name)
                    requests=[qr/'requests.jsonl']
                    if q.get('receipts_path'):
                        inherited=Path(q['receipts_path']);allowed=Path(artifact['complete']['path']).parents[1]
                        if not inherited.resolve().is_relative_to(allowed.resolve()):raise ValueError('Original native attribution path is outside the same method and book')
                        requests.append(inherited)
                    receipts[qid]=[]
                    for request_path in dict.fromkeys(requests):
                        if request_path.exists():
                            attributions.append(file_record(request_path));receipts[qid]+=[r['receipt'] for r in (json.loads(line) for line in request_path.read_text('utf-8').splitlines() if line.strip()) if r.get('status')=='completed' and isinstance(r.get('receipt'),dict)]
                    if req['method']=='lightrag' and q['status']=='completed' and run_root is not None and private_root is not None:
                        answer=checked(native[qid])['answer'];matched=any(any(c.get('finish_reason')=='stop' and c.get('message',{}).get('content')==answer for c in checked(r).get('response',{}).get('choices',[])) for r in receipts[qid])
                        if not matched:
                            origins[qid]=light_cached_answer_origin(artifact,contract,byq[qid],checked(native[qid]),run_root,private_root);receipts[qid].append(origins[qid]['receipt'])
                observations[req['id']]=dict(adapter='qa',contract=artifact['contract'],result=artifact['complete'],dataset=contract['datasets']['confirmation'],questions=questions,judge_raw=judges,native_qa=native,provider_receipts=receipts,track=track,native_index_sources=proof['sources'],request_attributions=attributions,native_cache_origins=origins,published_inventory=inventory_record)
            except (ValueError,KeyError,TypeError,OSError) as exc:errors.append(dict(requirement_id=req['id'],error_type=type(exc).__name__))
    return observations,errors


def collect_DVKG_runs(matrix,contracts):
    observations={};errors=[]
    for record in contracts:
        contract=checked(record);split={'regression':'regression313','independent':'existing_independent123','blind64':'new_blind64'}.get(contract['split'])
        if split=='new_blind64':continue
        for req in matrix['requirements']:
            if req['category']!='DVKG_control' or req.get('split')!=split:continue
            state=req['variant'].split('_prior')[0] if req['variant'].startswith(('G0_','T_')) else 'F_corrected'
            folder=Path(record['path']).parent/(req['cohort']+'__'+req['target_book']+'__'+state);result=folder/'result.json'
            if not result.exists():continue
            try:
                questions=[file_record(p) for p in sorted((folder/'questions').glob('*/record.json'))];judges={}
                for r in questions:
                    value=checked(r);path=Path(r['path']).parent/'judge_raw.json'
                    if path.exists():judges[value['question_id']]=file_record(path)
                if req['id'] in observations:raise ValueError('Multiple original DVKG contracts match one condition; explicit selection required')
                observations[req['id']]=dict(adapter='qa',contract=record,result=file_record(result),dataset=contract['datasets'][req['target_book']],questions=questions,judge_raw=judges)
            except (ValueError,KeyError,TypeError,OSError) as exc:errors.append(dict(requirement_id=req['id'],error_type=type(exc).__name__))
    return observations,errors


def ledger_snapshot(database):
    with closing(sqlite3.connect('file:'+str(Path(database))+'?mode=ro',uri=True,timeout=30)) as db:
        db.execute('BEGIN');rows=db.execute('SELECT id,created,key,stage,status FROM events').fetchall()
    counts=Counter(r[4] for r in rows);groups=Counter();stages=Counter()
    for _,_,_,stage,status in rows:
        if status!='request_started':continue
        group=next((g for prefix,g in [('echa_v1/','echa'),('identity_single_v2.','identity'),('blind64_20261022_v1/','blind64')] if stage.startswith(prefix)),'native');groups[group]+=1;stages[stage]+=1
    return dict(actual_HTTP=counts['request_started'],successful_responses=counts['response'],cache_hits=counts['cache_hit'],groups=dict(groups),stages=dict(stages),
        event_watermark=max((r[0] for r in rows),default=0),read_snapshot_digest=digest(rows),database_path=str(database),
        first_event_unix=min((r[1] for r in rows),default=None),last_event_unix=max((r[1] for r in rows),default=None),
        retries_counted_individually=True,unique_request_keys_are_not_actual_HTTP=True,monetary_cost=None,EDC='separate_ledger_not_inferred')


def paired_tables(rows,book_courses,replicates=2000,*,_auxiliary=False):
    groups=defaultdict(dict);ab=defaultdict(dict)
    for row in rows:
        if row.get('status') not in {'completed','failed'} or row.get('counts',{}).get('pending',0) or not row.get('question_scores'):continue
        key=':'.join(str(row.get(k,'')) for k in ['category','method','variant','split']);book=row.get('target_book')
        if book not in book_courses:continue
        if row.get('cohort')=='compAB':ab[(key,book)]['AB']=row['question_scores'];continue
        if row.get('cohort')=='compABCD':ab[(key,book)]['ABCD']=row['question_scores']
        if book in groups[key]:raise ValueError('Duplicate main book condition cannot increase independent sample size')
        groups[key][book]=row['question_scores']
    contrasts=[];primary=[];extensions=[]
    for key,books in sorted(groups.items()):
        if set(books)!=set(book_courses):continue
        zero={b:{qid:0 for qid in scores} for b,scores in books.items()};stat=paired_course_macro(books,zero,book_courses,replicates=replicates)
        name='f1' if _auxiliary else 'llm_judge'
        primary.append(dict(method=key,**{'course_macro_'+name:stat['mean_difference'],'book_'+name:{b:sum(s.values())/len(s) for b,s in books.items()},'course_'+name:stat['course_difference']},paired_questions=stat['paired_questions']))
    for left,right in itertools.combinations(sorted(groups),2):
        if left.split(':')[0]!=right.split(':')[0] or left.split(':')[-1]!=right.split(':')[-1] or set(groups[left])!=set(book_courses) or set(groups[right])!=set(book_courses):continue
        try:stat=paired_course_macro(groups[left],groups[right],book_courses,replicates=replicates)
        except ValueError:continue
        contrasts.append(dict(left=left,right=right,**stat,p_value=None,multiple_comparison_claim='no_p_value_or_significance_claim;intervals_descriptive'))
    for (key,book),versions in sorted(ab.items()):
        if set(versions)=={'AB','ABCD'} and set(versions['AB'])==set(versions['ABCD']):extensions.append(dict(method=key,book=book,**paired(versions['ABCD'],versions['AB'],replicates=replicates),direction='ABCD_minus_AB'))
    value=dict(primary_course_macro=primary,paired_course_uncertainty=contrasts,AB_ABCD=extensions,metric='f1' if _auxiliary else 'original_llm_judge_score')
    if not _auxiliary:
        auxiliary=[dict(r,question_scores=r['question_f1_scores']) for r in rows if r.get('question_f1_scores')]
        value['auxiliary_f1']=paired_tables(auxiliary,book_courses,replicates,_auxiliary=True)
    return value


def current_requirements(matrix):
    value=copy.deepcopy(matrix);rows=value['requirements'];cohorts=value.get('scope',{}).get('cohorts',{})
    if set(cohorts)!={'compAB','compABCD','osAB','netAB'}:return value
    books=sorted({b for group in cohorts.values() for b in group});keys={(r['category'],r['method'],r.get('cohort'),r.get('target_book'),r.get('variant'),r.get('split')) for r in rows}
    for split in ['regression313','existing_independent123','new_blind64']:
        counts=value['datasets'][split].get('by_book',8)
        for book in books:
            row=dict(category='native_end_to_end_RAG',method='graphrag',cohort=None,target_book=book,variant='Global',split=split)
            key=tuple(row[k] for k in ['category','method','cohort','target_book','variant','split'])
            if key not in keys:
                row.update(id='::'.join(str(v) for v in key if v is not None),required=True,question_count=counts[book] if isinstance(counts,dict) else counts,evidence=[]);rows.append(row);keys.add(key)
        for co,active in cohorts.items():
            for state in ['G0','T']:
                for book in active:
                    row=dict(category='DVKG_control',method='DVKG',cohort=co,target_book=book,variant=state+'_prior_frozen_dual_not_ECHA',split=split)
                    key=tuple(row[k] for k in ['category','method','cohort','target_book','variant','split'])
                    if key not in keys:
                        row.update(id='::'.join(str(v) for v in key if v is not None),required=True,question_count=counts[book] if isinstance(counts,dict) else counts,evidence=[]);rows.append(row);keys.add(key)
    value['reporting_scope_append']='Original rows retained; already-approved GraphGlobal and original G0/T DVKG conditions explicit; no new method, graph or question generated'
    graphs={(r.get('cohort'),r.get('variant')):r.get('graph_sha256') for r in rows if r['category']=='construction' and r['method']=='own_concept_graph'}
    for row in rows:
        if row['category']=='DVKG_control':
            state=row['variant'].split('_prior')[0] if row['variant'].startswith(('G0_','T_')) else 'F_corrected'
            if graphs.get((row.get('cohort'),state)):row['graph_sha256']=graphs[(row['cohort'],state)]
    return value


def QA_coverage_receipt(matrix,index,rows):
    terminal=index.get('queue_terminal')
    if not isinstance(terminal,dict) or set(terminal)!={'path','sha256'}:return None
    queue=checked(terminal)
    if queue.get('status') not in {'fixed_matrix_queue_terminal','fixed_matrix_queue_terminal_with_failures'}:return None
    expected={r['id']:r for r in matrix['requirements'] if r['category'] in QA_CATEGORIES and r.get('required',True)}
    observed={r['id']:r for r in rows if r['id'] in expected}
    if not expected or set(observed)!=set(expected):return None
    coverage=[]
    for key,req in sorted(expected.items()):
        row=observed[key];counts=row.get('counts',{})
        if (row.get('status') not in {'completed','failed'} or row.get('original_raw_revalidated') is not True or
                counts.get('fixed')!=req['question_count'] or counts.get('pending')!=0 or
                counts.get('completed',0)+counts.get('failed',0)!=req['question_count']):return None
        sources=row.get('sources',[])
        if not sources:return None
        for record in sources:
            if file_record(record['path'])!=record:raise ValueError('Final QA raw coverage source changed')
        coverage.append(dict(requirement_id=key,counts=counts,sources=sources))
    value=dict(status='completed_all_required_QA_coverage_raw_verified',matrix=index['matrix'],original_queue_terminal=terminal,
        required_QA_ids=sorted(expected),coverage=coverage,pending_questions=0,failed_questions=sum(r['counts']['failed'] for r in coverage),
        completed_questions=sum(r['counts']['completed'] for r in coverage),all_questions_successful=all(not r['counts']['failed'] for r in coverage),raw_text_exported=False)
    value['coverage_receipt_id']=digest(value);return value


def assemble(matrix,index,*,private_output=None):
    rows=[];private=[];observations=index.get('observations',{});sources=[];verified={}
    for requirement in matrix['requirements']:
        row={k:requirement.get(k) for k in IDENTITY_FIELDS};row.update(status='pending',reason='receipt_verification_pending',metrics=None)
        observation=observations.get(requirement['id'])
        if observation:
            try:
                if observation.get('adapter')=='qa':value,details=qa_observation(requirement,observation,_verified=verified);private.extend(details)
                elif observation.get('adapter')=='geometry':value=geometry_observation(requirement,observation)
                elif observation.get('adapter')=='native_index':value=native_index_observation(requirement,observation['artifact'])
                else:value=dict(status='pending',reason='category_specific_complete_receipts_required',metrics=None)
                row.update(value)
            except (ValueError,KeyError,TypeError,OSError) as exc:row.update(status='invalid',reason='source_validation_failed',error_type=type(exc).__name__)
        rows.append(row);sources.extend(row.get('sources',[]))
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate required matrix ID')
    tables=dict(geometry=[r for r in rows if r['category']=='geometry'],KGE=[r for r in rows if r['category'] in {'kge_training','kge_qualification'}],RAG=[r for r in rows if r['category'] in QA_CATEGORIES])
    paired_result=paired_tables(rows,index.get('book_courses',{})) if index.get('book_courses') else {}
    QA_proof=QA_coverage_receipt(matrix,index,rows);terminal=QA_proof is not None
    required=[r for r in rows if r.get('required',True)]
    covered=lambda row:row['status']=='completed' or (row['category'] in QA_CATEGORIES and row.get('original_raw_revalidated') is True and row.get('counts',{}).get('pending')==0 and row['status']=='failed')
    complete=bool(required) and terminal and all(covered(r) for r in required)
    value=dict(version='current_matrix_readonly_report_v1',full_matrix_complete=complete,coverage_complete=complete,
        all_questions_successful=terminal and QA_proof['all_questions_successful'],QA_original_coverage_receipt=QA_proof,
        QA_queue_terminal_original_receipts_verified=terminal,required_positions=len(required),status_counts=dict(Counter(r['status'] for r in rows)),rows=rows,tables=tables,
        **paired_result,question_errors=[dict(requirement_id=r['id'],**e) for r in rows for e in r.get('question_errors',[])],
        source_index=list({digest(r):r for r in sources}.values()),historical_matrix=matrix.get('historical_matrix'),cost=index.get('cost'),collection_errors=index.get('collection_errors',[]),
        limitations=['Required positions are obligations, not independent samples or completed experiments.','Missing questions remain pending, not executed zeros.','Geometry is structural description; native/controlled and Local/Global remain separate.','No winner, statistical significance or generalization to new courses is assumed.'],
        model_API_calls=0,raw_text_exported=False)
    if private_output is not None:
        path=Path(private_output)/'detailed_errors_primary_only.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in private),encoding='utf-8');value['private_error_table']=file_record(path)
    return value


def collect_pinned_evidence(matrix,matrix_record,*,database=None):
    observations={};collection_errors=[]
    for req in matrix['requirements']:
        evidence=req.get('evidence',[])
        # 尚未封存的新题没有可读入口；不扫描candidate/source或凭目录猜最新结果。
        if req.get('split')=='new_blind64':continue
        try:
            if req['category']=='geometry' and evidence:
                result=checked(evidence[0]);observations[req['id']]=dict(adapter='geometry',result=evidence[0],graph=result['graph'],helper=result['helper'])
            elif req['category']=='controlled_graph_RAG' and len(evidence)>=2:
                result,contract=evidence[:2];value=checked(contract);checked(result);book=req['target_book'];root=Path(result['path']).parent/book
                questions=[file_record(p) for p in sorted((root/'questions').glob('*/record.json'))];judges={}
                for record in questions:
                    q=checked(record);path=Path(record['path']).parent/'judge_raw.json'
                    if path.exists():judges[q['question_id']]=file_record(path)
                observations[req['id']]=dict(adapter='qa',contract=contract,result=result,dataset=value['datasets'][book],questions=questions,judge_raw=judges)
        except (ValueError,KeyError,TypeError,OSError) as exc:collection_errors.append(dict(requirement_id=req['id'],error_type=type(exc).__name__))
    index=dict(version='current_matrix_report_index_v1',matrix=matrix_record,observations=observations,collection_errors=collection_errors,model_API_calls=0,new64_question_sources_read=False)
    cohorts=matrix.get('scope',{}).get('cohorts',{})
    if set(cohorts)=={'compAB','compABCD','osAB','netAB'}:
        index['book_courses']={b:course for co,course in [('compABCD','computer_organization'),('osAB','operating_systems'),('netAB','computer_networks')] for b in cohorts[co]}
    if database:index['cost']=ledger_snapshot(database)
    return index


def render(value,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    chunks=['<!doctype html><meta charset="utf-8"><title>当前全矩阵实证汇总</title><style>body{font:14px sans-serif;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%;margin-bottom:30px}td,th{border:1px solid #ddd;padding:6px;overflow-wrap:anywhere}small{color:#555}</style><h1>当前全矩阵实证汇总</h1>',f'<p>完整验收：{value["full_matrix_complete"]}。义务数：{value["required_positions"]}。已深验 {value["status_counts"].get("completed",0)}，待收据核实 {value["status_counts"].get("pending",0)}；此数不代表已运行实验总数。失败和训练不足保留。成对与课程等权主指标为原 LLM 评分，F1 为辅助；证据与 grounding 不适用项不计为零。</p>']
    for name,rows in value['tables'].items():
        chunks.append('<h2>'+html.escape(name)+'</h2><table><tr><th>方法/状态</th><th>组/教材/题集</th><th>真实状态</th><th>固定分母/覆盖</th><th>指标</th></tr>')
        for row in rows:
            cells=[str(row.get('method'))+'/'+str(row.get('variant')),str(row.get('cohort'))+'/'+str(row.get('target_book'))+'/'+str(row.get('split')),'待收据核实' if row['status']=='pending' else row['status'],row.get('counts'),row.get('metrics')]
            chunks.append('<tr>'+''.join('<td>'+html.escape(json.dumps(c,ensure_ascii=False) if isinstance(c,(dict,list)) else str(c))+'</td>' for c in cells)+'</tr>')
        chunks.append('</table>')
    chunks.append('<small>原文、问题、答案、逐字引文与详细错误留在主服务器。来源SHA索引和所有缺口见summary.json。</small>')
    (output/'index.html').write_text(''.join(chunks),encoding='utf-8')
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--matrix',type=Path,required=True);parser.add_argument('--matrix-sha',required=True);parser.add_argument('--index',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--private-output',type=Path);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv);record=dict(path=str(args.matrix.resolve()),sha256=args.matrix_sha);matrix=checked(record)
    while True:
        index=read_document(args.index)
        if index.get('matrix')!=record:raise ValueError('Result index belongs to another fixed expected matrix')
        result=assemble(matrix,index,private_output=args.private_output or args.output/'private');render(result,args.output)
        if not args.watch or result['full_matrix_complete'] or time.time()>=1792950136.907112:break
        time.sleep(30)
    print(json.dumps(dict(full_matrix_complete=result['full_matrix_complete'],status_counts=result['status_counts'],summary=file_record(args.output/'summary.json'),html=file_record(args.output/'index.html'),model_API_calls=0,raw_text_exported=False)))


if __name__=='__main__':main()
