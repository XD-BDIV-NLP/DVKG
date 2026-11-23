"""冻结修正版的逐题评测；全题分母、原始收据和断点均保留。"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable
from .chat import call_json,account_status
from .rag import BUDGET,citations_score,evidence_recall,score_answer
from . import rag_repair_v3 as repair


def answerable(question,available):
    return set(question['evidence_ids'])<=available


def request_identity(prompt,payload,max_tokens,config):
    return digest(dict(prompt=prompt,payload=payload,max_tokens=max_tokens,runtime_config=config,
                       temperature=0,seed=42,enable_thinking=False))


def summarize(records,expected):
    ids=[r['question_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate question checkpoint')
    if set(ids)-set(expected):raise ValueError('Unknown question checkpoint')
    completed=[r for r in records if r['status']=='completed']
    absent=[r for r in records if r['status']=='unanswerable_active_sources']
    n=len(expected);citations=sum(r['scores']['citation_count'] for r in completed)
    metrics={key:sum(r['scores'][key] for r in completed)/n if n else None for key in ['f1','em','evidence_recall']}
    metrics.update(machine_judge_mean=sum(r['machine_judge']['score'] for r in completed)/n if n else None,
        citation_count=citations,citation_correctness=sum(r['scores']['valid_citation_count'] for r in completed)/citations if citations else None)
    return dict(status='completed' if len(records)==n and len(completed)+len(absent)==n else 'partial',
        question_count=n,completed_count=len(completed),unanswerable_count=len(absent),
        checkpoint_count=len(records),answerability_coverage=(n-len(absent))/n if n else None,
        metrics=metrics,conditional_metrics={k:sum(r['scores'][k] for r in completed)/len(completed) if completed else None for k in ['f1','em','evidence_recall']},
        denominator_policy='all_fixed_questions_unanswerable_zero_not_silently_dropped')


def limited_runtime(runtime,base):
    base=Path(base);limits=read_document(base/'limits.json');original=runtime._record
    def record(key,stage,status,value):
        code=value.get('status_code') if isinstance(value,dict) else None
        if status=='http_error' and code in {401,402,403}:
            write_immutable(base/'halt.json',dict(status='account_or_permission',status_code=code))
        with sqlite3.connect(base/'budget.sqlite',timeout=60) as db:
            db.execute('BEGIN IMMEDIATE')
            if status=='request_started':
                if (base/'halt.json').exists():raise RuntimeError('Finite experiment halted; preserve checkpoints')
                count,first=db.execute("SELECT COUNT(*),MIN(created) FROM events WHERE status='request_started'").fetchone()
                if count>=limits['max_http_attempts'] or first and time.time()-first>limits['max_seconds']:
                    raise RuntimeError('Finite experiment budget exhausted; preserve checkpoints')
            db.execute('INSERT INTO events(created,stage,status,code) VALUES(?,?,?,?)',(time.time(),stage,status,code))
        original(key,stage,status,value)
    runtime._record=record
    return runtime


def shared_call(runtime,prompt,payload,max_tokens,root,base,attempt=0):
    root=Path(root);key=request_identity(prompt,payload,max_tokens,runtime.config)
    expected_body=dict(runtime.config['llm'].get('extra_body') or {})
    expected_body.update(model=runtime.config['llm']['model'],temperature=0,max_tokens=max_tokens,seed=42,enable_thinking=False,
        response_format={'type':'json_object'},messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    with sqlite3.connect(Path(base)/'request_cache.sqlite',timeout=60) as db:
        cached=db.execute('SELECT record FROM responses WHERE key=?',(key,)).fetchone()
    if cached:
        stored=json.loads(cached[0]);source=stored['source']
        if file_record(source['path'])!=source:raise ValueError('Reused model receipt changed')
        raw=read_document(source['path'])
        if raw['request']!=expected_body or digest(raw['request'])!=stored['request_digest'] or digest(raw['response'])!=stored['response_digest']:
            raise ValueError('Reused model receipt identity mismatch')
        value=json.loads(raw['response']['choices'][0]['message']['content'])
        write_immutable(root/('reused_'+key+'.json'),dict(semantic_request_id=key,source=source,
                        request_digest=stored['request_digest'],response_digest=stored['response_digest']))
        return value,dict(reused=True,semantic_request_id=key,source=source)
    value,record=call_json(runtime,prompt,payload,max_tokens,root,attempt)
    if max_tokens==2048 and (not isinstance(value.get('answer'),str) or not isinstance(value.get('citations'),list)):
        raise ValueError('Invalid candidate cannot enter shared cache')
    if max_tokens==1024 and (type(value.get('score')) not in (int,float) or not math.isfinite(value['score']) or not 0<=value['score']<=1):
        raise ValueError('Invalid judge cannot enter shared cache')
    body=record['request'];cachekey=digest(dict(body=body,runtime_config=runtime.config,attempt=attempt,root=str(root)))
    source=file_record(root/(cachekey+'.json'))
    stored=dict(source=source,request_digest=digest(body),response_digest=digest(record['response']))
    with sqlite3.connect(Path(base)/'request_cache.sqlite',timeout=60) as db:
        db.execute('INSERT OR IGNORE INTO responses(key,record) VALUES(?,?)',(key,json.dumps(stored)))
    return value,dict(reused=False,semantic_request_id=key,source=source)


def evaluate(job,phase,output,runtime,base,*,query_vectors=None,stop_after=None):
    from .protocol import ROOT,bind_runtime
    bind_runtime(phase,runtime)
    for record in phase['files'].values():
        if file_record(record['path'])!=record:raise ValueError('Frozen evaluation input changed')
    index=read_document(phase['files']['index']['path']);graph=read_document(phase['files']['graph']['path']);reference=read_document(phase['files']['reference']['path'])
    rows=[r for r in index['rows'] if r['book_id'] in job['active_books']]
    available={e for r in rows for e in r['evidence_ids']}
    retriever=repair.RepairedRetriever(rows,graph)
    import functools
    retriever.graph.lexical=functools.lru_cache(maxsize=1024)(retriever.graph.lexical)
    identity=digest(dict(job=job,phase=phase['phase_id'],policy=repair.POLICY))
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    write_immutable(root/'identity.json',dict(result_id=identity,job=job,phase_id=phase['phase_id'],policy=repair.POLICY,
        reference_id=reference.get('reference_id',reference.get('dataset_id')),budget=BUDGET))
    answer_prompt=(ROOT/'prompts/evaluation/answer.txt').read_text('utf-8')
    judge_prompt=(ROOT/'prompts/evaluation/judge.txt').read_text('utf-8')
    records=[];failed=[]
    vector_units=None;matrix=None
    if job['control']=='vector':
        import numpy as np
        unit=read_document(phase['files']['unit_index']['path'])
        vector_units=unit['rows'];matrix=np.load(unit['vectors']['path'],mmap_mode='r')
        mask=[i for i,r in enumerate(vector_units) if r['book_id'] in job['active_books']]
        vector_units=[vector_units[i] for i in mask];matrix=matrix[mask]
    for q in reference['questions']:
        if stop_after is not None and len(records)>=stop_after:break
        path=root/'questions'/(q['question_id']+'.json')
        if path.exists():
            record=read_document(path)
            if record['result_id']!=identity or record['checkpoint_digest']!=digest({k:v for k,v in record.items() if k!='checkpoint_digest'}):
                raise ValueError('Question checkpoint changed')
            records.append(record);continue
        record=None
        if not answerable(q,available):
            record=dict(question_id=q['question_id'],result_id=identity,status='unanswerable_active_sources',
                missing_evidence_ids=sorted(set(q['evidence_ids'])-available),question_type=q['type'],completed_unix=time.time())
        else:
            for attempt in range(BUDGET['retries']+1):
                try:
                    if job['control']=='graph':selected,trace=retriever.retrieve(q['question'])
                    elif job['control']=='keyword':
                        scores=retriever.units.lexical(q['question'])
                        selected=[r for _,r in sorted(zip(scores,retriever.units.rows),key=lambda x:(-x[0],x[1]['book_id'],x[1]['chunk_id'],x[1]['evidence_ids'][0]))[:5]]
                        trace=dict(control='unit_keyword')
                    elif job['control']=='vector':
                        import numpy as np
                        vector=np.asarray((query_vectors or {}).get(q['question_id']) if q['question_id'] in (query_vectors or {}) else runtime.embed([q['question']])[0],dtype=np.float32)
                        vector=vector/(np.linalg.norm(vector) or 1.)
                        scores=matrix@vector
                        order=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),vector_units[i]['book_id'],vector_units[i]['chunk_id'],vector_units[i]['evidence_ids'][0]))[:5]
                        selected=[vector_units[i] for i in order];trace=dict(control='unit_vector')
                    else:raise ValueError('Unknown repaired control')
                    context=repair.pack_context(selected,q['question'])
                    write_immutable(root/'retrieval'/(q['question_id']+'.json'),dict(result_id=identity,question_id=q['question_id'],context=context,trace=trace))
                    candidate,answer_receipt=shared_call(runtime,answer_prompt,dict(question=q['question'],context=context),2048,root/'requests'/q['question_id']/'answer',base,attempt)
                    if not isinstance(candidate.get('answer'),str) or not isinstance(candidate.get('citations'),list):raise ValueError('Invalid candidate')
                    judge,judge_receipt=shared_call(runtime,judge_prompt,dict(question=q['question'],reference_answer=q['answer'],reference_citations=q['citations'],candidate=candidate),1024,root/'requests'/q['question_id']/'judge',base,attempt)
                    if type(judge.get('score')) not in (int,float) or not math.isfinite(judge['score']) or not 0<=judge['score']<=1:raise ValueError('Invalid judge')
                    record=dict(question_id=q['question_id'],result_id=identity,status='completed',question_type=q['type'],question=q['question'],
                        context=context,retrieval=trace,candidate=candidate,machine_judge=judge,
                        receipts=dict(answer=answer_receipt,judge=judge_receipt),
                        scores=dict(score_answer(candidate['answer'],q['answer']),**citations_score(candidate['citations'],context,rows),evidence_recall=evidence_recall(q,context)),
                        context_bytes=len(json.dumps(context,ensure_ascii=False).encode('utf-8')),human_audited=False,completed_unix=time.time())
                    break
                except (ValueError,KeyError,TypeError) as exc:
                    write_immutable(root/'errors'/(q['question_id']+'-'+str(attempt)+'.json'),dict(error_type=type(exc).__name__))
                except Exception as exc:
                    if account_status(exc):raise
                    raise
        if record is None:failed.append(q['question_id']);continue
        record['checkpoint_digest']=digest(record);write_immutable(path,record);records.append(record)
    result=dict(summarize(records,[q['question_id'] for q in reference['questions']]),result_id=identity,job=job,
        phase_id=phase['phase_id'],failed_questions=failed,human_audited=False)
    result['by_type']={t:summarize([r for r in records if r['question_type']==t],[q['question_id'] for q in reference['questions'] if q['type']==t]) for t in sorted({q['type'] for q in reference['questions']})}
    write_immutable(root/('result.json' if result['status']=='completed' else 'partial_'+digest(result)+'.json'),result)
    return result
