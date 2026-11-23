from __future__ import annotations
import json
import math
import re
from collections import Counter
from pathlib import Path
from numbers import Real
from urllib.error import HTTPError
from ..manifest import digest,read_document
from ..contributions import write_immutable
from .reference import source_units,validate_reference
from .protocol import ROOT,validate_phase,bound_input,bind_runtime
from .chat import call_json

BUDGET=dict(seed=42,retrieval_top_k=5,context_token_limit=3000,context_counting_policy='UTF8 bytes ceiling including source IDs and separators; conservative BPE byte upper bound, not exact model tokenizer',max_tokens=2048,temperature=0,judge_max_tokens=1024,judge_temperature=0,retries=2,embedding_model='BAAI/bge-m3',answer_model='deepseek-ai/DeepSeek-V4-Flash',judge_model='deepseek-ai/DeepSeek-V4-Flash')

def result_identity(reference_id,index_id,graph_id,books,control,*,phase_id=None,runtime_id=None):
    return digest(dict(reference_id=reference_id,index_id=index_id,graph_id=graph_id,active_books=sorted(books),control=control,budget=BUDGET,phase_id=phase_id,runtime_id=runtime_id))

def numeric_vector(value):
    if getattr(value, 'ndim', 1)!=1 or not hasattr(value, '__len__') or len(value)==0:
        raise ValueError('Embedding must be a nonempty numeric vector')
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in value):
        raise ValueError('Embedding must contain finite numeric scalars')
    return [float(x) for x in value]


def cosine(a,b):
    a,b=numeric_vector(a),numeric_vector(b)
    if len(a)!=len(b):raise ValueError('Embedding dimensions differ/empty')
    aa=sum(x*x for x in a);bb=sum(x*x for x in b)
    return sum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb) if aa and bb else 0.

def build_index(corpora,runtime,output):
    units=source_units(corpora);source_id=digest(units);path=Path(output)/('index_'+source_id+'.json')
    if path.exists():
        old=read_document(path)
        if old['index_id']!=digest({k:v for k,v in old.items() if k!='index_id'}) or old['source_id']!=source_id:raise ValueError('Index changed')
        dimensions={len(numeric_vector(r['vector'])) for r in old['rows']}
        if len(dimensions)!=1 or len(old['rows'])!=len(units) or old['embedding_model']!=BUDGET['embedding_model']:raise ValueError('Invalid cached index dimensions/model/rows')
        return old
    rows=[]
    for i in range(0,len(units),16):
        vectors=runtime.embed([u['text'] for u in units[i:i+16]])
        if len(vectors)!=len(units[i:i+16]):raise ValueError('Missing index embedding')
        for u,v in zip(units[i:i+16],vectors):
            v=numeric_vector(v)
            if rows and len(v)!=len(rows[0]['vector']):raise ValueError('Embedding dimensions differ')
            rows.append(dict(u,evidence_ids=[e['id'] for e in u['evidence']],vector=v))
    result=dict(source_id=source_id,embedding_model=BUDGET['embedding_model'],rows=rows);result['index_id']=digest(result);write_immutable(path,result);return result

def retrieve(question,query_vector,index,graph,control):
    if control not in ('graph','vector_only','no_retrieval'):raise ValueError('Unknown RAG control')
    if control=='no_retrieval':return []
    names={}
    for e in graph.get('entities',graph.get('nodes',[])):
        if e.get('type') in {'Book','Chapter','Section','Chunk','Evidence'}:continue
        names[e.get('entity_id',e.get('id'))]=e.get('canonical_name',e.get('name'))
    seeds={nid for nid,name in names.items() if name and name.casefold() in question.casefold()}
    neighbors=set(seeds)
    for r in graph.get('relations',graph.get('edges',[])):
        h,t=r.get('head_entity_id',r.get('source')),r.get('tail_entity_id',r.get('target'))
        if h in seeds and t in names:neighbors.add(t)
        if t in seeds and h in names:neighbors.add(h)
    graph_terms=[names[n] for n in neighbors]
    scores=[]
    for row in index:
        similarity=cosine(query_vector,row['vector'])
        navigation=sum(term in row['text'] for term in graph_terms)/max(1,len(graph_terms)) if control=='graph' else 0.
        scores.append((similarity+.25*navigation,row))
    return [r for _,r in sorted(scores,key=lambda s:(-s[0],s[1]['book_id'],s[1]['chunk_id']))[:BUDGET['retrieval_top_k']]]

def context_rows(rows):
    kept=[]
    for row in rows:
        clean=dict(chunk_id=row['chunk_id'],book_id=row['book_id'],evidence_ids=row['evidence_ids'],text=row['text'])
        # 统一用序列化后 UTF-8 字节数上界计入证据 ID 和分隔符。
        original=clean['text'];lo,hi=0,len(original)
        while lo<hi:
            mid=(lo+hi+1)//2;clean['text']=original[:mid]
            if len(json.dumps(kept+[clean],ensure_ascii=False).encode('utf-8'))<=BUDGET['context_token_limit']:lo=mid
            else:hi=mid-1
        clean['text']=original[:lo]
        if clean['text']:kept.append(clean)
    return kept

def tokens(text):
    return re.findall(r'[\u3400-\u9fff]|[a-z0-9]+',text.casefold())

def score_answer(candidate,answer):
    a,b=tokens(candidate),tokens(answer);common=sum((Counter(a)&Counter(b)).values())
    precision=common/len(a) if a else 0;recall=common/len(b) if b else 0
    return dict(em=int(a==b),f1=2*precision*recall/(precision+recall) if precision+recall else 0,tokenization='Chinese Han characters + Latin alphanumeric words, lowercase, punctuation ignored')

def citations_score(citations,contexts,index):
    raw={e['id']:e['text'] for row in index for e in row.get('evidence',[])}
    valid=[]
    for c in citations:
        eid,quote=c.get('evidence_id'),c.get('quote','')
        valid.append(bool(quote and eid in raw and quote in raw[eid] and any(eid in r['evidence_ids'] and quote in r['text'] for r in contexts)))
    return dict(citation_count=len(valid),valid_citation_count=sum(valid),citation_correctness=sum(valid)/len(valid) if valid else None,checks=valid)

def evidence_recall(question,contexts):
    cites=question['citations']
    return sum(any(c['evidence_id'] in r['evidence_ids'] and c['quote'] in r['text'] for r in contexts) for c in cites)/len(cites) if cites else None

def evaluate(reference,index,graph,active_books,control,phase_path,output,runtime):
    phase=validate_phase(phase_path)
    runtime_id=bind_runtime(phase,runtime)
    validate_reference(reference)
    for role,value in [('reference',reference),('index',index),('graph',graph)]:bound_input(phase,role,value)
    if phase['parameters']!=dict(BUDGET,control=control,active_books=sorted(active_books)):raise ValueError('RAG budget/control/bookset changed')
    if index['index_id']!=digest({k:v for k,v in index.items() if k!='index_id'}):raise ValueError('Index digest mismatch')
    rows=[r for r in index['rows'] if r['book_id'] in active_books]
    if {r['book_id'] for r in rows}!=set(active_books):raise ValueError('Active corpus missing')
    identity=result_identity(reference['reference_id'],digest(rows),digest(graph),active_books,control,phase_id=phase['phase_id'],runtime_id=runtime_id);root=Path(output)/identity;root.mkdir(parents=True,exist_ok=True)
    write_immutable(root/'identity.json',dict(result_id=identity,phase_id=phase['phase_id'],runtime_id=runtime_id,reference_id=reference['reference_id'],index_id=index['index_id'],graph_id=digest(graph),active_books=sorted(active_books),budget=BUDGET,control=control))
    answer_prompt=(ROOT/'prompts/evaluation/answer.txt').read_text('utf-8');judge_prompt=(ROOT/'prompts/evaluation/judge.txt').read_text('utf-8');records=[];errors=[]
    available={eid for r in rows for eid in r['evidence_ids']}
    for q in reference['questions']:
        checkpoint=root/'questions'/(q['question_id']+'.json')
        if checkpoint.exists():records.append(read_document(checkpoint));continue
        answerable=set(q['evidence_ids'])<=available
        if not answerable:
            result=dict(question_id=q['question_id'],status='unanswerable_active_sources',scores=None,reason='Required independent source evidence outside active books');write_immutable(checkpoint,result);records.append(result);continue
        result=None
        for attempt in range(BUDGET['retries']+1):
            try:
                bind_runtime(validate_phase(phase_path),runtime)
                query=[] if control=='no_retrieval' else runtime.embed([q['question']])[0]
                context=context_rows(retrieve(q['question'],query,rows,graph,control));payload=dict(question=q['question'],context=context)
                candidate,answer_raw=call_json(runtime,answer_prompt,payload,2048,root/'requests'/q['question_id']/'answer',attempt)
                if not isinstance(candidate.get('answer'),str) or not isinstance(candidate.get('citations'),list):raise ValueError('Invalid answer object')
                for c in candidate['citations']:
                    if not isinstance(c,dict):raise ValueError('Invalid answer citation')
                judge,judge_raw=call_json(runtime,judge_prompt,dict(question=q['question'],reference_answer=q['answer'],reference_citations=q['citations'],candidate=candidate),1024,root/'requests'/q['question_id']/'judge',attempt)
                if type(judge.get('score')) not in (int,float) or not math.isfinite(judge['score']) or not 0<=judge['score']<=1:raise ValueError('Invalid judge score')
                retrieved={eid for r in context for eid in r['evidence_ids']}
                result=dict(question_id=q['question_id'],status='completed',question=q['question'],reference_answer=q['answer'],context=context,candidate=candidate,answer_raw=answer_raw,judge_raw=judge_raw,
                    scores=dict(score_answer(candidate['answer'],q['answer']),**citations_score(candidate['citations'],context,rows),evidence_recall=evidence_recall(q,context)),machine_judge=judge,human_audited=False)
                write_immutable(checkpoint,result);records.append(result);break
            except HTTPError as exc:
                write_immutable(root/'errors'/f"{q['question_id']}-{attempt}.json",dict(error_type='HTTPError',status_code=exc.code))
                if exc.code in {401,402,403}:raise
            except (ValueError,KeyError,TypeError) as exc:
                write_immutable(root/'errors'/f"{q['question_id']}-{attempt}.json",dict(error_type=type(exc).__name__,reason=str(exc)))
        if result is None:errors.append(q['question_id'])
    completed=[r for r in records if r['status']=='completed']
    summary=dict(result_id=identity,course_id=reference['course_id'],evaluation='rag',graph_id=digest(graph),input_digest=phase['phase_id'],split_id=reference['reference_id'],seed=42,status='partial' if errors else 'completed',control=control,active_books=sorted(active_books),question_count=len(reference['questions']),completed_count=len(completed),unanswerable_count=sum(r['status']=='unanswerable_active_sources' for r in records),failed_questions=errors,answerability_coverage=(len(completed)+len(errors))/len(reference['questions']) if reference['questions'] else None,
        metrics={k:sum(r['scores'][k] for r in completed)/len(completed) if completed else None for k in ('em','f1','evidence_recall')},machine_judge_mean=sum(r['machine_judge']['score'] for r in completed)/len(completed) if completed else None,human_audited=False,raw_paths=[str(root/'questions'),str(root/'requests')])
    count=sum(r['scores']['citation_count'] for r in completed)
    summary['metrics']['citation_correctness']=sum(r['scores']['valid_citation_count'] for r in completed)/count if count else None
    write_immutable(root/('result.json' if not errors else 'partial_result_'+digest(summary)+'.json'),summary);return summary
