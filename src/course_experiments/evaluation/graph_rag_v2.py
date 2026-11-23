from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError

from ..contributions import write_immutable
from ..manifest import digest, read_document
from .chat import account_status, call_json
from .protocol import ROOT, bind_runtime, bound_input, validate_phase
from .rag import BUDGET as OLD_BUDGET, citations_score, cosine, evidence_recall, score_answer
from .reference import validate_reference

POLICY = dict(version='evidence_graph_rag_v2',max_seeds=8,max_hops=2,branch_limit=8,
              path_limit=64,bm25_k1=1.2,bm25_b=.75,
              graph_ranking='seed_coverage_then_original_question_BM25_then_shortest_path',
              source_resolution='exact_id_or_course_book_evidence_namespace',
              prior_policy='endpoint_lexical_navigation_not_source_evidence',
              context_policy='equal_serialized_byte_quota_one_question_selected_source_unit_per_chunk',
              tuning='synthetic_fixtures_only_posthoc_supplement_not_preregistered')
BUDGET = dict(OLD_BUDGET,retrieval_policy=POLICY)
STRUCTURAL = {'Book','Chapter','Section','Chunk','Evidence','Document'}


def normalize(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).casefold()).strip()


def terms(text):
    found=[]
    for part in re.findall(r'[\u3400-\u9fff]+|[a-z0-9_]+',normalize(text)):
        found.extend([part] if len(part)==1 or part.isascii() else
                     [part[i:i+2] for i in range(len(part)-1)])
    return found


def matches(name, text):
    name=normalize(name)
    if not name:return False
    left=r'(?<![a-z0-9_])' if name[0].isascii() and name[0].isalnum() else ''
    right=r'(?![a-z0-9_])' if name[-1].isascii() and name[-1].isalnum() else ''
    return re.search(left+re.escape(name)+right,normalize(text)) is not None


class EvidenceRetriever:
    def __init__(self,index,graph):
        self.rows=sorted(index,key=lambda r:(r['book_id'],r['chunk_id']))
        self.keys=[(r['book_id'],r['chunk_id']) for r in self.rows]
        self.counts=[Counter(terms(r['text'])) for r in self.rows]
        self.lengths=[sum(c.values()) for c in self.counts]
        self.average=sum(self.lengths)/max(1,len(self.rows)) or 1.
        self.postings=defaultdict(list)
        self.evidence=defaultdict(set)
        for i,(r,c) in enumerate(zip(self.rows,self.counts)):
            for term in c:self.postings[term].append(i)
            for eid in r['evidence_ids']:self.evidence[eid].add(i)
        self.entities={e.get('entity_id',e.get('id')):e for e in graph.get('entities',graph.get('nodes',[]))
                       if e.get('type') not in STRUCTURAL}
        self.adjacency=defaultdict(list)
        for edge in graph.get('relations',graph.get('edges',[])):
            h=edge.get('head_entity_id',edge.get('source'));t=edge.get('tail_entity_id',edge.get('target'))
            if h in self.entities and t in self.entities and h!=t:
                self.adjacency[h].append((t,edge,'outgoing'))
                self.adjacency[t].append((h,edge,'incoming'))

    def lexical(self,question):
        scores=[0.]*len(self.rows);n=len(scores)
        for term in sorted(set(terms(question))):
            posting=self.postings.get(term,[])
            idf=math.log(1+(n-len(posting)+.5)/(len(posting)+.5))
            for i in posting:
                count=self.counts[i][term]
                scores[i]+=idf*count*(POLICY['bm25_k1']+1)/(count+POLICY['bm25_k1']*(1-POLICY['bm25_b']+POLICY['bm25_b']*self.lengths[i]/self.average))
        return scores

    def source_rows(self,value):
        ids=list(value.get('source_node_ids',value.get('evidence_ids',[])))
        ids.extend(e.get('id') for e in value.get('evidence',[]) if isinstance(e,dict))
        found=set()
        for eid in ids:
            if not isinstance(eid,str):continue
            if eid in self.evidence:found.update(self.evidence[eid]);continue
            parts=eid.split('/evidence/',1)
            if len(parts)!=2:continue
            prefix=parts[0].split('/')
            if len(prefix)!=2:continue
            book=prefix[1]
            found.update(i for i in self.evidence.get(parts[1],[]) if self.rows[i]['book_id']==book)
        return found

    def names(self,nid):
        e=self.entities[nid]
        return sorted({v for v in [e.get('canonical_name',e.get('name','')),*e.get('aliases',[])] if isinstance(v,str) and v})

    def retrieve(self,question,control,query_vector=None):
        if not isinstance(question,str):raise TypeError('Retriever takes question text only')
        if control not in {'graph','keyword','vector'}:raise ValueError('Unknown retrieval control')
        lexical=self.lexical(question)
        trace=dict(control=control,seed_ids=[],paths=[],fallback=None,source_anchored_candidates=0,navigation_candidates=0)
        if control=='vector':
            scores=[cosine(query_vector,r['vector']) for r in self.rows]
            order=sorted(range(len(self.rows)),key=lambda i:(-scores[i],self.keys[i]))
        elif control=='keyword':
            order=sorted((i for i,s in enumerate(lexical) if s>0),key=lambda i:(-lexical[i],self.keys[i]))
        else:
            seeds=[]
            for nid in sorted(self.entities):
                hit=[len(normalize(name)) for name in self.names(nid) if matches(name,question)]
                if hit:seeds.append((max(hit),nid))
            seeds=[nid for _,nid in sorted(seeds,key=lambda x:(-x[0],x[1]))[:POLICY['max_seeds']]]
            trace['seed_ids']=seeds
            candidates=defaultdict(lambda:dict(seeds=set(),depth=99))
            navigation=set();supported=set()
            def add(indices,seed,depth,anchored):
                for i in indices:
                    candidates[i]['seeds'].add(seed);candidates[i]['depth']=min(candidates[i]['depth'],depth)
                (supported if anchored else navigation).update(indices)
            def navigate(nid,seed,depth):
                anchored=self.source_rows(self.entities[nid])
                if anchored:add(anchored,seed,depth,True)
                else:
                    scores=self.lexical(' '.join(self.names(nid)))
                    top=sorted((i for i,s in enumerate(scores) if s>0),key=lambda i:(-scores[i],self.keys[i]))[:BUDGET['retrieval_top_k']]
                    add(top,seed,depth,False)
            for seed in seeds:
                navigate(seed,seed,0)
                frontier=[(seed,[seed],[],[])]
                for depth in range(1,POLICY['max_hops']+1):
                    next_frontier=[]
                    for nid,visited,path,directions in frontier:
                        adjacent=[]
                        for target,e,direction in self.adjacency.get(nid,[]):
                            if target in visited:continue
                            relevance=len(set(terms(question))&set(terms(' '.join(self.names(target))+' '+e.get('relation',''))))
                            rid=e.get('relation_id') or e.get('id') or digest(e)
                            adjacent.append((-relevance,len(self.adjacency[target]),str(rid),target,direction,e))
                        for _,_,rid,target,direction,e in sorted(adjacent,key=lambda x:x[:5])[:POLICY['branch_limit']]:
                            if len(trace['paths'])>=POLICY['path_limit']:break
                            edges=path+[rid];dirs=directions+[direction]
                            anchors=self.source_rows(e);add(anchors,seed,depth,True)
                            navigate(target,seed,depth)
                            trace['paths'].append(dict(seed_id=seed,node_ids=visited+[target],relation_ids=edges,directions=dirs,
                                                       relation_type=e.get('relation','native_adjacency'),source_anchored=bool(anchors)))
                            next_frontier.append((target,visited+[target],edges,dirs))
                    frontier=next_frontier
            trace['source_anchored_candidates']=len(supported);trace['navigation_candidates']=len(navigation)
            if candidates:
                order=sorted(candidates,key=lambda i:(-len(candidates[i]['seeds']),-lexical[i],candidates[i]['depth'],self.keys[i]))
            else:
                trace['fallback']='keyword_no_seed' if not seeds else 'keyword_no_graph_evidence'
                order=sorted((i for i,s in enumerate(lexical) if s>0),key=lambda i:(-lexical[i],self.keys[i]))
        selected=[self.rows[i] for i in order[:BUDGET['retrieval_top_k']]]
        trace['retrieved_chunk_ids']=[r['chunk_id'] for r in selected]
        return selected,trace


def window(text,question,width=800):
    if len(text)<=width:return text
    anchors=set(terms(question));positions={0}
    folded=text.casefold()
    for term in sorted(anchors):
        start=0
        for _ in range(16):
            pos=folded.find(term,start)
            if pos<0:break
            positions.add(max(0,pos-width//3));start=pos+max(1,len(term))
    start=min(positions,key=lambda p:(-len(anchors&set(terms(text[p:p+width]))),p))
    return text[start:start+width]


def pack_context(rows,question):
    if not rows:return []
    quota=(BUDGET['context_token_limit']-2-2*len(rows))//len(rows);kept=[]
    for row in rows:
        units=[e for e in row.get('evidence',[]) if e.get('id') in row['evidence_ids'] and isinstance(e.get('text'),str) and e['text']]
        if not units:raise ValueError('Context requires exact source evidence units')
        question_terms=set(terms(question))
        unit=min(units,key=lambda e:(-len(question_terms&set(terms(e['text']))),e['id']))
        clean=dict(book_id=row['book_id'],chunk_id=row['chunk_id'],evidence_ids=[unit['id']])
        clean['text']=''
        overhead=len(json.dumps(clean,ensure_ascii=False).encode('utf-8'))
        text=window(unit['text'],question,min(800,max(1,(quota-overhead)//4)));lo,hi=0,len(text)
        while lo<hi:
            mid=(lo+hi+1)//2;clean['text']=text[:mid]
            if len(json.dumps(clean,ensure_ascii=False).encode('utf-8'))<=quota:lo=mid
            else:hi=mid-1
        clean['text']=text[:lo]
        if clean['text']:kept.append(clean)
    assert len(json.dumps(kept,ensure_ascii=False).encode('utf-8'))<=BUDGET['context_token_limit']
    return kept


def retrieval_audit(reference_question,selected,context):
    return dict(evidence_recall_before_packing=evidence_recall(reference_question,selected),
                evidence_recall=evidence_recall(reference_question,context),
                context_rows=len(context),context_bytes=len(json.dumps(context,ensure_ascii=False).encode('utf-8')))


def evaluate(reference,index,graph,active_books,control,phase_path,output,runtime,*,stop_after=None):
    phase=validate_phase(phase_path);runtime_id=bind_runtime(phase,runtime);validate_reference(reference)
    for role,value in [('reference',reference),('index',index),('graph',graph)]:bound_input(phase,role,value)
    if phase['parameters']!=dict(BUDGET,control=control,active_books=sorted(active_books)):raise ValueError('New retrieval budget differs')
    if index['index_id']!=digest({k:v for k,v in index.items() if k!='index_id'}):raise ValueError('Index hash differs')
    rows=[r for r in index['rows'] if r['book_id'] in active_books]
    if {r['book_id'] for r in rows}!=set(active_books):raise ValueError('Missing active textbook')
    identity=digest(dict(phase_id=phase['phase_id'],runtime_id=runtime_id,control=control,version=POLICY['version']))
    root=Path(output)/identity;root.mkdir(parents=True,exist_ok=True)
    write_immutable(root/'identity.json',dict(result_id=identity,phase_id=phase['phase_id'],reference_id=reference['reference_id'],
                                           graph_id=digest(graph),index_id=index['index_id'],control=control,budget=BUDGET,active_books=sorted(active_books)))
    retriever=EvidenceRetriever(rows,graph);records=[];errors=[]
    answer_prompt=(ROOT/'prompts/evaluation/answer.txt').read_text('utf-8')
    judge_prompt=(ROOT/'prompts/evaluation/judge.txt').read_text('utf-8')
    available={eid for r in rows for eid in r['evidence_ids']}
    for q in reference['questions']:
        if stop_after is not None and len(records)>=stop_after:break
        checkpoint=root/'questions'/(q['question_id']+'.json')
        if checkpoint.exists():
            record=read_document(checkpoint)
            if record['question_id']!=q['question_id'] or record['result_id']!=identity:raise ValueError('Checkpoint identity differs')
            if record.get('checkpoint_digest')!=digest({k:v for k,v in record.items() if k!='checkpoint_digest'}):raise ValueError('Checkpoint content changed')
            if record['status']!='completed' or record['question']!=q['question']:raise ValueError('Checkpoint question/status differs')
            records.append(record);continue
        if not set(q['evidence_ids'])<=available:raise ValueError('Full cohort has unavailable reference sources')
        record=None
        for attempt in range(BUDGET['retries']+1):
            try:
                vector=runtime.embed([q['question']])[0] if control=='vector' else None
                selected,trace=retriever.retrieve(q['question'],control,vector)
                context=pack_context(selected,q['question']);audit=retrieval_audit(q,selected,context)
                write_immutable(root/'retrieval'/(q['question_id']+'.json'),dict(question_id=q['question_id'],result_id=identity,trace=trace,audit=audit,context=context))
                candidate,answer_raw=call_json(runtime,answer_prompt,dict(question=q['question'],context=context),2048,root/'requests'/q['question_id']/'answer',attempt)
                if not isinstance(candidate.get('answer'),str) or not isinstance(candidate.get('citations'),list) or any(not isinstance(c,dict) for c in candidate['citations']):raise ValueError('Invalid answer object')
                judge,judge_raw=call_json(runtime,judge_prompt,dict(question=q['question'],reference_answer=q['answer'],reference_citations=q['citations'],candidate=candidate),1024,root/'requests'/q['question_id']/'judge',attempt)
                score=judge.get('score')
                if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid judge score')
                record=dict(question_id=q['question_id'],result_id=identity,status='completed',question=q['question'],question_type=q['type'],
                            context=context,retrieval=trace,audit=audit,candidate=candidate,answer_raw=answer_raw,judge_raw=judge_raw,
                            scores=dict(score_answer(candidate['answer'],q['answer']),**citations_score(candidate['citations'],context,rows),evidence_recall=audit['evidence_recall']),
                            machine_judge=judge,human_audited=False,completed_unix=time.time())
                record['checkpoint_digest']=digest(record)
                write_immutable(checkpoint,record);records.append(record);break
            except (ValueError,KeyError,TypeError) as exc:
                write_immutable(root/'errors'/f'{q["question_id"]}-{attempt}-{time.time_ns()}.json',dict(error_type=type(exc).__name__,observed_unix=time.time()))
            except (RuntimeError,HTTPError,URLError,TimeoutError,ConnectionError) as exc:
                code=account_status(exc)
                write_immutable(root/'errors'/f'{q["question_id"]}-{attempt}-{time.time_ns()}.json',dict(error_type=type(exc).__name__,status_code=code,observed_unix=time.time()))
                if code or (isinstance(exc,RuntimeError) and str(exc) not in {'Model service transport retries exhausted'} and not str(exc).startswith('Model service returned HTTP ')):raise
                if isinstance(exc,HTTPError) and exc.code not in {408,409,429,500,502,503,504}:raise
                if attempt<BUDGET['retries']:time.sleep(min(2**attempt,4))
        if record is None:errors.append(q['question_id'])
    completed=[r for r in records if r['status']=='completed']
    def metrics(group):
        n=len(group)
        citation_count=sum(r['scores']['citation_count'] for r in group)
        return dict(count=n,machine_judge_mean=sum(r['machine_judge']['score'] for r in group)/n if n else None,
                    citation_count=citation_count,citation_correctness=sum(r['scores']['valid_citation_count'] for r in group)/citation_count if citation_count else None,
                    seed_hit_count=sum(bool(r['retrieval']['seed_ids']) for r in group),path_hit_count=sum(bool(r['retrieval']['paths']) for r in group),
                    fallback_count=sum(r['retrieval']['fallback'] is not None for r in group),
                    context_rows_mean=sum(r['audit']['context_rows'] for r in group)/n if n else None,
                    context_bytes_mean=sum(r['audit']['context_bytes'] for r in group)/n if n else None,
                    evidence_recall_before_packing=sum(r['audit']['evidence_recall_before_packing'] for r in group)/n if n else None,
                    **{key:sum(r['scores'][key] for r in group)/n if n else None for key in ['em','f1','evidence_recall']})
    finished=len(completed)==len(reference['questions']) and not errors
    summary=dict(result_id=identity,status='completed' if finished else 'smoke_completed' if stop_after is not None and not errors else 'partial',course_id=reference['course_id'],control=control,
                 question_count=len(reference['questions']),completed_count=len(completed),failed_questions=errors,
                 active_books=sorted(active_books),reference_id=reference['reference_id'],phase_id=phase['phase_id'],
                 metrics=metrics(completed),by_type={kind:metrics([r for r in completed if r['question_type']==kind]) for kind in sorted({q['type'] for q in reference['questions']})},
                 retrieval_diagnostics=dict(seed_hit_count=sum(bool(r['retrieval']['seed_ids']) for r in completed),
                                            path_hit_count=sum(bool(r['retrieval']['paths']) for r in completed),
                                            fallback_count=sum(r['retrieval']['fallback'] is not None for r in completed),
                                            context_rows_mean=sum(r['audit']['context_rows'] for r in completed)/len(completed) if completed else None,
                                            context_bytes_mean=sum(r['audit']['context_bytes'] for r in completed)/len(completed) if completed else None),
                 human_audited=False,raw_text_exported=False)
    write_immutable(root/('result.json' if finished else summary['status']+'_'+digest(summary)+'.json'),summary)
    return summary
