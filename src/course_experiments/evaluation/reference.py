from __future__ import annotations
import json
from copy import deepcopy
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from ..manifest import digest,read_document,validate
from ..contributions import write_immutable
from .protocol import ROOT,validate_phase,bound_input,runtime_for,bind_runtime
from .chat import call_json
from ..splits import canonical_reference

TYPES=['definition','relation','prerequisite','multihop','conditions','evidence']
PARAMETERS=dict(seed=42,target_questions=100,target_facts=400,questions_per_chunk=4,facts_per_chunk=16,max_entities_per_chunk=32,max_facts_per_chunk=16,max_questions_per_chunk=4,smoke_max_chunks=16,smoke_max_generation_calls=28,smoke_retry_allocation='reserve_one_call_per_remaining_source_unit',smoke_targets={'questions':8,'facts':24},curation_policy='source_validated_subset_v1',max_tokens=12000,retries=6,selection='seed42 hash order interleaved books',audit_status='machine_generated_source_validated_not_human_audited')

def reference_targets(smoke):
    if smoke is None:return dict(questions=PARAMETERS['target_questions'],facts=PARAMETERS['target_facts'])
    if type(smoke) is not int or not 1<=smoke<=PARAMETERS['smoke_max_chunks']:raise ValueError('Smoke chunks must be1..16')
    return dict(PARAMETERS['smoke_targets'])

def source_units(corpora):
    units=[]
    for corpus in corpora:
        evidence={e['id']:e for e in corpus['evidence']}
        for chunk in corpus['chunks']:
            rows=[dict(id=eid,text=evidence[eid].get('text',evidence[eid].get('content',''))) for eid in chunk['evidence_ids']]
            if not rows or any(not e['text'] for e in rows):raise ValueError('Source chunk lacks raw evidence')
            units.append(dict(book_id=chunk['book_id'],chunk_id=chunk['id'],evidence=rows,text='\n'.join(e['text'] for e in rows)))
    return units

def selected_units(corpora,seed=42,smoke=None):
    units=source_units(corpora);books=sorted({u['book_id'] for u in units})
    bybook={b:sorted([u for u in units if u['book_id']==b],key=lambda u:digest([seed,u])) for b in books}
    selected=[]
    for i in range(max(map(len,bybook.values()))):
        for b in books:
            if i<len(bybook[b]):selected.append(bybook[b][i])
    return selected if smoke is None else selected[:smoke]

class ReferenceValidationError(ValueError):
    def __init__(self,message,row,field,rule_id,endpoint=None):
        super().__init__(message)
        self.diagnostic=dict(local_id=str(row.get('local_id',''))[:128],field=field,rule_id=rule_id,
            evidence_ids=[str(c.get('evidence_id',''))[:128] for c in row.get('citations',[])[:8]],
            origin='untrusted_previous_output_identifiers_not_source_evidence')
        if endpoint is not None:self.diagnostic['endpoint_local_id']=str(endpoint)[:128]


def validate_batch(value,unit,schema):
    for field in ('entities','facts','questions'):
        if not isinstance(value.get(field),list):raise ValueError('Reference '+field+' must be a list')
        if len(value[field])>PARAMETERS['max_'+field+'_per_chunk']:raise ValueError('Reference output exceeds '+field+' limit')
    evidence={e['id']:e['text'] for e in unit['evidence']}
    def citations(row):
        cites=row.get('citations',[])
        if not cites:raise ReferenceValidationError('Missing source citations',row,'citations','exact_source_citation')
        for c in cites:
            if c.get('evidence_id') not in evidence or not c.get('quote') or c['quote'] not in evidence[c['evidence_id']]:raise ReferenceValidationError('Citation not exact source substring',row,'citations','exact_source_citation')
        return cites
    entities={e['local_id']:e for e in value['entities']}
    if len(entities)!=len(value['entities']):raise ValueError('Duplicate local entity IDs')
    for e in entities.values():
        citations(e)
        if not e.get('canonical_name') or not e.get('scope'):raise ReferenceValidationError('Independent entity requires scope/name',e,'scope/canonical_name','explicit_scope_and_name')
        if not any(e['canonical_name'] in c['quote'] for c in e['citations']):raise ReferenceValidationError('Entity name not source quoted',e,'canonical_name','literal_canonical_name')
        for alias in e.get('aliases',[]):
            if not alias or not any(alias in c['quote'] and e['canonical_name'] in c['quote'] for c in e['citations']):raise ReferenceValidationError('Alias not explicitly co-quoted',e,'aliases','explicit_coquoted_alias')
    facts={f['local_id']:f for f in value['facts']}
    if len(facts)!=len(value['facts']):raise ValueError('Duplicate fact IDs')
    for f in facts.values():
        cites=citations(f)
        if f.get('head') not in entities or f.get('tail') not in entities or f.get('relation') not in schema['relation_types']:raise ValueError('Invalid reference fact endpoints/predicate')
        for field in ('head','tail'):
            endpoint=f[field]
            if not any(any(name in c['quote'] for name in [entities[endpoint]['canonical_name'],*entities[endpoint].get('aliases',[])]) for c in cites):raise ReferenceValidationError('Fact endpoints not quoted in fact evidence',f,field,'fact_endpoint_own_citation',endpoint)
        if f.get('conditions') and not any(f['conditions'] in c['quote'] for c in cites):raise ValueError('Conditions not quoted')
    for q in value['questions']:
        citations(q)
        if q.get('type') not in TYPES or not q.get('question') or not q.get('answer'):raise ValueError('Invalid question')
        path=q.get('fact_path',[])
        if any(x not in facts for x in path):raise ValueError('Question path missing fact')
        if q['type']=='multihop':
            if len(path)<2 or len(set(path))!=len(path):raise ValueError('Multihop needs unique source fact path')
            for a,b in zip(path,path[1:]):
                if not {facts[a]['head'],facts[a]['tail']}&{facts[b]['head'],facts[b]['tail']}:raise ValueError('Disconnected multihop path')
    return value

def curate_batch(value,unit,schema):
    # 先拒绝畸形整包；仅对形状完整的候选逐项应用原验证器。
    fields=('entities','facts','questions')
    if not isinstance(value,dict):raise ValueError('Reference packet must be an object')
    for field in fields:
        rows=value.get(field)
        if not isinstance(rows,list):raise ValueError('Reference '+field+' must be a list')
        if len(rows)>PARAMETERS['max_'+field+'_per_chunk']:raise ValueError('Reference output exceeds '+field+' limit')
        for row in rows:
            if not isinstance(row,dict):raise ValueError('Malformed reference candidate')
            if field!='questions' or 'local_id' in row:
                if not isinstance(row.get('local_id'),str) or not row['local_id']:raise ValueError('Malformed reference local ID')
            cites=row.get('citations',[])
            if not isinstance(cites,list) or any(not isinstance(c,dict) or not isinstance(c.get('evidence_id'),str) or not isinstance(c.get('quote'),str) for c in cites):raise ValueError('Malformed reference citations')
            names={'entities':('canonical_name','scope'),'facts':('head','tail','relation','scope','conditions'),'questions':('type','question','answer')}[field]
            for name in names:
                if name in row and not isinstance(row[name],str):raise ReferenceValidationError('Malformed reference text field: '+name+' must be a JSON string',row,name,'json_string_field')
            for key in ('aliases','fact_path'):
                if key in row and (not isinstance(row[key],list) or any(not isinstance(x,str) for x in row[key])):raise ValueError('Malformed reference '+key)
    collisions={field:{key for key,count in Counter(r['local_id'] for r in value[field] if 'local_id' in r).items() if count>1} for field in fields}
    whole_error=None
    try:
        validate_batch(value,unit,schema)
        if any(collisions.values()):raise ValueError('Duplicate local IDs')
    except ValueError as exc:whole_error=str(exc)
    if whole_error is None:
        selected={field:deepcopy(value[field]) for field in fields};rejections=[]
    else:
        selected={field:[] for field in fields};rejections=[]
        for field in fields:
            for index,row in enumerate(value[field]):
                try:
                    if row.get('local_id') in collisions[field]:raise ValueError('Ambiguous duplicate local ID')
                    candidate=dict(selected,**{field:[*selected[field],row]})
                    validate_batch(candidate,unit,schema)
                except ValueError as exc:
                    rejections.append(dict(collection=field,index=index,local_id=row.get('local_id'),candidate_hash=digest(row),reason=str(exc),diagnostic=getattr(exc,'diagnostic',None)))
                else:selected[field].append(deepcopy(row))
        validate_batch(selected,unit,schema)
    audit=dict(policy=PARAMETERS['curation_policy'],whole_response_valid=whole_error is None,whole_response_error=whole_error,
        retained_counts={field:len(selected[field]) for field in fields},rejected_counts={field:len(value[field])-len(selected[field]) for field in fields},rejections=rejections)
    return selected,audit

def validate_reference(value):
    if value.get('status')!='complete' or value.get('reference_id')!=digest({k:v for k,v in value.items() if k!='reference_id'}):raise ValueError('Only complete hash-verified reference can freeze')
    if value.get('human_audited') is not False:raise ValueError('Machine reference cannot claim human audit')
    if len(value['questions'])<value['targets']['questions'] or len(value['facts'])<value['targets']['facts'] or value.get('failed_chunks'):raise ValueError('Incomplete reference counts/checkpoints')
    if len({q['question_id'] for q in value['questions']})!=len(value['questions']):raise ValueError('Duplicate reference question IDs')
    return value

def freeze_reference(value,output):
    validate_reference(value);write_immutable(output,value);return value

def assemble(course,records,scope,identity,targets):
    entities={};facts={};questions={}
    for unit,batch in records:
        mapping={}
        for e in batch['entities']:
            key=digest([course,e['canonical_name'],e['scope']]);mapping[e['local_id']]=key
            if key not in entities:entities[key]=dict(entity_id=key,canonical_name=e['canonical_name'],scope=e['scope'],aliases=[],citations=[])
            entities[key]['aliases']=sorted(set(entities[key]['aliases']+e.get('aliases',[])))
            entities[key]['citations']+=e['citations']
        fm={}
        for f in batch['facts']:
            row=dict(head=mapping[f['head']],relation=f['relation'],tail=mapping[f['tail']],conditions=f.get('conditions',''),scope=f.get('scope',''),citations=f['citations'],book_id=unit['book_id'],chunk_id=unit['chunk_id'],evidence_ids=[c['evidence_id'] for c in f['citations']])
            key=digest(row);fm[f['local_id']]=key;facts[key]=dict(row,fact_id=key)
        for q in batch['questions']:
            row=dict(q,book_id=unit['book_id'],chunk_id=unit['chunk_id'],fact_path=[fm[f] for f in q.get('fact_path',[])],evidence_ids=[c['evidence_id'] for c in q['citations']])
            key=digest([course,q['question']]);questions.setdefault(key,dict(row,question_id=key))
    complete=len(questions)>=targets['questions'] and len(facts)>=targets['facts'] and set(q['type'] for q in questions.values())==set(TYPES)
    result=dict(course_id=course,scope=scope,input_id=identity,entities=list(entities.values()),facts=list(facts.values()),questions=list(questions.values()),status='complete' if complete else 'partial',targets=targets,audit_status=PARAMETERS['audit_status'],human_audited=False)
    result=canonical_reference(result);result['reference_id']=digest(result);return result

def _request_identity(runtime,prompt,payload,root,attempt):
    body=dict(runtime.config['llm'].get('extra_body') or {})
    body.update(model=runtime.config['llm']['model'],temperature=0,max_tokens=PARAMETERS['max_tokens'],seed=42,enable_thinking=False,response_format={'type':'json_object'},messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    return body,digest(dict(body=body,runtime_config=runtime.config,attempt=attempt,root=str(root/'requests')))

def _check_attempt_record(path,value,required):
    if path.exists():
        if value is None or read_document(path)!=value:raise ValueError('Reference curation audit mismatch: '+path.name)
    elif value is not None:
        if required:raise ValueError('Reference curation audit record missing: '+path.name)
        write_immutable(path,value)

def generate(construction_path,phase_path,course_id,output,runtime=None,smoke=None):
    construction=validate(construction_path);phase=validate_phase(phase_path);bound_input(phase,'construction',construction)
    if phase['parameters']!=dict(PARAMETERS,smoke_chunks=smoke):raise ValueError('Reference generation parameters not frozen')
    course=next(c for c in construction['courses']['courses'] if c['course_id']==course_id)
    schema=read_document(construction['files']['assets']['schema']['path'])
    corpora=[read_document(construction['files']['sources'][b['book_id']]['path']) for b in course['books']]
    targets=reference_targets(smoke);units=selected_units(corpora,smoke=smoke)
    identity=digest(dict(phase_id=phase['phase_id'],course_id=course_id,units=units,targets=targets));root=Path(output)/course_id/('full' if smoke is None else 'smoke')/identity
    root.mkdir(parents=True,exist_ok=True);write_immutable(root/'identity.json',dict(input_id=identity,phase_id=phase['phase_id'],source_passage_hashes=[digest(u) for u in units]))
    owned=runtime is None;runtime=runtime or runtime_for(construction,root/'runtime');records=[];errors=[];logical_calls=0;audits={}
    call_limit=PARAMETERS['smoke_max_generation_calls'] if smoke is not None else len(units)*(PARAMETERS['retries']+1)
    prompt=(ROOT/'prompts/evaluation/reference.txt').read_text('utf-8')
    try:
        bind_runtime(phase,runtime)
        for i,unit in enumerate(units):
            p=root/'checkpoints'/(digest(unit)+'.json')
            cached=read_document(p) if p.exists() else None
            batch=None;feedback=None;attempt_hashes=[]
            attempts=min(PARAMETERS['retries']+1,call_limit-logical_calls-(len(units)-i-1))
            if cached is not None:
                validate_batch(cached,unit,schema)
                count=cached['curation']['attempts']
                if type(count) is not int or not 1<=count<=attempts:raise ValueError('Reference curation audit attempt count mismatch')
                attempts=count
            for attempt in range(max(0,attempts)):
                bind_runtime(validate_phase(phase_path),runtime);validate(construction_path)
                payload=dict(unit=unit,schema=schema,course_description=course['description'],requested_types=TYPES[i%6:]+TYPES[:i%6],output_limits={k:PARAMETERS['max_'+k+'_per_chunk'] for k in ('entities','facts','questions')})
                if feedback is not None:payload['validation_feedback']=feedback
                body,request_id=_request_identity(runtime,prompt,payload,root,attempt)
                request_ref='requests/'+request_id+'.json';audit_ref='audits/'+digest(unit)+'-'+str(attempt)+'.json'
                request_path=root/request_ref;error_path=root/'errors'/f'{i}-{attempt}.json'
                saved_error=read_document(error_path) if error_path.exists() else None
                required=cached is not None or saved_error is not None or (root/audit_ref).exists()
                request=read_document(request_path) if request_path.exists() else None
                if request is not None and request['request']!=body:raise ValueError('Reference curation audit request body mismatch')
                replay_http=request is None and saved_error is not None and saved_error.get('error_type')=='HTTPError'
                if required and request is None and not replay_http:raise ValueError('Reference curation audit request missing: '+request_id)
                audit=None;error=None;selected=None;fatal_http=None
                logical_calls+=1
                try:
                    if replay_http:raise HTTPError('',saved_error['status_code'],'Replayed HTTP error',None,None)
                    if request is None:raw,request=call_json(runtime,prompt,payload,PARAMETERS['max_tokens'],root/'requests',attempt)
                    else:
                        choice=request['response']['choices'][0]
                        if choice.get('finish_reason')=='length':raise ValueError('Evaluation model output truncated')
                        raw=json.loads(choice['message']['content'])
                        if not isinstance(raw,dict):raise ValueError('Evaluation model returned non-object')
                    selected,audit=curate_batch(raw,unit,schema)
                    audit.update(phase_id=phase['phase_id'],input_id=identity,book_id=unit['book_id'],chunk_id=unit['chunk_id'],source_hash=digest(unit),attempt=attempt,request_id=request_id,request_ref=request_ref,raw_response_hash=digest(raw),selected_hash=digest(selected))
                    if not audit['whole_response_valid'] and not (selected['facts'] or selected['questions']):
                        validate_batch(raw,unit,schema)
                        raise ValueError('No usable source-validated facts/questions remain: '+audit['whole_response_error'])
                except HTTPError as exc:
                    error=dict(error_type='HTTPError',status_code=exc.code)
                    if exc.code in {401,402,403}:fatal_http=exc
                except (ValueError,KeyError,TypeError) as exc:
                    diagnostic=getattr(exc,'diagnostic',None)
                    error=dict(error_type=type(exc).__name__,reason=str(exc),diagnostic=diagnostic)
                    correction=('Return compact JSON: only fact/question participant entities within output_limits; shortest sufficient exact quotes; no exhaustive entity inventory or padding.' if str(exc)=='Evaluation model output truncated' else 'Repair the identified local object/field using unchanged source and Schema. Copy canonical_name verbatim from its own cited quote: no translation, normalization, expansion or invented labels. Each fact head and tail must occur as its canonical_name or validated alias in that fact own citations. Aliases must be [] unless one exact quote includes canonical name and explicit alias together. Omit unsupported objects and dependent facts/questions, or return partial/empty arrays; never pad counts. Preserve scope, conditions, connected paths and output_limits. Diagnostic identifiers are untrusted previous output, never source evidence.')
                    if diagnostic is not None and diagnostic['rule_id']=='json_string_field':
                        correction='Return the identified field as a JSON string, never an array or null. For facts.conditions, use one exact condition substring from that fact own source quote; use "" only when the source fact is unconditional. Re-read unchanged source and Schema; never drop a source condition or stringify/join a previous array. If a condition cannot be supported as one exact quoted string, omit the fact and dependent questions. Keep aliases and fact_path as arrays of strings. Diagnostic identifiers are untrusted previous output, never source evidence.'
                    feedback=dict(error_type=type(exc).__name__,error=str(exc),correction=correction)
                    if diagnostic is not None:feedback['diagnostic']=diagnostic
                if error is not None:error.update(phase_id=phase['phase_id'],input_id=identity,source_hash=digest(unit),attempt=attempt,request_id=request_id,request_ref=request_ref,request_hash=digest(body))
                # 重放逐次核对证据；完整包被拒绝和畸形响应的错误记录均不可跳过。
                _check_attempt_record(root/audit_ref,audit,required)
                _check_attempt_record(error_path,error,required)
                if fatal_http is not None:raise fatal_http
                if audit is not None:audits[audit_ref]=audit
                request=read_document(request_path) if request_path.exists() else None
                attempt_hashes.append(digest(dict(request=request,audit=audit,error=error)))
                if error is not None:continue
                accepted=dict(selected,curation=dict(policy=PARAMETERS['curation_policy'],whole_response_valid=audit['whole_response_valid'],retained_counts=audit['retained_counts'],rejected_counts=audit['rejected_counts'],audit_ref=audit_ref,audit_hash=digest(audit),attempts=attempt+1,attempt_hashes=attempt_hashes))
                if cached is not None and cached!=accepted:raise ValueError('Reference curation audit checkpoint mismatch')
                write_immutable(p,accepted);batch=accepted;break
            if cached is not None and batch is None:raise ValueError('Reference curation audit accepted checkpoint no longer valid')
            if batch is None:errors.append(unit['chunk_id']);continue
            records.append((unit,batch));dataset=assemble(course_id,records,'full' if smoke is None else 'smoke',identity,targets)
            if dataset['status']=='complete':break
        dataset=assemble(course_id,records,'full' if smoke is None else 'smoke',identity,targets);dataset['failed_chunks']=errors
        if errors:dataset['status']='partial'
        discovered={str(p.relative_to(root)).replace('\\','/') for p in (root/'audits').glob('*.json')}
        if discovered!=set(audits):raise ValueError('Reference curation audit set contains unconsumed records')
        audits=sorted(audits.items())
        dataset['curation']=dict(policy=PARAMETERS['curation_policy'],audit_refs=[name for name,_ in audits],
            audit_hashes={name:digest(audit) for name,audit in audits},
            accepted_retained_counts={field:sum(len(batch[field]) for _,batch in records) for field in ('entities','facts','questions')},
            rejected_candidate_counts={field:sum(a['rejected_counts'][field] for _,a in audits) for field in ('entities','facts','questions')})
        dataset['generation_budget']=dict(logical_calls=logical_calls,max_logical_calls=call_limit,smoke_max_chunks=PARAMETERS['smoke_max_chunks'] if smoke is not None else None)
        dataset['reference_id']=digest({k:v for k,v in dataset.items() if k!='reference_id'})
        write_immutable(root/('reference.json' if dataset['status']=='complete' else 'partial_reference_'+dataset['reference_id']+'.json'),dataset)
        return dataset,root
    finally:
        if owned:runtime.db.close()
