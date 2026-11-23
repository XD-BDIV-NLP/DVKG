from __future__ import annotations
import copy
import json
import re
import time
from pathlib import Path

from ..manifest import digest
from ..contributions import write_immutable, read_contribution
from .native import NativeChat
from .protocol import semantic_view, COMMITS
from .runner import error_code

PROMPT=Path(__file__).with_name('alignment_prompt.txt')
BATCH_POLICY=dict(version=2,max_nodes=32,max_edges=24,order='sorted_edge_id_then_remaining_node_id',
                  request_node_fields=['name','type','entity_type'],request_edge_fields=['predicate','type'],
                  id_policy='sorted_batch_bijective_n_e_handles_strict_rebind',evidence_policy='current_source_text_only')


def decode_response(content: str) -> dict:
    value=content.strip()
    fenced=re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```',value)
    if fenced: value=fenced.group(1)
    response=json.loads(value)
    if not isinstance(response,dict): raise ValueError('Alignment response arrays required')
    return response


def alignment_batches(nodes: list[dict], edges: list[dict]) -> list[dict]:
    by_node={n['id']:n for n in nodes}
    if len(by_node)!=len(nodes) or len({e['id'] for e in edges})!=len(edges):
        raise ValueError('Duplicate native alignment IDs')
    batches=[]; current_edges=[]; current_nodes=set(); covered=set()
    def flush():
        batches.append(dict(nodes=[by_node[n] for n in sorted(current_nodes)],edges=list(current_edges)))
        covered.update(current_nodes)
    for edge in sorted(edges,key=lambda e:e['id']):
        endpoints={edge['source'],edge['target']}
        if not endpoints<=set(by_node): raise ValueError('Alignment native endpoint missing')
        if current_edges and (len(current_edges)>=BATCH_POLICY['max_edges'] or len(current_nodes|endpoints)>BATCH_POLICY['max_nodes']):
            flush(); current_edges=[]; current_nodes=set()
        current_edges.append(edge); current_nodes.update(endpoints)
    if current_edges: flush()
    remaining=sorted(set(by_node)-covered)
    for start in range(0,len(remaining),BATCH_POLICY['max_nodes']):
        batches.append(dict(nodes=[by_node[n] for n in remaining[start:start+BATCH_POLICY['max_nodes']]],edges=[]))
    return batches


def model_batch(nodes: list[dict], edges: list[dict]) -> tuple[dict, dict]:
    mapping={}; encoded={}
    for field,rows,prefix,allowed in [('nodes',nodes,'n',BATCH_POLICY['request_node_fields']),
                                      ('edges',edges,'e',BATCH_POLICY['request_edge_fields'])]:
        ids=[r['id'] for r in rows]
        if any(not isinstance(value,str) for value in ids) or len(set(ids))!=len(ids):
            raise ValueError('Duplicate or invalid native alignment IDs')
        ordered=sorted(rows,key=lambda row:row['id'])
        mapping[field]={prefix+str(i):row['id'] for i,row in enumerate(ordered)}
        encoded[field]=[]
        for i,row in enumerate(ordered):
            item={'id':prefix+str(i)}
            for key in allowed:
                if key in row:
                    if not isinstance(row[key],str) and row[key] is not None:
                        raise ValueError('Invalid native alignment label')
                    item[key]=row[key]
            if field=='edges':
                reverse={value:handle for handle,value in mapping['nodes'].items()}
                for key in ('source','target'):
                    if not isinstance(row.get(key),str) or row[key] not in reverse:
                        raise ValueError('Alignment native endpoint missing')
                    item[key]=reverse[row[key]]
            encoded[field].append(item)
    return encoded,mapping


def rebind_response(response: dict, mapping: dict) -> dict:
    if not isinstance(response,dict) or any(not isinstance(response.get(field),list) for field in ('entities','edges')):
        raise ValueError('Alignment response arrays required')
    rebound=copy.deepcopy(response)
    for field,kind in [('entities','nodes'),('edges','edges')]:
        rows=rebound[field]; handles=[r.get('id') if isinstance(r,dict) else None for r in rows]
        if (any(not isinstance(handle,str) for handle in handles) or len(set(handles))!=len(handles)
                or set(handles)!=set(mapping[kind])):
            raise ValueError('Alignment changed native ID coverage')
        for row in rows:
            row['id']=mapping[kind][row['id']]
            if field=='edges':
                for key in ('head','tail','source','target'):
                    if row.get(key) is not None:
                        if not isinstance(row[key],str) or row[key] not in mapping['nodes']:
                            raise ValueError('Alignment added or changed native endpoints')
                        row[key]=mapping['nodes'][row[key]]
    return rebound


def validation_feedback(exc, response, nodes, edges, attempt):
    categories={
        'Native response reached output limit':'output_limit',
        'Empty native model response':'json_format',
        'Alignment response arrays required':'json_format',
        'Alignment changed native ID coverage':'id_coverage',
        'Alignment quote not exact source substring':'source_quote',
        'Alignment condition must be preserved as source quote substring':'source_condition',
        'Alignment entity type outside Schema':'schema_type',
        'Alignment predicate outside Schema':'schema_predicate',
        'Alignment added or changed native endpoints':'native_endpoints',
        'Unmapped edge cannot get endpoints':'native_endpoints',
        'Explicit alignment inclusion/reason required':'decision_fields',
        'Excluded native entity cannot get type':'decision_fields',
        'Alignment edge reason required':'decision_fields'}
    category='json_format' if isinstance(exc,json.JSONDecodeError) else categories.get(str(exc),'request_failure') if type(exc) is ValueError else 'request_failure'
    feedback=dict(validation_category=category,failed_attempt=attempt+1,
                  instruction='Return one complete compact JSON object. Cover every input ID exactly once; use only Schema values and exact source_text quotes. Exclude unsupported nodes or leave edges unmapped with a short reason. Do not invent evidence or change endpoints.')
    if category=='id_coverage' and isinstance(response,dict):
        for field,expected in [('entities',nodes),('edges',edges)]:
            rows=response.get(field,[])
            present={r['id'] for r in rows if isinstance(r,dict) and isinstance(r.get('id'),str)} if isinstance(rows,list) else set()
            missing=sorted({r['id'] for r in expected}-present)
            feedback['missing_'+field+'_ids']=missing[:8]
            feedback['missing_'+field+'_count']=len(missing)
    return feedback


def validate_response(response: dict, nodes: list[dict], edges: list[dict], text: str, schema: dict) -> None:
    if not isinstance(response,dict) or not isinstance(response.get('entities'),list) or not isinstance(response.get('edges'),list):
        raise ValueError('Alignment response arrays required')
    node_ids={n['id'] for n in nodes}; edge_by_id={e['id']:e for e in edges}
    def cover(rows,expected):
        ids=[r.get('id') for r in rows if isinstance(r,dict)]
        if len(ids)!=len(rows) or len(ids)!=len(set(ids)) or set(ids)!=expected: raise ValueError('Alignment changed native ID coverage')
    cover(response['entities'],node_ids); cover(response['edges'],set(edge_by_id))
    def quoted(row):
        quote=row.get('quote')
        if not isinstance(quote,str) or len(quote.strip())<2 or quote not in text: raise ValueError('Alignment quote not exact source substring')
    for row in response['entities']:
        if type(row.get('include')) is not bool or not isinstance(row.get('reason'),str): raise ValueError('Explicit alignment inclusion/reason required')
        if row['include']:
            if row.get('entity_type') not in schema['entity_types']: raise ValueError('Alignment entity type outside Schema')
            quoted(row)
        elif row.get('entity_type') is not None: raise ValueError('Excluded native entity cannot get type')
    for row in response['edges']:
        if not isinstance(row.get('reason'),str): raise ValueError('Alignment edge reason required')
        if row.get('predicate') is None:
            if row.get('head') is not None or row.get('tail') is not None: raise ValueError('Unmapped edge cannot get endpoints')
        else:
            edge=edge_by_id[row['id']]
            if row['predicate'] not in schema['relation_types']: raise ValueError('Alignment predicate outside Schema')
            if {row.get('head'),row.get('tail')}!={edge['source'],edge['target']} or row['head']==row['tail']:
                raise ValueError('Alignment added or changed native endpoints')
            quoted(row)
            conditions=row.get('conditions')
            if not isinstance(conditions,str) or conditions and conditions not in row['quote']:
                raise ValueError('Alignment condition must be preserved as source quote substring')


def build_projection(native: dict, audits: list[dict], schema: dict, method: str) -> dict:
    nodes=copy.deepcopy(native['nodes']); edges=copy.deepcopy(native['edges'])
    by_node={}; by_edge={}
    for audit in audits:
        for row in audit['response']['entities']: by_node.setdefault(row['id'],[]).append(dict(row,unit_id=audit['unit_id']))
        for row in audit['response']['edges']: by_edge.setdefault(row['id'],[]).append(dict(row,unit_id=audit['unit_id']))
    kept=[]; entity_audit=[]
    for node in nodes:
        rows=by_node.get(node['id'],[])
        types={r['entity_type'] for r in rows if r['include']}
        accepted=len(types)==1
        entity_audit.append(dict(native_id=node['id'],decisions=rows,accepted=accepted,
                                 reason='source_type_consensus' if accepted else 'conflicting_types' if len(types)>1 else 'excluded_or_no_source_type'))
        if accepted:
            node['native_type']=node.get('type'); node['type']=next(iter(types)); node['type_status']='source_backed_schema_alignment'
            node['evidence_ids']=sorted({r['unit_id'] for r in rows if r['include']})
            kept.append(node)
    kept_ids={n['id'] for n in kept}; aligned=[]; edge_audit=[]
    for edge in edges:
        eid=digest(edge); rows=by_edge.get(eid,[])
        mappings={(r['predicate'],r['head'],r['tail'],r['conditions']) for r in rows if r['predicate'] is not None}
        accepted=len(mappings)==1 and edge['source'] in kept_ids and edge['target'] in kept_ids
        edge_audit.append(dict(native_id=eid,original=edge,decisions=rows,accepted=accepted,
                               reason='source_mapping_consensus' if accepted else 'conflicting_mappings' if len(mappings)>1 else 'unmapped_or_excluded_endpoint'))
        if accepted:
            predicate,head,tail,conditions=next(iter(mappings))
            mapped=copy.deepcopy(edge); mapped.update(source=head,target=tail,predicate=predicate,conditions=conditions,native_edge_id=eid,
                 evidence_ids=sorted({r['unit_id'] for r in rows if r['predicate'] is not None}),
                 source_texts=sorted({r['quote'] for r in rows if r['predicate'] is not None}),
                 provenance_status='source_backed_schema_alignment')
            aligned.append(mapped)
    graph=dict(nodes=kept,edges=aligned,directed=True)
    view=semantic_view(graph,schema,method+'_schema_aligned')
    return dict(method_id=method+'_schema_aligned',native_method_id=method,native_graph_unchanged=True,
                graph=graph,semantic_view=view,entity_audit=entity_audit,edge_audit=edge_audit,
                alignment_status='model_source_backed_not_human_verified',
                coverage=dict(native_nodes=len(nodes),aligned_nodes=len(kept),native_edges=len(edges),aligned_edges=len(aligned)))


def align(native: dict, units: list[dict], schema: dict, method: str, params: dict, runtime, root: Path, identity: str,
          metadata: dict | None=None) -> dict:
    audits=[]; failed=[]; coverage=[]; graph_id=digest(native); nodes={n['id']:n for n in native['nodes']}
    unit_ids={u['unit_id'] for u in units}
    if len(unit_ids)!=len(units) or len(nodes)!=len(native['nodes']): raise ValueError('Duplicate alignment input IDs')
    if any(not set(obj.get('evidence_ids',[]))<=unit_ids for obj in native['nodes']+native['edges']):
        raise ValueError('Unknown alignment source unit')
    prompt=PROMPT.read_text('utf-8')
    for unit in units:
        uid=unit['unit_id']; selected_nodes={n['id'] for n in native['nodes'] if uid in n.get('evidence_ids',[])}
        edges=[dict(e,id=digest(e)) for e in native['edges'] if uid in e.get('evidence_ids',[])]
        selected_nodes.update(end for e in edges for end in (e['source'],e['target']))
        subset=[nodes[nid] for nid in sorted(selected_nodes)]
        batches=alignment_batches(subset,edges); unit_audits=[]
        for index,batch in enumerate(batches):
            batch_nodes=batch['nodes']; batch_edges=batch['edges']
            encoded,mapping=model_batch(batch_nodes,batch_edges)
            policy_id=digest(BATCH_POLICY)
            request=dict(schema=schema,**encoded,source_text=unit['text'],unit_id=uid,batch_index=index,request_policy_digest=policy_id)
            request_id=digest(dict(input_id=identity,native_graph_id=graph_id,unit_id=uid,batch_index=index,
                                   batch_policy=BATCH_POLICY,parameters=params,prompt_digest=digest(prompt),request=request,request_mapping=mapping))
            directory=root/request_id; checkpoint=directory/'complete.json'
            write_immutable(directory/'request.json',dict(request_id=request_id,request=request,
                            request_mapping=mapping,request_policy_digest=policy_id))
            if checkpoint.exists():
                audit=read_contribution(checkpoint)
                if audit['request_id']!=request_id or audit.get('unit_id')!=uid or audit.get('batch_index')!=index:
                    raise ValueError('Alignment checkpoint identity mismatch')
                validate_response(audit['response'],batch_nodes,batch_edges,unit['text'],schema)
                if (audit.get('request_mapping')!=mapping or audit.get('request_policy_digest')!=policy_id
                        or rebind_response(audit.get('raw_response'),mapping)!=audit['response']):
                    raise ValueError('Alignment checkpoint mapping mismatch')
                unit_audits.append(audit); continue
            feedback=None; response=None
            for attempt in range(params['retries']+1):
                attempt_root=directory/'attempts'/str(time.time_ns())
                chat=NativeChat(runtime,params,attempt_root/'responses',response_format={'type':'json_object'})
                response=None; raw_response=None
                try:
                    prefix=('Previous attempt validation feedback: '+json.dumps(feedback,ensure_ascii=False)+'\n') if feedback else ''
                    content=chat.generate(prefix+prompt+json.dumps(request,ensure_ascii=False))
                    raw_response=decode_response(content)
                    response=rebind_response(raw_response,mapping)
                    validate_response(response,batch_nodes,batch_edges,unit['text'],schema); break
                except Exception as exc:
                    code=error_code(exc)
                    feedback=validation_feedback(exc,raw_response,encoded['nodes'],encoded['edges'],attempt)
                    write_immutable(attempt_root/'error.json',dict(error_type=type(exc).__name__,status_code=code,
                                    validation_category=feedback['validation_category'],feedback=feedback))
                    if code==402: raise RuntimeError('Model service returned HTTP 402 during alignment') from None
                    recoverable=isinstance(exc,(ValueError,TimeoutError,ConnectionError)) or code in {408,409,429,500,502,503,504}
                    if not recoverable or attempt==params['retries']:
                        failed.append(uid+':batch:'+str(index)); response=None; break
            if response is None: continue
            audit=dict(request_id=request_id,unit_id=uid,batch_index=index,response=response,raw_response=raw_response,
                       request_mapping=mapping,request_policy_digest=policy_id); audit['artifact_id']=digest(audit)
            write_immutable(checkpoint,audit); unit_audits.append(audit)
        covered_nodes={r['id'] for a in unit_audits for r in a['response']['entities']}
        covered_edges=[r['id'] for a in unit_audits for r in a['response']['edges']]
        if len(unit_audits)==len(batches):
            if covered_nodes!=selected_nodes or sorted(covered_edges)!=sorted(e['id'] for e in edges):
                raise ValueError('Alignment batch coverage incomplete')
        coverage.append(dict(unit_id=uid,batch_count=len(batches),native_nodes=len(subset),native_edges=len(edges),
                             covered_nodes=len(covered_nodes),covered_edges=len(covered_edges)))
        audits.extend(unit_audits)
    if failed: raise RuntimeError('Unfinished source alignment units/batches: '+','.join(failed))
    result=build_projection(native,audits,schema,method)
    result.update(input_id=identity,native_graph_id=graph_id,parameters=params,runtime_stats=runtime.stats(),
                  audit_files=[a['request_id'] for a in audits],status='complete',native_commit=COMMITS[method],
                  graph_state='native_schema_aligned',native_graph_state='native',**(metadata or {}))
    result.update(batch_policy=copy.deepcopy(BATCH_POLICY),request_policy_digest=digest(BATCH_POLICY),batch_coverage=coverage,alignment_extra_api_cost=True,
                  untraceable_native_objects=dict(nodes=sum(not n.get('evidence_ids') for n in native['nodes']),
                                                  edges=sum(not e.get('evidence_ids') for e in native['edges'])))
    result['artifact_id']=digest(result)
    return write_immutable(root/'aligned_artifact.json',result)
