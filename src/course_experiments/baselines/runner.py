from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path

from ..construction import output_lock, PROMPTS
from ..contributions import write_immutable, read_contribution
from ..manifest import digest, read_document
from ..textbook import scoped_corpus
from .protocol import COMMITS, freeze_phase, validate_phase, semantic_view


def error_code(exc: Exception) -> int | None:
    code=getattr(exc,'status_code',None) or getattr(exc,'code',None)
    match=re.search(r'\b(402|408|409|429|500|502|503|504)\b',str(exc))
    return code if isinstance(code,int) else int(match[1]) if match else None


def process_units(units: list[dict], driver, root: Path, identity: str, retries: int) -> list[dict]:
    rows=[]; failed=[]
    for unit in units:
        unit_root=root/digest([unit['book_id'],unit['chunk_id']])
        metadata=dict(input_id=identity,book_id=unit['book_id'],chunk_id=unit['chunk_id'],text_digest=unit['text_digest'])
        write_immutable(unit_root/'identity.json',metadata)
        artifact=unit_root/'complete.json'
        if artifact.exists():
            row=read_contribution(artifact)
            if any(row['provenance'].get(k)!=v for k,v in metadata.items()): raise ValueError('Native checkpoint provenance mismatch')
            rows.append(row); continue
        attempts=unit_root/'attempts'; attempts.mkdir(parents=True,exist_ok=True)
        for attempt in range(retries+1):
            directory=attempts/str(time.time_ns()); directory.mkdir()
            try:
                native=driver.extract(unit,directory)
                ids={n['id'] for n in native['nodes']}
                if any(e['source'] not in ids or e['target'] not in ids for e in native['edges']): raise ValueError('Native edge has unknown endpoint')
                row=dict(provenance=dict(metadata,unit_id=unit.get('unit_id',digest(metadata))),native=native)
                row['artifact_id']=digest(row); write_immutable(artifact,row); rows.append(row); break
            except Exception as exc:
                code=error_code(exc)
                # SDK异常文本可能含请求参数；仅持久化异常类别与状态码。
                write_immutable(directory/'error.json',dict(error_type=type(exc).__name__,status_code=code))
                if code==402: raise RuntimeError('Model service returned HTTP 402; checkpoints preserved') from None
                recoverable=isinstance(exc,(ValueError,TimeoutError,ConnectionError)) or code in {408,409,429,500,502,503,504}
                if not recoverable or attempt==retries:
                    failed.append(unit['chunk_id']); break
        else: failed.append(unit['chunk_id'])
    if failed: raise RuntimeError('Unfinished native chunks: '+','.join(failed))
    return rows


def corpus_units(manifest: dict, course: dict, aliases: list[str], smoke: int | None) -> list[dict]:
    units=[]
    if sorted(aliases) not in (['A'],['B'],['A','B']): raise ValueError('Declared bookset A, B or AB required')
    for book in course['books']:
        if book['alias'] not in aliases: continue
        corpus=scoped_corpus(read_document(manifest['files']['sources'][book['book_id']]['path']),smoke)
        evidence={e['id']:e for e in corpus['evidence']}
        for chunk in corpus['chunks']:
            rows=[evidence[eid] for eid in chunk['evidence_ids']]
            text='\n\n'.join(e.get('text',e.get('content','')) for e in rows)
            if not text.strip(): raise ValueError('Baseline input chunk has no source text')
            unit=dict(book_id=book['book_id'],chunk_id=chunk['id'],text=text,text_digest=digest(text),
                      evidence=rows,scope=corpus['construction_scope'],source_fingerprint=corpus['fingerprint'])
            unit['unit_id']=digest(unit); units.append(unit)
    return units


def aggregate_complete(driver, rows: list[dict], root: Path, identity: str, retries: int) -> dict:
    checkpoint=root/'aggregate_complete.json'
    if checkpoint.exists():
        result=read_contribution(checkpoint)
        if result['input_id']!=identity: raise ValueError('Native aggregate identity mismatch')
        return result['native_graph']
    for attempt in range(retries+1):
        directory=root/'aggregate'/str(time.time_ns())
        try:
            native=driver.aggregate(rows,directory)
            result=dict(input_id=identity,native_graph=native); result['artifact_id']=digest(result)
            write_immutable(checkpoint,result); return native
        except Exception as exc:
            code=error_code(exc)
            write_immutable(directory/'error.json',dict(error_type=type(exc).__name__,status_code=code))
            if code==402: raise RuntimeError('Model service returned HTTP 402; aggregate checkpoints preserved') from None
            recoverable=isinstance(exc,(ValueError,TimeoutError,ConnectionError)) or code in {408,409,429,500,502,503,504}
            if not recoverable or attempt==retries: raise
    raise RuntimeError('Native aggregate did not complete')


def source_support(graph: dict, rows: list[dict], units: list[dict], method: str) -> tuple[dict,list[dict]]:
    graph=json.loads(json.dumps(graph,ensure_ascii=False))
    evidence=[dict(id=u['unit_id'],source_node_id=u['unit_id'],book_id=u['book_id'],chunk_id=u['chunk_id'],text=u['text'],content=u['text'],
                   text_digest=u['text_digest'],source_evidence_ids=[e['id'] for e in u['evidence']],
                   evidence_status='source_unit_provenance_not_verified_sentence') for u in units]
    names={n['id']:n['name'] for n in graph['nodes']}
    def labels(node):
        aliases=node.get('aliases',[])
        if isinstance(aliases,str): aliases=[aliases]
        return {node['name'],*[a for a in aliases if isinstance(a,str)]}
    normalize=(lambda s:' '.join(s.strip().lower().split())) if method=='k12_kgraph' else (lambda s:s)
    source_nodes={}; source_edges={}
    for row in rows:
        unit=row['provenance']['unit_id']; ng=row['native']; nn={n['id']:n['name'] for n in ng['nodes']}
        for node in ng['nodes']: source_nodes.setdefault(normalize(node['name']),set()).add(unit)
        for edge in ng['edges']:
            triple=(normalize(nn[edge['source']]),edge.get('predicate'),normalize(nn[edge['target']]))
            source_edges.setdefault(triple,set()).add(unit)
    nodes={n['id']:n for n in graph['nodes']}
    for node in graph['nodes']:
        support=set(node.get('source_units',[]))
        if not support:
            for name in labels(node): support.update(source_nodes.get(normalize(name),set()))
        node['evidence_ids']=sorted(support); node['source_state']='native_extraction_provenance'
    for edge in graph['edges']:
        support=set(edge.get('source_units',[]))
        if not support:
            for head in labels(nodes[edge['source']]):
                for tail in labels(nodes[edge['target']]):
                    for predicate in [edge.get('predicate'),*edge.get('native_predicate_aliases',[])]:
                        support.update(source_edges.get((normalize(head),predicate,normalize(tail)),set()))
        edge['evidence_ids']=sorted(support)
        edge['provenance_status']='native_unit_correlated' if support else 'aggregation_transformation_unresolved'
        edge['source_conditions_status']='not_modeled_by_native_method'
    valid={e['id'] for e in evidence}
    if any(not set(n['evidence_ids'])<=valid for n in graph['nodes']+graph['edges']): raise ValueError('Unknown native source unit')
    source_text={e['id']:e['text'] for e in evidence}
    for edge in graph['edges']: edge['source_texts']=[source_text[eid] for eid in edge['evidence_ids']]
    return graph,evidence


def execute(construction: Path, phase: Path, method: str, source: Path, course_id: str,
            aliases: list[str], output: Path, smoke: int | None=None, schema_align: bool=False) -> dict:
    manifest,frozen=validate_phase(construction,phase,method,source)
    courses=[c for c in manifest['courses']['courses'] if c['course_id']==course_id]
    if len(courses)!=1: raise ValueError('Unknown declared course')
    units=corpus_units(manifest,courses[0],aliases,smoke)
    scope='full' if smoke is None else 'smoke'
    identity=digest(dict(phase=frozen['phase_id'],course=course_id,scope=scope,smoke=smoke,units=[u['unit_id'] for u in units]))
    root=output/course_id/method/scope/identity
    params=frozen['parameters']; schema=read_document(manifest['files']['assets']['schema']['path'])
    if schema_align and 'alignment' not in params: raise ValueError('Source-backed alignment budget not frozen')
    with output_lock(root):
        write_immutable(root/'identity.json',dict(input_id=identity,phase_id=frozen['phase_id'],scope=scope,book_set=sorted(aliases)))
        artifact=root/'artifact.json'
        if artifact.exists() and not schema_align: return read_contribution(artifact)
        validate_phase(construction,phase,method,source)
        runtime_config=read_document(manifest['files']['assets']['runtime_config']['path'])
        for role in ('llm','embedding'):
            if runtime_config[role]['base_url'].rstrip('/')!=params['base_url']: raise ValueError('Runtime API must be official frozen URL')
        if (runtime_config['llm']['model']!=params['model'] or runtime_config['embedding']['model']!=params['embedding_model']
            or runtime_config['llm'].get('max_tokens')!=params['max_tokens'] or runtime_config['llm'].get('temperature')!=params['temperature']):
            raise ValueError('Actual runtime model or budget differs from phase')
        config=dict(runtime_config,workers=params['concurrency'],service_retries=params['retries'])
        from domain_ckg_workspace.ckg_incremental.runtime import ModelRuntime
        from .native import driver
        runtime=ModelRuntime(root/'runtime',PROMPTS,config)
        try:
            if artifact.exists():
                native_result=read_contribution(artifact)
                if native_result['input_id']!=identity: raise ValueError('Native artifact identity mismatch')
                native=native_result['native_graph']; evidence=native_result['evidence']; native_driver=None
            else:
                native_driver=driver(method,source,runtime,params,schema)
            # 单书缓存不依赖AB组合，输入与完整/小样本范围仍严格分离。
            if native_driver is not None:
                cache=output/course_id/method/'chunk_cache'/frozen['phase_id']/scope/str(smoke)
                with output_lock(cache):
                    rows=process_units(units,native_driver,cache,frozen['phase_id'],params['retries'])
                native=aggregate_complete(native_driver,rows,root,identity,params['retries'])
                native,evidence=source_support(native,rows,units,method)
            view=semantic_view(native,schema,method)
            result=native_result if native_driver is None else dict(version=1,input_id=identity,phase_id=frozen['phase_id'],course_id=course_id,method_id=method,
                        scope=scope,book_set=sorted(aliases),graph_state='native',status='complete',
                        native_commit=COMMITS[method],adapter_differences=native_driver.differences,
                        text_policy='primary_evidence_concat_no_context_or_g0',parameters=params,
                        provenance=[r['provenance'] for r in rows],native_graph=native,semantic_view=view,
                        evidence=evidence,runtime_stats=runtime.stats())
            if native_driver is not None: result['artifact_id']=digest(result)
            for name,data in [('entities.json',view['entities']),('relations.json',view['relations']),('evidence.json',evidence)]:
                write_immutable(root/name,data)
            write_immutable(root/'concept_kg.json',dict(entities=view['entities'],relations=view['relations']))
            write_immutable(root/'extractable_doc_nodes.json',evidence)
            semantic_ids={n['id'] for n in view['entities'] if n.get('type_status')!='native_untyped_candidate'}
            write_immutable(root/'native_topology.json',dict(directed=native['directed'],nodes=native['nodes'],edges=native['edges'],
                                                          metric_scope='native_adjacency_including_untyped_entities'))
            write_immutable(root/'native_concept_topology.json',dict(directed=native['directed'],nodes=view['entities'],
                edges=[e for e in native['edges'] if e['source'] in semantic_ids and e['target'] in semantic_ids],
                metric_scope='native_concept_adjacency_not_common_typed_semantic_topology'))
            write_immutable(artifact,result)
            if schema_align:
                from .alignment import align
                alignment_params=dict(params,**params['alignment'])
                alignment_config=dict(config,service_retries=params['alignment']['retries'])
                alignment_runtime=ModelRuntime(root/'alignment_runtime',PROMPTS,alignment_config)
                try:
                    aligned=align(native,units,schema,method,alignment_params,alignment_runtime,root/'schema_aligned',identity,
                                  dict(course_id=course_id,phase_id=frozen['phase_id'],scope=scope,book_set=sorted(aliases)))
                    aligned_dir=root/'schema_aligned'
                    aligned_view=aligned['semantic_view']
                    for name,data in [('entities.json',aligned_view['entities']),('relations.json',aligned_view['relations']),('evidence.json',evidence),
                                      ('extractable_doc_nodes.json',evidence),('concept_kg.json',dict(entities=aligned_view['entities'],relations=aligned_view['relations']))]:
                        write_immutable(aligned_dir/name,data)
                    return aligned
                finally: alignment_runtime.close()
            return result
        finally: runtime.close()


def main():
    parser=argparse.ArgumentParser(description='Frozen native baseline execution')
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('freeze','run'):
        command=sub.add_parser(action); command.add_argument('--construction-manifest',required=True,type=Path)
        command.add_argument('--method',required=True,choices=sorted(COMMITS)); command.add_argument('--native-source',required=True,type=Path)
        command.add_argument('--phase',required=True,type=Path)
        if action=='freeze':
            command.add_argument('--parameters',required=True,type=Path); command.add_argument('--dependency-lock',required=True,type=Path)
        else:
            command.add_argument('--course-id',required=True); command.add_argument('--book-set',required=True,choices=['A','B','AB'])
            command.add_argument('--output-root',required=True,type=Path); command.add_argument('--smoke-chunks',type=int)
            command.add_argument('--schema-align',action='store_true')
    args=parser.parse_args()
    if args.action=='freeze':
        result=freeze_phase(args.construction_manifest,args.method,args.native_source,read_document(args.parameters),args.dependency_lock,args.phase)
        print(result['phase_id'])
    else:
        result=execute(args.construction_manifest,args.phase,args.method,args.native_source,args.course_id,list(args.book_set),args.output_root,args.smoke_chunks,args.schema_align)
        print(result['artifact_id'])


if __name__=='__main__': main()
