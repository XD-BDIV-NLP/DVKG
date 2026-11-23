"""全方法身份候选与资格/成本预检；不调用模型，不启用正式映射。"""
from __future__ import annotations

from collections import defaultdict
import copy

from ..splits import RELATIONS
from ..manifest import digest
from .scope_identity import names,prepare_pair


SEMANTIC_TYPES = {'Concept','Method','Principle','Formula','Algorithm','Structure','Process','Metric','Tool','Phenomenon'}


def source_evidence_index(base, artifact_evidence):
    result=dict(base)
    rows=artifact_evidence.items() if isinstance(artifact_evidence,dict) else (
        (None,row) for row in artifact_evidence)
    for key,row in rows:
        text=row.get('text',row.get('content',''))
        if not text or row.get('text_digest') != digest(text):
            raise ValueError('Native source-unit evidence content digest mismatch')
        for eid in {key,row.get('id'),row.get('source_node_id')} - {None,''}:
            if eid in result:
                previous=result[eid]
                previous=previous if isinstance(previous,str) else previous.get('text',previous.get('content',''))
                if previous != text:
                    raise ValueError('Native evidence ID conflicts with a different source text')
            result[eid]=copy.deepcopy(row)
    return result


def entity_sources(entity):
    result=copy.deepcopy(entity)
    syntax=[]
    if result.get('aliases') is None:
        result['aliases']=[];syntax.append('aliases_null_to_empty')
    for row in result.get('identity_provenance') or []:
        if row.get('aliases') is None:row['aliases']=[];syntax.append('member_aliases_null_to_empty')
    result['input_syntax_normalizations']=syntax
    result['source_node_ids']=sorted({eid for field in ['source_node_ids','evidence_ids','merged_source_node_ids']
        for eid in entity.get(field) or []})
    return result


def qualification(graph, capabilities=None):
    entities=graph.get('entities',[]);relations=graph.get('relations',[])
    ids={e['entity_id'] for e in entities}
    typed=[r for r in relations if r.get('relation') in RELATIONS and
        r.get('head_entity_id') in ids and r.get('tail_entity_id') in ids]
    untyped=any(e.get('type_status')=='native_untyped_candidate' or
        e.get('entity_type',e.get('type')) not in SEMANTIC_TYPES for e in entities)
    if untyped:
        status='not_applicable_untyped_concept_identity'
    elif capabilities is not None and capabilities.get('kge') is not True:
        status='not_applicable_native_capability_requires_separate_alignment'
    elif not typed:
        status='insufficient_no_supported_typed_edges'
    else:
        status='eligible_typed_graph'
    return dict(status=status,entities=len(entities),declared_edges=len(relations),supported_typed_edges=len(typed),
        no_types_or_predicates_invented=True)


def prepare_graph(course_id, graph, reference_entities, reference_evidence, graph_evidence, capabilities=None):
    eligible=qualification(graph,capabilities)
    index=defaultdict(dict)
    for row in graph.get('entities',[]):
        entity=entity_sources(row)
        for name in names(entity):index[name][entity['entity_id']]=entity
    packets={};invalid=0;without=0
    for reference in reference_entities:
        candidates={eid:e for name in names(reference) for eid,e in index[name].items()}
        if not candidates:without+=1
        for entity in candidates.values():
            try:packet=prepare_pair(course_id,reference,entity,reference_evidence,graph_evidence)
            except ValueError:
                invalid+=1;continue
            packets[packet['candidate_id']]=packet
    source=sum(p['both_sides_have_source'] for p in packets.values())
    authorized=source if eligible['status']=='eligible_typed_graph' else 0
    return dict(qualification=eligible,reference_entities=len(reference_entities),candidate_pairs=len(packets),
        reference_entities_without_exact_name_or_alias=without,two_sided_source_pairs=source,
        prior_or_missing_graph_source_pairs=len(packets)-source,invalid_source_candidates=invalid,
        maximum_initial_and_reverse_logical_judgments=2*authorized,actual_HTTP_model_retries_not_estimated=True,
        model_calls=0,mapper_enabled=False,new_split_created=False,
        prerequisite='independent_development_gate_review_then_frozen_uniform_rules_all_applicable_methods'),list(packets.values())
