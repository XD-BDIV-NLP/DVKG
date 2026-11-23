from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from ..manifest import file_record,digest

EVALUATOR_SHA='e438c6118167e411cc7a75819dab58b88d9290db9e913a9da8cf78d6966342ad'
STRUCTURAL={'Book','Chapter','Section','Chunk','Evidence','Document'}
CONCEPT_TYPES={'Concept','Method','Principle','Formula','Algorithm','Structure','Process','Metric','Tool','Phenomenon'}

def load_evaluator(path,expected=EVALUATOR_SHA):
    path=Path(path)
    if file_record(path)['sha256']!=expected:raise ValueError('Existing evaluator source SHA differs')
    name='course_existing_evaluator_'+expected[:12];spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    utils_path=path.parent.parent/'src/utils.py';old=sys.modules.get('src.utils')
    if utils_path.exists():
        utils_spec=importlib.util.spec_from_file_location('src.utils',utils_path);utils=importlib.util.module_from_spec(utils_spec);utils_spec.loader.exec_module(utils)
        sys.modules['src.utils']=utils
    try:spec.loader.exec_module(module)
    finally:
        if utils_path.exists():
            if old is None:sys.modules.pop('src.utils',None)
            else:sys.modules['src.utils']=old
    module._course_utils_file=module.normalize_whitespace.__globals__['__file__']
    return module

def evaluate(graph,evaluator_path,*,native_undirected=False):
    native='nodes' in graph
    entities=graph.get('entities',[dict(e,entity_id=e.get('entity_id',e.get('id')),canonical_name=e.get('canonical_name',e.get('name'))) for e in graph.get('nodes',[])])
    selected=[e for e in entities if e.get('type') in CONCEPT_TYPES and e.get('type_status')!='native_untyped_candidate']
    ids={e['entity_id'] for e in selected}
    raw_relations=graph.get('relations',[dict(e,head_entity_id=e['source'],tail_entity_id=e['target'],relation='native_adjacency') for e in graph.get('edges',[])])
    relations=[r for r in raw_relations if r.get('head_entity_id') in ids and r.get('tail_entity_id') in ids]
    if native_undirected:
        relations=[v for r in relations for v in [r,dict(r,head_entity_id=r['tail_entity_id'],tail_entity_id=r['head_entity_id'])] if r['head_entity_id']!=r['tail_entity_id']]
    module=load_evaluator(evaluator_path);metrics=module.evaluate_graph(dict(entities=selected,relations=relations))
    result=dict(status='completed' if selected else 'insufficient',graph_id=digest(graph),evaluator=file_record(evaluator_path),utils_dependency=file_record(module._course_utils_file),entity_count=len(selected),relation_count=len(relations),metrics=metrics if selected else None,layer='native_undirected_bidirectional_projection' if native_undirected else 'native_directed_adjacency' if native else 'concept_semantic',excluded_structural_or_untyped=len(entities)-len(selected))
    if native_undirected:result['directional_metrics_applicability']='not_applicable; directed density,degree,reciprocity,SCC,PageRank are projection-only; use weak components,isolates,kcore main comparison'
    return result
