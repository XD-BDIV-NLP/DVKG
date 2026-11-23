from __future__ import annotations
import hashlib
import importlib.metadata
from pathlib import Path

from ..manifest import digest, file_record, read_document, validate
from ..contributions import write_immutable

COMMITS = {'k12_kgraph':'865bc3556768224d929825d7c258a5e77ac1aab0',
           'kggen_lm_dedup':'e0cb917241654cb820f9dd3ba64c397f329fbfcd',
           'graphrag_extract':'56a865bff0cac9fbbc24f343cf52d0a1850abba6'}
ROOT = Path(__file__).resolve().parent
EXCLUDED = {'.git','__pycache__','.pytest_cache','*.egg-info'}


def source_tree(root: Path) -> dict:
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(Path(root).rglob('*')) if p.is_file() and
           not any(part in EXCLUDED or part.endswith('.egg-info') for part in p.relative_to(root).parts)
           and p.suffix!='.pyc'}
    return dict(files=files,tree_digest=digest(files))


def check_source(method: str, root: Path) -> dict:
    if method not in COMMITS: raise ValueError('Baseline unavailable: Tree-KG source missing; BookQA excluded')
    expected=read_document(ROOT/'source_pins.json')[method]
    actual=source_tree(root)
    if actual != expected: raise ValueError('Official pinned source tree mismatch')
    return dict(path=str(Path(root).resolve()),commit=COMMITS[method],**actual)


def validate_parameters(params: dict) -> None:
    expected=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',
                  embedding_model='BAAI/bge-m3',max_tokens=12000,temperature=0,seed=42)
    if set(params)-{'alignment'}!=set(expected)|{'concurrency','retries','max_gleanings'}: raise ValueError('Unexpected baseline parameter; credentials must not be frozen')
    if any(params.get(k)!=v for k,v in expected.items()): raise ValueError('Frozen model/provider/budget mismatch')
    for key in ('concurrency','retries','max_gleanings'):
        if type(params.get(key)) is not int or not (1 <= params[key] <= 8 if key=='concurrency' else 0 <= params[key] <= 6):
            raise ValueError('Explicit bounded baseline parameter required: '+key)
    if 'alignment' in params and params['alignment']!=dict(max_tokens=12000,temperature=0,retries=6):
        raise ValueError('Alignment requires explicit 12000-token, temperature0, retries6 budget')


def assets() -> dict:
    return {p.name:file_record(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in {'.py','.json','.txt'}}


def environment_versions() -> dict:
    names={dist.metadata['Name'].lower() for dist in importlib.metadata.distributions()}
    return {name:importlib.metadata.version(name) for name in sorted(names)}


def freeze_phase(construction: Path, method: str, source: Path, parameters: dict,
                 dependency_lock: Path, output: Path) -> dict:
    manifest=validate(construction); validate_parameters(parameters)
    payload=dict(version=1,stage='native_baseline',construction_manifest_id=manifest['manifest_id'],
                 method_id=method,native_source=check_source(method,source),parameters=parameters,
                 implementation=assets(),dependency_lock=file_record(dependency_lock),
                 installed_versions=environment_versions())
    payload['phase_id']=digest(payload)
    return write_immutable(output,payload)


def validate_phase(construction: Path, phase: Path, method: str, source: Path) -> tuple[dict,dict]:
    manifest=validate(construction); frozen=read_document(phase)
    if frozen.get('phase_id') != digest({k:v for k,v in frozen.items() if k!='phase_id'}): raise ValueError('Phase digest mismatch')
    if frozen.get('stage')!='native_baseline' or frozen.get('method_id')!=method or frozen['construction_manifest_id']!=manifest['manifest_id']:
        raise ValueError('Construction/method phase identity mismatch')
    validate_parameters(frozen['parameters'])
    if frozen['native_source'] != check_source(method,source) or frozen['implementation'] != assets(): raise ValueError('Frozen baseline source/implementation mismatch')
    if frozen['dependency_lock'] != file_record(frozen['dependency_lock']['path']): raise ValueError('Dependency lock changed')
    if frozen['installed_versions'] != environment_versions(): raise ValueError('Frozen installed package versions changed')
    if method=='kggen_lm_dedup' and frozen['installed_versions'].get('dspy')!='2.6.27': raise ValueError('KGGen requires DSPy 2.6.27 chat backend')
    return manifest,frozen


def semantic_view(graph: dict, schema: dict, method: str) -> dict:
    types={k.lower():k for k in schema.get('entity_types',{'Concept':''})}
    semantic=set(types)
    allowed={'concept'} if method=='k12_kgraph' else semantic
    entities=[]
    for node in graph['nodes']:
        if node.get('type','').lower() not in allowed: continue
        entities.append(dict(node,id=node['id'],entity_type='Concept' if method=='k12_kgraph' else types[node['type'].lower()],
                             definition=node.get('definition',node.get('description')),
                             entity_id=node['id'],canonical_name=node['name'],
                             merged_raw_entity_ids=[node['id']],merged_source_node_ids=node.get('evidence_ids',[]),
                             merged_definitions=[node.get('definition',node.get('description'))] if node.get('definition',node.get('description')) else []))
        entities[-1]['native_type']=node['type']; entities[-1]['type']=entities[-1]['entity_type']
    ids={n['id'] for n in entities}; names={n['id']:n['name'] for n in entities}
    relations=[]; unmapped=[]
    maps={'is_a':'is_a','prerequisites_for':'prerequisite_of'} if method=='k12_kgraph' else {k:k for k in schema['relation_types']}
    for edge in graph['edges']:
        predicate=maps.get(edge.get('predicate'))
        if not graph['directed'] or edge['source'] not in ids or edge['target'] not in ids or predicate not in schema['relation_types']:
            unmapped.append(edge); continue
        relations.append(dict(edge,head=names[edge['source']],tail=names[edge['target']],relation=predicate,
                              relation_id=digest(edge),head_entity_id=edge['source'],tail_entity_id=edge['target'],
                              head_name=names[edge['source']],tail_name=names[edge['target']],
                              conditions=edge.get('conditions',''),evidence_ids=edge.get('evidence_ids',[]),
                              source_node_ids=edge.get('evidence_ids',[]),evidences=edge.get('source_texts',[]),
                              is_inferred=False))
    typed=graph['directed']
    untyped_entities=any(n.get('type_status')=='native_untyped_candidate' for n in entities)
    applicable=typed and not untyped_entities
    reasons={}
    if not typed:
        reasons=dict(topology='Native undirected descriptions have no typed semantic predicates',
                     kge='Native undirected descriptions have no directed typed facts')
    elif untyped_entities:
        reasons=dict(topology='Native untyped entity strings require separately frozen concept-type alignment',
                     kge='Common concept-level KGE requires separately frozen concept-type alignment')
    return dict(entities=entities,relations=relations,unmapped_relations=unmapped,
                projection_coverage=dict(native_edges=len(graph['edges']),mapped_edges=len(relations)),
                capabilities=dict(rag=True,native_topology=True,topology=True if applicable else None,kge=True if applicable else None),
                capability_reasons=reasons)
