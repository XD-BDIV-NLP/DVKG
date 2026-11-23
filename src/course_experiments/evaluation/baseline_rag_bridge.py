"""完整原生构图产物到受控RAG的薄桥；保留未定型原节点/边，仅导航到同S原文。"""
from __future__ import annotations
import copy
from pathlib import Path
from ..manifest import digest,file_record,read_document
from ..textbook import scoped_corpus
from . import state_dual_view as state
from .dual_view_rag import asset
from .dvkg_development import CompletedMappingsRetriever

METHODS=('k12_kgraph','kggen_lm_dedup','edc','tree_kg','graphrag_extract')
POLICY=dict(version='complete_native_graph_controlled_RAG_bridge_v1',method_matrix_version='five_methods_four_cohorts_v2',
    graphrag_extract='legacy official GraphExtractor and raw-response joint _process_results construction track; distinct from per-book Microsoft native local/global QA',
    graph_source='one_completed_native_joint_cohort_artifact;no_A_plus_B_union;no_partial_or_gate_substitution',
    untyped='all_native_nodes_and_edges_retained;missing_type_remains_missing;semantic_view_not_used',
    undirected='native_open_edge_kept_as_undirected_navigation_adjacency;original_predicate_preserved_without_directional_typed_relation_claim',
    provenance='source_unit_to_exact_original_S_evidence_navigation;not_per_fact_support_or_correctness_gold',
    retrieval='unchanged_development_frozen_DualView_or_CompletedMappingsRetriever',
    evidence_budget=dict(units=5,serialized_UTF8_bytes=3000),answer=state.ANSWER,judge=state.JUDGE,
    variants=['unchanged_v1','occurrence_mapping_v2'],cohorts=['compAB','compABCD','osAB','netAB'])


def pending_matrix(ready,known_reasons):
    declared={m+':'+c for m in METHODS for c in POLICY['cohorts']}
    if not set(ready)<=declared:raise ValueError('Complete artifact outside the declared five-method/four-cohort matrix')
    result={key:known_reasons.get(key,'complete_joint_cohort_artifact_missing;no_single_book_union_or_gate_substitution') for key in sorted(declared-set(ready))}
    for cohort in POLICY['cohorts']:
        key='graphrag_extract:'+cohort
        if key in result:
            result[key]=('legacy_original381_or787_units_await_all1168_final_joint_materialization;not_Microsoft_native_QA' if cohort in ['compAB','osAB'] else
                'pending_original_unit_and_raw_response_joint_aggregate_qualification;native_single_book_600token_graphs_not_unioned')
    return result


def require_complete_scope(artifact,method,books,aliases):
    if method not in METHODS or artifact.get('status') not in ['complete','completed']:
        raise ValueError('A complete native baseline artifact is required')
    if artifact.get('method_id',method)!=method:raise ValueError('Native artifact method differs')
    if artifact.get('scope')!='full' and not (method=='tree_kg' and 'book_ids' in artifact):
        raise ValueError('Gate/smoke extraction cannot substitute for a full cohort artifact')
    if 'book_ids' in artifact:actual=set(artifact['book_ids'])
    else:
        labels=artifact.get('book_set',[]);labels=list(labels) if isinstance(labels,str) else labels
        if any(a not in aliases for a in labels):raise ValueError('Unknown native source book alias')
        actual={aliases[a] for a in labels}
    if actual!=set(books):raise ValueError('Native joint aggregation book set differs; no single-book unions')
    return actual


def original_units(assets,cohort):
    structures,unit_assets,bindings=state.validate_book_bindings(assets,cohort);expected=[]
    for book in assets['cohorts'][cohort]['active_books']:
        corpus=scoped_corpus(asset(assets['corpora'][book]),None);by_id={e['id']:e for e in corpus['evidence']}
        for chunk in corpus['chunks']:
            evidence=[by_id[e] for e in chunk['evidence_ids']];text='\n\n'.join(e.get('text',e.get('content','')) for e in evidence)
            row=dict(book_id=book,chunk_id=chunk['id'],text=text,text_digest=digest(text),evidence=evidence,
                scope=corpus['construction_scope'],source_fingerprint=corpus['fingerprint'])
            row['unit_id']=digest(row);expected.append(row)
    return expected,structures,unit_assets,bindings


def source_unit_mapping(evidence,units,structures):
    expected={u['unit_id']:u for u in units}
    if len(expected)!=len(units):raise ValueError('Original source unit identifiers are duplicated')
    supplied={e['id']:e for e in evidence}
    if len(supplied)!=len(evidence) or set(supplied)!=set(expected):raise ValueError('Native artifact does not cover every complete original source unit')
    result={}
    for key,unit in expected.items():
        row=supplied[key];book=unit['book_id'];ids=[e['id'] for e in unit['evidence']]
        if (book not in structures or row.get('book_id')!=book or row.get('chunk_id')!=unit['chunk_id'] or
            row.get('text',row.get('content'))!=unit['text'] or row.get('text_digest')!=unit['text_digest'] or
            row.get('source_evidence_ids')!=ids):raise ValueError('Native source unit differs from exact original chunk/evidence')
        for source in unit['evidence']:
            original=structures[book]['evidence'].get(source['id'])
            if original is None or original['book_id']!=book or original['text']!=source.get('text',source.get('content','')):
                raise ValueError('Native source unit does not resolve to exact original S evidence')
        result[key]={book:list(ids)}
    return result


def tree_source_mapping(evidence,structures):
    expected={book+'/evidence/'+eid:(book,eid) for book,s in structures.items() for eid in s['evidence']}
    if len({e['id'] for e in evidence})!=len(evidence) or {e['id'] for e in evidence}!=set(expected):
        raise ValueError('Tree artifact is not the full original structural-evidence scope')
    result={}
    for row in evidence:
        book,eid=expected[row['id']];original=structures[book]['evidence'][eid]
        if row.get('book_id')!=book or row.get('text')!=original['text'] or row.get('evidence_id',eid)!=eid:
            raise ValueError('Tree evidence does not match original S source')
        result[row['id']]={book:[eid]}
    return result


def native_navigation_graph(artifact,source_map,*,method):
    native=artifact['native_graph'];nodes=native['nodes'];edges=native['edges']
    ids=[n['id'] for n in nodes]
    if any(not isinstance(k,str) or not k for k in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Native node identity is missing or duplicated')
    def anchors(row):
        declared=row.get('evidence_ids') or row.get('source_units') or []
        if not isinstance(declared,list) or any(k not in source_map for k in declared):raise ValueError('Unknown native source-unit provenance')
        return sorted({book+'/evidence/'+eid for key in declared for book,eids in source_map[key].items() for eid in eids})
    entities=[]
    for node in nodes:
        if not isinstance(node.get('name'),str) or not node['name'].strip():raise ValueError('Native node has no source name')
        row=copy.deepcopy(node);row.update(entity_id=node['id'],canonical_name=node['name'],source_node_ids=anchors(node),
            native_record=copy.deepcopy(node),provenance_interpretation='source_unit_correlated_navigation_only')
        if 'definition' not in row and isinstance(node.get('description'),str):row['definition']=node['description']
        if isinstance(row.get('aliases'),str):row['aliases']=[row['aliases']]
        if row.get('aliases') is not None and (not isinstance(row['aliases'],list) or any(not isinstance(a,str) for a in row['aliases'])):
            raise ValueError('Native aliases cannot be interpreted without alteration')
        entities.append(row)
    relations=[];directed=native.get('directed') is True
    for i,edge in enumerate(edges):
        if edge.get('source') not in ids or edge.get('target') not in ids:raise ValueError('Native edge endpoint is absent')
        predicate=edge.get('predicate');predicate=predicate if isinstance(predicate,str) else ''
        relation=predicate if directed else ''
        relations.append(dict(edge,relation_id=digest(dict(native_edge_index=i,edge=edge)),head_entity_id=edge['source'],tail_entity_id=edge['target'],
            relation=relation,native_predicate=predicate,native_directed=directed,source_node_ids=anchors(edge),
            native_record=copy.deepcopy(edge),provenance_interpretation='source_unit_correlated_navigation_only',
            typed_semantic_fact_claimed=False))
    graph=dict(entities=entities,relations=relations,native_method=method,native_graph_digest=digest(native),
        artifact_id=artifact.get('artifact_id'),bridge_policy=POLICY['version'])
    audit=dict(native_nodes=len(nodes),retained_native_nodes=len(entities),native_edges=len(edges),retained_native_edges=len(relations),
        native_untyped_nodes=sum(not n.get('type') or n.get('type_status')=='native_untyped_candidate' for n in nodes),
        unanchored_native_nodes=sum(not n['source_node_ids'] for n in entities),unanchored_native_edges=sum(not e['source_node_ids'] for e in relations),
        native_directed=directed,typed_semantic_facts_claimed=False,source_unit_correlated_is_fact_support=False)
    return graph,audit


class BaselineRetriever(CompletedMappingsRetriever):
    def __init__(self,method,graph,structures,mappings,unit_assets,concept_ids,concept_vectors,*,variant='occurrence_mapping_v2'):
        if method not in METHODS or variant not in POLICY['variants']:raise ValueError('Explicit native method and frozen development variant required')
        if mappings.get('concept_graph_id')!=digest(graph) or mappings.get('structure_ids')!={b:asset(s)['structure_id'] for b,s in structures.items()}:
            raise ValueError('Baseline mapping differs from own complete graph and same S')
        self.method,self.variant,self.semantic_state=method,variant,'controlled_'+method
        super().__init__(graph,structures,mappings,unit_assets,concept_ids,concept_vectors)

    def retrieve(self,question,target_book_id,query_vector,method='dual'):
        if method!='dual':raise ValueError('The full baseline bridge uses the same frozen dual retrieval configuration')
        cls=CompletedMappingsRetriever if self.variant=='occurrence_mapping_v2' else state.DualViewRetriever
        selected,trace=cls.retrieve(self,question,target_book_id,query_vector,'dual')
        return selected,dict(trace,native_method=self.method,method=self.method+'+S+M',controlled_bridge_policy=POLICY['version'],
            source_unit_correlated_is_fact_support=False,native_graph_used_as_answer_evidence=False)


def load_complete_artifact(record,assets,cohort,method):
    artifact=asset(record)
    if artifact.get('artifact_id')!=digest({k:v for k,v in artifact.items() if k!='artifact_id'}):raise ValueError('Complete native artifact signature differs')
    cohort_row=assets['cohorts'][cohort];construction=asset(cohort_row['construction']);books=cohort_row['active_books']
    courses=[c for c in construction['courses']['courses'] if set(books)<={b['book_id'] for b in c['books']}]
    if len(courses)!=1:raise ValueError('Cohort must correspond to one original independent course')
    course=courses[0];aliases={b['alias']:b['book_id'] for b in course['books']}
    if artifact.get('course_id')!=course['course_id']:raise ValueError('Native graph course differs from original source cohort')
    require_complete_scope(artifact,method,books,aliases)
    units,structures,unit_assets,bindings=original_units(assets,cohort)
    source_map=tree_source_mapping(artifact['evidence'],structures) if method=='tree_kg' else source_unit_mapping(artifact['evidence'],units,structures)
    graph,audit=native_navigation_graph(artifact,source_map,method=method);mappings=state.state_mappings(graph,structures)
    return dict(method=method,cohort=cohort,active_books=books,artifact=record,graph=graph,mappings=mappings,
        structure_units=bindings,source_map=source_map,source_unit_count=len(units),source_scope_verified=True,
        audit=audit,construction=cohort_row['construction'],provenance_policy=POLICY['provenance'],
        native_commit=artifact.get('native_commit'),adapter_differences=artifact.get('adapter_differences',artifact.get('reconstruction_differences')))
