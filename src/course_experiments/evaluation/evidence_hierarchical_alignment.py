"""证据约束层级对齐的离线核心；核验结论与真实token计数由冻结运行提供。"""
from __future__ import annotations

import hashlib
import math
from numbers import Real

from ..manifest import digest
from .graph_rag_v2 import matches

POLICY = dict(version='evidence_hierarchical_alignment_prototype_v1',
    candidates='union_original_sources_exact_names_aliases_top_positive_semantic',
    verification='external_frozen_verifier_then_exact_source_span_and_role_checks',
    hierarchy='native_textbook_tree_maximum_verified_passage_weight',
    routing='marginal_concept_coverage_per_token_antichain_then_best_single',
    semantic_accuracy_claimed=False,optimality_guarantee=False,
    implementation_status='offline_core_not_full_evaluated_pipeline')
ROLES = {'mention','definition','explanation','example'}


def _score(value):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):
        raise ValueError('Finite numeric score required')
    if not -1.000001 <= value <= 1.000001:
        raise ValueError('Cosine or coverage score outside normalized range')
    return max(0.,min(1.,float(value)))


def candidate_paragraphs(concept,structure,paragraph_scores,*,origins=(),limit=8):
    if type(limit) is not int or limit<1:
        raise ValueError('Positive per-channel candidate limit required')
    evidence=structure['evidence'];book=structure['book_id']
    if set(paragraph_scores)-evidence.keys() or set(origins)-evidence.keys():
        raise ValueError('Unknown target-book candidate or origin')
    if any(row['book_id']!=book for row in evidence.values()):
        raise ValueError('Foreign evidence in target-book structure')
    scores={key:_score(value) for key,value in paragraph_scores.items()}
    names=[concept.get('canonical_name',concept.get('name','')),*(concept.get('aliases') or [])]
    names={name.strip() for name in names if isinstance(name,str)
           and len(name.strip()) >= (3 if name.strip().isascii() else 2)}
    lexical=[key for key,row in evidence.items() if any(matches(n,row['text']) for n in names)]
    dense=[key for key,value in scores.items() if value>0]
    order=lambda key:(-scores.get(key,0.),key)
    channels={}
    for channel,keys in [('origin',sorted(set(origins))),
                         ('lexical',sorted(lexical,key=order)[:limit]),
                         ('semantic',sorted(dense,key=order)[:limit])]:
        for key in keys:channels.setdefault(key,[]).append(channel)
    return [dict(book_id=book,evidence_id=key,semantic_score=scores.get(key,0.),
                 channels=channels[key],status='candidate_not_verified')
            for key in sorted(channels)]


def qualify_mapping(graph_id,concept_id,structure,candidate,verdict):
    book=structure['book_id'];eid=candidate['evidence_id']
    if candidate['book_id']!=book or eid not in structure['evidence']:
        raise ValueError('Mapping candidate outside target textbook')
    source=structure['evidence'][eid]
    if source['book_id']!=book or not graph_id or not concept_id:
        raise ValueError('Invalid source or concept identity')
    flags=['same_sense','scope_consistent','evidence_supported']
    if (verdict.get('decision') not in {'accept','reject','unresolved'}
            or any(type(verdict.get(key)) is not bool for key in flags)
            or not isinstance(verdict.get('reason'),str)):
        raise ValueError('Explicit verification decision and semantic flags required')
    accepted=verdict['decision']=='accept' and all(verdict[key] for key in flags)
    role=verdict.get('role');citations=verdict.get('citations',[])
    if accepted:
        if role not in ROLES or not isinstance(citations,list) or not citations:
            raise ValueError('Accepted source mapping requires an explicit role and quotation')
        for quote in citations:
            start,end=quote.get('start'),quote.get('end')
            if (quote.get('evidence_id')!=eid or type(start) is not int or type(end) is not int
                    or not 0<=start<end<=len(source['text'])
                    or source['text'][start:end]!=quote.get('quote')):
                raise ValueError('Quotation must exactly equal the addressed original source span')
    row=dict(policy=POLICY['version'],concept_graph_id=graph_id,concept_id=concept_id,
        book_id=book,structure_id=structure['structure_id'],evidence_id=eid,
        source_text_sha256=hashlib.sha256(source['text'].encode('utf-8')).hexdigest(),
        status='accepted' if accepted else 'rejected_or_unresolved',role=role,
        navigation_only=role=='mention',weight=_score(candidate['semantic_score']) if accepted else 0.,
        verification=verdict)
    return dict(row,mapping_id=digest(row))


def _hierarchy(structure):
    nodes=structure['nodes'];book=structure['book_id'];ancestors={};desc={key:set() for key in nodes}
    roots=[key for key,node in nodes.items() if not node.get('parent')]
    if len(roots)!=1 or nodes[roots[0]]['type']!='Book':
        raise ValueError('A single native textbook root is required')
    for key in nodes:
        chain=[];cursor=key
        while cursor is not None:
            if cursor in chain or cursor not in nodes or nodes[cursor]['book_id']!=book:
                raise ValueError('Cyclic, unknown or foreign native hierarchy node')
            chain.append(cursor);cursor=nodes[cursor].get('parent')
        ancestors[key]=set(chain[1:])
    owners={}
    for key,node in nodes.items():
        if node['type']!='Evidence':continue
        eid=node['source_id']
        if eid in owners or eid not in structure['evidence']:
            raise ValueError('Evidence must have one unique native leaf owner')
        owners[eid]=key
        for parent in ancestors[key]|{key}:desc[parent].add(eid)
    if set(owners)!=set(structure['evidence']):
        raise ValueError('Every original paragraph requires a native leaf')
    return ancestors,desc,owners


def lift_coverage(graph,structure,mappings):
    ancestors,_,owners=_hierarchy(structure)
    graph_id=digest(graph);concepts={row.get('entity_id',row.get('id')) for row in graph['entities']}
    result={}
    for row in mappings:
        if row.get('mapping_id')!=digest({k:v for k,v in row.items() if k!='mapping_id'}):
            raise ValueError('Mapping record changed after verification')
        if (row['concept_graph_id']!=graph_id or row['structure_id']!=structure['structure_id']
                or row['book_id']!=structure['book_id'] or row['concept_id'] not in concepts):
            raise ValueError('Mapping graph, textbook or concept version differs')
        eid=row['evidence_id']
        if eid not in owners or row['source_text_sha256']!=hashlib.sha256(structure['evidence'][eid]['text'].encode('utf-8')).hexdigest():
            raise ValueError('Original evidence changed; revalidation required')
        if row['status']!='accepted':continue
        for key in ancestors[owners[eid]]|{owners[eid]}:
            values=result.setdefault(key,{})
            values[row['concept_id']]=max(values.get(row['concept_id'],0.),_score(row['weight']))
    return result


def select_regions(structure,coverage,query_weights,paragraph_tokens,*,budget,max_regions):
    ancestors,desc,_=_hierarchy(structure)
    if type(budget) is not int or budget<0 or type(max_regions) is not int or max_regions<1:
        raise ValueError('Nonnegative token budget and positive region count required')
    if set(paragraph_tokens)!=set(structure['evidence']) or any(type(n) is not int or n<0 for n in paragraph_tokens.values()):
        raise ValueError('Exact frozen tokenizer counts required for every paragraph')
    if set(coverage)-structure['nodes'].keys():raise ValueError('Unknown coverage region')
    values={key:{c:_score(v) for c,v in row.items()} for key,row in coverage.items()}
    weights={key:float(value) for key,value in query_weights.items()}
    if any(not math.isfinite(v) or v<0 for v in weights.values()):raise ValueError('Invalid query concept weight')
    total=sum(weights.values());weights={k:v/total for k,v in weights.items()} if total else {}
    costs={key:sum(paragraph_tokens[e] for e in ids) for key,ids in desc.items()}
    eligible=sorted(key for key in values if desc[key] and structure['nodes'][key]['type']!='Book')
    def utility(keys):
        return sum(weight*max((values[key].get(c,0.) for key in keys),default=0.)
                   for c,weight in sorted(weights.items()))
    selected=[];spent=0;steps=[]
    while len(selected)<max_regions:
        choices=[];current=utility(selected)
        for key in eligible:
            if key in selected or costs[key]+spent>budget:continue
            if any(key in ancestors[old] or old in ancestors[key] for old in selected):continue
            gain=utility(selected+[key])-current
            if gain>0:choices.append((gain/costs[key] if costs[key] else math.inf,gain,costs[key],key))
        if not choices:break
        _,gain,cost,key=min(choices,key=lambda row:(-row[0],-row[1],row[2],row[3]))
        selected.append(key);spent+=cost;steps.append(dict(region_id=key,gain=gain,tokens=cost))
    single=sorted((key for key in eligible if costs[key]<=budget and utility([key])>0),
                  key=lambda key:(-utility([key]),costs[key],key))
    replaced=bool(single and utility([single[0]])>utility(selected))
    if replaced:selected=[single[0]];spent=costs[single[0]]
    evidence=sorted({eid for key in selected for eid in desc[key]})
    return dict(policy=POLICY,book_id=structure['book_id'],region_ids=selected,
        evidence_ids=evidence,tokens=spent,objective=utility(selected),
        greedy_steps=steps,best_single_replaced_greedy=replaced,
        fallback_required=not bool(selected),token_count_source='caller_frozen_tokenizer')
