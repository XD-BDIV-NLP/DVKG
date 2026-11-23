from __future__ import annotations
import copy
from collections import defaultdict
from .manifest import digest


def build_structure(corpus):
    if len(corpus['books'])!=1:raise ValueError('One textbook per structure graph required')
    book=corpus['books'][0]['id'];root=book+'/book';nodes={root:dict(type='Book',book_id=book)};edges=set();evidence={}
    chunks={}
    for chunk in corpus['chunks']:
        if chunk['book_id']!=book:raise ValueError('Foreign source chunk')
        for eid in chunk['evidence_ids']:chunks.setdefault(eid,chunk['id'])
    for source in corpus['evidence']:
        if source['book_id']!=book or source['id'] in evidence:raise ValueError('Foreign or duplicate evidence')
        if source['id'] not in chunks:raise ValueError('Evidence absent from declared source chunks')
        path=list(source.get('section_path_ids') or [source.get('section_id') or book])
        titles=list(source.get('title_path') or []);parent=root
        for i,section in enumerate(path):
            key=book+'/section/'+section
            node=dict(type='Chapter' if section==source.get('chapter_id') else 'Section',book_id=book,source_id=section,parent=parent,title=titles[i] if i<len(titles) else None)
            if key in nodes and nodes[key]['parent']!=parent:raise ValueError('Conflicting original section parent')
            nodes.setdefault(key,node);edges.add((parent,key,'contains'));parent=key
        key=book+'/evidence/'+source['id'];nodes[key]=dict(type='Evidence',book_id=book,parent=parent,source_id=source['id']);edges.add((parent,key,'contains'))
        evidence[source['id']]=dict(book_id=book,evidence_id=source['id'],chunk_id=chunks[source['id']],section_id=source.get('section_id') or path[-1],
            section_path_ids=path,chapter_id=source.get('chapter_id'),text=source['text'],source=copy.deepcopy(source))
    ordered=sorted(evidence.values(),key=lambda r:(r['source'].get('paragraph_order',0),r['source'].get('start',0),r['evidence_id']))
    for left,right in zip(ordered,ordered[1:]):
        edges.add((book+'/evidence/'+left['evidence_id'],book+'/evidence/'+right['evidence_id'],'next'))
    value=dict(book_id=book,nodes=nodes,edges=[list(e) for e in sorted(edges)],evidence=evidence,
        provenance='original_corpus_section_paths_paragraph_order_and_offsets; specialized_formula_table_types_not_inferred')
    value['structure_id']=digest(value);return value


def build_mappings(graph,structures):
    by_id=defaultdict(list)
    for book,structure in structures.items():
        for eid in structure['evidence']:by_id[eid].append(book)
    audit=dict(resolved_anchors=0,missing_anchors=0,ambiguous_anchors=0)
    def resolve(anchor):
        if not isinstance(anchor,str):audit['missing_anchors']+=1;return None
        if '/evidence/' in anchor:
            prefix,eid=anchor.split('/evidence/',1);book=prefix.split('/')[-1]
            if book in structures and eid in structures[book]['evidence']:
                audit['resolved_anchors']+=1;return book,eid
        elif len(by_id.get(anchor,[]))==1:
            audit['resolved_anchors']+=1;return by_id[anchor][0],anchor
        elif len(by_id.get(anchor,[]))>1:
            audit['ambiguous_anchors']+=1;return None
        audit['missing_anchors']+=1;return None
    def anchors(row):
        found=defaultdict(set)
        ids=set(row.get('source_node_ids') or row.get('evidence_ids') or [])
        ids.update(e.get('id') for e in row.get('evidence',[]) if isinstance(e,dict))
        for anchor in sorted(ids,key=str):
            value=resolve(anchor)
            if value:found[value[0]].add(value[1])
        return {b:sorted(ids) for b,ids in sorted(found.items())}
    concepts={e.get('entity_id',e.get('id')):anchors(e) for e in graph.get('entities',graph.get('nodes',[]))}
    relations={e.get('relation_id',e.get('id',digest(e))):anchors(e) for e in graph.get('relations',graph.get('edges',[]))}
    value=dict(concepts=concepts,relations=relations,audit=audit,basis='existing_extraction_provenance; occurrence_and_relation_support_kept_separate_not_independent_gold',
               concept_graph_id=digest(graph),structure_ids={b:s['structure_id'] for b,s in structures.items()})
    value['mapping_id']=digest(value);return value
