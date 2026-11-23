from __future__ import annotations
import json
import math
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from ..manifest import digest,file_record,read_document
from .graph_rag_v2 import EvidenceRetriever,STRUCTURAL,terms,matches

POLICY=dict(version='target_book_dual_view_v1',seed=42,concept_top_k=6,concept_pool=32,semantic_floor=.25,
    expanded_concepts=20,neighbors_per_seed=4,section_top_k=8,rrf_constant=60,
    retrieval_top_k=5,context_bytes=3000,embedding_text_chars=1024,
    fallback='target_book_global_hybrid_only_when_no_coarse_candidates',
    development='synthetic_scope_and_navigation_tests; no_observed_test_score_tuning')
METHODS=('dual','keyword','vector','hybrid','no_structure','no_mapping','no_semantic','no_relations')


def concept_text(entity):
    return json.dumps(dict(name=entity.get('canonical_name',entity.get('name','')),aliases=(entity.get('aliases') or [])[:16],
        scope=entity.get('scope') or '',definition=entity.get('definition') or '\n'.join(entity.get('merged_definitions',[])[:2])),ensure_ascii=False)[:POLICY['embedding_text_chars']]


def asset(value):
    if isinstance(value,dict) and 'path' in value and 'sha256' in value:
        if file_record(value['path'])!=value:raise ValueError('Frozen asset changed')
        return read_document(value['path'])
    return value


def reciprocal(orders):
    scores=defaultdict(float)
    for order in orders:
        for rank,key in enumerate(order,1):scores[key]+=1/(POLICY['rrf_constant']+rank)
    return scores


class DualViewRetriever:
    def __init__(self,graph,structures,mappings,unit_assets,concept_ids,concept_vectors):
        self.graph=graph;self.graph_id=digest(graph);self.structures=structures;self.mappings=mappings;self.unit_assets=unit_assets
        self.entities={e.get('entity_id',e.get('id')):e for e in graph['entities'] if e.get('type') not in STRUCTURAL}
        self.ids=list(concept_ids)
        if self.ids!=sorted(self.entities):raise ValueError('Concept vector identity/order differs')
        self.concept_vectors=np.asarray(concept_vectors,dtype=np.float32)
        if self.concept_vectors.ndim!=2 or self.concept_vectors.shape[0]!=len(self.ids):raise ValueError('Concept vectors do not cover graph')
        self.position={key:i for i,key in enumerate(self.ids)}
        self.concept_lexical=EvidenceRetriever([dict(book_id='concept',chunk_id=k,text=concept_text(self.entities[k]),evidence_ids=[]) for k in self.ids],dict(entities=[],relations=[]))
        self.adjacency=defaultdict(list)
        for edge in graph['relations']:
            head=edge.get('head_entity_id');tail=edge.get('tail_entity_id')
            if head in self.entities and tail in self.entities and head!=tail:
                self.adjacency[head].append((tail,edge,'outgoing'));self.adjacency[tail].append((head,edge,'incoming'))
        self.active_book=None;self.candidate_ids=()

    def load_book(self,book):
        if book not in self.structures or book not in self.unit_assets:raise ValueError('Unknown target textbook')
        if self.active_book==book:return 0.
        start=time.monotonic();structure=asset(self.structures[book]);units=asset(self.unit_assets[book])
        if structure['book_id']!=book or structure['structure_id']!=self.mappings['structure_ids'][book]:raise ValueError('Structure mapping identity mismatch')
        vectors=units['vectors']
        if isinstance(vectors,dict):
            if file_record(vectors['path'])!=vectors:raise ValueError('Book vector matrix changed')
            vectors=np.load(vectors['path'],mmap_mode='r')
        self.lexical=EvidenceRetriever(units['rows'],dict(entities=[],relations=[]));self.rows=self.lexical.rows
        if any(r['book_id']!=book for r in self.rows):raise ValueError('Foreign book in target index')
        keys=[(r['book_id'],r['evidence_ids'][0]) for r in units['rows']]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate target source unit')
        bykey={key:i for i,key in enumerate(keys)}
        self.vectors=np.asarray(vectors)[[bykey[(book,r['evidence_ids'][0])] for r in self.rows]]
        if self.vectors.shape[0]!=len(self.rows):raise ValueError('Target vectors incomplete')
        self.structure=structure;self.evidence_index={r['evidence_ids'][0]:i for i,r in enumerate(self.rows)}
        self.section_indices=defaultdict(set)
        for i,row in enumerate(self.rows):
            source=structure['evidence'][row['evidence_ids'][0]]
            if source['text']!=row['text']:raise ValueError('Target evidence text differs from original corpus')
            for section in source['section_path_ids']:self.section_indices[section].add(i)
        self.active_book=book;return time.monotonic()-start

    def seeds(self,question,query,semantic):
        lexical=self.concept_lexical.lexical(question)
        term_order=sorted((i for i,s in enumerate(lexical) if s>0),key=lambda i:(-lexical[i],self.ids[i]))[:POLICY['concept_pool']]
        dense=self.concept_vectors@query
        semantic_order=sorted((i for i,s in enumerate(dense) if s>=POLICY['semantic_floor']),key=lambda i:(-float(dense[i]),self.ids[i]))[:POLICY['concept_pool']] if semantic else []
        scores=reciprocal([term_order,semantic_order] if semantic else [term_order])
        exact=set()
        for i in set(term_order+semantic_order):
            entity=self.entities[self.ids[i]]
            if any(matches(n,question) for n in [entity.get('canonical_name',entity.get('name','')),*(entity.get('aliases') or [])] if isinstance(n,str)):
                exact.add(i)
        ranked=sorted(scores,key=lambda i:(i not in exact,-scores[i],self.ids[i]))[:POLICY['concept_top_k']]
        return [self.ids[i] for i in ranked],dense

    def retrieve(self,question,target_book_id,query_vector,method='dual'):
        if not isinstance(question,str) or method not in METHODS:raise ValueError('Invalid question or method')
        started=time.monotonic();load=self.load_book(target_book_id)
        query=np.asarray(query_vector,dtype=np.float32);query=query/(np.linalg.norm(query) or 1.)
        if query.shape!=(self.concept_vectors.shape[1],):raise ValueError('Query vector dimensions differ')
        lexical=self.lexical.lexical(question);seeds=[];expanded=[];paths=[];sections={};fallback=False
        if method in {'keyword','vector','hybrid'}:
            candidates=set(range(len(self.rows)))
        else:
            seeds,dense=self.seeds(question,query,method!='no_semantic');expanded=list(seeds)
            if method!='no_relations':
                for seed in seeds:
                    options=[]
                    for target,edge,direction in self.adjacency.get(seed,[]):
                        predicate=edge.get('relation','')
                        if any(t in question for t in ['原因','前提','依赖']) and predicate in {'causes','prerequisite_of','enables'} and direction!='incoming':continue
                        if any(t in question for t in ['导致','造成','结果']) and predicate in {'causes','enables'} and direction!='outgoing':continue
                        overlap=len(set(terms(question))&set(terms(predicate+' '+self.entities[target].get('canonical_name',''))))
                        options.append((overlap,float(dense[self.position[target]]),target,edge,direction))
                    for _,_,target,edge,direction in sorted(options,key=lambda x:(-x[0],-x[1],x[2]))[:POLICY['neighbors_per_seed']]:
                        if len(expanded)>=POLICY['expanded_concepts']:break
                        if target not in expanded:expanded.append(target)
                        paths.append(dict(seed=seed,target=target,relation=edge.get('relation'),relation_id=edge.get('relation_id'),direction=direction,conditions=edge.get('conditions','')))
            anchors=set()
            if method=='no_mapping':
                names=' '.join(self.entities[k].get('canonical_name','') for k in expanded)
                scores=self.lexical.lexical(names)
                anchors={self.rows[i]['evidence_ids'][0] for i in sorted((i for i,s in enumerate(scores) if s>0),key=lambda i:(-scores[i],self.rows[i]['evidence_ids'][0]))[:32]}
            else:
                for rank,key in enumerate(expanded):
                    for eid in self.mappings['concepts'].get(key,{}).get(target_book_id,[]):
                        anchors.add(eid)
                        section=self.structure['evidence'][eid]['section_id'];sections[section]=sections.get(section,0.)+1/(rank+1)
                for path in paths:
                    for eid in self.mappings['relations'].get(path['relation_id'],{}).get(target_book_id,[]):anchors.add(eid)
            if method=='no_structure':
                candidates={self.evidence_index[eid] for eid in anchors if eid in self.evidence_index}
            else:
                if method=='no_mapping':
                    for eid in anchors:
                        section=self.structure['evidence'][eid]['section_id'];sections[section]=sections.get(section,0.)+1
                chosen=sorted(sections,key=lambda s:(-sections[s]/math.sqrt(max(1,len(self.section_indices[s]))),s))[:POLICY['section_top_k']]
                sections={s:sections[s] for s in chosen};candidates=set()
                for section in chosen:candidates.update(self.section_indices[section])
                # 有出处的关系端点作为额外候选，不把路径文本充当答案证据。
                candidates.update(self.evidence_index[eid] for eid in anchors if eid in self.evidence_index)
            if not candidates:fallback=True;candidates=set(range(len(self.rows)))
        self.candidate_ids=tuple(sorted(self.rows[i]['evidence_ids'][0] for i in candidates))
        term_order=sorted((i for i in candidates if lexical[i]>0),key=lambda i:(-lexical[i],self.rows[i]['evidence_ids'][0]))
        if method=='keyword':order=term_order or sorted(candidates)
        else:
            positions=sorted(candidates);scores=self.vectors[positions]@query
            dense_order=[i for _,i in sorted(zip(scores,positions),key=lambda x:(-float(x[0]),self.rows[x[1]]['evidence_ids'][0]))]
            if method=='vector':order=dense_order
            else:
                combined=reciprocal([term_order,dense_order]);order=sorted(combined,key=lambda i:(-combined[i],self.rows[i]['evidence_ids'][0]))
        selected=[self.rows[i] for i in order[:POLICY['retrieval_top_k']]]
        if any(r['book_id']!=target_book_id for r in selected):raise AssertionError('Target book boundary broken')
        return selected,dict(policy=POLICY['version'],method=method,target_book_id=target_book_id,concept_graph_id=self.graph_id,
            seed_ids=seeds,expanded_concept_ids=expanded,paths=paths,coarse_sections=sorted(sections),candidate_count=len(candidates),
            total_target_units=len(self.rows),candidate_digest=digest(self.candidate_ids),fallback=fallback,
            loaded_structure_books=[target_book_id],structure_load_seconds=load,retrieval_seconds=time.monotonic()-started)
