"""ECHA的有界核验与精确token计数；原始响应仅落运行服务器。"""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path

import numpy as np

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import evidence_hierarchical_alignment as core
from .echa_rag import MODEL_PROMPT,qualify_verdict
from .graph_rag_v2 import normalize

POLICY=dict(version='echa_alignment_runtime_v1',batch_pairs=8,max_output_tokens=4096,
    format_repairs=1,source_neighbor_limit=1,candidate_k=8,scope='all_selected_graph_concepts_target_book_original_paragraphs',
    invalid='unresolved_zero_support_preserve_raw',embeddings='exact_pinned_existing_concept_and_original_paragraph_vectors',
    model_prompt=MODEL_PROMPT)


class FrozenTokenizer:
    def __init__(self,manifest,*,loader=None):
        self.manifest=manifest
        if (manifest.get('version')!='echa_frozen_tokenizer_v1' or manifest.get('model')!='deepseek-ai/DeepSeek-V4-Flash'
                or manifest.get('add_special_tokens') is not False):raise ValueError('Frozen answer tokenizer identity differs')
        for record in manifest['files'].values():
            if any(file_record(record['path']).get(k)!=v for k,v in record.items() if k in {'path','sha256'}):
                raise ValueError('Frozen tokenizer file changed')
        if loader is None:
            from tokenizers import Tokenizer
            loader=Tokenizer.from_file
        self.encoder=loader(manifest['files']['tokenizer.json']['path'])

    def count(self,text):return len(self.encoder.encode(text,add_special_tokens=False).ids)


def occurrences(entities,structure):
    trie={};end=object();result=defaultdict(set)
    for concept in entities:
        cid=concept.get('entity_id',concept.get('id'))
        for name in [concept.get('canonical_name',concept.get('name','')),*(concept.get('aliases') or [])]:
            if not isinstance(name,str) or len(name.strip())<(3 if name.strip().isascii() else 2):continue
            key=normalize(name);node=trie
            for char in key:node=node.setdefault(char,{})
            node.setdefault(end,set()).add(cid)
    def word(c):return c.isascii() and (c.isalnum() or c=='_')
    for eid,row in structure['evidence'].items():
        text=normalize(row['text'])
        for start in range(len(text)):
            node=trie;stop=start
            while stop<len(text) and text[stop] in node:
                node=node[text[stop]];stop+=1
                if end not in node:continue
                if text[start].isascii() and text[start].isalnum() and start and word(text[start-1]):continue
                if text[stop-1].isascii() and text[stop-1].isalnum() and stop<len(text) and word(text[stop]):continue
                for cid in node[end]:result[cid].add(eid)
    return result


def candidate_sets(graph,structure,origins,concept_ids,concept_vectors,rows,paragraph_vectors,*,only_concepts=None):
    entities={r.get('entity_id',r.get('id')):r for r in graph['entities']}
    if set(concept_ids)!=set(entities):raise ValueError('Concept vector identity differs')
    eids=[r['evidence_ids'][0] for r in rows]
    if len(set(eids))!=len(eids) or set(eids)!=set(structure['evidence']):raise ValueError('Original paragraph vectors incomplete')
    if any(r['book_id']!=structure['book_id'] or r['text']!=structure['evidence'][eid]['text'] for r,eid in zip(rows,eids)):
        raise ValueError('Original paragraph vector text/target book differs')
    matrix=np.asarray(paragraph_vectors);vectors=np.asarray(concept_vectors)
    if matrix.ndim!=2 or vectors.shape!=(len(concept_ids),matrix.shape[1]) or not np.isfinite(matrix).all() or not np.isfinite(vectors).all():
        raise ValueError('Finite dimension-compatible vectors required')
    if not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-4) or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
        raise ValueError('Frozen normalized vectors required')
    lexical=occurrences(graph['entities'],structure);book=structure['book_id'];positions={eid:i for i,eid in enumerate(eids)}
    selected=set(concept_ids) if only_concepts is None else set(only_concepts)
    if not selected<=set(concept_ids):raise ValueError('Unknown requested concept')
    for first in range(0,len(concept_ids),32):
        chosen=[i for i in range(first,min(first+32,len(concept_ids))) if concept_ids[i] in selected]
        scores=vectors[chosen]@matrix.T
        for i,values in zip(chosen,scores):
            cid=concept_ids[i];order=lambda eid:(-max(0.,min(1.,float(values[positions[eid]]))),eid)
            source=origins['concepts'].get(cid,{}).get(book,[])
            if set(source)-set(eids):raise ValueError('Unknown original extraction anchor')
            dense=sorted((eid for eid,value in zip(eids,values) if value>0),key=order)[:8]
            names=sorted(lexical.get(cid,[]),key=order)[:8];channels={}
            for channel,keys in [('origin',sorted(set(source))),('lexical',names),('semantic',dense)]:
                for eid in keys:channels.setdefault(eid,[]).append(channel)
            candidates=[dict(book_id=book,evidence_id=eid,semantic_score=core._score(float(values[positions[eid]])),
                channels=channels[eid],status='candidate_not_verified') for eid in sorted(channels)]
            yield dict(concept_id=cid,candidates=candidates)


def _pairs(graph,structure,plans):
    entities={r.get('entity_id',r.get('id')):r for r in graph['entities']}
    ordered=sorted(structure['evidence'],key=lambda eid:(structure['evidence'][eid]['source'].get('paragraph_order',0),eid))
    positions={eid:i for i,eid in enumerate(ordered)}
    for row in plans:
        cid=row['concept_id'];concept=entities[cid]
        description={k:concept.get(k) for k in ['canonical_name','name','aliases','scope','definition','definition_status','merged_definitions']}
        for candidate in row['candidates']:
            eid=candidate['evidence_id'];evidence=structure['evidence'][eid];i=positions[eid]
            adjacent=[ordered[j] for j in [i-1,i+1] if 0<=j<len(ordered) and structure['evidence'][ordered[j]]['section_id']==evidence['section_id']]
            payload=dict(concept=description,candidate=dict(evidence_id=eid,book_id=structure['book_id'],text=evidence['text']),
                neighbors=[dict(evidence_id=e,text=structure['evidence'][e]['text']) for e in adjacent])
            pair_id=digest([cid,payload])
            yield dict(pair_id=pair_id,concept_id=cid,candidate=candidate,payload=dict(pair_id=pair_id,**payload))


def verify_candidates(provider,graph,structure,plans,root,stage):
    root=Path(root);contract=dict(graph=digest(graph),structure=digest(structure),plans=digest(plans),policy=POLICY)
    write_immutable(root/'contract.json',contract);pairs=list(_pairs(graph,structure,plans));done={};pending=[]
    for pair in pairs:
        key=digest([pair['concept_id'],pair['candidate']]);pair['key']=key;p=root/'mappings'/(key+'.json')
        if p.exists():
            record=read_document(p)
            if record.get('mapping_id')!=digest({k:v for k,v in record.items() if k!='mapping_id'}):raise ValueError('Mapping checkpoint changed')
            core.lift_coverage(graph,structure,[record]);done[key]=record
        else:pending.append(pair)
    def save(pair,verdict):
        mapping=qualify_verdict(digest(graph),pair['concept_id'],structure,pair['candidate'],verdict)
        write_immutable(root/'mappings'/(pair['key']+'.json'),mapping);done[pair['key']]=mapping
    for start in range(0,len(pairs),8):
        remaining=pairs[start:start+8];batch_id=digest([p['key'] for p in remaining])
        if all(p['key'] in done for p in remaining):continue
        for attempt in range(2):
            if not remaining:break
            payload=dict(target_book_id=structure['book_id'],pairs=[p['payload'] for p in remaining])
            prompt=MODEL_PROMPT+'\n本批次逐对返回JSON {"verdicts":[{"pair_id":"输入ID",...上述判定字段}]}，每对必须且仅有一个判定。'
            if attempt:payload['format_repair']='前次字段、逐字引文或结构无效；仍只根据原始输入重新给出本批次有效JSON，不推断缺失证据。'
            raw_path=root/'raw'/batch_id/(str(attempt)+'.json')
            if raw_path.exists():
                stored=read_document(raw_path)
                if stored['payload']!=payload:raise ValueError('Saved mapping response input changed')
                raw=stored['response']
            else:
                raw=provider.chat_response([dict(role='system',content=prompt),dict(role='user',content=json.dumps(payload,ensure_ascii=False))],
                    max_tokens=4096,response_format={'type':'json_object'},stage=stage+'/role_verification')
                raw={k:v for k,v in raw.items() if k!='_native_cache_hit'}
                write_immutable(raw_path,dict(payload=payload,response=raw))
            failed=[];errors=[]
            try:
                choice=raw['choices'][0]
                if choice.get('finish_reason')!='stop':raise ValueError('Mapping batch output incomplete')
                verdicts=json.loads(choice['message']['content'])['verdicts'];by_id={r['pair_id']:r for r in verdicts}
                if len(by_id)!=len(verdicts) or set(by_id)!={p['pair_id'] for p in remaining}:raise ValueError('Mapping batch pair identities differ')
            except (ValueError,KeyError,TypeError) as exc:
                failed=remaining;by_id={};errors.append(dict(error_type=type(exc).__name__,reason=str(exc)))
            if by_id:
                for pair in remaining:
                    try:save(pair,by_id[pair['pair_id']])
                    except (ValueError,KeyError,TypeError) as exc:
                        failed.append(pair);errors.append(dict(pair_id=pair['pair_id'],error_type=type(exc).__name__,reason=str(exc)))
            write_immutable(root/'raw'/batch_id/(str(attempt)+'_validation.json'),dict(errors=errors,unresolved_pairs=[p['pair_id'] for p in failed]))
            remaining=failed
        for pair in remaining:
            save(pair,dict(decision='unresolved',same_sense=False,scope_consistent=False,evidence_supported=False,
                role=None,reason='Bounded format or source-span validation unresolved; original responses retained',citations=[]))
    return [done[p['key']] for p in pairs]
