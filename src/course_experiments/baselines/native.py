from __future__ import annotations
import asyncio
import copy
import importlib
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ..manifest import digest, read_document
from ..contributions import write_immutable
from .k12_native_projection import flat_native_edges


def native_import(name: str, source: Path, import_root: Path):
    top=name.split('.')[0]
    if top in sys.modules:
        location=Path(sys.modules[top].__file__).resolve()
        if not location.is_relative_to(source.resolve()): raise ValueError('Foreign native package already imported: '+top)
    sys.path.insert(0,str(import_root.resolve()))
    module=importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(source.resolve()): raise ValueError('Native import outside frozen source')
    return module


class NativeChat:
    def __init__(self,runtime,params,root,response_format=None):
        self.runtime,self.params,self.root=runtime,params,root
        self.response_format=copy.deepcopy(response_format)
        self.records=[]

    def complete(self,messages):
        body=copy.deepcopy(self.runtime.config['llm'].get('extra_body',{}))
        body.update(model=self.params['model'],messages=messages,temperature=self.params['temperature'],max_tokens=self.params['max_tokens'])
        if self.response_format is not None: body['response_format']=self.response_format
        # 片段失败后的新尝试允许重新请求；同一尝试内仍复用完整响应。
        key=digest(dict(body=body,attempt=str(self.root.resolve())))
        with self.runtime.gate:
            raw=self.runtime._raw_cached(key)
            if raw is None: raw=self.runtime._post('/chat/completions',body,self.runtime.config['llm'],key,'native_baseline')
        choice=raw['choices'][0]
        content=choice['message']['content']
        # 只保存允许字段，不记录请求头、SDK参数或凭据。
        record=dict(messages=messages,content=content,usage=raw.get('usage'),finish_reason=choice.get('finish_reason'))
        credential=os.environ.get('SILICONFLOW_API_KEY','')
        if credential and credential in json.dumps(record,ensure_ascii=False): raise RuntimeError('Native response contained credential; not persisted')
        write_immutable(self.root/(key+'.json'),record)
        if choice.get('finish_reason')=='length': raise ValueError('Native response reached output limit')
        if not isinstance(content,str) or not content.strip(): raise ValueError('Empty native model response')
        return content

    def generate(self,prompt,**kwargs):
        if kwargs.get('temperature',0)!=self.params['temperature'] or kwargs.get('max_tokens',self.params['max_tokens'])!=self.params['max_tokens']:
            raise ValueError('Native caller changed fixed generation budget')
        return self.complete([dict(role='user',content=prompt)])

    async def achat(self,prompt,history=None,**kwargs):
        messages=list(history or [])+[dict(role='user',content=prompt)]
        content=await asyncio.to_thread(self.complete,messages)
        if not kwargs.get('name','').startswith('extract-loopcheck'): self.records.append(content)
        return SimpleNamespace(output=SimpleNamespace(content=content),history=messages+[dict(role='assistant',content=content)])


class BgeEncoder:
    def __init__(self,runtime): self.runtime=runtime
    def encode(self,texts,show_progress_bar=False,**kwargs):
        import numpy as np
        if not isinstance(texts,list) or not all(isinstance(t,str) for t in texts): raise ValueError('BGE native adapter requires ordered list of strings')
        result=np.asarray(self.runtime.embed(texts),dtype=np.float32)
        if result.ndim!=2 or result.shape[0]!=len(texts) or not np.isfinite(result).all(): raise ValueError('Invalid BGE native adapter matrix')
        return result


def kggen_json(graph):
    return dict(entities=sorted(graph.entities),edges=sorted(graph.edges),relations=sorted([list(r) for r in graph.relations]),
                entity_clusters={k:sorted(v) for k,v in (graph.entity_clusters or {}).items()},
                edge_clusters={k:sorted(v) for k,v in (graph.edge_clusters or {}).items()})


def kggen_closure(graph):
    entities=set(graph.entities); labels=set(graph.edges)
    missing=[list(r) for r in sorted(graph.relations) if r[0] not in entities or r[2] not in entities]
    endpoints={end for h,_,t in graph.relations for end in (h,t)}
    added=sorted(endpoints-entities); missing_labels=sorted({r for _,r,_ in graph.relations}-labels)
    closed=copy.deepcopy(graph)
    closed.entities=entities|endpoints; closed.edges=labels|set(missing_labels)
    audit=dict(native_entity_table_count=len(entities),closed_entity_count=len(closed.entities),
               added_endpoint_candidate_count=len(added),added_endpoint_candidates=added,
               relations_with_missing_endpoint=len(missing),missing_endpoint_relations=missing,
               missing_edge_labels=missing_labels,status='explicit_native_tuple_closure_unverified')
    return closed,audit


def normalized_kggen(graph):
    value=kggen_json(graph)
    closed,audit=kggen_closure(graph)
    return dict(nodes=[dict(id=n,name=n,type='Concept',type_status='native_untyped_candidate',aliases=value['entity_clusters'].get(n,[])) for n in sorted(closed.entities)],
                edges=[dict(source=h,target=t,predicate=r,native_predicate_aliases=value['edge_clusters'].get(r,[])) for h,r,t in value['relations']],
                directed=True,raw=value,endpoint_closure_audit=audit)


def nx_json(graph,source_map):
    def attrs(attributes):
        value=dict(attributes)
        value['source_units']=[source_map[x.strip()] for x in str(value.get('source_id','')).split(',') if x.strip() in source_map]
        return value
    return dict(nodes=[dict(attrs(a),id=n,name=n) for n,a in graph.nodes(data=True)],
                edges=[dict(attrs(a),source=h,target=t) for h,t,a in graph.edges(data=True)],directed=False)


class K12Driver:
    differences=['Prepared corpus chunks replace native Markdown segmentation', 'Frozen DeepSeek generation budget',
                 'Course-local native merge; no existing subject/global graphs']
    def __init__(self,source,runtime,params):
        self.source,self.runtime,self.params=source,runtime,params
        self.extractor=native_import('kg.extract_kg_from_textbook',source,source/'src')
        self.merger=native_import('kg.merge_kg',source,source/'src')
        self.config_class=native_import('utils.config',source,source/'src').PipelineConfig
        self.parser=native_import('utils.llm_client',source,source/'src').OpenAIClient.parse_response
        self.prompt=(source/'src/kg/prompt.txt').read_text('utf-8')

    def extract(self,unit,root):
        config=self.config_class(dict(paths=dict(workspace_dir=str(root/'workspace'),output_dir=str(root/'data')),
                                      llm=dict(temperature=self.params['temperature'])),root)
        prefix='course_'+digest([unit['book_id'],unit['chunk_id']])[:16]
        directory=config.sections_dir_for(prefix); directory.mkdir(parents=True,exist_ok=True)
        file='unit.md'; (directory/file).write_text(unit['text'],'utf-8')
        config.build_graph_book_dir(prefix).mkdir(parents=True,exist_ok=True)
        (config.chapter_kg_dir/prefix).mkdir(parents=True,exist_ok=True)
        client=NativeChat(self.runtime,self.params,root/'responses')
        client.parse_response=lambda response:self.parser(client,response)
        result=self.extractor.process_section(config,client,prefix,dict(file=file,chapter_num='1',chapter_title='',section_num='1',section_title=''),
                                              self.prompt,self.params['max_tokens'],overwrite=False)
        if result['status'] not in ('ok','skipped_existing'):
            if '402' in result.get('message',''): raise RuntimeError('Model service returned HTTP 402')
            raise ValueError('Native K12 section failure: '+result['status'])
        raw=read_document(result['output_path'])
        return dict(nodes=[dict(n['properties'] if isinstance(n.get('properties'),dict) else {},id=n['id'],name=n['name'],type=n['label']) for n in raw['nodes']],
                    edges=flat_native_edges(raw),directed=True,raw=raw)

    def aggregate(self,rows,root):
        paths=[]
        for row in rows:
            path=root/'inputs'/(digest(row['provenance'])+'.json'); write_immutable(path,row['native']['raw']); paths.append(path)
        config=self.config_class(dict(paths=dict(workspace_dir=str(root/'workspace'),output_dir=str(root/'data'))),root)
        raw=self.merger.merge_graphs(paths,config,False,root/'native_merged.json')
        if raw is None: raw=dict(nodes=[],edges=[])
        return dict(nodes=[dict(n.get('properties',{}),id=n['id'],name=n['name'],type=n['label']) for n in raw['nodes']],
                    edges=flat_native_edges(raw),directed=True,raw=raw)


class KGGenDriver:
    differences=['Frozen chat model and DSPy 2.6.27', 'LM_BASED with BGE-M3 API encoder',
                 'Bounded native dedup thread pool', 'No local retrieval weights', 'Native untyped entity names are candidates, not verified concept types',
                 'Audited explicit tuple endpoint/label closure before native dedup and after extraction/dedup; original raw graph retained',
                 'Seed42; native unordered sets may affect determinism']
    def __init__(self,source,runtime,params):
        self.source,self.runtime,self.params=source,runtime,params
        module=native_import('kg_gen.kg_gen',source,source/'src')
        self.dedup=native_import('kg_gen.steps._3_deduplicate',source,source/'src')
        self.llmdedup=native_import('kg_gen.utils.llm_deduplicate',source,source/'src')
        self.kg=module.KGGen(model='openai/'+params['model'],api_key=os.environ['SILICONFLOW_API_KEY'],api_base=params['base_url'],
                             max_tokens=params['max_tokens'],temperature=params['temperature'],retrieval_model=None,disable_cache=True)
        extra_body=copy.deepcopy(runtime.config['llm'].get('extra_body',{}))
        for key in ('model','messages','temperature','max_tokens'): extra_body.pop(key,None)
        self.kg.lm.kwargs.update(timeout=240,extra_body=extra_body)
        self.kg.lm.num_retries=params['retries']

    def history(self,root,reject_length=True):
        key=os.environ.get('SILICONFLOW_API_KEY','')
        truncated=False
        def field(value,name,default=None):
            return value.get(name,default) if isinstance(value,dict) else getattr(value,name,default)
        def plain_usage(value):
            if isinstance(value,dict): return {k:plain_usage(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)): return [plain_usage(v) for v in value]
            if callable(getattr(value,'model_dump',None)): return plain_usage(value.model_dump(mode='json'))
            return value
        for row in self.kg.lm.history:
            record={k:row[k] for k in ('prompt','messages','outputs','usage','cost') if k in row}
            if 'usage' in record: record['usage']=plain_usage(record['usage'])
            choices=field(row.get('response'),'choices',[])
            safe_choices=[]
            for choice in choices:
                content=field(field(choice,'message'),'content',field(choice,'text'))
                reason=field(choice,'finish_reason')
                if content is not None and not isinstance(content,str): raise ValueError('Native response content must be text')
                if reason is not None and not isinstance(reason,str): raise ValueError('Native finish reason must be text')
                safe_choices.append(dict(content=content,finish_reason=reason))
            record.update(response_choices=safe_choices,finish_reasons=[c['finish_reason'] for c in safe_choices])
            truncated=truncated or 'length' in record['finish_reasons']
            encoded=__import__('json').dumps(record,ensure_ascii=False,allow_nan=False)
            if key and key in encoded: raise RuntimeError('Native history contained credential; not persisted')
            write_immutable(root/(digest(record)+'.json'),record)
        self.kg.lm.history.clear()
        if truncated and reject_length: raise ValueError('Native KGGen response reached output limit (finish_reason=length)')

    def extract(self,unit,root):
        try: graph=self.kg.generate(unit['text'],cluster=False,chunk_size=None)
        finally: self.history(root/'responses',reject_length=sys.exc_info()[0] is None)
        return normalized_kggen(graph)

    def aggregate(self,rows,root):
        import numpy as np
        np.random.seed(self.params['seed'])
        graphs=[self.kg.from_dict(row['native']['raw']) for row in rows]
        graph=self.kg.aggregate(graphs)
        write_immutable(root/'native_before_dedup.json',kggen_json(graph))
        graph,closure=kggen_closure(graph)
        write_immutable(root/'native_endpoint_closure.json',closure)
        if not graph.entities or not graph.edges:
            result=normalized_kggen(graph); result['dedup_status']='empty_or_no_relations'; return result
        futures=[]; pools=[]
        class TrackedPool(ThreadPoolExecutor):
            def submit(self,*args,**kwargs):
                future=super().submit(*args,**kwargs); futures.append(future); return future
        def bounded_pool(*args,**kwargs):
            kwargs['max_workers']=self.params['concurrency']; pool=TrackedPool(*args,**kwargs); pools.append(pool); return pool
        class SafeNativeError(logging.Filter):
            def filter(self,record):
                if record.levelno>=logging.ERROR:
                    record.msg='Native KGGen cluster failure; exception text redacted'; record.args=(); record.exc_info=None
                return True
        guard=SafeNativeError(); logger=self.llmdedup.LLMDeduplicate.logger; logger.addFilter(guard)
        try:
            with patch.object(self.llmdedup,'ThreadPoolExecutor',bounded_pool):
                graph=self.dedup.run_deduplication(self.kg.lm,graph,method=self.dedup.DeduplicateMethod.LM_BASED,retrieval_model=BgeEncoder(self.runtime))
            errors=[future.exception() for future in futures if future.exception() is not None]
            if errors:
                from .runner import error_code
                if any(error_code(exc)==402 for exc in errors): raise RuntimeError('Model service returned HTTP 402')
                raise ValueError('Native KGGen swallowed cluster failures; incomplete result refused')
        finally:
            for pool in pools: pool.shutdown(wait=True)
            logger.removeFilter(guard)
            self.history(root/'responses',reject_length=sys.exc_info()[0] is None)
        result=normalized_kggen(graph); result['dedup_status']='lm_based_complete'; return result


class GraphRAGDriver:
    differences=['Prepared corpus chunks as native input documents', 'Frozen DeepSeek model and generation budget',
                 'Schema entity type vocabulary; native prompts, parser and gleaning retained',
                 'Native undirected descriptive edges preserved; typed KGE unavailable']
    def __init__(self,source,runtime,params,schema):
        self.source,self.runtime,self.params,self.schema=source,runtime,params,schema
        self.extractor_class=native_import('graphrag.index.operations.extract_graph.graph_extractor',source,source).GraphExtractor

    def extractor(self,chat):
        def on_error(error,stack,data): raise error
        return self.extractor_class(model_invoker=chat,max_gleanings=self.params['max_gleanings'],on_error=on_error)

    def extract(self,unit,root):
        chat=NativeChat(self.runtime,self.params,root/'responses')
        result=asyncio.run(self.extractor(chat)([unit['text']],dict(entity_types=list(self.schema['entity_types']))))
        if set(result.source_docs)!={0}: raise ValueError('Native GraphRAG document incomplete')
        graph=nx_json(result.output,{'0':unit['unit_id']}); graph['raw_records']=''.join(chat.records)
        return graph

    def aggregate(self,rows,root):
        chat=NativeChat(self.runtime,self.params,root/'responses')
        records={i:r['native']['raw_records'] for i,r in enumerate(rows)}
        graph=asyncio.run(self.extractor(chat)._process_results(records,'<|>','##'))
        return nx_json(graph,{str(i):r['provenance']['unit_id'] for i,r in enumerate(rows)})


def driver(method,source,runtime,params,schema):
    if method=='k12_kgraph': return K12Driver(source,runtime,params)
    if method=='kggen_lm_dedup': return KGGenDriver(source,runtime,params)
    if method=='graphrag_extract': return GraphRAGDriver(source,runtime,params,schema)
    raise ValueError('Baseline has no runnable source')
