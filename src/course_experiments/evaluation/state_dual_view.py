"""G0/纯T的独立语义状态；只导航至原S证据，不借用F事实或覆盖正在运行的评测器。"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from ..dual_view import build_mappings,build_structure
from ..manifest import digest,file_record,read_document
from .dual_view_rag import DualViewRetriever,POLICY as RETRIEVAL,STRUCTURAL,concept_text,asset
from .dvkg_development import CompletedMappingsRetriever
from . import run_frozen_dual_view as frozen

ANSWER=frozen.old.ANSWER
JUDGE=frozen.JUDGE
POLICY=dict(RETRIEVAL,version='independent_semantic_states_v1',methods=['G0+S+M','T+S+M'],
 state_graph='separately_frozen_original_G0_or_native_textbook_only_view;never_filter_F',
 mapping='own_state_existing_provenance_anchors_plus_development_frozen_exact_name_alias_occurrence_navigation',
 occurrence_is_fact=False,answer_prompt=ANSWER,judge_prompt=JUDGE,judge_version=frozen.POLICY['judge_version'],
 vector_reuse='exact_concept_text_and_full_BGE_embedding_configuration;identity_alone_insufficient',
 cohort_pairing='compAB_and_compABCD_remain_separate_matched_jobs')


def verify_state_graph(graph,fused,state,active_books):
    if state not in ['G0','T','F']:raise ValueError('Unknown semantic state')
    if (graph.get('view_id')!=fused.get('view_id') or graph.get('entities')!=fused.get('entities') or
        graph.get('relations')!=fused.get('relations')):raise ValueError('State graph is not the complete independently exported view')
    native=fused['native_contributions'];prior=native['g0']['graph']
    if state=='T' and any(prior.get(k) for k in ['nodes','claims','evidence','books']):
        raise ValueError('Pure T requires its independently constructed empty prior carrier')
    expected=[] if state=='G0' else sorted(active_books)
    if sorted(fused['active_books'])!=expected or set(native)!={'g0',*expected}:
        raise ValueError('State construction book set differs')
    if state=='G0' and (prior.get('evidence') or prior.get('books')):raise ValueError('G0 includes textbook evidence')
    for book in expected:
        if native[book]['book_id']!=book or fused['input_hashes'][book]!=native[book]['artifact_id']:
            raise ValueError('Independent native textbook contribution identity differs')
    return dict(state=state,view_id=fused['view_id'],course_id=fused.get('course_id'),active_books=expected,
        input_hashes=fused['input_hashes'],prior_nodes=len(prior.get('nodes',{})),prior_claims=len(prior.get('claims',{})))


def state_mappings(graph,structures):
    result=build_mappings(graph,structures)
    result['navigation_disclosure']='existing own-state extraction anchors; exact canonical/alias occurrences may augment navigation only; no new semantic facts'
    return result


class StateRetriever(CompletedMappingsRetriever):
    def __init__(self,state,graph,structures,mappings,unit_assets,concept_ids,concept_vectors,*,variant='occurrence_mapping_v2'):
        if state not in ['G0','T'] or variant not in ['unchanged_v1','occurrence_mapping_v2']:
            raise ValueError('Explicit supported state and prior development selection required')
        if mappings.get('concept_graph_id')!=digest(graph) or mappings.get('structure_ids')!={b:asset(s)['structure_id'] for b,s in structures.items()}:
            raise ValueError('Cross-state mapping or structure identity differs')
        self.semantic_state,self.variant=state,variant
        super().__init__(graph,structures,mappings,unit_assets,concept_ids,concept_vectors)

    def retrieve(self,question,target_book_id,query_vector,method=None):
        expected=self.semantic_state+'+S+M'
        if method not in [None,expected]:raise ValueError('State method differs from its own concept graph')
        base=CompletedMappingsRetriever if self.variant=='occurrence_mapping_v2' else DualViewRetriever
        selected,trace=base.retrieve(self,question,target_book_id,query_vector,'dual')
        return selected,dict(trace,method=expected,state=self.semantic_state,state_policy=POLICY['version'],
            mapping_graph_id=self.original_mappings['concept_graph_id'],mapping_structures=self.original_mappings['structure_ids'],
            mapping_id=self.original_mappings.get('mapping_id'),development_variant=self.variant,
            semantic_graph_used_as_answer_evidence=False)


def packed_retrieval(engine,question,book,vector):
    selected,trace=engine.retrieve(question,book,vector)
    context=frozen.old.pack_context(selected,question);frozen.old.validate_context(context,book)
    return context,dict(trace,selected_evidence_ids=[r['evidence_ids'][0] for r in selected])


def vector_key(text,embedding_configuration):
    return digest(dict(text=text,embedding_configuration=embedding_configuration,normalization='L2_float32'))


def plan_vectors(graph,embedding_configuration,catalog):
    entities={e.get('entity_id',e.get('id')):e for e in graph['entities'] if e.get('type') not in STRUCTURAL}
    if len(entities)!=sum(e.get('type') not in STRUCTURAL for e in graph['entities']):raise ValueError('Duplicate semantic state entity IDs')
    ids=sorted(entities);texts=[concept_text(entities[k]) for k in ids];entries=[]
    for key,text in zip(ids,texts):
        text_key=vector_key(text,embedding_configuration)
        entries.append(dict(concept_id=key,text_key=text_key,reuse=catalog.get(text_key)))
    missing={r['text_key']:text for r,text in zip(entries,texts) if r['reuse'] is None}
    return dict(concept_ids=ids,input_text_digest=digest(texts),embedding_configuration=embedding_configuration,
        entries=entries,reusable_count=sum(r['reuse'] is not None for r in entries),missing_count=sum(r['reuse'] is None for r in entries),
        unique_missing_count=len(missing),missing_texts=missing)


def existing_vector_catalog(assets):
    catalog={};sources=[]
    for cohort,row in assets['cohorts'].items():
        construction=asset(row['construction']);config=asset(construction['files']['assets']['runtime_config'])['embedding']
        if config['model']!='BAAI/bge-m3' or config['base_url']!='https://api.siliconflow.cn/v1':raise ValueError('Frozen embedding provider differs')
        index=asset(assets['concept_indices'][cohort]);graph=asset(row['graph'])
        entities={e['entity_id']:e for e in graph['entities'] if e.get('type') not in STRUCTURAL};ids=sorted(entities)
        texts=[concept_text(entities[k]) for k in ids]
        if (index['concept_ids']!=ids or index['input_text_digest']!=digest(texts) or index['graph']!=row['graph'] or
            index['embedding_model']!='BAAI/bge-m3' or file_record(index['vectors']['path'])!=index['vectors']):raise ValueError('Old concept vector input binding differs')
        vectors=np.load(index['vectors']['path'],mmap_mode='r')
        if vectors.shape!=(len(ids),1024) or not np.isfinite(vectors).all():raise ValueError('Old BGE matrix invalid')
        for i,text in enumerate(texts):
            key=vector_key(text,config)
            catalog.setdefault(key,dict(vectors=index['vectors'],row=i,concept_index=assets['concept_indices'][cohort],
                concept_id=ids[i],input_text_digest=index['input_text_digest'],embedding_configuration=config))
        sources.append(dict(cohort=cohort,concept_index=assets['concept_indices'][cohort],rows=len(ids),configuration_digest=digest(config)))
    return catalog,sources


def materialize_reused(plan):
    if plan['missing_count']:raise ValueError('New state vectors are missing; no silent substitutions or model calls')
    matrices={};rows=[]
    for entry in plan['entries']:
        proof=entry['reuse'];path=proof['vectors']['path']
        if path not in matrices:
            if file_record(path)!=proof['vectors']:raise ValueError('Reused vector matrix changed')
            matrices[path]=np.load(path,mmap_mode='r')
        rows.append(matrices[path][proof['row']])
    result=np.asarray(rows,dtype=np.float32)
    if result.shape!=(len(plan['concept_ids']),1024) or not np.isfinite(result).all():raise ValueError('Reused state BGE vectors invalid')
    return result


def validate_book_bindings(assets,cohort):
    row=assets['cohorts'][cohort];construction=asset(row['construction']);structures={};units={};metadata={}
    for book in row['active_books']:
        if construction['files']['sources'][book]!=assets['corpora'][book]:raise ValueError('State source corpus differs from B assets')
        corpus=asset(assets['corpora'][book]);structure=asset(assets['structures'][book]);unit=asset(assets['unit_assets'][book])
        if build_structure(corpus)!=structure:raise ValueError('S differs from exact original corpus hierarchy/evidence')
        if (unit['book_id']!=book or {r['evidence_ids'][0] for r in unit['rows']}!=set(structure['evidence']) or
            any(r['book_id']!=book or r['text']!=structure['evidence'][r['evidence_ids'][0]]['text'] for r in unit['rows'])):
            raise ValueError('Target units/evidence differ from exact original sources')
        if file_record(unit['vectors']['path'])!=unit['vectors']:raise ValueError('Frozen source vectors differ')
        structures[book]=structure;units[book]=assets['unit_assets'][book]
        metadata[book]=dict(structure=assets['structures'][book],units=assets['unit_assets'][book],corpus=assets['corpora'][book],
            structure_id=structure['structure_id'],source_units=len(unit['rows']),source_vectors=unit['vectors'])
    return structures,units,metadata


def state_entry(assets,matrix,cohort,state):
    matches=[j for j in matrix['jobs'] if j['name']==cohort+'_'+state]
    if len(matches)!=1:raise ValueError('One frozen independent state phase required')
    job=matches[0];phase=asset(job['phase']);record=phase['files']['graph'];graph=asset(record)
    if sorted(job['active_books'])!=sorted(assets['cohorts'][cohort]['active_books']):raise ValueError('State phase books differ from paired B cohort')
    return dict(cohort=cohort,state=state,method=state+'+S+M',graph_record=record,graph=graph,
        phase_record=job['phase'],phase=phase,active_books=job['active_books'],construction=phase['files']['construction'])


def evaluate_state_question(engine,question,query_vector,provider,folder,identity,stage):
    # 保留共同生成/盲评/指标；参考答案仅进入评分，不进入图导航或检索。
    context,trace=packed_retrieval(engine,question['question'],question['target_book_id'],query_vector)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    candidate,receipt=frozen.answer(provider,question,context,folder,stage+'/answer')
    judgment=frozen.judge(provider,question,context,candidate,folder,identity)
    scores=dict(frozen.old.score_answer(candidate['answer'],question['answer']),
        **frozen.old.citations_score(candidate['citations'],context,engine.rows),
        **frozen.old.reference_metrics(question,context,trace['selected_evidence_ids'],engine.candidate_ids,engine.structure))
    return dict(status='completed',question_id=question['question_id'],question_type=question['type'],target_book_id=question['target_book_id'],
        candidate=candidate,context=context,retrieval=trace,machine_judge=judgment,scores=scores,answer_receipt=receipt,
        state=engine.semantic_state,state_policy=POLICY['version'])


def prior_carrier(path):
    # 只流式解析原视图中的首个G0贡献；避免完整加载巨大的native快照。
    import json,re
    decoder=json.JSONDecoder();buffer='';found=False
    with Path(path).open(encoding='utf-8') as stream:
        while True:
            part=stream.read(65536)
            if not part:raise ValueError('Original view lacks complete native prior carrier')
            buffer+=part
            if not found:
                match=re.search(r'(?<!\\)"native_contributions"\s*:\s*\{\s*"g0"\s*:\s*',buffer)
                if match:buffer=buffer[match.end():];found=True
                elif len(buffer)>1000000:buffer=buffer[-1000:]
            if found:
                try:value,_=decoder.raw_decode(buffer)
                except json.JSONDecodeError:continue
                if value.get('artifact_id')!=digest({k:v for k,v in value.items() if k!='artifact_id'}):
                    raise ValueError('Original prior carrier artifact differs')
                return value


def inspect_state(entry,reference_source_hashes):
    graph=entry['graph'];folder=Path(entry['graph_record']['path']).parent
    summary=read_document(folder/'summary.json');sources=read_document(folder/'sources.json');prior=prior_carrier(folder/'fused.json')
    expected=[] if entry['state']=='G0' else sorted(entry['active_books'])
    if (graph['view_id']!=summary['view_id'] or folder.name!=summary['view_id'] or
        sorted(summary['active_books'])!=expected or sorted(sources['active_books'])!=expected or
        set(sources['input_hashes'])!={'g0',*expected} or sources['input_hashes']['g0']!=prior['artifact_id']):
        raise ValueError('Independent original state view provenance differs')
    snapshot=prior['graph']
    if entry['state']=='T' and any(snapshot.get(k) for k in ['nodes','claims','evidence','books']):
        raise ValueError('Pure T prior carrier is not empty')
    if entry['state']=='G0' and (snapshot['evidence'] or snapshot['books']):raise ValueError('G0 contains source facts')
    if any(sources['input_hashes'][book]!=reference_source_hashes[book] for book in expected):
        raise ValueError('Pure T source contribution differs from paired F construction')
    if entry['state']=='G0' and sources['input_hashes']['g0']!=reference_source_hashes['g0']:
        raise ValueError('Independent G0 differs from paired F prior')
    return dict(graph=entry['graph_record'],summary=file_record(folder/'summary.json'),sources=file_record(folder/'sources.json'),
        prior_artifact_id=prior['artifact_id'],prior_nodes=len(snapshot['nodes']),prior_claims=len(snapshot['claims']),
        active_books=expected,course_id=summary['course_id'],input_hashes=sources['input_hashes'],
        independent_original_view=True,filtered_F=False)


def inventory(assets,matrix):
    catalog,catalog_sources=existing_vector_catalog(assets);states={};missing={};bindings={}
    for cohort,row in assets['cohorts'].items():
        structures,units,bound=validate_book_bindings(assets,cohort);bindings[cohort]=bound
        reference=read_document(Path(row['graph']['path']).parent/'sources.json')['input_hashes']
        for state in ['G0','T']:
            entry=state_entry(assets,matrix,cohort,state);proof=inspect_state(entry,reference)
            config=asset(asset(entry['construction'])['files']['assets']['runtime_config'])['embedding']
            plan=plan_vectors(entry['graph'],config,catalog);maps=state_mappings(entry['graph'],structures)
            key=cohort+'_'+state
            states[key]=dict(cohort=cohort,state=state,method=state+'+S+M',active_books=row['active_books'],graph=entry['graph_record'],
                phase=entry['phase_record'],provenance=proof,construction=entry['construction'],vector_plan=plan,mappings=maps,
                concept_count=len(plan['concept_ids']),relation_count=len(entry['graph']['relations']),structure_units=bound,
                anchor_audit=maps['audit'],mapped_concepts=sum(bool(v) for v in maps['concepts'].values()))
            for k,text in plan['missing_texts'].items():missing[k]=dict(text=text,embedding_configuration=config)
    return dict(policy=POLICY,states=states,bindings=bindings,existing_catalog_sources=catalog_sources,
        unique_missing_embeddings=missing,new_vector_count=len(missing),embedding_http_base_upper=(len(missing)+15)//16,
        reuse_rule=POLICY['vector_reuse'],jobs=20,model_api_calls=0)
