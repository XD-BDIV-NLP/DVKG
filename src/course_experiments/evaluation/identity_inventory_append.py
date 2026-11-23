"""完整新图的有限零API追加准备；旧库存、旧判定和独立参考拆分不变。"""
from __future__ import annotations

import argparse
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import time

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import baseline_rag_bridge as bridge,identity_single_v2 as identity
from .scope_mapping_preflight import qualification,SEMANTIC_TYPES
from .run_scope_identity_batch import prepare_population
from .identity_single_v2_g0 import _manifest,reference_only_qualification
from .run_identity_single_v2 import OfficialMessageCounter,scan_exact_cache,RUNTIME_CONFIG_ID
from .identity_source_full_execution import START,AUTHORIZATION_SHA,ALLOCATION_SHA
from .run_native_full import prepare_plan


def native_projection_matches(raw,graph):
    native=raw.get('native_graph',{});nodes={n['id']:n for n in native.get('nodes',[])};edges={digest(e):e for e in native.get('edges',[])}
    for entity in graph['entities']:
        original=nodes.get(entity['entity_id'])
        if (not original or not isinstance(original.get('type'),str) or original.get('type_status')=='native_untyped_candidate' or
                original['type'].lower()!=str(entity.get('entity_type',entity.get('type'))).lower()):return False
    for relation in graph['relations']:
        original=edges.get(relation.get('relation_id'))
        if not original:return False
        predicate=original.get('predicate')
        if raw.get('method_id')=='k12_kgraph':predicate={'is_a':'is_a','prerequisites_for':'prerequisite_of'}.get(predicate)
        if (relation.get('head_entity_id')!=original.get('source') or relation.get('tail_entity_id')!=original.get('target') or
                relation.get('relation')!=predicate or relation.get('conditions','')!=original.get('conditions','')):return False
    return True


def classify_complete(record):
    if file_record(record['path'])!=record:raise ValueError('Published native artifact changed')
    raw=read_document(record['path'])
    if raw.get('artifact_id')!=digest({k:v for k,v in raw.items() if k!='artifact_id'}):raise ValueError('Native complete artifact identity differs')
    evidence=dict(artifact=record,native_schema_present='native_graph' in raw,semantic_view_present='semantic_view' in raw)
    if raw.get('status') not in {'complete','completed'} or (raw.get('scope')!='full' and not (raw.get('method_id')=='tree_kg' and raw.get('book_ids'))):
        return dict(evidence,status='pending_incomplete_native_artifact')
    semantic=raw.get('semantic_view');native=raw.get('native_graph',{});nodes=native.get('nodes',[])
    if not semantic:return dict(evidence,status='pending_common_schema_semantic_qualification')
    cap=semantic.get('capabilities',{});graph={k:semantic.get(k,[]) for k in ['entities','relations']};q=qualification(graph,cap)
    evidence.update(common_schema_qualification=q,native_nodes=len(nodes),native_untyped_nodes=sum(not n.get('type') or n.get('type_status')=='native_untyped_candidate' for n in nodes),kge_capability=cap.get('kge'))
    if q['status']=='eligible_typed_graph' and native.get('directed') is True and native_projection_matches(raw,graph):
        return dict(evidence,status='eligible_complete_original_semantic_projection')
    if (nodes and 'kge' in cap and cap['kge'] in {None,False} and all(not n.get('type') or n.get('type_status')=='native_untyped_candidate' for n in nodes) and
            all(e.get('type_status')=='native_untyped_candidate' or e.get('entity_type',e.get('type')) not in SEMANTIC_TYPES for e in graph['entities'])):
        return dict(evidence,status='not_applicable_proven_native_untyped',original_schema_evidence=record,no_types_or_predicates_invented=True)
    if (native.get('directed') is False and native.get('edges') and 'kge' in cap and cap['kge'] in {None,False} and not graph['relations']):
        return dict(evidence,status='not_applicable_proven_native_undirected_relations',original_schema_evidence=record,
            native_edge_count=len(native['edges']),native_directed=False,no_types_or_predicates_invented=True)
    return dict(evidence,status='pending_common_schema_semantic_qualification')


def original_cohort_source(inventory,assets,cohort,course):
    asset=assets['cohorts'][cohort]
    if file_record(asset['graph']['path'])!=asset['graph']:raise ValueError('Original frozen fused graph changed')
    evidence=str(Path(asset['graph']['path']).parent/'evidence.json')
    rows=[r for r in inventory['graphs'] if r['course_id']==course and cohort in r.get('cohorts',[]) and r.get('evidence_path')==evidence]
    if (not rows or len({tuple(r['book_set']) for r in rows})!=1 or any(r['reference']!=asset['reference'] for r in rows)):
        raise ValueError('Append must inherit original cohort reference, pinned fused evidence and book aliases')
    if file_record(asset['reference']['path'])!=asset['reference']:raise ValueError('Original cohort course reference changed')
    return copy.deepcopy(rows[0])


def corrected_rows(inventory,record,expected_graph_shas):
    """只追加已冻结的四个corrected F；不调用ECHA或重新构图/对齐。"""
    if file_record(record['path'])!=record:raise ValueError('Frozen corrected inventory changed')
    value=read_document(record['path']);assets_record=value['assets']
    if value.get('version')!='echa_asset_inventory_v1' or set(expected_graph_shas)!={'compAB','compABCD','osAB','netAB'}:raise ValueError('The four original corrected F pins are required')
    if file_record(assets_record['path'])!=assets_record:raise ValueError('Original corrected source assets changed')
    assets=read_document(assets_record['path']);entries=[r for r in value['states'].values() if r.get('state')=='F_corrected']
    if len(entries)!=4 or {r['cohort'] for r in entries}!=set(expected_graph_shas):raise ValueError('Exactly one frozen corrected F per original cohort required')
    rows=[];audits=[];seen={(r['method'],r['course_id'],file_record(r['path'])['sha256']) for r in inventory['graphs']}
    for entry in entries:
        co=entry['cohort'];graph_record=entry['graph'];asset=assets['cohorts'][co]
        if graph_record['sha256']!=expected_graph_shas[co] or file_record(graph_record['path'])!=graph_record:raise ValueError('Current corrected F exact graph SHA changed')
        if entry['active_books']!=asset['active_books']:raise ValueError('Corrected F original source cohort changed')
        old=[r for r in inventory['graphs'] if co in r.get('cohorts',[]) and r['reference']==asset['reference']]
        courses={r['course_id'] for r in old}
        if len(courses)!=1:raise ValueError('Corrected F must inherit one original course')
        if file_record(asset['reference']['path'])!=asset['reference']:raise ValueError('Original corrected course reference changed')
        course=next(iter(courses));source=original_cohort_source(inventory,assets,co,course);q=qualification(read_document(graph_record['path']))
        if q['status']!='eligible_typed_graph':raise ValueError('Frozen corrected F lacks original typed semantic graph qualification')
        audits.append(dict(method='F_corrected',cohort=co,graph=graph_record,inventory=record,original_assets=assets_record,status='eligible_frozen_corrected_semantic_graph',common_schema_qualification=q,no_types_or_predicates_invented=True))
        marker=('F_corrected',course,graph_record['sha256'])
        if marker in seen:continue
        rows.append(dict(method='F_corrected',course_id=course,path=graph_record['path'],reference=source['reference'],evidence_path=source['evidence_path'],cohorts=[co],book_set=source['book_set']))
        seen.add(marker)
    return rows,audits


def append_preparation(previous_preparation,new_rows,output,*,evidence,token_assets_root):
    previous_preparation=Path(previous_preparation);output=Path(output);old=_manifest(previous_preparation)
    inventory=read_document(previous_preparation/'input_inventory.json');inputs=read_document(previous_preparation/'input_records.json')
    for record in inputs.values():
        if file_record(record['path'])!=record:raise ValueError('Original frozen graph/reference/source changed')
    source,prior,source_origins,prior_origins=prepare_population(dict(graphs=new_rows))
    populations={};origins={};additional={}
    for branch,new_packets,new_origins in [('source',source,source_origins),('prior',prior,prior_origins)]:
        old_packets=read_document(previous_preparation/(branch+'_full_packets_server_only.json'));old_origins=read_document(previous_preparation/(branch+'_full_origins_server_only.json'))
        merged={p['candidate_id']:p for p in old_packets};old_ids=set(merged)
        for packet in new_packets:
            cid=packet['candidate_id']
            if cid!=digest({k:v for k,v in packet.items() if k!='candidate_id'}) or (cid in merged and packet!=merged[cid]):raise ValueError('An inherited original candidate changed in place')
            merged[cid]=packet
        populations[branch]=[merged[key] for key in sorted(merged)];origins[branch]=list({digest(r):r for r in [*old_origins,*new_origins]}.values());additional[branch]=len(set(merged)-old_ids)
    inventory=copy.deepcopy(inventory);inventory['graphs'].extend(copy.deepcopy(new_rows))
    for row in new_rows:
        for path in [row['path'],row['reference']['path'],row.get('evidence_path')]:
            if path:inputs[path]=file_record(path)
    values={'input_inventory.json':inventory,'input_records.json':inputs}
    for branch in populations:
        values[branch+'_full_packets_server_only.json']=populations[branch];values[branch+'_full_origins_server_only.json']=origins[branch]
    for name,value in values.items():write_immutable(output/name,value)
    value=dict(status='prepared_new_inventory_candidate_qualification_pending',previous_preparation=file_record(previous_preparation/'manifest.json'),
        files={name:file_record(output/name) for name in values},requests={branch+'-full':dict(actual_pairs=len(rows)) for branch,rows in populations.items()},
        token_assets_root=str(token_assets_root),additional_source_candidates=additional['source'],additional_prior_candidates=additional['prior'],
        previous_graph_count=len(inventory['graphs'])-len(new_rows),new_graph_count=len(new_rows),inventory_graph_count=len(inventory['graphs']),
        unknown_reclassified_as_negative=False,old_decisions_policy='inherit_every_original_classification_including_UNKNOWN_without_new_HTTP',
        future_mapping_executor_required=True,new_candidates_qualified=False,qualification_created=False,new_split_created=False,model_calls=0,raw_text_exported=False,
        original_reference_split_fixed=True,old_failed_development_gates_passed=False,published_complete_evidence=evidence)
    value['preparation_id']=digest(value);write_immutable(output/'manifest.json',value);return value


def prepare_append(run_root,mapping_preparation,reference_preparation,processed,parent_package,inventories,output,tokenizer_manifest,*,corrected_inventory=None,corrected_graph_shas=None):
    run_root=Path(run_root);mapping_preparation=Path(mapping_preparation);processed=Path(processed);output=Path(output)
    previous=read_document(parent_package)
    if previous.get('package_id')!=digest({k:v for k,v in previous.items() if k!='package_id'}) or previous.get('current_inventory_source_full_qualified') is not True:
        raise ValueError('Entire current inventory qualification must precede future append preparation')
    original=identity.qualify_all(mapping_preparation,reference_preparation,processed)
    refqual=reference_only_qualification(reference_preparation,processed)
    if refqual['qualification_id']!=previous['reference_qualification_id']:raise ValueError('Original independent reference qualification changed')
    for course,record in refqual['reference_records'].items():
        reference,split=identity.freeze_reference(read_document(record['path']),refqual)
        if (previous['reference_courses'][course]['split_id']!=split['split_id'] or previous['reference_courses'][course]['original_fact_count']!=len(reference['facts'])):
            raise ValueError('Future append may not alter original reference-only splits or fact denominators')
    old=_manifest(mapping_preparation);inventory=read_document(mapping_preparation/'input_inventory.json');seen={(r['method'],r['course_id'],file_record(r['path'])['sha256']) for r in inventory['graphs']}
    rows=[];audits=[];published=[]
    for record in inventories:
        if file_record(record['path'])!=record:raise ValueError('Published bridge inventory changed')
        value=read_document(record['path'])
        if value.get('identity')!=digest({k:v for k,v in value.items() if k!='identity'}):raise ValueError('Published bridge inventory identity changed')
        assets=read_document(value['asset_manifest']['path'])
        if file_record(value['asset_manifest']['path'])!=value['asset_manifest']:raise ValueError('Published original S assets changed')
        published.append(record)
        for key,entry in value['ready'].items():
            method,cohort=entry['method'],entry['cohort'];q=classify_complete(entry['artifact']);audits.append(dict(method=method,cohort=cohort,**q))
            if q['status']!='eligible_complete_original_semantic_projection':continue
            bridge.load_complete_artifact(entry['artifact'],assets,cohort,method)
            raw=read_document(entry['artifact']['path']);course=raw['course_id'];marker=(method,course,entry['artifact']['sha256'])
            if marker in seen:continue
            if course not in refqual['reference_records']:raise ValueError('New graph course is outside the fixed independent reference')
            source=original_cohort_source(inventory,assets,cohort,course)
            row=dict(method=method,course_id=course,path=entry['artifact']['path'],artifact=True,reference=refqual['reference_records'][course],
                evidence_path=source['evidence_path'],cohorts=[cohort],book_set=source['book_set'])
            rows.append(row);seen.add(marker)
    if corrected_inventory:
        corrected,corrected_audits=corrected_rows(dict(graphs=inventory['graphs']+rows),corrected_inventory,corrected_graph_shas)
        rows.extend(corrected);audits.extend(corrected_audits)
    evidence=dict(published_inventories=published,frozen_corrected_inventory=corrected_inventory,original_package=file_record(parent_package),original_full_qualification_id=original['qualification_id'],
        reference_qualification_id=refqual['qualification_id'],schema_audits=audits,pending_inventories_not_NA=True)
    if not rows:
        return dict(status='waiting_new_complete_typed_semantic_graph',**evidence,new_graph_count=0,model_calls=0,qualification_created=False,raw_text_exported=False)
    preparation=output/'preparation';manifest=append_preparation(mapping_preparation,rows,preparation,evidence=evidence,token_assets_root=old['token_assets_root'])
    config=read_document(prepare_plan(run_root,'hipporag','book_49bb8d31ddb9')['construction']['files']['assets']['runtime_config']['path'])
    if digest(config)!=RUNTIME_CONFIG_ID:raise ValueError('Original runtime config changed')
    formal=run_root/'formal_provider'
    if file_record(formal/'budget_extension_authorization_20261023.json')['sha256']!=AUTHORIZATION_SHA or file_record(formal/'bounded_allocations_20261023.json')['sha256']!=ALLOCATION_SHA:
        raise ValueError('Original approved inherited allocation changed')
    with closing(sqlite3.connect('file:'+str(formal/'audit.sqlite')+'?mode=ro',uri=True)) as db:
        records={key:json.loads(value) for key,value in db.execute('SELECT key,record FROM receipts')};attempts=dict(db.execute("SELECT key,COUNT(*) FROM events WHERE status='request_started' GROUP BY key"))
        used=db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND id>102848 AND stage LIKE 'identity_single_v2.%'").fetchone()[0]
    counter=OfficialMessageCounter(tokenizer_manifest,Path(old['token_assets_root'])/'encoding/encoding_dsv4.py');new={}
    for branch in ['source','prior']:
        original_ids={p['candidate_id'] for p in read_document(mapping_preparation/(branch+'_full_packets_server_only.json'))}
        new[branch]=[p for p in read_document(preparation/(branch+'_full_packets_server_only.json')) if p['candidate_id'] not in original_ids]
    cache=scan_exact_cache(new,config,records,attempts=attempts,counter=counter)
    available=max(0,min(33000-used,200000-sum(attempts.values())));required=cache['execution_actual_HTTP_cap2_upper']
    deadline_reached=time.time()>=START+345600
    value=dict(status='prepared_deadline_reached' if deadline_reached else 'prepared_budget_shortfall' if required>available else 'new_candidates_prepared_within_existing_identity_allocation',
        preparation=file_record(preparation/'manifest.json'),**evidence,additional_source_candidates=manifest['additional_source_candidates'],additional_prior_candidates=manifest['additional_prior_candidates'],
        cache=cache,identity_incremental_actual_HTTP=used,identity_incremental_cap=33000,available_actual_HTTP=available,required_cap2_upper=required,
        actual_HTTP_shortfall=max(0,required-available),deadline_unix=START+345600,deadline_reached=deadline_reached,original_reference_split_fixed=True,model_calls=0,new_candidates_qualified=False,
        qualification_created=False,training_started=False,raw_text_exported=False)
    write_immutable(output/'append_plan.json',value);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['run-root','mapping-preparation','reference-preparation','processed','parent-package','inventory-root','output','tokenizer-manifest']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    from .run_repaired_full import atomic
    last_key=None;value=None
    while time.time()<START+345600:
        records=[file_record(p) for p in sorted(args.inventory_root.glob('*/inventory.json'))]
        if not args.parent_package.exists():value=dict(status='waiting_complete_current_inventory_package',model_calls=0,raw_text_exported=False)
        else:
            key=digest(dict(parent=file_record(args.parent_package),inventories=records))
            if key!=last_key:
                value=prepare_append(args.run_root,args.mapping_preparation,args.reference_preparation,args.processed,args.parent_package,records,args.output/key,args.tokenizer_manifest);last_key=key
        atomic(args.output/'state.json',dict(value,updated_unix=time.time(),deadline_unix=START+345600))
        if not args.watch or value['status'] not in {'waiting_complete_current_inventory_package','waiting_new_complete_typed_semantic_graph'}:
            print(json.dumps(value));return
        time.sleep(30)
    raise RuntimeError('Original-start 96-hour finite append-preparation deadline reached')


if __name__=='__main__':main()
