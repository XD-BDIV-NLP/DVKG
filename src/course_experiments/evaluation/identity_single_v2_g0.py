"""全prior及独立reference完成后的G0限定零API导出；不放行source依赖图。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import identity_single_v2 as identity
from .identity_kge import project_identity_graph
from .run_identity_kge import inspect_inventory,load_graph,training_jobs,TRAINING_MATRIX
from .run_scope_identity_gate import identity_projection


def _manifest(folder):
    value=read_document(Path(folder)/'manifest.json')
    if value.get('preparation_id')!=digest({k:v for k,v in value.items() if k!='preparation_id'}):raise ValueError('Frozen original preparation changed')
    for record in value['files'].values():
        if file_record(record['path'])!=record:raise ValueError('Frozen original preparation file changed')
    return value


def reference_only_qualification(preparation,processed):
    preparation=Path(preparation);processed=Path(processed);manifest=_manifest(preparation)
    for record in [*manifest['references'].values(),*manifest['source_records'].values()]:
        if file_record(record['path'])!=record:raise ValueError('Frozen independent reference/source changed')
    packets=read_document(preparation/'full_packets_server_only.json');origins=read_document(preparation/'full_origins_server_only.json')
    if len(packets)!=manifest['full_count']:raise ValueError('Entire independent reference population is required')
    index={p['candidate_id']:p for p in packets}
    for row in origins:
        p=index[row['candidate_id']]
        if (row['course_id']!=p['course_id'] or row['left_reference_entity_id']!=p['left']['entity_id'] or row['right_reference_entity_id']!=p['right']['entity_id']):
            raise ValueError('Original independent reference pair endpoints changed')
    result=identity.validate_saved_population(packets,'reference',processed/'reference');proof=identity.reference_components(origins,result)
    value=dict(protocol=identity.POLICY['version'],status='qualified_complete_independent_reference_only',components=proof['components'],
        reference_component_proof=proof,all_candidates_processed=True,original_reference_digests={c:digest(read_document(r['path'])) for c,r in manifest['references'].items()},
        reference_records=manifest['references'],original_reference_preparation=file_record(preparation/'manifest.json'),
        result_sha256=file_record(processed/'reference/result.json')['sha256'],branch_counts={k:result[k] for k in ['candidate_count','accepted_count','negative_count','insufficient_count','unknown_count']},
        graph_inputs=False,old_batch_gates_passed=False,all_original_facts_preserved=True)
    value['qualification_id']=digest(value);return value


def _export(root,graph,reference,split,projection,qualified,reference_qualification,source_record,*,stage='strict_single_v2_G0_sanitized_export_only'):
    from .protocol import freeze_phase
    from .kge import export_kge_package
    if reference['identity_kge_protocol']['qualification_id']!=reference_qualification['qualification_id'] or qualified['reference_qualification_id']!=reference_qualification['qualification_id']:
        raise ValueError('Separate independent reference and graph mapping qualifications are required')
    root=Path(root);primary=root/'primary'
    values={'graph.json':graph,'reference.json':reference,'fact_split.json':split,'projection.json':projection,
        'identity_qualification.json':qualified,'reference_qualification.json':reference_qualification,
        'construction.json':dict(format='identity_only_projection_of_unchanged_native_graph',original_source=source_record,
            qualification_id=qualified['qualification_id'],reference_qualification_id=reference_qualification['qualification_id'],
            graph_id=digest(graph),new_edges_added=0,prior_identity_is_textbook_support=False,policy=identity.POLICY)}
    for name,value in values.items():write_immutable(primary/name,value)
    paths={name.removesuffix('.json'):primary/name for name in values};phase=primary/'primary_phase.json'
    freeze_phase(paths,dict(seed=42,stage=stage,policy=identity.POLICY),phase)
    return export_kge_package(projection,split,phase,root/'gpu_inputs')['paths']


def inventory_append_request(previous,new_graphs,reference_qualification_id):
    if previous.get('package_id')!=digest({k:v for k,v in previous.items() if k!='package_id'}):raise ValueError('Previous inventory package changed')
    if previous.get('reference_qualification_id')!=reference_qualification_id:raise ValueError('Inventory extension must retain the exact independent reference-only split identity')
    def key(row):return digest({k:row[k] for k in ['method','course_id','graph','book_set']})
    old={key(r) for r in previous['graphs']};new={key(r) for r in new_graphs}
    if not old<=new:raise ValueError('New inventory version must retain every previous frozen graph record')
    value=dict(status='pending_new_inventory_version_candidate_qualification',previous_package_id=previous['package_id'],
        reference_qualification_id=reference_qualification_id,previous_graph_count=len(previous['graphs']),new_inventory_graph_count=len(new_graphs),additional_graph_count=len(new-old),
        new_graphs_qualified=False,source_full_qualified=False,new_candidate_HTTP='exact_cached_raw_revalidated_only_new_bodies_charge_existing_ledger',
        new_split_created=False,model_calls=0,types_invented=False)
    return dict(value,request_id=digest(value))


def prepare_g0(mapping_preparation,reference_preparation,processed,output,*,parent_package=None):
    mapping_preparation=Path(mapping_preparation);processed=Path(processed);output=Path(output)
    manifest=_manifest(mapping_preparation);inventory,metas=inspect_inventory(mapping_preparation)
    refqual=reference_only_qualification(reference_preparation,processed)
    append=None
    if parent_package:
        append=inventory_append_request(read_document(parent_package),metas,refqual['qualification_id'])
        write_immutable(output/'inventory_append_request.json',append)
    packets=read_document(mapping_preparation/'prior_full_packets_server_only.json');origins=read_document(mapping_preparation/'prior_full_origins_server_only.json')
    source_origins=read_document(mapping_preparation/'source_full_origins_server_only.json')
    if len(packets)!=manifest['requests']['prior-full']['actual_pairs']:raise ValueError('Entire fixed prior population is required')
    prior=identity.validate_saved_population(packets,'prior',processed/'prior');identity.validate_population(origins,prior)
    index={p['candidate_id']:p for p in packets};records=read_document(mapping_preparation/'input_records.json')
    for row in origins:
        p=index[row['candidate_id']]
        if (row['course_id']!=p['course_id'] or row['graph_entity_id']!=p['right']['entity_id'] or row['reference_entity_id']!=p['left']['entity_id'] or records.get(row['graph_record']['path'])!=row['graph_record']):
            raise ValueError('Frozen full prior graph/reference endpoints changed')
    g0={r['graph']['path'] for r in metas if r['method']=='G0'}
    if not g0 or any(r['graph_record']['path'] in g0 for r in source_origins):raise ValueError('G0 source identity dependencies cannot be hidden by prior-only qualification')
    canonical={c:{e:sorted(ids)[0] for ids in groups for e in ids} for c,groups in refqual['components'].items()};normalized=[]
    for origin in origins:
        row=copy.deepcopy(origin);old=row['reference_entity_id'];row.update(original_reference_entity_id=old,reference_entity_id=canonical.get(row['course_id'],{}).get(old,old),identity_basis='prior_generated_definition');normalized.append(row)
    joint=identity_projection(normalized,prior['decisions'])
    value=dict(protocol=identity.POLICY['version'],policy=identity.POLICY,status='qualified_G0_only_complete_prior_and_independent_reference',usable_for_KGE=True,
        qualification_scope='only_frozen_G0_graph_records_without_source_identity_candidates',qualified_graph_records=[r['graph'] for r in metas if r['method']=='G0'],
        all_prior_candidates_processed=True,prior_counts={k:prior[k] for k in ['candidate_count','accepted_count','negative_count','insufficient_count','unknown_count']},
        prior_result=file_record(processed/'prior/result.json'),original_mapping_preparation=file_record(mapping_preparation/'manifest.json'),
        reference_qualification_id=refqual['qualification_id'],mapping=[r for r in joint['mapping'] if r['graph_record']['path'] in g0],
        rejections=joint['rejections'],full_matrix_qualified=False,source_full_qualified=False,old_batch_gates_passed=False,old_source32_calibration_passed=False)
    value['qualification_id']=digest(value);write_immutable(output/'G0_qualification.json',value);write_immutable(output/'reference_only_qualification.json',refqual)
    references={};splits={}
    for course,record in refqual['reference_records'].items():
        references[course],splits[course]=identity.freeze_reference(read_document(record['path']),refqual)
        write_immutable(output/'references'/course/'reference.json',references[course]);write_immutable(output/'references'/course/'fact_split.json',splits[course])
    rows=[]
    for source,meta in zip(inventory['graphs'],metas):
        row=copy.deepcopy(meta);sha=row['graph']['sha256'];course=row['course_id'];row['entry_id']=digest(dict(method=row['method'],course=course,graph_sha256=sha,book_set=row['book_set']))
        if row['qualification']['status'].startswith('not_applicable'):
            row.update(status=row['qualification']['status'],usable_for_KGE=False,no_schema_types_invented=True)
        elif row['method']!='G0':row.update(status='incomplete_source_identity_qualification',usable_for_KGE=False)
        elif row['qualification']['status']!='eligible_typed_graph':row.update(status='insufficient_native_typed_graph',usable_for_KGE=False)
        else:
            if source['reference']!=refqual['reference_records'][course]:raise ValueError('Graph/reference frozen records differ')
            graph,cap=load_graph(source);mapping=[r for r in value['mapping'] if r['graph_record']==row['graph'] and r['course_id']==course]
            projection=project_identity_graph(graph,references[course],splits[course],mapping,graph_sha=sha,capabilities=cap)
            projection['identity_disclosure']['policy']=identity.POLICY;projection['identity_disclosure'].update(mapping_qualification_id=value['qualification_id'],reference_qualification_id=refqual['qualification_id'])
            projection['projection_id']=digest({k:v for k,v in projection.items() if k!='projection_id'})
            paths=_export(output/'graphs'/row['entry_id'],graph,references[course],splits[course],projection,value,refqual,row['graph'])
            ready=len(projection['training'])>=10 and any(projection['covered_test_mask']) and any(projection['covered_validation_mask'])
            row.update(status='prepared_for_review_training_not_authorized' if ready else 'frozen_insufficient_training_coverage',usable_for_KGE=True,
                trainable_by_existing_adapter=ready,projection_id=projection['projection_id'],graph_id=projection['graph_id'],split_id=projection['split_id'],
                training_triples=len(projection['training']),covered_test_count=sum(projection['covered_test_mask']),fixed_test_count=len(projection['test']),
                covered_validation_count=sum(projection['covered_validation_mask']),fixed_validation_count=len(projection['validation']),
                accepted_native_entities=len(projection['audit']['identity_mapping_provenance']),paths=paths,prior_identity_is_textbook_support=False)
        rows.append(row);write_immutable(output/'entries'/(row['entry_id']+'.json'),row)
    jobs=training_jobs(rows)
    package=dict(protocol=identity.POLICY['version'],status='G0_only_inputs_prepared_for_review',policy=identity.POLICY,qualification_id=value['qualification_id'],
        reference_qualification_id=refqual['qualification_id'],graphs=rows,inventory_graph_count=len(rows),qualified_graph_count=sum(r['usable_for_KGE'] for r in rows),
        reference_courses={c:dict(reference_id=s['reference_id'],split_id=s['split_id'],original_fact_count=len(references[c]['facts']),
            folds={k:len(v) for k,v in s['folds'].items()},holdout_block_policy=s.get('identity_holdout_block_policy')) for c,s in splits.items()},
        existing_training_matrix=dict(policy=TRAINING_MATRIX,jobs=jobs,total_jobs=len(jobs)),full_matrix_qualified=False,old_batch_gates_passed=False,
        training_started=False,training_authorized=False,model_calls=0,raw_text_exported=False,common_intersection_not_primary=True,zero_uncovered_fixed_denominator=True)
    if append:
        previous=read_document(parent_package)
        if any(package['reference_courses'].get(course)!=row for course,row in previous['reference_courses'].items()):
            raise ValueError('New inventory must preserve every actual frozen reference split, folds and denominator')
        package['inventory_append_request']=append
    package['package_id']=digest(package);write_immutable(output/'package.json',package);return package


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['mapping-preparation','reference-preparation','processed','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--parent-package',type=Path)
    args=parser.parse_args(argv);value=prepare_g0(args.mapping_preparation,args.reference_preparation,args.processed,args.output,parent_package=args.parent_package)
    print(json.dumps({k:value[k] for k in ['protocol','status','package_id','inventory_graph_count','qualified_graph_count','reference_courses','training_started','model_calls','raw_text_exported']}))


if __name__=='__main__':main()
