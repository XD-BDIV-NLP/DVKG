"""全source完成后追加完整冻结库存身份投影；参考拆分只来自独立reference。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from . import identity_single_v2 as identity
from .identity_single_v2_g0 import _manifest,_export,reference_only_qualification,inventory_append_request
from .identity_source_full_execution import validate_actual_proofs
from .identity_source_full_execution import START,AUTHORIZATION_SHA,ALLOCATION_SHA,_validate_allocation_numbers
from .identity_kge import project_identity_graph
from .run_identity_kge import inspect_inventory,load_graph,training_jobs,TRAINING_MATRIX


def prepare_full(mapping_preparation,reference_preparation,processed,output,*,database,config,parent_package):
    mapping_preparation=Path(mapping_preparation);processed=Path(processed);output=Path(output)
    mapping_manifest=_manifest(mapping_preparation);inventory,metas=inspect_inventory(mapping_preparation)
    packets=read_document(mapping_preparation/'source_full_packets_server_only.json')
    authorized=read_document(processed/'source_execution_authorization.json');allocation=authorized.get('allocation',{})
    _validate_allocation_numbers(allocation)
    if (authorized.get('execution_identity')!=digest({k:v for k,v in authorized.items() if k!='execution_identity'}) or
            allocation.get('authorization_sha256')!=AUTHORIZATION_SHA or allocation.get('allocation_sha256')!=ALLOCATION_SHA or
            allocation.get('watermark_event_id')!=102848 or authorized.get('worker_count')!=4 or
            authorized.get('source_candidate_count')!=len(packets) or authorized.get('source_packets_digest')!=digest(packets)):
        raise ValueError('Complete original source execution authorization and population changed')
    proof=validate_actual_proofs(packets,processed/'source',config,database,read_document(processed/'source/contract.json'))
    full=identity.qualify_all(mapping_preparation,reference_preparation,processed)
    refqual=reference_only_qualification(reference_preparation,processed);previous=read_document(parent_package)
    append=inventory_append_request(previous,metas,refqual['qualification_id'])
    value=copy.deepcopy(full);value.pop('qualification_id');value.update(reference_qualification_id=refqual['qualification_id'],
        original_full_qualification_id=full['qualification_id'],source_actual_HTTP_proof=proof,
        qualification_scope='all_and_only_current_frozen_inventory_graph_records',qualified_graph_records=[r['graph'] for r in metas],future_graphs_qualified=False)
    value['qualification_id']=digest(value)
    references={};splits={};courses={}
    for course,record in refqual['reference_records'].items():
        references[course],splits[course]=identity.freeze_reference(read_document(record['path']),refqual);s=splits[course]
        courses[course]=dict(reference_id=s['reference_id'],split_id=s['split_id'],original_fact_count=len(references[course]['facts']),
            folds={k:len(v) for k,v in s['folds'].items()},holdout_block_policy=s.get('identity_holdout_block_policy'))
    if courses!=previous['reference_courses']:raise ValueError('Complete source mapping must retain exact original independent reference-only splits and all fact denominators')
    write_immutable(output/'full_inventory_qualification.json',value);write_immutable(output/'reference_only_qualification.json',refqual)
    for course in references:
        write_immutable(output/'references'/course/'reference.json',references[course]);write_immutable(output/'references'/course/'fact_split.json',splits[course])
    rows=[]
    for source,meta in zip(inventory['graphs'],metas):
        row=copy.deepcopy(meta);sha=row['graph']['sha256'];course=row['course_id'];row['entry_id']=digest(dict(method=row['method'],course=course,graph_sha256=sha,book_set=row['book_set']))
        if row['qualification']['status'].startswith('not_applicable'):row.update(status=row['qualification']['status'],usable_for_KGE=False,no_schema_types_invented=True)
        elif row['qualification']['status']!='eligible_typed_graph':row.update(status='insufficient_native_typed_graph',usable_for_KGE=False)
        else:
            if source['reference']!=refqual['reference_records'][course]:raise ValueError('Frozen graph/reference record changed')
            graph,cap=load_graph(source);mapping=[r for r in value['mapping'] if r['graph_record']==row['graph'] and r['course_id']==course]
            projection=project_identity_graph(graph,references[course],splits[course],mapping,graph_sha=sha,capabilities=cap)
            projection['identity_disclosure']['policy']=identity.POLICY;projection['identity_disclosure'].update(mapping_qualification_id=value['qualification_id'],reference_qualification_id=refqual['qualification_id'])
            projection['projection_id']=digest({k:v for k,v in projection.items() if k!='projection_id'})
            paths=_export(output/'graphs'/row['entry_id'],graph,references[course],splits[course],projection,value,refqual,row['graph'],stage='strict_single_v2_full_inventory_sanitized_export_only')
            ready=len(projection['training'])>=10 and any(projection['covered_test_mask']) and any(projection['covered_validation_mask'])
            row.update(status='prepared_qualified_full_inventory' if ready else 'frozen_insufficient_training_coverage',usable_for_KGE=True,
                trainable_by_existing_adapter=ready,projection_id=projection['projection_id'],graph_id=projection['graph_id'],split_id=projection['split_id'],
                training_triples=len(projection['training']),covered_test_count=sum(projection['covered_test_mask']),fixed_test_count=len(projection['test']),
                covered_validation_count=sum(projection['covered_validation_mask']),fixed_validation_count=len(projection['validation']),
                accepted_native_entities=len(projection['audit']['identity_mapping_provenance']),paths=paths,prior_identity_is_textbook_support=False)
        rows.append(row);write_immutable(output/'entries'/(row['entry_id']+'.json'),row)
    jobs=training_jobs(rows)
    package=dict(protocol=identity.POLICY['version'],status='complete_frozen_inventory_identity_inputs_prepared',policy=identity.POLICY,
        qualification_id=value['qualification_id'],reference_qualification_id=refqual['qualification_id'],reference_courses=courses,
        graphs=rows,inventory_graph_count=len(rows),qualified_graph_count=sum(r['usable_for_KGE'] for r in rows),
        existing_training_matrix=dict(policy=TRAINING_MATRIX,jobs=jobs,total_jobs=len(jobs)),inventory_append_request=append,
        current_inventory_source_full_qualified=True,future_graphs_qualified=False,user_full_matrix_complete=False,
        old_batch_gates_passed=False,old_source32_calibration_passed=False,training_started=False,model_calls=0,raw_text_exported=False,
        common_intersection_not_primary=True,zero_uncovered_fixed_denominator=True,source_actual_HTTP_proof=proof,
        original_population_counts={b:full['branch_counts'][b]['candidate_count'] for b in full['branch_counts']},
        execution_authorization=dict(original_started_unix=START,deadline_unix=START+345600,authorized_total_seconds=345600,
            authorized_cumulative_HTTP=200000,authorization_sha256=AUTHORIZATION_SHA,allocation_sha256=ALLOCATION_SHA),
        qualification=file_record(output/'full_inventory_qualification.json'),previous_package=file_record(parent_package))
    package['source_execution_authorization']=file_record(processed/'source_execution_authorization.json')
    if mapping_manifest.get('previous_preparation'):
        from .identity_append_execution import validate_adoption
        old_preparation=Path(mapping_manifest['previous_preparation']['path']).parent
        if file_record(old_preparation/'manifest.json')!=mapping_manifest['previous_preparation']:raise ValueError('Inherited original preparation changed')
        inherited={b:validate_adoption(read_document(old_preparation/(b+'_full_packets_server_only.json')),b,processed/b,database=database,config=config) for b in ['source','prior']}
        proof=dict(parent_package_id=previous['package_id'],parent_package=file_record(parent_package),
            original_reference_splits_unchanged=True,all_original_classifications_and_raw_unchanged=True,original_HTTP_not_reasked=True,
            previous_population_counts=previous['original_population_counts'],current_population_counts=package['original_population_counts'],
            adopted_candidates={b:r['inherited_candidates'] for b,r in inherited.items()},adoption_receipts={b:r['receipt'] for b,r in inherited.items()})
        proof['proof_id']=digest(proof);package.update(status='complete_extended_inventory_identity_inputs_prepared',inventory_append_execution=proof)
    package['package_id']=digest(package);write_immutable(output/'package.json',package);return package


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['mapping-preparation','reference-preparation','processed','output','database','runtime-config','parent-package']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args(argv);value=prepare_full(args.mapping_preparation,args.reference_preparation,args.processed,args.output,database=args.database,config=read_document(args.runtime_config),parent_package=args.parent_package)
    print(json.dumps({k:value[k] for k in ['status','package_id','inventory_graph_count','qualified_graph_count','reference_courses','current_inventory_source_full_qualified','future_graphs_qualified','user_full_matrix_complete','training_started','model_calls','raw_text_exported']}))


if __name__=='__main__':main()
