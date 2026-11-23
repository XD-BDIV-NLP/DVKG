"""冻结全参考事实的新身份KGE包；仅零API准备，不启动GPU或模型。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from .identity_kge import POLICY,STAGES,freeze_reference,project_identity_graph,qualify_mappings
from .scope_mapping_preflight import qualification


TRAINING_MATRIX = dict(version='identity_kge_reduced_matrix_v1',primary_models=['TransE','TransR'],
    primary_seeds=[42,43,44],auxiliary_models=['DistMult','ComplEx','RotatE'],auxiliary_seeds=[42],
    epochs=[1000],jobs_per_trainable_unique_graph_split=9,
    deduplication='actual_graph_sha256_split_model_seed_epoch',untrainable_graphs='retained_fixed_denominator_no_training_job')


def training_jobs(rows):
    jobs={}
    for row in rows:
        if row.get('trainable_by_existing_adapter') is not True:continue
        for role in ('primary','auxiliary'):
            for model in TRAINING_MATRIX[role+'_models']:
                for seed in TRAINING_MATRIX[role+'_seeds']:
                    identity=dict(graph_sha256=row['graph']['sha256'],split_id=row['split_id'],model=model,seed=seed,epochs=1000)
                    key=digest(identity)
                    if key in jobs and jobs[key]['projection_id']!=row['projection_id']:
                        raise ValueError('Shared graph/split has conflicting projection contents')
                    job=jobs.setdefault(key,dict(identity,role=role,training_job_id=key,name='identity_'+key[:20],
                        projection_id=row['projection_id'],entry_ids=[],cohorts=[]))
                    job['entry_ids']=sorted(set(job['entry_ids']+[row['entry_id']]))
                    job['cohorts']=sorted(set(job['cohorts']+row.get('cohorts',[])))
    return [jobs[key] for key in sorted(jobs)]


def inspect_dependencies(mapping_root):
    from .run_scope_identity_batch import read_verified_checkpoint
    stages={};missing=[];incomplete=[]
    for name in STAGES:
        path=Path(mapping_root)/name/'result.json'
        if not path.exists():missing.append(name);continue
        row=read_verified_checkpoint(path);stages[name]=row
        if row.get('status')!='completed':incomplete.append(name)
    return dict(ready=not missing and not incomplete,missing_stages=missing,incomplete_stages=incomplete,
        stages=stages,model_calls=0,training_started=False)


def load_graph(row):
    raw=read_document(row['path'])
    if row.get('artifact'):
        if raw.get('status')!='complete':raise ValueError('Incomplete native artifact cannot become a KGE graph')
        return {key:copy.deepcopy(raw['semantic_view'][key]) for key in ('entities','relations')},raw['semantic_view'].get('capabilities')
    return raw,None


def export_projection(root,graph,reference,split,projection,qualified,source_record):
    if qualified.get('usable_for_KGE') is not True or not qualified.get('qualification_id'):
        raise ValueError('Completed identity qualification required before KGE export')
    if reference.get('identity_kge_protocol',{}).get('qualification_id')!=qualified['qualification_id']:
        raise ValueError('Frozen new reference qualification required; legacy split cannot substitute')
    from .protocol import freeze_phase
    from .kge import export_kge_package
    root=Path(root);primary=root/'primary'
    values={'graph.json':graph,'reference.json':reference,'fact_split.json':split,'projection.json':projection,
        'identity_qualification.json':qualified,'construction.json':dict(format='identity_only_projection_of_unchanged_native_graph',
            original_source=source_record,qualification_id=qualified['qualification_id'],graph_id=digest(graph),
            new_edges_added=0,prior_identity_is_textbook_support=False,policy=POLICY)}
    for name,value in values.items():write_immutable(primary/name,value)
    paths={role:primary/name for role,name in [('graph','graph.json'),('reference','reference.json'),
        ('fact_split','fact_split.json'),('projection','projection.json'),('construction','construction.json'),
        ('identity_qualification','identity_qualification.json')]}
    phase=primary/'primary_phase.json'
    freeze_phase(paths,dict(seed=42,stage='qualified_identity_primary_sanitized_export',policy=POLICY),phase)
    return export_kge_package(projection,split,phase,root/'gpu_inputs')['paths']


def validate_completed_decision(packet,decision):
    from . import scope_identity_batch as core
    if decision.get('status')!='completed' or decision.get('candidate_id')!=packet['candidate_id']:
        raise ValueError('Incomplete or mismatched identity decision')
    canonical=core._anonymous(packet);first=decision.get('primary_record',{})
    validated=core._validate(packet,canonical,first.get('judgment',{}),False)
    if first!=validated:raise ValueError('Stored identity primary validation changed')
    reverse=decision.get('reverse')
    if first['eligible_identity_edge']:
        if not reverse:raise ValueError('Positive primary lacks a reverse judgment')
        # 保存的是已还原左右的判定，原方向重验引句即可。
        expected=core._validate(packet,canonical,copy.deepcopy(reverse['judgment']),False)
        expected['reverse_input']=True
        if reverse!=expected:raise ValueError('Stored reverse identity validation changed')
    if decision.get('eligible_identity_edge') and not (first['eligible_identity_edge'] and reverse and reverse['eligible_identity_edge']):
        raise ValueError('Positive identity lacks two validated same-sense judgments')


def validate_mapping_preparation(preparation):
    from .run_identity_recovery import validate_original_preparation
    return validate_original_preparation(Path(preparation))


def completed_qualification(preparation,mapping_root,single119_root,*,reference_qualification=None):
    from . import run_scope_identity_batch as batch
    from . import scope_identity_batch as core
    manifest=validate_mapping_preparation(preparation);dependency=inspect_dependencies(mapping_root)
    if not dependency['ready']:raise ValueError('Full source/prior identity mapping dependencies are not complete')
    stages=dependency['stages'];source_plan=read_document(Path(preparation)/'source32_plan.json')
    single119=batch.load_single119(single119_root,source_plan['single_population_candidate_ids'])
    computed=core.compare_development(source_plan,single119,stages['source32']['decisions'])
    if computed!=stages['source32'].get('comparison'):raise ValueError('Source32 frozen gate attestation differs')
    prior_plan=read_document(Path(preparation)/'prior24_plan.json')
    computed=batch.compare_prior24(prior_plan,stages['prior24-single']['decisions'],stages['prior24-batch']['decisions'])
    if computed!=stages['prior24-batch'].get('comparison'):raise ValueError('Prior24 frozen gate attestation differs')
    for name in STAGES:
        packets,origins=batch.stage_inputs(name,Path(preparation))
        if set(stages[name]['decisions'])!={p['candidate_id'] for p in packets}:raise ValueError('Stage fixed population changed')
        # 重新验证逐对双侧引句；不重读模型、不重判sameSense、不允许改判负结果。
        by_id={p['candidate_id']:p for p in packets}
        for cid,decision in stages[name]['decisions'].items():
            validate_completed_decision(by_id[cid],decision)
    source_origins=read_document(Path(preparation)/'source_full_origins_server_only.json')
    prior_origins=read_document(Path(preparation)/'prior_full_origins_server_only.json')
    for name,origins,old_origins,old in [('source-full',source_origins,
            read_document(Path(preparation)/'source119_origins_server_only.json'),single119),
            ('prior-full',prior_origins,read_document(Path(preparation)/'prior24_origins_server_only.json'),stages['prior24-single']['decisions'])]:
        if batch.apply_negative_veto(origins,stages[name]['decisions'],old_origins,old)!=stages[name]['decisions']:
            raise ValueError('A frozen negative identity decision was not preserved')
    qualified=qualify_mappings(stages,source_origins,prior_origins,manifest['preparation_id'],reference_qualification=reference_qualification)
    qualified['stage_receipts']={name:file_record(Path(mapping_root)/name/'result.json') for name in STAGES}
    qualified['qualification_id']=digest({k:v for k,v in qualified.items() if k!='qualification_id'})
    return qualified,manifest


def inspect_inventory(preparation):
    preparation=Path(preparation);manifest=read_document(preparation/'manifest.json')
    if manifest.get('preparation_id')!=digest({k:v for k,v in manifest.items() if k!='preparation_id'}):
        raise ValueError('Preparation identity changed')
    for name in ('input_inventory.json','input_records.json'):
        if file_record(preparation/name)['sha256']!=manifest['files'][name]['sha256']:raise ValueError('Frozen inventory changed')
    inputs=read_document(preparation/'input_records.json')
    for record in inputs.values():
        if file_record(record['path'])!=record:raise ValueError('Original graph/reference/source file changed')
    inventory=read_document(preparation/'input_inventory.json');rows=[]
    for row in inventory['graphs']:
        record=file_record(row['path'])
        if record['path'] not in inputs or inputs[record['path']]!=record:raise ValueError('Graph outside frozen inventory')
        graph,cap=load_graph(row);status=qualification(graph,cap)
        rows.append(dict(method=row['method'],course_id=row['course_id'],cohorts=row.get('cohorts',[]),
            book_set=row.get('book_set',[]),graph=record,reference=row['reference'],qualification=status))
    return inventory,rows


def finalize_gpu_entry(package_path,entry_id,gpu_inputs,assets,output):
    from .protocol import freeze_phase,validate_phase
    from .kge import prepare_protocol,validate_protocol
    package=read_document(package_path)
    if package.get('package_id')!=digest({k:v for k,v in package.items() if k!='package_id'}):
        raise ValueError('Frozen identity KGE package changed')
    entry=next((r for r in package['graphs'] if r['entry_id']==entry_id),None)
    if not entry or entry.get('usable_for_KGE') is not True or entry.get('trainable_by_existing_adapter') is not True:
        raise ValueError('Unqualified or untrainable KGE entry')
    from .kge import BUDGET
    jobs=[j for j in package['existing_training_matrix']['jobs'] if entry_id in j['entry_ids'] and
        j['model']==BUDGET['model'] and j['seed']==BUDGET['seed'] and j['epochs']==1000]
    if len(jobs)!=1:raise ValueError('Actual native model/seed is outside the frozen reduced training matrix')
    required={'construction','gpu_environment','dependency_lock','src_utils','evaluator_original',
        'sweep_original','evaluator_patched','sweep_patched'}
    if not required<=assets.keys():raise ValueError('Actual existing GPU/evaluator assets required')
    for record in assets.values():
        if file_record(record['path'])!=record:raise ValueError('GPU execution asset changed')
    gpu_inputs=Path(gpu_inputs);output=Path(output)
    paths={key:gpu_inputs/name for key,name in [('graph','graph_attestation.json'),('projection','projection.json'),
        ('fact_split','fact_split.json'),('primary_attestation','primary_attestation.json')]}
    paths.update({key:record['path'] for key,record in assets.items()});paths['identity_package']=package_path
    projection=read_document(paths['projection']);split=read_document(paths['fact_split'])
    if any(projection[key]!=entry[key] for key in ('projection_id','graph_id','split_id')):
        raise ValueError('Relocated GPU projection differs from frozen full-population entry')
    phase_path=output/'gpu_phase.json'
    freeze_phase(paths,dict(seed=42,training_seed=BUDGET['seed'],model=BUDGET['model'],stage='qualified_identity_KGE',
        package_id=package['package_id'],entry_id=entry_id,full_fixed_denominators=True,common_intersection_primary=False),phase_path)
    phase=validate_phase(phase_path);phase['path']=str(phase_path)
    protocol=prepare_protocol(projection,split,phase,output/'protocol_main.json',common_mask=None)
    validate_protocol(protocol)
    value=dict(status='ready_for_existing_run_external',package_id=package['package_id'],entry_id=entry_id,
        protocol=file_record(output/'protocol_main.json'),graph=file_record(paths['graph']),
        training_algorithm='existing_evaluate_kg_quality_external_protocol_and_PyKEEN',budget=BUDGET,
        execution_epochs=[1000],training_job_id=jobs[0]['training_job_id'],
        covered_test_count=sum(projection['covered_test_mask']),fixed_test_count=len(projection['test']),
        provider_calls=0,training_started=False,multi_model_adapter='derive_native unchanged existing model/seed-only adapter')
    write_immutable(output/'prepared_gpu_entry.json',value);return value


def freeze_population(preparation,mapping_root,single119_root,output,*,reference_qualification=None):
    from .reference import validate_reference
    inventory,inventory_rows=inspect_inventory(preparation)
    qualified,manifest=completed_qualification(preparation,mapping_root,single119_root,reference_qualification=reference_qualification)
    output=Path(output);write_immutable(output/'identity_qualification.json',qualified)
    references={};splits={};reference_records={}
    for row in inventory['graphs']:
        course=row['course_id'];record=row['reference']
        if course in reference_records and reference_records[course]!=record:raise ValueError('Different independent references for one course')
        reference_records[course]=record
        if course not in references:
            original=validate_reference(read_document(record['path']))
            references[course],splits[course]=freeze_reference(original,qualified['qualification_id'],reference_qualification=reference_qualification)
            for name,value in [('reference.json',references[course]),('fact_split.json',splits[course])]:
                write_immutable(output/'references'/course/name,value)
    rows=[];known_graphs={r['graph']['sha256'] for r in inventory_rows}
    if any(r['graph_record']['sha256'] not in known_graphs for r in qualified['mapping']):raise ValueError('Qualified mapping refers to an uninventoried graph')
    for source,meta in zip(inventory['graphs'],inventory_rows):
        row=copy.deepcopy(meta);sha=row['graph']['sha256'];course=row['course_id']
        row['entry_id']=digest(dict(method=row['method'],course=course,graph_sha256=sha,book_set=row['book_set']))
        if row['qualification']['status']!='eligible_typed_graph':
            row.update(status='not_applicable' if row['qualification']['status'].startswith('not_applicable') else 'insufficient',
                usable_for_KGE=False,no_schema_types_invented=True)
        else:
            graph,cap=load_graph(source)
            mapping=[r for r in qualified['mapping'] if r['graph_record']['sha256']==sha and r['course_id']==course]
            projection=project_identity_graph(graph,references[course],splits[course],mapping,graph_sha=sha,capabilities=cap)
            folder=output/'graphs'/row['entry_id']
            paths=export_projection(folder,graph,references[course],splits[course],projection,qualified,row['graph'])
            ready=len(projection['training'])>=10 and any(projection['covered_test_mask']) and any(projection['covered_validation_mask'])
            row.update(status='frozen_ready_for_existing_KGE' if ready else 'frozen_insufficient_training_coverage',
                usable_for_KGE=True,trainable_by_existing_adapter=ready,projection_id=projection['projection_id'],
                graph_id=projection['graph_id'],split_id=projection['split_id'],training_triples=len(projection['training']),
                covered_test_count=sum(projection['covered_test_mask']),fixed_test_count=len(projection['test']),
                covered_validation_count=sum(projection['covered_validation_mask']),fixed_validation_count=len(projection['validation']),
                accepted_native_entities=len(projection['audit']['identity_mapping_provenance']),paths=paths,
                complete_fixed_denominators_retained=True,prior_identity_is_textbook_support=False)
        rows.append(row);write_immutable(output/'entries'/ (row['entry_id']+'.json'),row)
    jobs=training_jobs(rows)
    result=dict(status='frozen_identity_KGE_population',policy=POLICY,preparation_id=manifest['preparation_id'],
        qualification_id=qualified['qualification_id'],graphs=rows,inventory_graph_count=len(rows),
        reference_courses={c:dict(reference_id=s['reference_id'],split_id=s['split_id'],groups=len(s['groups']),
            original_fact_count=len(references[c]['facts']),folds={k:len(v) for k,v in s['folds'].items()},
            holdout_block_policy=s.get('identity_holdout_block_policy')) for c,s in splits.items()},
        existing_training_matrix=dict(policy=TRAINING_MATRIX,jobs=jobs,total_jobs=len(jobs),
            trainable_unique_graph_splits=len({(j['graph_sha256'],j['split_id']) for j in jobs})),
        training_started=False,model_calls=0,old_results_reused=False,common_intersection_not_primary=True,
        corrected_graph_extension='project_identity_graph accepts only that exact graph SHA and separately completed identity mapping; old F identity/scores never substitute',
        raw_text_exported=False,human_audited=False)
    result['package_id']=digest(result);write_immutable(output/'package.json',result);return result


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='gpu-finalize':
        parser=argparse.ArgumentParser(description='Bind actual GPU paths/assets to the existing KGE protocol; no training')
        for name in ['package','gpu-inputs','assets','output']:
            parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--entry-id',required=True)
        args=parser.parse_args(argv[1:])
        result=finalize_gpu_entry(args.package,args.entry_id,args.gpu_inputs,read_document(args.assets),args.output)
        print(json.dumps(result));return
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['preparation-root','mapping-root','single119-root','output-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--reference-preparation-root',type=Path)
    parser.add_argument('--reference-output-root',type=Path)
    args=parser.parse_args(argv);inventory,rows=inspect_inventory(args.preparation_root)
    dependencies=inspect_dependencies(args.mapping_root)
    reference_qualification=None;reference_ready=False
    if args.reference_preparation_root and args.reference_output_root:
        from .run_reference_identity import load_qualification,DependencyNotReady
        try:
            reference_qualification=load_qualification(args.reference_preparation_root,args.reference_output_root);reference_ready=True
        except DependencyNotReady:pass
    if args.check_only:
        print(json.dumps(dict(status='check_only',dependencies_ready=dependencies['ready'] and reference_ready,
            mapping_dependencies_ready=dependencies['ready'],reference_identity_dependencies_ready=reference_ready,
            missing_stages=dependencies['missing_stages'],incomplete_stages=dependencies['incomplete_stages'],
            inventory_graph_count=len(rows),typed_graphs=sum(r['qualification']['status']=='eligible_typed_graph' for r in rows),
            not_applicable_graphs=sum(r['qualification']['status'].startswith('not_applicable') for r in rows),
            model_calls=0,training_started=False,new_split_created=False,policy_id=digest(POLICY),
            training_matrix_policy=TRAINING_MATRIX,actual_training_jobs='not_created_until_full_mapping_qualification')))
        return
    if not dependencies['ready'] or not reference_ready:raise SystemExit('Identity mapping/reference qualification incomplete; no new split or training package created')
    result=freeze_population(args.preparation_root,args.mapping_root,args.single119_root,args.output_root,reference_qualification=reference_qualification)
    print(json.dumps(dict(status=result['status'],package_id=result['package_id'],inventory_graph_count=len(result['graphs']),
        model_calls=0,training_started=False)))


if __name__=='__main__':main()
