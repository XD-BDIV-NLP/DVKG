"""严格v2资格限定GPU控制器；原PyKEEN算法，仅既有模型/seed薄适配。"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from .identity_single_v2 import POLICY
from .run_identity_kge import training_jobs,TRAINING_MATRIX


def execution_scope(package):
    if package.get('status')=='G0_only_inputs_prepared_for_review':
        if any(r.get('usable_for_KGE') and r['method']!='G0' for r in package['graphs']):raise ValueError('Source dependent graph cannot enter G0 execution')
        return dict(original_deadline_unix=1792777336.907112,source_graphs_forbidden=True,scope='G0_only')
    from .identity_source_full_execution import START,AUTHORIZATION_SHA,ALLOCATION_SHA
    authorization=dict(original_started_unix=START,deadline_unix=START+345600,authorized_total_seconds=345600,
        authorized_cumulative_HTTP=200000,authorization_sha256=AUTHORIZATION_SHA,allocation_sha256=ALLOCATION_SHA)
    proof=package.get('source_actual_HTTP_proof',{})
    counts=package.get('original_population_counts',{});extended=package.get('status')=='complete_extended_inventory_identity_inputs_prepared'
    if extended:
        inherit=package.get('inventory_append_execution',{});previous=inherit.get('previous_population_counts',{})
        if (inherit.get('proof_id')!=digest({k:v for k,v in inherit.items() if k!='proof_id'}) or inherit.get('current_population_counts')!=counts or
                inherit.get('original_reference_splits_unchanged') is not True or inherit.get('all_original_classifications_and_raw_unchanged') is not True or
                inherit.get('original_HTTP_not_reasked') is not True or previous.get('source',0)<7519 or previous.get('prior',0)<200 or previous.get('reference')!=485 or
                counts.get('source',0)<previous['source'] or counts.get('prior',0)<previous['prior'] or counts.get('reference')!=485 or
                inherit.get('adopted_candidates')!={b:previous[b] for b in ['source','prior']}):
            raise ValueError('Extended inventory requires complete immutable inherited classifications, original HTTP and reference-split proof')
    if (package.get('status') not in {'complete_frozen_inventory_identity_inputs_prepared','complete_extended_inventory_identity_inputs_prepared'} or package.get('current_inventory_source_full_qualified') is not True or
            package.get('future_graphs_qualified') is not False or package.get('user_full_matrix_complete') is not False or
            (not extended and counts!={'source':7519,'prior':200,'reference':485}) or
            package.get('reference_qualification_id')!='5fe6c8630a70897424f72ffc95484bb40170d24ae02f3fe8d443d99e245dbf06' or
            package.get('execution_authorization')!=authorization or proof.get('candidate_count')!=counts.get('source') or
            proof.get('zero_HTTP_without_raw')!=0 or proof.get('all_original_raw_revalidated_at_finalization') is not True):
        raise ValueError('Separately authorized complete-source frozen-inventory qualification and original reference-only split required')
    return dict(original_deadline_unix=START+345600,source_graphs_forbidden=False,scope='complete_current_frozen_inventory',execution_authorization=authorization)


def frozen_jobs(package):
    if package.get('package_id')!=digest({k:v for k,v in package.items() if k!='package_id'}) or package.get('protocol')!=POLICY['version']:
        raise ValueError('Frozen complete v2 qualified package required')
    execution_scope(package)
    matrix=package['existing_training_matrix'];expected=training_jobs(package['graphs'])
    if matrix['policy']!=TRAINING_MATRIX or matrix['jobs']!=expected or matrix['total_jobs']!=len(expected):raise ValueError('Frozen G0 model/seed/1000 epoch matrix changed')
    return expected


def fixed_denominator_metrics(metrics,*,covered,fixed):
    if (not isinstance(metrics,dict) or type(covered) is not int or type(fixed) is not int or not 0<=covered<=fixed or fixed<1):
        raise ValueError('Successful native metrics and complete fixed denominator required')
    fraction=covered/fixed;value=dict(covered_test_count=covered,fixed_test_count=fixed,coverage=fraction,uncovered_contribution=0)
    for native,label in [('inverse_harmonic_mean_rank','mrr'),('hits_at_1','hits_at_1'),('hits_at_3','hits_at_3'),('hits_at_10','hits_at_10')]:
        score=metrics[native]
        if type(score) not in (float,int) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid native normalized rank metric')
        value['covered_'+label]=score;value['full_fixed_'+label]=score*fraction
    return value


def software():return {name:importlib.metadata.version(name) for name in ['torch','pykeen','numpy','pandas','scikit-learn']}


def _checked(record):
    if file_record(record['path'])!=record:raise ValueError('Frozen execution asset changed')
    return Path(record['path'])


def training_artifacts(root,expected_epochs):
    root=Path(root);history=read_document(root/'model/results.json');losses=history.get('losses')
    if (type(expected_epochs) is not int or expected_epochs not in {2,1000} or not isinstance(losses,list) or len(losses)!=expected_epochs or
            any(type(loss) not in (int,float) or not math.isfinite(loss) for loss in losses)):
        raise ValueError('Saved native training loss history does not prove the complete actual epoch count')
    names=['model/trained_model.pkl','external_kge_identity.json','external_kge_protocol.json','pykeen_metrics.json','model/results.json']
    mappings=sorted(p for p in (root/'model/training_triples').rglob('*') if p.is_file())
    if not mappings:raise ValueError('Saved actual training mappings and triples are missing')
    names.extend(str(p.relative_to(root)).replace('\\','/') for p in mappings)
    return {name:file_record(root/name) for name in names}


def validate_saved_receipt(receipt,root,job,plan_id,smoke):
    if (receipt.get('result_id')!=digest({k:v for k,v in receipt.items() if k!='result_id'}) or receipt['plan_id']!=plan_id or
            receipt['job']!=job or receipt['smoke'] is not smoke or receipt['actual_epochs']!=(2 if smoke else 1000)):
        raise ValueError('Existing training receipt identity differs')
    for record in receipt['artifacts'].values():_checked(record)
    if training_artifacts(root,2 if smoke else 1000)!=receipt['artifacts']:
        raise ValueError('Saved actual training mapping/triples/history artifact population changed')
    if receipt.get('training_reused'):
        source=_checked(receipt['inherited_actual_model_receipt']);previous=read_document(source)
        validate_saved_receipt(previous,source.parent,previous['job'],previous['plan_id'],smoke)
    return receipt


def evaluation_signature(projection):
    return digest({k:projection[k] for k in ['training','validation','test','test_ids','validation_ids','entity_vocabulary',
        'relation_vocabulary','covered_test_mask','covered_validation_mask','reference_id','split_id','condition_projection']})


def inherit_actual_training(collection,root,job,projection,software,adapter_sha,*,smoke=False):
    from .kge import validate_protocol
    from .protocol import validate_phase
    import errno
    root=Path(root)
    for source in collection.get('smoke_results' if smoke else 'results',[]):
        path=_checked(source['result']);receipt=read_document(path);old_job=receipt['job']
        if old_job.get('training_signature')!=job['training_signature']:continue
        validate_saved_receipt(receipt,path.parent,old_job,receipt['plan_id'],smoke)
        if receipt['software']!=software:raise ValueError('Inherited actual model software differs')
        protocol=validate_protocol(read_document(path.parent/'external_kge_protocol.json'));phase=validate_phase(protocol['phase_path'])
        if phase['files']['actual_kge_adapter']['sha256']!=adapter_sha:raise ValueError('Inherited actual training algorithm differs')
        if evaluation_signature(protocol['projection'])!=evaluation_signature(projection):
            raise ValueError('Existing actual training matches but fixed evaluation changed; preserve model without duplicate training')
        for relative,record in receipt['artifacts'].items():
            if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Inherited artifact path outside verified checkpoint')
            source_path=_checked(record);target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if file_record(target)['sha256']!=record['sha256']:raise ValueError('Existing new-attempt artifact differs; preserve it')
            else:
                try:os.link(source_path,target)
                except OSError as exc:
                    if exc.errno!=errno.EXDEV:raise
                    shutil.copy2(source_path,target)
        return dict(training_reused=True,new_training_run=False,inherited_actual_model_receipt=file_record(path),
            inherited_actual_epochs=receipt['actual_epochs'],fixed_evaluation_signature=evaluation_signature(projection),native_status=receipt['native_status'])
    return None


def await_resources(try_guard,probe_busy,deadline,notify,*,pause=time.sleep,now=time.time):
    while now()<deadline:
        guard=try_guard()
        if guard is not None:
            try:
                if not probe_busy():return guard
            except BaseException:guard.close();raise
            guard.close()
        notify('waiting_existing_GPU_execution');pause(30)
    raise RuntimeError('Original finite deadline reached while waiting for shared GPU resources')


def validate_native_completion(result,smoke):
    if result.get('status') not in {'completed','completed_with_diagnostic_failure'} or result.get('training_status')!='completed':
        raise RuntimeError('Actual existing-framework model training did not complete')
    if smoke and (result['status']!='completed' or result.get('diagnostics_status')!='completed'):
        raise RuntimeError('A failed or unavailable native smoke diagnostic cannot unlock the formal matrix')


def prepare(bundle,output):
    from .multi_model_kge import derive_native,training_signature
    bundle=Path(bundle).resolve();output=Path(output).resolve();package=read_document(bundle/'package.json');jobs=frozen_jobs(package)
    transfer=read_document(bundle/'transfer_manifest.json')
    for record in transfer['files']:_checked(record)
    expected=read_document(bundle/'metadata/gpu_environment.json')
    if software()!=expected['packages']:raise ValueError('Actual GPU software differs from frozen environment')
    contract=dict(package=file_record(bundle/'package.json'),transfer=file_record(bundle/'transfer_manifest.json'),bundle=str(bundle),software=software(),
        runner=file_record(Path(__file__)),maximum_seconds=100000,per_job_seconds=7200,physical_gpu_slots=4,
        epochs=[1000],smoke_epochs=2,raw_text_received=False,**execution_scope(package))
    inherited=bundle/'metadata/inherited_training_collection.json'
    if inherited.exists():contract['inherited_training_collection']=file_record(inherited)
    write_immutable(output/'contract.json',contract);prepared=[]
    for original in jobs:
        job=dict(original);entry=next(r for r in package['graphs'] if r['entry_id']==job['entry_ids'][0]);folder=bundle/'inputs'/entry['entry_id']
        projection=read_document(folder/'projection.json')
        if any(projection[k]!=entry[k] for k in ['projection_id','graph_id','split_id']):raise ValueError('Sanitized GPU projection differs from qualified package')
        private=output/'private'/job['training_job_id'];code=private/'code/app'
        if not code.exists():shutil.copytree(bundle/'code/app',code)
        source=(bundle/'code/app/course_experiments/evaluation/kge.py').read_bytes();derived,proof=derive_native(source,job['model'],job['seed'])
        destination=code/'course_experiments/evaluation/kge.py'
        if destination.read_bytes()!=derived:
            if destination.read_bytes()!=source:raise ValueError('Private frozen model adapter changed')
            destination.write_bytes(derived)
        write_immutable(private/'model_control.json',proof)
        job.update(private=str(private),input_folder=str(folder),training_signature=training_signature(projection,job['model'],job['seed'],1000,contract['software']))
        prepared.append(job)
    plan=dict(contract=file_record(output/'contract.json'),package_id=package['package_id'],jobs=prepared,raw_text_received=False)
    plan['plan_id']=digest(plan);write_immutable(output/'prepared.json',plan)
    for job in prepared:
        env=dict(os.environ,PYTHONPATH=str(Path(job['private'])/'code/app'),PYTHONDONTWRITEBYTECODE='1')
        with (Path(job['private'])/'finalize.log').open('a') as log:
            result=subprocess.run([sys.executable,'-B','-m','course_experiments.evaluation.identity_kge_execution','finalize','--output',str(output),'--job',job['training_job_id']],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if result.returncode:raise RuntimeError('Private original-framework protocol finalization failed')
    return plan


def _plan(output):
    output=Path(output);plan=read_document(output/'prepared.json')
    if plan['plan_id']!=digest({k:v for k,v in plan.items() if k!='plan_id'}):raise ValueError('Frozen training plan changed')
    contract=read_document(_checked(plan['contract']));_checked(contract['package']);_checked(contract['transfer']);_checked(contract['runner'])
    if contract.get('inherited_training_collection'):_checked(contract['inherited_training_collection'])
    if software()!=contract['software']:raise ValueError('Frozen actual software changed')
    return plan,contract


def finalize(output,key):
    from .kge import prepare_protocol,validate_protocol,BUDGET
    from .protocol import freeze_phase,validate_phase
    plan,contract=_plan(output);job=next(j for j in plan['jobs'] if j['training_job_id']==key);private=Path(job['private']);bundle=Path(contract['bundle']);folder=Path(job['input_folder'])
    if BUDGET['model']!=job['model'] or BUDGET['seed']!=job['seed']:raise ValueError('Frozen derived model/seed differs')
    paths={k:folder/name for k,name in [('graph','graph_attestation.json'),('projection','projection.json'),('fact_split','fact_split.json'),('primary_attestation','primary_attestation.json')]}
    paths.update(construction=bundle/'metadata/construction.json',gpu_environment=bundle/'metadata/gpu_environment.json',dependency_lock=bundle/'metadata/dependency_lock.txt',
        src_utils=bundle/'code/reused_patched/src/utils.py',evaluator_original=bundle/'code/reused_original/scripts/evaluate_kg_quality.py',sweep_original=bundle/'code/reused_original/scripts/run_pykeen_epoch_sweep.py',
        evaluator_patched=bundle/'code/reused_patched/scripts/evaluate_kg_quality.py',sweep_patched=bundle/'code/reused_patched/scripts/run_pykeen_epoch_sweep.py',
        identity_package=bundle/'package.json',model_control=private/'model_control.json',actual_kge_adapter=private/'code/app/course_experiments/evaluation/kge.py',
        finite_execution_contract=Path(output)/'contract.json',finite_execution_plan=Path(output)/'prepared.json')
    phase_path=private/'gpu_phase.json';phase=freeze_phase(paths,dict(seed=42,training_seed=job['seed'],model=job['model'],stage='strict_single_v2_G0_KGE' if contract['source_graphs_forbidden'] else 'strict_single_v2_full_inventory_KGE'),phase_path)
    phase['path']=str(phase_path)
    for smoke in [True,False]:
        protocol=prepare_protocol(read_document(paths['projection']),read_document(paths['fact_split']),phase,private/('protocol_smoke.json' if smoke else 'protocol_full.json'),smoke=smoke)
        validate_protocol(protocol)
    validate_phase(phase_path)


def worker(output,key,*,smoke=False):
    import torch
    from .kge import validate_protocol,BUDGET
    from .topology import load_evaluator
    from .protocol import validate_phase
    from .multi_model_kge import training_signature
    plan,contract=_plan(output);job=next(j for j in plan['jobs'] if j['training_job_id']==key);private=Path(job['private']);folder=Path(job['input_folder'])
    protocol_path=private/('protocol_smoke.json' if smoke else 'protocol_full.json');protocol=validate_protocol(read_document(protocol_path));phase=validate_phase(protocol['phase_path']);projection=protocol['projection']
    if BUDGET['model']!=job['model'] or BUDGET['seed']!=job['seed'] or job['epochs']!=1000:raise ValueError('Frozen actual training model/seed/epochs changed')
    if training_signature(projection,job['model'],job['seed'],1000,contract['software'])!=job['training_signature']:raise ValueError('Actual training inputs differ from signature')
    root=Path(output)/('smoke' if smoke else 'training')/key;receipt=root/'verified_result.json'
    if receipt.exists():
        return validate_saved_receipt(read_document(receipt),root,job,plan['plan_id'],smoke)
    root.mkdir(parents=True,exist_ok=True)
    evaluator_record=phase['files']['evaluator_patched'];evaluator=load_evaluator(str(_checked(evaluator_record)),evaluator_record['sha256'])
    args=SimpleNamespace(seed=job['seed'],pykeen_model=job['model'],pykeen_epochs=2 if smoke else 1000,pykeen_device='cuda:0',external_protocol=protocol_path)
    reused=None
    if contract.get('inherited_training_collection'):
        reused=inherit_actual_training(read_document(_checked(contract['inherited_training_collection'])),root,job,projection,contract['software'],phase['files']['actual_kge_adapter']['sha256'],smoke=smoke)
    if reused:
        result=dict(status=reused['native_status'],training_status='completed',diagnostics_status='completed' if reused['native_status']=='completed' else 'failed')
    else:result=evaluator.run_pykeen(SimpleNamespace(concept_kg=read_document(folder/'graph_attestation.json')),args,root)
    validate_native_completion(result,smoke)
    model=torch.load(root/'model/trained_model.pkl',map_location='cuda:0',weights_only=False)
    if type(model).__name__!=job['model']:raise ValueError('Saved actual model class differs')
    from pykeen.triples import TriplesFactory
    triples=TriplesFactory.from_path_binary(root/'model/training_triples');em={e:i for i,e in enumerate(projection['entity_vocabulary'])};rm={r:i for i,r in enumerate(projection['relation_vocabulary'])}
    if triples.entity_to_id!=em or triples.relation_to_id!=rm or model.num_entities!=len(em) or model.num_relations!=len(rm):raise ValueError('Saved model entity/relation mappings differ')
    if {tuple(t) for t in triples.mapped_triples.tolist()}!={(em[h],rm[r],em[t]) for h,r,t in projection['training']}:raise ValueError('Saved model actual training triples differ')
    metrics=read_document(root/'pykeen_metrics.json')['both']['realistic']
    value=dict(status='completed_actual_model_checkpoint_verified',job=job,plan_id=plan['plan_id'],smoke=smoke,actual_epochs=2 if smoke else 1000,
        software=contract['software'],native_status=result['status'],raw_text_received=False,
        scores=fixed_denominator_metrics(metrics,covered=sum(projection['covered_test_mask']),fixed=len(projection['test'])),
        actual_training_history='native_PyKEEN_save_to_directory_results_json_loss_count',artifacts=training_artifacts(root,2 if smoke else 1000))
    value.update(reused or dict(training_reused=False,new_training_run=True))
    value['result_id']=digest(value);write_immutable(receipt,value);return value


def controller(output):
    import fcntl
    from .native_provider import _atomic
    output=Path(output);plan,contract=_plan(output);start=time.time();done=[]
    def state(status,**kw):_atomic(output/'state.json',dict(status=status,completed=done,updated_unix=time.time(),raw_text_received=False,**kw))
    if not plan['jobs']:
        state('no_trainable_qualified_graphs_fixed_population_retained',training_started=False);return
    def try_guard():
        lock_path = Path(os.environ.get('DVKG_KGE_LOCK_PATH', Path.home() / '.dvkg_kge_execution.lock'))
        guard=lock_path.open('a+b')
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);return guard
        except BlockingIOError:guard.close();return None
    def compute_busy():
        busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
        return bool(busy.stdout.strip())
    guard=None
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=4:raise RuntimeError('Authorized four actual GPUs unavailable')
        guard=await_resources(try_guard,compute_busy,min(start+contract['maximum_seconds'],contract['original_deadline_unix']),state)
        def run(job,gpu,smoke=False):
            if time.time()>=min(start+contract['maximum_seconds'],contract['original_deadline_unix']):raise RuntimeError('Original finite experiment deadline reached')
            private=Path(job['private']);env={k:v for k,v in os.environ.items() if not k.upper().endswith(('_API_KEY','_PROXY','_PASSWORD'))}
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(private/'code/app'),PYTHONHASHSEED=str(job['seed']),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            logs=output/'logs';logs.mkdir(exist_ok=True);command=[sys.executable,'-B','-m','course_experiments.evaluation.identity_kge_execution','worker','--output',str(output),'--job',job['training_job_id']]
            if smoke:command.append('--smoke')
            with (logs/(job['training_job_id']+('_smoke' if smoke else '')+'.log')).open('a') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(contract['per_job_seconds'],max(1,contract['original_deadline_unix']-time.time())))
            if result.returncode:raise RuntimeError('Original-framework worker failed; preserve all checkpoints')
            return dict(training_job_id=job['training_job_id'],physical_gpu=gpu,smoke=smoke,exit_code=0)
        state('five_model_two_epoch_smoke')
        for model in ['TransE','TransR','DistMult','ComplEx','RotatE']:run(next(j for j in plan['jobs'] if j['model']==model and j['seed']==42),0,True)
        state('running_qualified_1000_epoch_matrix')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending=iter(plan['jobs']);active={}
            for gpu in range(4):
                job=next(pending,None)
                if job:active[pool.submit(run,job,gpu)]=gpu
            while active:
                ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ready:
                    gpu=active.pop(future);done.append(future.result());state('running_qualified_1000_epoch_matrix')
                    job=next(pending,None)
                    if job:active[pool.submit(run,job,gpu)]=gpu
        state('completed_qualified_actual_1000_epoch_models',training_started=True)
    except BaseException as exc:state('stopped_preserve_all_checkpoints',error_type=type(exc).__name__);raise
    finally:
        if guard is not None:guard.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','finalize','worker','controller']);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bundle',type=Path);parser.add_argument('--job');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv)
    if args.mode=='prepare':prepare(args.bundle,args.output)
    elif args.mode=='finalize':finalize(args.output,args.job)
    elif args.mode=='worker':worker(args.output,args.job,smoke=args.smoke)
    else:controller(args.output)


if __name__=='__main__':main()
