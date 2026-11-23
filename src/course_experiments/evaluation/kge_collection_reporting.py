"""只读核对primary资格包与GPU脱敏收据；不加载模型、不训练、不调用API。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..manifest import digest,file_record,read_document
from . import identity_kge_execution as execution
from .run_identity_kge import training_jobs
from .scope_mapping_preflight import qualification

PROOF_VERSION='identity_kge_remote_verified_documents_v1'
COLLECTION_STATUS='all_actual_full_inventory_models_collected_and_checkpoint_SHA_reverified'
CORE_ARTIFACTS={'model/trained_model.pkl','external_kge_identity.json','external_kge_protocol.json','pykeen_metrics.json','model/results.json'}


def checked(record):
    if file_record(record['path'])!=record:raise ValueError('Original primary KGE evidence SHA changed')
    return read_document(record['path'])


def _identity(value,key):
    if value.get(key)!=digest({k:v for k,v in value.items() if k!=key}):raise ValueError('Original KGE document identity changed')


def _snapshot(path):
    path=Path(path);data=path.read_bytes()
    return dict(record=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest()),utf8=data.decode('utf-8'))


def _document(snapshot):
    # remote路径只作为原始身份字段；primary只解析已传回的小型JSON字节。
    if hashlib.sha256(snapshot['utf8'].encode('utf-8')).hexdigest()!=snapshot['record']['sha256']:
        raise ValueError('Remote verified JSON bytes differ from original file SHA')
    return json.loads(snapshot['utf8'])


def capture_remote_evidence(collection,plan_path):
    """仅在原GPU collect端调用：复用既有收据验收，回传JSON与工件SHA而不传模型。"""
    plan_path=Path(plan_path);plan=read_document(plan_path);_identity(plan,'plan_id')
    if collection.get('status')!=COLLECTION_STATUS or collection.get('package_id')!=plan['package_id']:
        raise ValueError('Original GPU collection and training plan differ')
    jobs={j['training_job_id']:j for j in plan['jobs']};results=collection['results']
    if len(jobs)!=len(plan['jobs']) or len(results)!=len(jobs) or {r['training_job_id'] for r in results}!=set(jobs):
        raise ValueError('Complete original GPU job population required')
    captured={}
    def capture(record,job=None,plan_id=None,ancestors=()):
        marker=(record['path'],record['sha256'])
        if marker in ancestors:raise ValueError('Cyclic inherited actual model receipt')
        if marker in captured:return
        value=checked(record)
        execution.validate_saved_receipt(value,Path(record['path']).parent,job or value['job'],plan_id or value['plan_id'],False)
        if value.get('status')!='completed_actual_model_checkpoint_verified' or value.get('native_status') not in {'completed','completed_with_diagnostic_failure'}:
            raise ValueError('Training failure cannot be collected as a completed model')
        history_record=value['artifacts']['model/results.json'];history=checked(history_record)
        losses=history.get('losses')
        if not isinstance(losses,list) or len(losses)!=1000 or any(type(x) not in (int,float) or not math.isfinite(x) for x in losses):
            raise ValueError('Actual saved1000 finite training losses required')
        captured[marker]=dict(receipt=_snapshot(record['path']),history=_snapshot(history_record['path']),verified_artifacts=value['artifacts'])
        if value.get('training_reused'):
            capture(value['inherited_actual_model_receipt'],ancestors=(*ancestors,marker))
    for result in results:capture(result['result'],jobs[result['training_job_id']],plan['plan_id'])
    value=dict(version=PROOF_VERSION,package_id=plan['package_id'],collection_digest=digest(collection),plan=_snapshot(plan_path),
        receipts=list(captured.values()),verifier=dict(capture=file_record(Path(__file__)),validator=file_record(Path(execution.__file__))),
        raw_text_exported=False,model_parameters_transferred=False,model_API_calls=0)
    value['proof_id']=digest(value);return value


def _remote_models(package,collection,proof):
    _identity(proof,'proof_id')
    if (proof.get('version')!=PROOF_VERSION or proof.get('package_id')!=package['package_id'] or
            proof.get('collection_digest')!=digest(collection) or collection.get('package_id')!=package['package_id'] or
            collection.get('status')!=COLLECTION_STATUS):raise ValueError('Exact original package/collection/proof binding required')
    for key,path in [('capture',Path(__file__)),('validator',Path(execution.__file__))]:
        if proof['verifier'][key]['sha256']!=file_record(path)['sha256']:raise ValueError('Original remote verification implementation differs')
    plan=_document(proof['plan']);_identity(plan,'plan_id')
    if plan['package_id']!=package['package_id']:raise ValueError('Remote plan belongs to another qualified package')
    expected={j['training_job_id']:j for j in execution.frozen_jobs(package)};jobs={j['training_job_id']:j for j in plan['jobs']}
    summaries={r['training_job_id']:r for r in collection['results']}
    if (len(jobs)!=len(plan['jobs']) or len(summaries)!=len(collection['results']) or set(jobs)!=set(expected) or set(summaries)!=set(expected)):
        raise ValueError('Remote model population is incomplete or duplicated')
    snapshots={};values={}
    for item in proof['receipts']:
        r=item['receipt']['record'];key=(r['path'],r['sha256'])
        if key in snapshots:raise ValueError('Duplicate remote receipt evidence')
        value=_document(item['receipt']);_identity(value,'result_id');history=_document(item['history']);losses=history.get('losses')
        if (value.get('smoke') is not False or value.get('actual_epochs')!=1000 or value.get('status')!='completed_actual_model_checkpoint_verified' or
                value.get('native_status') not in {'completed','completed_with_diagnostic_failure'} or
                not isinstance(losses,list) or len(losses)!=1000 or any(type(x) not in (int,float) or not math.isfinite(x) for x in losses)):
            raise ValueError('Successful actual model and1000 saved finite losses required')
        artifacts=value['artifacts']
        if (not CORE_ARTIFACTS<=set(artifacts) or not any(k.startswith('model/training_triples/') for k in artifacts) or
                artifacts!=item['verified_artifacts'] or artifacts['model/results.json']!=item['history']['record']):
            raise ValueError('Original actual model/mapping/loss artifact verification differs')
        if any(set(r)!={'path','sha256'} or len(r['sha256'])!=64 for r in artifacts.values()):raise ValueError('Complete original model artifact SHA population required')
        snapshots[key]=item;values[key]=value
    def ancestry(key,seen=()):
        if key in seen or key not in values:raise ValueError('Complete acyclic original model-reuse receipt chain required')
        value=values[key]
        if value.get('training_reused'):
            r=value['inherited_actual_model_receipt'];old=ancestry((r['path'],r['sha256']),(*seen,key))
            if (value.get('new_training_run') is not False or value['job'].get('training_signature')!=old['job'].get('training_signature') or
                    value.get('software')!=old.get('software') or
                    {k:r['sha256'] for k,r in value['artifacts'].items()}!={k:r['sha256'] for k,r in old['artifacts'].items()}):
                raise ValueError('Inherited model does not have identical frozen training inputs and actual artifacts')
        return value
    out={}
    for key,job in jobs.items():
        if any(job.get(k)!=v for k,v in expected[key].items()):raise ValueError('Remote graph/split/model/seed job differs from original package')
        summary=summaries[key];record=summary['result'];value=ancestry((record['path'],record['sha256']))
        if value['job']!=job or value['plan_id']!=plan['plan_id'] or any(summary.get(k)!=v for k,v in dict(model=job['model'],seed=job['seed'],epochs=1000,scores=value['scores'],status=value['status']).items()):
            raise ValueError('Original collection summary differs from actual result or model plan')
        out[key]=dict(value,remote_result=record,actual_loss_count=1000)
    return out


def _entry_evidence(package,entry):
    raw=checked(entry['graph']);graph=raw;cap=None;sources=[entry['graph']]
    if 'semantic_view' in raw:
        if raw.get('status') not in {'complete','completed'}:raise ValueError('Incomplete native artifact cannot prove KGE applicability')
        graph={k:raw['semantic_view'][k] for k in ['entities','relations']};cap=raw['semantic_view'].get('capabilities')
    current=qualification(graph,cap)
    if current!=entry['qualification']:raise ValueError('Original native schema qualification differs')
    if current['status'].startswith('not_applicable'):
        if entry.get('usable_for_KGE') is not False or entry.get('status')!=current['status']:raise ValueError('Original inapplicable native schema evidence required')
        return None,current,sources
    if current['status']!='eligible_typed_graph' or entry.get('usable_for_KGE') is not True:return None,current,sources
    paths=entry.get('paths',{});projection=read_document(paths['projection.json']);split=read_document(paths['fact_split.json'])
    sources += [file_record(paths['projection.json']),file_record(paths['fact_split.json'])]
    _identity(projection,'projection_id')
    from .kge import validate_kge_split
    validate_kge_split(split)
    course=package['reference_courses'][entry['course_id']]
    if (projection['projection_id']!=entry['projection_id'] or projection['graph_id']!=digest(graph) or projection['graph_id']!=entry['graph_id'] or
            projection['audit'].get('original_graph_file_sha256')!=entry['graph']['sha256'] or
            projection['split_id']!=entry['split_id'] or split['split_id']!=entry['split_id'] or course['split_id']!=entry['split_id'] or
            projection['reference_id']!=course['reference_id'] or split['reference_id']!=course['reference_id']):
        raise ValueError('Exact original graph/projection/reference split binding differs')
    for fold in ['test','validation']:
        expected=[split['groups'][k]['triple'] for k in split['folds'][fold]];mask=projection['covered_'+fold+'_mask']
        if (projection[fold]!=expected or projection[fold+'_ids']!=split['folds'][fold] or len(mask)!=len(expected) or any(type(x) is not bool for x in mask) or
                len(expected)!=course['folds'][fold] or entry['covered_'+fold+'_count']!=sum(mask) or entry['fixed_'+fold+'_count']!=len(mask)):
            raise ValueError('Original fixed fact denominator or coverage mask differs')
    ready=len(projection['training'])>=10 and any(projection['covered_test_mask']) and any(projection['covered_validation_mask'])
    if entry.get('trainable_by_existing_adapter') is not ready or entry['training_triples']!=len(projection['training']):raise ValueError('Original frozen training eligibility differs')
    if not ready and entry['status']!='frozen_insufficient_training_coverage':raise ValueError('Training failure cannot be relabelled frozen input insufficiency')
    return projection,current,sources


def collect_observations(requirements,package_record,collection_record=None,remote_proof_record=None):
    """输出给只读总表的独立行；缺远端证明保持pending，失败不能改为N/A。"""
    requirements=[r for r in requirements if r.get('category') in {'kge_qualification','kge_training'}]
    observations={};errors=[];sources=[package_record]
    try:
        package=checked(package_record);execution.frozen_jobs(package)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return dict(observations={r['id']:dict(adapter='verified_kge_collection',status='invalid',reason='qualified_package_validation_failed',error_type=type(exc).__name__,metrics=None,sources=sources) for r in requirements},pending_reasons=[],sources=sources)
    models=None;remote_error=None
    if collection_record and remote_proof_record:
        try:
            models=_remote_models(package,checked(collection_record),checked(remote_proof_record));sources += [collection_record,remote_proof_record]
        except (OSError,ValueError,KeyError,TypeError) as exc:remote_error=type(exc).__name__
    for req in requirements:
        row=dict(adapter='verified_kge_collection',status='pending',reason='qualified_exact_graph_not_yet_published',metrics=None,sources=[package_record],training_completed=False)
        observations[req['id']]=row
        if not req.get('graph_sha256'):row['reason']='frozen_required_graph_SHA_not_supplied';continue
        method=req.get('variant') if req['method']=='own_concept_graph' else req['method']
        entries=[e for e in package['graphs'] if e['method']==method and req.get('cohort') in e.get('cohorts',[]) and e['graph']['sha256']==req['graph_sha256']]
        if len(entries)!=1:
            if len(entries)>1:row.update(status='invalid',reason='ambiguous_original_graph_entry')
            continue
        entry=entries[0]
        try:
            projection,schema,ev=_entry_evidence(package,entry);row['sources']+=ev;row.update(graph_sha256=entry['graph']['sha256'],entry_id=entry['entry_id'],qualification_id=package.get('qualification_id'),reference_qualification_id=package['reference_qualification_id'])
            if schema['status'].startswith('not_applicable'):
                row.update(status='not_applicable',reason=schema['status'],applicability_evidence_verified=True);continue
            if projection is None:row['reason']='native_schema_or_source_identity_not_qualified';continue
            row.update(split_id=projection['split_id'],coverage=dict(training_triples=len(projection['training']),covered_test_count=sum(projection['covered_test_mask']),fixed_test_count=len(projection['test']),covered_validation_count=sum(projection['covered_validation_mask']),fixed_validation_count=len(projection['validation'])))
            if req['category']=='kge_qualification':row.update(status='completed',reason='original_graph_reference_split_and_coverage_verified');continue
            if not entry['trainable_by_existing_adapter']:
                row.update(status='not_applicable',reason='frozen_insufficient_training_coverage',applicability_evidence_verified=True);continue
            if models is None:
                row.update(reason='remote_actual_model_loss_and_artifact_proof_required',remote_proof_error_type=remote_error);continue
            jobs=training_jobs([entry]);selected=[models[j['training_job_id']] for j in jobs]
            for value in selected:
                score=value['scores'];fixed=len(projection['test']);covered=sum(projection['covered_test_mask'])
                if score['fixed_test_count']!=fixed or score['covered_test_count']!=covered or score['coverage']!=covered/fixed or score.get('uncovered_contribution')!=0:raise ValueError('Actual model metrics use another fact denominator')
                for label in ['mrr','hits_at_1','hits_at_3','hits_at_10']:
                    a=score['covered_'+label];b=score['full_fixed_'+label]
                    if type(a) not in (float,int) or not math.isfinite(a) or not 0<=a<=1 or b!=a*covered/fixed:raise ValueError('Original fixed-denominator model metric differs')
                if value.get('training_reused') and value.get('fixed_evaluation_signature')!=execution.evaluation_signature(projection):raise ValueError('Inherited model fixed evaluation differs')
            row.update(status='completed',reason='same_GPU_verification_original_models_losses_artifacts_and_fixed_split_verified',training_completed=True,remote_model_artifacts_verified=True,
                models=[dict(training_job_id=v['job']['training_job_id'],model=v['job']['model'],seed=v['job']['seed'],epochs=1000,actual_loss_count=v['actual_loss_count'],scores=v['scores'],training_reused=v.get('training_reused',False),remote_result=v['remote_result']) for v in selected],sources=row['sources']+[collection_record,remote_proof_record])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            row.update(status='invalid',reason='original_KGE_evidence_validation_failed',error_type=type(exc).__name__);errors.append(dict(requirement_id=req['id'],error_type=type(exc).__name__))
    return dict(observations=observations,pending_reasons=[dict(requirement_id=k,reason=v['reason']) for k,v in observations.items() if v['status']=='pending'],collection_errors=errors,sources=sources,model_API_calls=0,training_started=False)
