from __future__ import annotations
import difflib
import json
import traceback
from pathlib import Path
from ..manifest import file_record,digest,read_document
from ..contributions import write_immutable
from ..splits import canonical
from .topology import EVALUATOR_SHA

SWEEP_SHA='d88fee5332f7719fbd0337e26065968c5c57e31acad25cded7dacb99e1c60729'
BUDGET=dict(model='TransE',epochs=[500,1000],dimension=128,batch_size=256,negative_samples=16,learning_rate=.001,seed=42)

def validate_kge_split(split):
    if split.get('format')!='sanitized_kge_split':
        from ..splits import validate_split
        return validate_split(split)
    if split['export_id']!=digest({k:v for k,v in split.items() if k!='export_id'}):raise ValueError('Sanitized split hash mismatch')
    folds=[set(split['folds'][k]) for k in ('train','validation','test')]
    if any(folds[i]&folds[j] for i in range(3) for j in range(i)) or set.union(*folds)!=set(split['groups']):raise ValueError('Sanitized fold overlap/assignment')
    if any(digest(canonical(*g['triple']))!=key for key,g in split['groups'].items()):raise ValueError('Sanitized equivalence group changed')
    if any(set(g)!={'triple','fact_ids'} for g in split['groups'].values()):raise ValueError('Sanitized split includes raw source fields')
    return split

def export_kge_package(projection,split,primary_phase_path,output):
    from .protocol import validate_phase,bound_input
    phase=validate_phase(primary_phase_path);validate_kge_split(split)
    bound_input(phase,'projection',projection);bound_input(phase,'fact_split',split)
    graph=read_document(phase['files']['graph']['path'])
    if digest(graph)!=projection['graph_id'] or projection['projection_id']!=digest({k:v for k,v in projection.items() if k!='projection_id'}):raise ValueError('Primary graph/projection changed')
    if projection['split_id']!=split['split_id']:raise ValueError('Primary split changed')
    sanitized=dict(format='sanitized_kge_split',split_id=split['split_id'],original_split_digest=digest(split),reference_id=split['reference_id'],groups={k:dict(triple=v['triple'],fact_ids=v['fact_ids']) for k,v in split['groups'].items()},folds=split['folds'],seed=42,policy=split['policy'])
    sanitized['export_id']=digest(sanitized)
    attestation=dict(format='primary_kge_attestation',primary_phase_id=phase['phase_id'],graph_id=projection['graph_id'],projection_id=projection['projection_id'],split_id=split['split_id'],original_split_digest=digest(split),sanitized_split_digest=digest(sanitized),reference_id=split['reference_id'])
    attestation['attestation_id']=digest(attestation)
    graph_attestation=dict(format='kge_graph_attestation',graph_id=projection['graph_id'],projection_id=projection['projection_id'],primary_attestation_id=attestation['attestation_id'])
    output=Path(output)
    for name,value in [('projection.json',projection),('fact_split.json',sanitized),('primary_attestation.json',attestation),('graph_attestation.json',graph_attestation)]:write_immutable(output/name,value)
    return dict(attestation=attestation,paths={name:str(output/name) for name in ('projection.json','fact_split.json','primary_attestation.json','graph_attestation.json')})

def patch_evaluators(evaluator,sweep,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);diffs=[];records={}
    for role,path,sha in [('evaluator',Path(evaluator),EVALUATOR_SHA),('sweep',Path(sweep),SWEEP_SHA)]:
        if file_record(path)['sha256']!=sha:raise ValueError('Original evaluator/sweep SHA mismatch')
        old=path.read_text('utf-8');new=old
        if role=='evaluator':
            anchor='def run_pykeen(inputs: EvaluationInputs, args: argparse.Namespace, report_dir: Path) -> dict[str, Any]:\n'
            insertion=anchor+'    if getattr(args, "external_protocol", None):\n        from course_experiments.evaluation.kge import run_external\n        args._course_evaluator_utils_path = normalize_whitespace.__globals__["__file__"]\n        return run_external(inputs, args, report_dir)\n'
            if old.count(anchor)!=1:raise ValueError('Patch anchor changed')
            new=old.replace(anchor,insertion)
        anchor='    return parser.parse_args()'
        if old.count(anchor)!=1:raise ValueError('CLI patch anchor changed')
        new=new.replace(anchor,'    parser.add_argument("--external-protocol", type=Path, help="Frozen external course split/budget protocol; original split disabled.")\n'+anchor)
        if role=='sweep':
            anchor='            print(f"[sweep] running epochs={epoch} -> {run_dir}", flush=True)'
            new=new.replace(anchor,'            if args.external_protocol:\n                command.extend(["--external-protocol", str(args.external_protocol)])\n'+anchor)
            new=new.replace('if not (args.skip_existing and report_path.exists()):','if args.external_protocol or not (args.skip_existing and report_path.exists()):')
        target=output/path.name;write_immutable(output/(role+'_source_identity.json'),dict(original=file_record(path)))
        if target.exists() and target.read_text('utf-8')!=new:raise ValueError('Patched source changed')
        if not target.exists():target.write_text(new,'utf-8')
        diffs.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/'+path.name,tofile='external/'+path.name))
        records[role]=file_record(target)
    diffpath=output/'external_protocol.patch'
    if diffpath.exists() and diffpath.read_text('utf-8')!=''.join(diffs):raise ValueError('Patch diff changed')
    if not diffpath.exists():diffpath.write_text(''.join(diffs),'utf-8')
    records['diff_path']=str(diffpath);write_immutable(output/'patch_identity.json',records);return records

def prepare_protocol(projection,split,phase,output,*,smoke=False,common_mask=None):
    validate_kge_split(split)
    if projection['projection_id']!=digest({k:v for k,v in projection.items() if k!='projection_id'}) or projection['split_id']!=split['split_id']:raise ValueError('Projection hash/split differs')
    if projection['test']!=[split['groups'][g]['triple'] for g in split['folds']['test']]:raise ValueError('External test changed')
    if common_mask is not None and (len(common_mask)!=len(projection['test']) or any(m and not c for m,c in zip(common_mask,projection['covered_test_mask']))):raise ValueError('Invalid common coverage mask')
    from .protocol import bound_input
    bound_input(phase,'projection',projection);bound_input(phase,'fact_split',split)
    if common_mask is not None:
        bound_input(phase,'common_mask',read_document(phase['files']['common_mask']['path']))
        declared=read_document(phase['files']['common_mask']['path'])
        if declared['mask']!=common_mask or projection['projection_id'] not in declared['projection_ids'] or declared['split_id']!=split['split_id']:raise ValueError('Common mask not frozen for projection')
    p=dict(projection=projection,phase_id=phase['phase_id'],phase_path=phase['path'] if 'path' in phase else None,budget=dict(BUDGET,epochs=[2] if smoke else BUDGET['epochs']),scope='smoke' if smoke else 'full',common_test_mask=common_mask)
    p['protocol_id']=digest(p);write_immutable(output,p);return p

def validate_protocol(value):
    if value['protocol_id']!=digest({k:v for k,v in value.items() if k!='protocol_id'}):raise ValueError('External protocol changed')
    projection=value['projection']
    if projection['projection_id']!=digest({k:v for k,v in projection.items() if k!='projection_id'}):raise ValueError('Projection changed')
    sets=[{canonical(*t) for t in projection[k]} for k in ('training','validation','test')]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('External split leakage')
    if value['budget']!=dict(BUDGET,epochs=[2] if value['scope']=='smoke' else [500,1000]):raise ValueError('External fixed budget changed')
    if not value.get('phase_path'):raise ValueError('External protocol requires frozen phase path')
    from .protocol import validate_phase
    phase=validate_phase(value['phase_path'])
    if phase['phase_id']!=value['phase_id']:raise ValueError('External phase mismatch')
    required={'graph','projection','fact_split','construction','gpu_environment','dependency_lock','src_utils','evaluator_original','sweep_original','evaluator_patched','sweep_patched'}
    if not required<=set(phase['files']):raise ValueError('External phase lacks actual evaluator/software/input assets')
    from .protocol import bound_input
    bound_input(phase,'projection',projection)
    split=read_document(phase['files']['fact_split']['path'])
    validate_kge_split(split)
    if projection['split_id']!=split['split_id'] or projection['test']!=[split['groups'][g]['triple'] for g in split['folds']['test']] or projection['validation']!=[split['groups'][g]['triple'] for g in split['folds']['validation']]:raise ValueError('External reference folds changed')
    if projection['reference_id']!=split['reference_id'] or projection['test_ids']!=split['folds']['test'] or projection['validation_ids']!=split['folds']['validation']:raise ValueError('External reference identity/ordered folds changed')
    mask=value.get('common_test_mask')
    if mask is not None:
        if len(mask)!=len(projection['test']) or any(type(m) is not bool or m and not c for m,c in zip(mask,projection['covered_test_mask'])):raise ValueError('Invalid common coverage mask')
        if 'common_mask' not in phase['files']:raise ValueError('Common selection not phase bound')
        declared=read_document(phase['files']['common_mask']['path'])
        if declared['mask']!=mask or projection['projection_id'] not in declared['projection_ids'] or declared['split_id']!=split['split_id']:raise ValueError('Common selection differs from frozen declaration')
    if split.get('format')=='sanitized_kge_split':
        if 'primary_attestation' not in phase['files']:raise ValueError('Primary sanitized export attestation required')
        a=read_document(phase['files']['primary_attestation']['path'])
        if a['attestation_id']!=digest({k:v for k,v in a.items() if k!='attestation_id'}) or any(a[k]!=projection[k] for k in ('graph_id','projection_id','split_id','reference_id')) or a['sanitized_split_digest']!=digest(split) or a['original_split_digest']!=split['original_split_digest']:raise ValueError('Primary export identity differs')
    if phase['files']['evaluator_original']['sha256']!=EVALUATOR_SHA or phase['files']['sweep_original']['sha256']!=SWEEP_SHA:raise ValueError('Wrong existing evaluator version')
    return value

def run_external(inputs,args,report_dir):
    p=validate_protocol(read_document(args.external_protocol));projection=p['projection'];budget=p['budget']
    from .protocol import validate_phase
    phase=validate_phase(p['phase_path'])
    utils_path=getattr(args,'_course_evaluator_utils_path',None)
    if utils_path is None:
        import src.utils
        utils_path=src.utils.__file__
    if file_record(utils_path)!=phase['files']['src_utils']:raise ValueError('Actual src.utils not frozen')
    graph=inputs.concept_kg
    if graph.get('format')=='kge_graph_attestation':
        from .protocol import validate_phase,bound_input
        phase=validate_phase(p['phase_path']);bound_input(phase,'graph',graph)
        a=read_document(phase['files']['primary_attestation']['path'])
        if graph['graph_id']!=projection['graph_id'] or graph['projection_id']!=projection['projection_id'] or graph['primary_attestation_id']!=a['attestation_id']:raise ValueError('Sanitized graph identity mismatch')
    elif digest(graph)!=projection['graph_id']:raise ValueError('External graph identity mismatch')
    if args.seed!=42 or args.pykeen_model!=budget['model'] or args.pykeen_epochs not in budget['epochs']:raise ValueError('Caller changed frozen KGE options')
    testmask=p.get('common_test_mask') if p.get('common_test_mask') is not None else projection['covered_test_mask']
    testing=[t for t,covered in zip(projection['test'],testmask) if covered]
    validation=[t for t,covered in zip(projection['validation'],projection['covered_validation_mask']) if covered]
    training=projection['training'];report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol_id=p['protocol_id'],phase_id=p['phase_id'],graph_id=projection['graph_id'],split_id=projection['split_id'],epochs=args.pykeen_epochs,requested_device=args.pykeen_device,options=budget)
    identity['run_id']=digest(identity)
    write_immutable(report_dir/'external_kge_identity.json',identity)
    write_immutable(report_dir/'external_kge_protocol.json',p)
    summary=dict(enabled=True,run_id=identity['run_id'],protocol_id=p['protocol_id'],split_id=projection['split_id'],graph_id=projection['graph_id'],scope=p['scope'],test_ids=[x for x,m in zip(projection['test_ids'],testmask) if m],fixed_test_count=len(projection['test']),covered_test_count=sum(projection['covered_test_mask']),common_test_count=sum(testmask),condition_projection=projection['condition_projection'],options=budget)
    result_path=report_dir/'external_kge_result.json'
    if result_path.exists():
        old=read_document(result_path)
        if old.get('run_id')!=identity['run_id']:raise ValueError('KGE result belongs to different run identity')
        if old['status'] in {'completed','completed_with_diagnostic_failure'}:return old
        raise RuntimeError('Prior failed/insufficient attempt retained; use a new report directory after diagnosis')
    def publish(value):
        write_immutable(result_path,value);return value
    def failure(stage,exc,training_status):
        artifacts=[str(f) for f in sorted(report_dir.rglob('*')) if f.is_file()]
        value=dict(summary,status='failed',failure_stage=stage,error_type=type(exc).__name__,error_message=str(exc),traceback=traceback.format_exc(),training_status=training_status,existing_artifact_paths=artifacts)
        path=report_dir/('failure_'+stage+'.json');write_immutable(path,value)
        value['failure_record']=str(path);return value
    if len(training)<10 or not testing or not validation:return publish(dict(summary,status='insufficient',reason='Needs >=10 train triples and nonempty eligible fixed validation/test; no score'))
    try:
        import numpy as np
        import torch
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError as exc:return publish(dict(summary,status='missing_dependency',reason=str(exc),error_type='ImportError',failure_stage='dependency_import'))
    entity_to_id={e:i for i,e in enumerate(projection['entity_vocabulary'])};relation_to_id={r:i for i,r in enumerate(projection['relation_vocabulary'])}
    def factory(triples):return TriplesFactory.from_labeled_triples(np.array(triples,dtype=str),entity_to_id=entity_to_id,relation_to_id=relation_to_id,create_inverse_triples=False)
    stage='factories';training_status='not_started'
    try:
        write_immutable(report_dir/'stage_factories.json',dict(run_id=identity['run_id'],stage=stage))
        training_factory,validation_factory,testing_factory=map(factory,(training,validation,testing))
        device=args.pykeen_device
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        stage='training';training_status='running'
        write_immutable(report_dir/'stage_training.json',dict(run_id=identity['run_id'],stage=stage,device=str(device)))
        result=pipeline(training=training_factory,validation=validation_factory,testing=testing_factory,model=args.pykeen_model,
            model_kwargs={'embedding_dim':budget['dimension']},training_kwargs={'num_epochs':args.pykeen_epochs,'batch_size':budget['batch_size']},
            negative_sampler_kwargs={'num_negs_per_pos':budget['negative_samples']},optimizer_kwargs={'lr':budget['learning_rate']},random_seed=args.seed,device=device)
        training_status='completed';stage='model_save'
        write_immutable(report_dir/'stage_model_save.json',dict(run_id=identity['run_id'],stage=stage,training_status=training_status))
        result.save_to_directory(report_dir/'model')
        stage='metrics_save';metrics=result.metric_results.to_dict();write_immutable(report_dir/'pykeen_metrics.json',metrics)
    except Exception as exc:
        # 外部训练边界保存精确错误及已生成文件后原样抛出，不删除模型或掩盖失败。
        publish(failure(stage,exc,training_status));raise
    completed=dict(summary,status='completed',training_status='completed',diagnostics_status='completed',model=args.pykeen_model,epochs=args.pykeen_epochs,device=str(device),train_triples=len(training),validation_triples=len(validation),test_triples=len(testing),metrics_file=str(report_dir/'pykeen_metrics.json'),model_path=str(report_dir/'model'),raw_paths=[str(report_dir/'external_kge_protocol.json')])
    try:
        write_immutable(report_dir/'stage_rank_diagnostics.json',dict(run_id=identity['run_id'],stage='rank_diagnostics',training_status='completed'))
        # 使用原生 rank evaluator 可选导出逐三元组评估，不自行重写排序规则。
        from pykeen.evaluation import RankBasedEvaluator
        evaluator=RankBasedEvaluator(clear_on_finalize=False)
        evaluator.evaluate(model=result.model,mapped_triples=testing_factory.mapped_triples,additional_filter_triples=[training_factory.mapped_triples,validation_factory.mapped_triples],device=device)
        if hasattr(evaluator,'ranks'):
            rank_payload={str(k):[np.asarray(a).tolist() for a in v] for k,v in evaluator.ranks.items()}
            write_immutable(report_dir/'per_triple_ranks.json',dict(test_ids=summary['test_ids'],ranks=rank_payload,ordering='PyKEEN side/rank_type batches in evaluation order'))
            completed['raw_paths'].append(str(report_dir/'per_triple_ranks.json'))
        else:completed['diagnostics_status']='unavailable'
    except Exception as exc:
        diagnostic=failure('rank_diagnostics',exc,'completed')
        completed.update(status='completed_with_diagnostic_failure',diagnostics_status='failed',diagnostic_failure=diagnostic['failure_record'])
        completed['raw_paths'].append(diagnostic['failure_record'])
    return publish(completed)
