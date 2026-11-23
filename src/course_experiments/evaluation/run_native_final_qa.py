"""固定终评薄包装：冻结前模板不读新题，复用完整原生索引及官方问答。"""
from __future__ import annotations
import argparse,copy,importlib.util,json
from pathlib import Path
from ..manifest import digest,file_record,read_document

def checked(record):
    if file_record(record['path'])!=record:raise ValueError('Frozen final QA dependency changed')
    return read_document(record['path'])

def verify_file(record):
    if file_record(record['path'])!=record:raise ValueError('Frozen final QA code or asset changed')
    return record

def load_legacy123(sealed_record,selection_record):
    from .run_native_holdout import load_holdout
    verify_file(sealed_record);verify_file(selection_record)
    data=load_holdout(sealed_record['path'],selection_record['path'])
    if data['seal']['accepted_count']!=123:raise ValueError('Existing independent123 actual denominator changed')
    return data

def asset_records(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:return [value]
        return [record for item in value.values() for record in asset_records(item)]
    if isinstance(value,list):return [record for item in value for record in asset_records(item)]
    return []

def verify_preseal_assets(template,*,template_record=None,frozen=None):
    code=[template['worker'],*template['code_assets']];parameters=asset_records(template['parameters'])
    for record in code+parameters:verify_file(record)
    if frozen is None:return
    matches=[method for method in frozen['methods'] if method['method_id']==template['registered_method_id']]
    if len(matches)!=1:raise ValueError('Native target method is absent or duplicated in final freeze')
    method=matches[0]
    if template_record not in method['code'] or any(record not in method['code'] for record in code):
        raise ValueError('Native preseal template or execution code is not registered under its target frozen method')
    if any(record not in method['parameters'] for record in parameters):raise ValueError('Native parameters are not registered under their target frozen method')

def verify_execution_assets(base,template):
    from . import run_native_holdout,native_provider
    required=[file_record(__file__),file_record(run_native_holdout.__file__),file_record(native_provider.__file__),*base['contract']['code'].values()]
    if template['recipe']['kind']=='worker':required.append(template['recipe']['worker'])
    declared=[template['worker'],*template['code_assets']]
    for record in required:
        verify_file(record)
        if record not in declared:raise ValueError('An actual native preparation/query/judge/transport dependency is absent from the frozen code manifest')

def validate_preseal_template(spec):
    if spec.get('version')!='native_final_QA_preseal_v1':raise ValueError('Explicit native final QA preseal template required')
    if any(k in spec for k in ['sealed','algorithm_freeze','questions','dataset_index']):
        raise ValueError('Preseal template must not depend on a future seal, freeze or questions')
    if spec.get('method') not in ['hipporag','lightrag','graphrag'] or spec.get('split') not in ['legacy123','blind64','regression313']:
        raise ValueError('Explicit original native method and fixed split required')
    if spec['split']=='regression313' and (spec['method']!='graphrag' or spec['track']!='global'):raise ValueError('Old313 addition is only the missing GraphGlobal row')
    allowed=['local','global'] if spec['method']=='graphrag' else ['native']
    if spec.get('track') not in allowed:raise ValueError('Native method track differs from registered row')
    for key in ['book','app','python','recipe','worker','parameters','index_pin']:
        if key not in spec:raise ValueError('Incomplete native final QA template')
    return spec

def bind_final_plan(base,books,records,dependency,*,split,track,dataset_metadata):
    method,book=base['method'],base['book']
    if set(books)!=set(base['allowed_books']) or book not in books:raise ValueError('Final common dataset has foreign or missing books')
    if dependency['index_identity']!=digest(base['index_contract']) or base['contract']['index_identity']!=dependency['index_identity']:
        raise ValueError('Final QA changed the pinned native index identity')
    if track not in (['local','global'] if method=='graphrag' else ['native']):raise ValueError('Wrong native final QA track')
    if split=='regression313' and (method!='graphrag' or track!='global'):raise ValueError('Do not rerun completed native regression rows')
    ids=[q['question_id'] for value in books.values() for q in value['questions']]
    if len(ids)!=len(set(ids)):raise ValueError('Final common dataset has duplicate questions')
    name={'legacy123':'independent_confirmation','blind64':'blind64_confirmation','regression313':'confirmation'}[split]
    plan=dict(base);contract=copy.deepcopy(base['contract']);plan['datasets']={name:books[book]}
    contract['datasets']={name:records[book]};contract['global_question_counts']={b:{name:len(v['questions'])} for b,v in books.items()}
    contract['final_fixed_QA']=dict(split=split,track=track,dataset_metadata=dataset_metadata,reused_index_dependency=dependency,
        wrapper=file_record(__file__),index_rebuild_allowed=False,questions_shared_across_all_methods=True,
        failed_question_policy='fixed_all_sealed_questions_incomplete_zero_no_silent_drop')
    contract['policy']=dict(contract['policy'],final_quality_evaluation='same_fixed_sealed_dataset_native_official_query_common_judge')
    contract['code']['run_native_final_qa.py']=file_record(__file__)
    if method=='graphrag':
        contract['tracks']=[track]
        contract['policy']['microsoft_global']='separate_final_QA_row_official_global_query_same_full_faithful_index'
    plan['contract']=contract
    return plan

def completed_dependency(base,run_root,contract_record,complete_record):
    from . import run_native_full as shared
    from .run_native_holdout import DependencyPending,alive
    c=checked(contract_record);result=checked(complete_record);identity=digest(base['index_contract']);root=Path(run_root)/'native_indices'/base['method']/base['book']/identity[:16]
    if c.get('index_identity')!=identity or c.get('method')!=base['method'] or c.get('book')!=base['book']:
        raise ValueError('Full native QA dependency binds a different index or book')
    if c.get('job_identity')!=digest({k:v for k,v in c.items() if k!='job_identity'}) or result.get('job_identity')!=c['job_identity']:
        raise ValueError('Completed native QA dependency digest changed')
    if result.get('status')!='completed':raise DependencyPending('Original native QA is incomplete')
    tracks=result.get('tracks') if base['method']=='graphrag' else {'native':result}
    expected_tracks=set(c.get('tracks',[])) if base['method']=='graphrag' else {'native'}
    if set(tracks)!=expected_tracks:raise ValueError('Original native QA track denominator changed')
    for value in tracks.values():
        splits=value.get('splits',{});counts=c['global_question_counts'][base['book']]
        if set(splits)!=set(counts) or any(splits[k].get('completed_count')!=n or splits[k].get('question_count')!=n or splits[k].get('failed_count') or splits[k].get('pending_count') for k,n in counts.items()):
            raise DependencyPending('Original native QA has unfinished fixed questions')
    state_path=Path(complete_record['path']).parent/'state.json';state=read_document(state_path)
    if alive(state.get('pid')):raise DependencyPending('Original native worker still alive; index reuse waits')
    index_contract=read_document(root/'contract.json')
    if index_contract!={'index_identity':identity,**base['index_contract']}:raise ValueError('Pinned original full native index contract differs')
    receipt=shared._read_checkpoint(root/'index_receipt.json',identity)
    return dict(index_root=str(root),index_identity=identity,native_index=receipt['native_index'],index_receipt=file_record(root/'index_receipt.json'),
        index_contract=file_record(root/'contract.json'),regression_contract=contract_record,regression_result=complete_record,worker_exit_state=file_record(state_path),worker_pid=state.get('pid'))

def load_blind64(sealed_record,freeze_record):
    from .blind64_pipeline import require_freeze,POLICY
    _,frozen=require_freeze(freeze_record['path'],freeze_record['sha256']);seal=checked(sealed_record)
    if seal.get('status')!='sealed_complete' or seal.get('target_count')!=64 or seal.get('accepted_count')!=64 or seal.get('shortage')!=0 or seal.get('released_for_quality_evaluation') is not True:
        from .run_native_holdout import DependencyPending
        raise DependencyPending('All64 new questions must be sealed after final algorithm freeze')
    contract=checked(seal['contract'])
    if contract.get('algorithm_freeze')!=frozen:raise ValueError('Blind64 generation used a different algorithm freeze')
    index=checked(seal['dataset_index']);books={b:checked(r) for b,r in index['books'].items()}
    targets=POLICY['quotas']
    if len(books)!=8 or sum(len(v['questions']) for v in books.values())!=64:raise ValueError('Fixed new64 eight-book denominator changed')
    for b,value in books.items():
        if value['target_book_id']!=b or len(value['questions'])!=8:raise ValueError('New64 original book scope changed')
        counts={kind:sum(q['type']==kind for q in value['questions']) for kind in targets}
        if counts!=targets:raise ValueError('New64 fixed3/3/2 type counts changed')
    return books,index['books'],dict(sealed=sealed_record,algorithm_freeze=frozen,dataset_index=seal['dataset_index'],accepted_count=64,target_count=64,shortage=0)

def prepare_from_spec(spec):
    template=checked(spec['template']);validate_preseal_template(template)
    verify_preseal_assets(template)
    if template['split']=='blind64':
        from .blind64_pipeline import require_freeze
        frozen,_=require_freeze(spec['algorithm_freeze']['path'],spec['algorithm_freeze']['sha256'])
        verify_preseal_assets(template,template_record=spec['template'],frozen=frozen)
    recipe=template['recipe'];run_root=Path(template['run_root']);method,book=template['method'],template['book']
    if recipe['kind']=='faithful_graph':
        from .run_native_microsoft_faithful import prepare_plan
        base=prepare_plan(run_root,book)
    elif recipe['kind']=='worker':
        r=recipe['worker'];verify_file(r);s=importlib.util.spec_from_file_location('frozen_native_final_recipe',r['path']);module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
        base=module.prepare(*recipe['prepare_args'])
    elif recipe['kind']=='original':
        from .run_native_full import prepare_plan
        base=prepare_plan(run_root,method,book)
    else:raise ValueError('Unknown native index recipe')
    if digest(base['index_contract'])!=template['index_pin']['index_identity']:raise ValueError('Frozen native recipe did not reproduce the exact original index')
    verify_execution_assets(base,template)
    if template['parameters']['runtime']!=base['contract']['runtime_config'] or template['parameters']['official_index_parameters']!={'index_identity':digest(base['index_contract']),**base['index_contract']}:
        raise ValueError('Native execution parameters differ from the actual frozen recipe')
    dep=completed_dependency(base,run_root,spec['regression_contract'],spec['regression_complete'])
    if dep['index_receipt']!=spec['index_receipt'] or dep['index_contract']!=spec['index_contract']:raise ValueError('Final QA dependency differs from immutable index pins')
    if template['split']=='legacy123':
        data=load_legacy123(spec['sealed'],spec['selection'])
        books,records=data['books'],data['book_records'];metadata=dict(sealed=spec['sealed'],selection=spec['selection'],accepted_count=123,target_count=128,shortage=5,dataset_index=data['seal']['dataset_index'])
        from .dual_view_tasks import validate_task
        for q in books[book]['questions']:
            canonical=validate_task(q,base['structure']['evidence'],book,q['type'])
            if any(q.get(k)!=v for k,v in canonical.items()):raise ValueError('Existing independent source-grounded question changed')
    elif template['split']=='blind64':
        books,records,metadata=load_blind64(spec['sealed'],spec['algorithm_freeze'])
        from .blind64_pipeline import _validated_task
        for q in books[book]['questions']:
            canonical=_validated_task(q,base['structure']['evidence'],book,q['type'])
            if any(q.get(k)!=v for k,v in canonical.items()):raise ValueError('Blind64 original source question changed')
    else:
        index=checked(base['contract']['dataset_index']);records={b:r['confirmation'] for b,r in index['books'].items()};books={b:checked(r) for b,r in records.items()}
        if len(books)!=8 or sum(len(v['questions']) for v in books.values())!=313 or any(not 37<=len(v['questions'])<=40 for v in books.values()):raise ValueError('Exact original313 eight-book confirmation denominator required')
        if records[book]!=base['contract']['datasets']['confirmation'] or books[book]!=base['datasets']['confirmation']:raise ValueError('Original confirmation records differ')
        metadata=dict(dataset_index=base['contract']['dataset_index'],accepted_count=313,by_book_counts={b:len(v['questions']) for b,v in books.items()},already_complete_other_native_regression_rows_not_rerun=True)
    plan=bind_final_plan(base,books,records,dep,split=template['split'],track=template['track'],dataset_metadata=metadata)
    return template,plan,dep

async def run_existing(plan,run_root,provider,dependency):
    from . import run_native_holdout as existing
    if plan['method']!='graphrag':return await existing.run_existing(plan,run_root,provider,dependency)
    from . import run_native_microsoft_full as official
    from .run_native_microsoft_faithful import query
    from unittest.mock import patch
    # 单进程只选一条官方轨道，沿用同一完整索引与官方query。
    with patch.object(official,'TRACKS',tuple(plan['contract']['tracks'])),patch.object(existing,'query_with_source_mapping',query):
        return await existing.run_existing(plan,run_root,provider,dependency)
