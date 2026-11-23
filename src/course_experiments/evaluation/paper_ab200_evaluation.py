"""新增 200 题的薄绑定；原索引、方法和生成参数不变。"""
from __future__ import annotations
import copy
from pathlib import Path
from ..manifest import digest,file_record,read_document
from ..contributions import write_immutable

BOOKS=('book_25f718562685','book_49bb8d31ddb9')
TYPES=('definition','paraphrase','conditions','cross_chapter')

def validate_books(books):
    if set(books)!=set(BOOKS):raise ValueError('Exact AB dataset required')
    ids=[]
    for book,dataset in books.items():
        questions=dataset['questions']
        if dataset['target_book_id']!=book or len(questions)!=100 or any(q['target_book_id']!=book for q in questions):
            raise ValueError('Each target textbook requires exactly 100 questions')
        if {k:sum(q['type']==k for q in questions) for k in TYPES}!={k:25 for k in TYPES}:
            raise ValueError('Each textbook requires four exact quotas of25')
        ids.extend(q['question_id'] for q in questions)
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate question identifier')
    return len(ids)

def checked(ref):
    if file_record(ref['path'])!=ref:raise ValueError('Frozen AB200 dependency changed')
    return read_document(ref['path'])

def release(generation,output,wrapper):
    from . import dual_view_tasks as tasks
    from .run_independent_holdout import validate_audit_quotes
    generation=Path(generation);sealed=read_document(generation/'sealed.json')
    if sealed['status']!='sealed_complete' or sealed['accepted_count']!=200 or sealed['shortage']!=0:
        raise ValueError('Only the complete sealed200 may enter evaluation')
    contract=checked(sealed['contract']);index=checked(sealed['dataset_index']);books={b:checked(ref) for b,ref in index['books'].items()};validate_books(books)
    structures={b:checked(contract['source_records'][b]['structure']) for b in BOOKS};accepted={}
    for b,data in books.items():
        evidence=structures[b]['evidence']
        for ref in data['decisions']:
            d=checked(ref)
            if d['decision_digest']!=digest({k:v for k,v in d.items() if k!='decision_digest'}):raise ValueError('Candidate audit digest changed')
            if not d['accepted']:continue
            q=d['question'];canonical=tasks.validate_task(q,evidence,b,q['type'])
            if canonical!=q:raise ValueError('Question differs from original source validation')
            audit=d['source_audit']
            if audit.get('supported')is not True or audit.get('coherent')is not True or q['type']=='conditions' and audit.get('explicit_conditions')is not True:
                raise ValueError('Source support audit incomplete')
            validate_audit_quotes(audit,evidence,b,required=q['evidence_ids'])
            if q['type']=='cross_chapter':
                if set(d['necessity'])!=set(q['chapter_ids']):raise ValueError('Cross chapter audit incomplete')
                for chapter,a in d['necessity'].items():
                    if a.get('answerable')is not False:raise ValueError('Cross chapter evidence not necessary')
                    validate_audit_quotes(a,{e:r for e,r in evidence.items() if e in d['source_bundle'] and r['chapter_id']!=chapter},b)
            for receipt in d['model_checkpoints']:checked(receipt)
            accepted[q['question_id']]=q
        if any(accepted.get(q['question_id'])!=q for q in data['questions']):raise ValueError('Dataset differs from accepted source audits')
    if len(accepted)!=200:raise ValueError('Exact200 accepted source audits required')
    registry=contract['method_registry']
    for name in ('own_context','controlled_K12','native_pins','recall_guard_candidate','frozen_generation_worker'):
        if file_record(registry[name]['path'])!=registry[name]:raise ValueError('Pre-generation method registration changed')
    value=dict(status='released_complete_AB200_for_same_questions_evaluation',sealed=file_record(generation/'sealed.json'),
        dataset_index=sealed['dataset_index'],datasets=index['books'],accepted_count=200,by_book={b:100 for b in BOOKS},
        registry=registry,source_contract=sealed['contract'],wrapper=wrapper,method_parameters_changed=False,
        old80_not_in_new200=True,human_audited=False,source_checks='all200_exact_quotes_and_support_audits;cross_chapter_necessity')
    write_immutable(output,value);return value

def bind_native(base,books,refs,sealed,wrapper):
    validate_books(books);book=base['book']
    if book not in BOOKS:raise ValueError('Foreign native target textbook')
    plan=dict(base);contract=copy.deepcopy(base['contract'])
    plan['datasets']={'paper200_confirmation':books[book]}
    contract['datasets']={'paper200_confirmation':refs[book]}
    contract['global_question_counts']={b:{'paper200_confirmation':100} for b in BOOKS}
    contract['dataset_index']=sealed
    contract['paper200_evaluation']=dict(sealed=sealed,wrapper=wrapper,original_index_unchanged=True,
        same_questions_for_all_methods=True,protocol='official_native_separate_table',target_count=200)
    if base['method']=='graphrag':contract['tracks']=['local']
    plan['contract']=contract
    return plan
