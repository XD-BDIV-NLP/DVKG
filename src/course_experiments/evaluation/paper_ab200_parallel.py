"""只拆分执行调度，保留原200题结果身份和累计额度。"""
from ..manifest import digest

VARIANTS={'own':{'ECHA_original':'echa','ECHA_recall_guard':'echa'},
          'lexical':{'BM25':'keyword','BGE-M3':'vector','Hybrid':'hybrid'}}

def own_book_tasks(books):
    if len(books)!=2 or set(books)!={'book_25f718562685','book_49bb8d31ddb9'}:
        raise ValueError('Full own experiment must retain exactly the authorized AB books')
    return [dict(variant=v,book=b,kind='own') for v in VARIANTS['own'] for b in books]

def selected_variant(kind,label):
    if kind not in VARIANTS or label not in VARIANTS[kind]:
        raise ValueError('Variant does not belong to its original method group')
    return label,VARIANTS[kind][label]

def resumed_result_id(original,label,refs):
    if original['identity']!=digest({k:v for k,v in original.items() if k!='identity'}):
        raise ValueError('Original AB200 job identity changed')
    return digest(dict(job=original['identity'],label=label,refs=refs))
