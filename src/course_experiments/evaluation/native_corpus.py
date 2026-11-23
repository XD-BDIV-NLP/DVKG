"""把细粒度引用段恢复成原始来源片段，不删正文，不合并不同来源。"""
from collections import defaultdict

from ..manifest import digest


POLICY = dict(version='native_original_source_chunks_v1',
    grouping='exact_target_book_and_original_chunk_id',
    ordering='original_paragraph_order_then_evidence_id',separator='\n',
    truncation=False,summary=False,answer_information=False)


def native_passages(rows, structure, book_id):
    groups=defaultdict(list);seen=set();characters=0
    for row in rows:
        ids=row['evidence_ids']
        if row['book_id']!=book_id or len(ids)!=1 or ids[0] in seen:
            raise ValueError('Native corpus expects unique single-evidence target-book units')
        eid=ids[0];source=structure['evidence'][eid]
        if row['text']!=source['text'] or row['chunk_id']!=source['chunk_id']:
            raise ValueError('Native corpus evidence differs from frozen structure source')
        order=source['source'].get('paragraph_order')
        if type(order) is not int:
            raise ValueError('Original paragraph order is required for lossless passage restoration')
        seen.add(eid);characters+=len(row['text'])
        groups[row['chunk_id']].append((order,eid,source))
    result=[]
    for chunk_id,values in sorted(groups.items()):
        spans=[];parts=[];offset=0;chapters=set()
        for _,eid,source in sorted(values,key=lambda r:(r[0],r[1])):
            text=source['text'];parts.append(text);chapters.add(source['chapter_id'])
            spans.append(dict(evidence_id=eid,start=offset,end=offset+len(text)))
            offset+=len(text)+1
        result.append(dict(book_id=book_id,chunk_id=chunk_id,text='\n'.join(parts),
            evidence_ids=[s['evidence_id'] for s in spans],evidence_spans=spans,
            chapter_ids=sorted(chapters),chapter_id=next(iter(chapters)) if len(chapters)==1 else None))
    restored={e for r in result for e in r['evidence_ids']}
    if restored!=seen:raise AssertionError('Native passage restoration lost evidence')
    return result,dict(policy=POLICY,target_book_id=book_id,source_unit_count=len(rows),
        native_passage_count=len(result),source_characters=characters,
        all_evidence_covered=True,evidence_identifiers_digest=digest(sorted(seen)),
        restored_passages_digest=digest(result))
