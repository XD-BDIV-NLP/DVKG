"""补充任务仅接受独立原文与必要性审查，不读取待测图。"""
from __future__ import annotations
from .manifest import digest


def eligible(rows,excluded_chunks,excluded_evidence):
    return [r for r in rows if r['chunk_id'] not in excluded_chunks and
        not ({e['id'] for e in r['evidence']} & excluded_evidence)]


def validate(value,rows,kind):
    sources={e['id']:(r,e) for r in rows for e in r['evidence']}
    if not isinstance(value.get('question'),str) or not value['question'].strip() or not isinstance(value.get('answer'),str) or not value['answer'].strip():
        raise ValueError('Missing grounded question/answer')
    citations=value.get('citations',[])
    if not citations:raise ValueError('Missing exact citations')
    for c in citations:
        if not isinstance(c,dict) or c.get('evidence_id') not in sources or not isinstance(c.get('quote'),str) or len(c['quote'])<4 or c['quote'] not in sources[c['evidence_id']][1]['text']:
            raise ValueError('Citation is not exact complete source substring')
    cited={c['evidence_id'] for c in citations};chunks={sources[e][0]['chunk_id'] for e in cited}
    if kind=='cross_source' and len(chunks)<2:raise ValueError('Cross-source requires two source chunks')
    if kind=='conditions' and not all(isinstance(value.get(k),str) and value[k].strip() for k in ['condition','applicable_case','inapplicable_case']):
        raise ValueError('Explicit condition and contrasting cases required')
    result=dict(value,type=kind,evidence_ids=sorted(cited),source_chunk_ids=sorted(chunks),
        source_book_ids=sorted({sources[e][0]['book_id'] for e in cited}),human_audited=False)
    result['question_id']='supp_'+digest(result)
    return result
