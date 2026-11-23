"""只并发原始八对批次，不改请求内容、核验规则或输出顺序。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def verify(original, provider, graph, structure, plans, root, stage, *, workers=4):
    if type(workers) is not int or not 1<=workers<=4:
        raise ValueError('Use at most four inherited shared request slots')
    root=Path(root)
    digest=original.__globals__['digest']
    seal=original.__globals__['write_immutable']
    seal(root/'full_original_contract.json',dict(graph=digest(graph),structure=digest(structure),
         plans=digest(plans),policy=original.__globals__['POLICY'],workers=workers,
         batches='original contiguous eight pairs; unchanged request body and repair rule'))
    pairs=[(row['concept_id'],candidate) for row in plans for candidate in row['candidates']]
    batches=[pairs[first:first+8] for first in range(0,len(pairs),8)]
    def one(item):
        ordinal,batch=item
        subset=[]
        for cid,candidate in batch:
            if not subset or subset[-1]['concept_id']!=cid:
                subset.append(dict(concept_id=cid,candidates=[]))
            subset[-1]['candidates'].append(candidate)
        result=original(provider,graph,structure,subset,root/'batches'/str(ordinal),stage)
        if [(r['concept_id'],r['evidence_id']) for r in result]!=[(cid,c['evidence_id']) for cid,c in batch]:
            raise ValueError('Original batch result order or coverage changed')
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [row for batch in pool.map(one,enumerate(batches)) for row in batch]
