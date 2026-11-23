from __future__ import annotations

import copy
from .manifest import digest


def _validate(value):
    if value.get('format_version') != 1 or value.get('artifact_id') != digest({k:v for k,v in value.items() if k!='artifact_id'}):
        raise ValueError('Invalid contribution digest or format')
    if value['scope'] not in {'full','smoke'}:
        raise ValueError('Invalid contribution scope')
    prefix = f"{value['course_id']}/{value['book_id'] or 'g0'}"
    graph = value['graph']
    for table, singular in (('nodes','node'),('claims','claim'),('evidence','evidence'),('books','book')):
        for key,row in graph[table].items():
            if key != f"{prefix}/{singular}/{row['native_id']}" or row['id'] != key:
                raise ValueError('Invalid namespace')
        if value['native_id_map'][table] != {row['native_id']:key for key,row in graph[table].items()}:
            raise ValueError('Invalid native identity mapping')
    for node in graph['nodes'].values():
        ids = [eid for key in ('evidence_ids','definition_evidence_ids','mention_evidence_ids') for eid in node.get(key, [])]
        ids += [eid for definition in node.get('definitions', []) for eid in definition.get('evidence_ids', [])]
        if not set(ids) <= graph['evidence'].keys():
            raise ValueError('Unknown node evidence')
    supports = {row['claim_id']:row for row in value['supports']}
    if len(supports) != len(value['supports']) or supports.keys() != graph['claims'].keys():
        raise ValueError('Invalid support ledger')
    for claim in graph['claims'].values():
        support = supports[claim['id']]
        if (support['evidence_ids'] != claim.get('evidence_ids',[]) or
                support['book_id'] != value['book_id'] or
                support['native_claim_id'] != claim['native_id']):
            raise ValueError('Support ledger mismatch')
        if not set(claim.get('evidence_ids',[])) <= graph['evidence'].keys():
            raise ValueError('Unknown claim evidence')
        for key in ('head','tail','node_id'):
            if key in claim['content'] and claim['content'][key] not in graph['nodes']:
                raise ValueError('Unknown claim endpoint')


def build_view(g0: dict, contributions: list[dict], active_books: list[str], *,
               alignment: dict | None = None, runtime=None) -> dict:
    _validate(g0)
    if g0['book_id'] is not None or g0['scope'] != 'full':
        raise ValueError('G0 must be a prior contribution')
    books = {}
    for value in contributions:
        _validate(value)
        if value['course_id'] != g0['course_id']:
            raise ValueError('Course mismatch')
        if value['book_id'] is None or value['book_id'] in books:
            raise ValueError('Duplicate or missing book identity')
        books[value['book_id']] = value
    active = sorted(set(active_books))
    if not set(active) <= books.keys():
        raise ValueError('Unknown active book')
    scopes = {books[b]['scope'] for b in active}
    if len(scopes) > 1:
        raise ValueError('Selected textbook scope mismatch')
    scope = next(iter(scopes), 'full')
    selected = [g0] + [books[b] for b in active]
    native = {v['book_id'] or 'g0':copy.deepcopy(v) for v in selected}
    # 原生事件仅保留一份，各事实通过摘要引用同一贡献的完整历史。
    audit_refs = {key:dict(artifact='fused.json',contribution_key=key,
        artifact_id=value['artifact_id'],events_digest=digest(value['graph'].get('events',[])),
        event_count=len(value['graph'].get('events',[]))) for key,value in native.items()}
    nodes = {k:copy.deepcopy(r) for v in selected for k,r in v['graph']['nodes'].items()}
    evidence = {k:copy.deepcopy(r) for v in selected for k,r in v['graph']['evidence'].items()}
    claims = {k:copy.deepcopy(r) for v in selected for k,r in v['graph']['claims'].items()}
    audit=[]
    decisions={k:v for k,v in (alignment or {}).items() if k in nodes}
    if runtime is not None:
        if alignment:
            raise ValueError('Choose runtime or fixture alignment')
        from domain_ckg_workspace.ckg_incremental.semantic_merge import SemanticMerger
        merger=SemanticMerger(runtime,evidence,{k:r for v in selected for k,r in v['graph']['books'].items()})
        decisions,audit=merger.deduplicate(dict(nodes=nodes,claims=claims))
    else:
        audit=[dict(source_id=k,canonical_id=v,decision='merge',method='explicit_fixture') for k,v in sorted(decisions.items())]
    parent={k:k for k in nodes}
    cluster_types={k:{row['entity_type']} for k,row in nodes.items()}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    prior=set(g0['graph']['nodes'])
    for source,target in sorted(decisions.items()):
        if target not in nodes:raise ValueError('Unknown alignment endpoint')
        a,b=root(source),root(target)
        if a==b:continue
        types=cluster_types[a] | cluster_types[b]
        # 与原生合并器一致：通用 Concept 可补充单个非 Metric 的具体类型。
        compatible=(len(types)==1 or len(types)==2 and 'Concept' in types and 'Metric' not in types)
        if not compatible:
            if runtime is None:
                raise ValueError('Alignment type mismatch')
            audit.append(dict(source_id=source,canonical_id=target,decision='separate',
                method='adapter_type_compatibility',reason='Incompatible specific cluster types',
                entity_types=sorted(types)))
            continue
        representative=min((a,b),key=lambda k:(k not in prior,k))
        parent[a]=parent[b]=representative
        cluster_types[representative]=types
    mapping={k:root(k) for k in sorted(nodes)}
    groups={}
    for value in selected:
        for cid,claim in sorted(value['graph']['claims'].items()):
            if claim['status']!='active':continue
            content=copy.deepcopy(claim['content'])
            for key in ('head','tail','node_id'):
                if key in content:content[key]=mapping[content[key]]
            identity=digest(content)
            row=groups.setdefault(identity,dict(fact_id='fact_'+identity,content=content,supports=[]))
            row['supports'].append(dict(claim_id=cid,book_id=value['book_id'],evidence_ids=claim['evidence_ids'],revision=claim.get('revision',1),native_source_state=claim.get('source_state'),audit_ref=dict(audit_refs[value['book_id'] or 'g0'])))
    facts=[]
    for key,row in sorted(groups.items()):
        row['supports'].sort(key=lambda s:s['claim_id'])
        has_prior=any(s['book_id'] is None for s in row['supports'])
        has_book=any(s['book_id'] is not None for s in row['supports'])
        row['source_state']='prior_textbook_supported' if has_prior and has_book else 'textbook_only_addition' if has_book else 'prior_only_unverified'
        row['evidence_ids']=sorted({e for s in row['supports'] for e in s['evidence_ids']})
        facts.append(row)
    # 不同定义或关系仅保留候选修订，缺少教材证据不能否定先验。
    alternatives=[]
    for i,left in enumerate(facts):
        c=left['content']
        for right in facts[i+1:]:
            d=right['content']
            same_conditions=d.get('conditions','')==c.get('conditions','')
            definitions=(c.get('kind')==d.get('kind')=='node_fact' and
                c.get('field')==d.get('field') and c.get('field') in ('definition','definition_detail') and
                c.get('node_id')==d.get('node_id') and c.get('value')!=d.get('value'))
            relations=(c.get('kind')==d.get('kind')=='relation' and
                c.get('head')==d.get('head') and c.get('tail')==d.get('tail') and
                c.get('relation')!=d.get('relation'))
            if same_conditions and (definitions or relations):
                alternatives.append(dict(state='potential_revision' if definitions else 'potential_relation_alternative',
                    fact_ids=[left['fact_id'],right['fact_id']],
                    evidence_ids=sorted(set(left['evidence_ids']+right['evidence_ids'])),retires_prior=False))
    result=dict(format_version=1,course_id=g0['course_id'],scope=scope,active_books=active,input_hashes={v['book_id'] or 'g0':v['artifact_id'] for v in selected},alignment_mapping=mapping,alignment_audit=audit,native_contributions=native,facts=facts,evidence=evidence,potential_revisions=alternatives)
    from .views import semantic_view
    result.update(semantic_view(nodes,mapping,facts,evidence))
    result['view_id']=digest(result)
    return result
