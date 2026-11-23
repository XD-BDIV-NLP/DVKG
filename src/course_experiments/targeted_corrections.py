"""重放既有决策并生成独立修订视图，不修改原生贡献。"""
from __future__ import annotations
import copy
from .manifest import digest


def replay(nodes,prior,decisions,cannot_link):
    parent={k:k for k in nodes};types={k:{v['entity_type']} for k,v in nodes.items()}
    members={k:{k} for k in nodes};forbidden={k:set() for k in nodes}
    for a,b in cannot_link:
        if a not in nodes or b not in nodes or a==b:raise ValueError('Invalid native cannot-link pair')
        forbidden[a].add(b);forbidden[b].add(a)
    def root(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    blocked=[]
    for source,target in sorted(decisions.items()):
        a,b=root(source),root(target)
        if a==b:continue
        if any(forbidden[k]&members[b] for k in members[a]):
            blocked.append(dict(source=source,target=target,reason='cannot_link_cluster_conflict'));continue
        merged=types[a]|types[b]
        if not (len(merged)==1 or len(merged)==2 and 'Concept' in merged and 'Metric' not in merged):
            blocked.append(dict(source=source,target=target,reason='incompatible_type'));continue
        canonical=min((a,b),key=lambda k:(k not in prior,k))
        parent[a]=parent[b]=canonical;types[canonical]=merged;members[canonical]=members[a]|members[b]
    mapping={k:root(k) for k in nodes}
    if any(mapping[a]==mapping[b] for a,b in cannot_link):raise ValueError('Cannot-link violated after replay')
    return mapping,blocked


def facts_from_native(view,mapping,quarantined):
    groups={};known=set()
    for key,value in view['native_contributions'].items():
        graph=value['graph'];ref=dict(artifact='fused.json',contribution_key=key,artifact_id=value['artifact_id'],
            events_digest=digest(graph.get('events',[])),event_count=len(graph.get('events',[])))
        for cid,claim in sorted(graph['claims'].items()):
            known.add(cid)
            if claim['status']!='active' or cid in quarantined:continue
            content=copy.deepcopy(claim['content'])
            for field in ['head','tail','node_id']:
                if field in content:content[field]=mapping[content[field]]
            identity=digest(content);row=groups.setdefault(identity,dict(fact_id='fact_'+identity,content=content,supports=[]))
            row['supports'].append(dict(claim_id=cid,book_id=value['book_id'],evidence_ids=claim['evidence_ids'],
                revision=claim.get('revision',1),native_source_state=claim.get('source_state'),audit_ref=ref))
    if set(quarantined)-known:raise ValueError('Unknown quarantined native claim')
    facts=[]
    for _,row in sorted(groups.items()):
        row['supports'].sort(key=lambda s:s['claim_id']);prior=any(s['book_id'] is None for s in row['supports']);book=any(s['book_id'] is not None for s in row['supports'])
        row['source_state']='prior_textbook_supported' if prior and book else 'textbook_only_addition' if book else 'prior_only_unverified'
        row['evidence_ids']=sorted({e for s in row['supports'] for e in s['evidence_ids']});facts.append(row)
    return facts


def quotes_valid(response,payload):
    sources={s['evidence_id']:s['text'] for s in payload.get('sources',[])}
    quotes=response.get('quotes',[])
    return bool(quotes) and all(isinstance(c,dict) and isinstance(c.get('quote'),str) and c['quote'] and
        c.get('evidence_id') in sources and c['quote'] in sources[c['evidence_id']] for c in quotes)


def derive(view,cannot_link,quarantine,review_record):
    nodes={k:r for v in view['native_contributions'].values() for k,r in v['graph']['nodes'].items()}
    prior=set(view['native_contributions']['g0']['graph']['nodes'])
    decisions={a['source_id']:a['canonical_id'] for a in view['alignment_audit'] if a['decision']=='merge'}
    baseline,original_blocked=replay(nodes,prior,decisions,[])
    if baseline!=view['alignment_mapping']:raise ValueError('Original mapping cannot be reproduced exactly')
    baseline_facts=facts_from_native(view,baseline,set())
    if digest(baseline_facts)!=digest(view['facts']):raise ValueError('Original native facts cannot be reproduced exactly')
    mapping,blocked=replay(nodes,prior,decisions,cannot_link)
    result={k:copy.deepcopy(v) for k,v in view.items() if k not in {'view_id','entities','relations','facts','alignment_mapping','potential_revisions'}}
    result['alignment_mapping']=mapping;result['facts']=facts_from_native(view,mapping,set(quarantine))
    # 原候选修订留在父版本，避免引用已经改变的事实ID。
    result['potential_revisions']=[]
    result['correction']=dict(parent_view_id=view['view_id'],review=review_record,cannot_link=[list(p) for p in sorted(cannot_link)],
        quarantined_native_claims=sorted(quarantine),blocked_decisions=blocked,original_type_blocks=original_blocked,
        mapping_changes=sum(mapping[k]!=baseline[k] for k in mapping),original_potential_revisions_preserved_in_parent=True,
        policy='targeted_full_source_review_no_test_scores_no_original_mutation')
    from .views import semantic_view
    result.update(semantic_view(nodes,mapping,result['facts'],result['evidence']));result['view_id']=digest(result)
    return result
