"""参考实体跨scope重名与事实分组风险的零API审计；不宣称候选语义等价。"""
from __future__ import annotations

from collections import Counter,defaultdict
from itertools import combinations
import math

from ..manifest import digest
from ..splits import norm,canonical,canonical_reference,freeze_split
from .scope_identity import scope_category


POLICY=dict(version='source_only_reference_identity_risk_audit_v1',
    candidate_selection='same_course_exact_NFKC_name_or_existing_explicit_alias_different_scoped_components',
    graph_inputs=False,rank_inputs=False,coverage_selection=False,semantic_merges=False,
    interpretation='lexical_cross_scope_candidates_may_be_homonyms_not_confirmed_equivalence',
    activation='source_only_definition_scope_judgments_then_conflict_checked_frozen_components_before_new_split')


def audit_reference(reference):
    course=reference['course_id'];ref=canonical_reference(reference)
    entities={r['entity_id']:r for r in ref['entities']};names=defaultdict(set)
    for eid,row in entities.items():
        for name in [row['canonical_name'],*row.get('aliases',[])]:names[norm(name)].add(eid)
    pairs=defaultdict(set);parents={eid:eid for eid in entities}
    def find(key):
        while parents[key]!=key:
            parents[key]=parents[parents[key]];key=parents[key]
        return key
    for name,ids in names.items():
        for left,right in combinations(sorted(ids),2):
            a,b=find(left),find(right);parents[max(a,b)]=min(a,b)
            pairs[(left,right)].add(name)
    candidates=[]
    for (left,right),overlap in sorted(pairs.items()):
        a,b=entities[left],entities[right]
        if str(a.get('scope',''))==str(b.get('scope','')):
            raise ValueError('Scoped explicit aliases were not canonicalized before risk audit')
        identity=dict(course_id=course,reference_entity_ids=[left,right],policy=POLICY['version'])
        candidates.append(dict(identity,candidate_id=digest(identity),name_overlap_digest=digest(sorted(overlap)),
            scope_category=scope_category(a.get('scope',''),b.get('scope','')),
            scope_digests=[digest(a.get('scope','')),digest(b.get('scope',''))],
            both_have_reference_source_citations=bool(a.get('citations') and b.get('citations')),
            exact_source_substring_revalidation_required_before_judgment=True,semantic_equivalence_confirmed=False))
    split=freeze_split(reference);fold={gid:key for key,ids in split['folds'].items() for gid in ids}
    lexical=defaultdict(list)
    for fact in ref['facts']:
        h,r,t=fact['head'],fact['relation'],fact['tail'];actual=digest(canonical(h,r,t))
        lexical_key=digest(canonical(find(h),r,find(t)))
        lexical[lexical_key].append(dict(fact_id=fact['fact_id'],actual_group_id=actual,fold=fold[actual]))
    risks=[]
    for key,rows in sorted(lexical.items()):
        groups=sorted({r['actual_group_id'] for r in rows})
        if len(groups)>1:
            risks.append(dict(lexical_endpoint_pattern_digest=key,actual_group_ids=groups,
                fact_ids=sorted(r['fact_id'] for r in rows),folds=sorted({r['fold'] for r in rows}),
                semantic_equivalence_confirmed=False))
    summary=dict(policy=POLICY,course_id=course,reference_digest=digest(reference),
        canonical_reference_entities=len(entities),independent_fact_count=len(reference['facts']),
        existing_equivalence_groups=len(split['groups']),cross_scope_reference_pairs=len(candidates),
        source_citation_present_pairs=sum(r['both_have_reference_source_citations'] for r in candidates),
        scope_categories=dict(Counter(r['scope_category'] for r in candidates)),
        candidate_population_digest=digest(candidates),
        potential_equivalent_fact_clusters_across_actual_groups=len(risks),
        potential_equivalent_fact_clusters_across_folds=sum(len(r['folds'])>1 for r in risks),
        potentially_affected_fact_count=sum(len(r['fact_ids']) for r in risks),
        potential_fact_risk_digest=digest(risks),semantic_equivalence_confirmed=False,
        minimum_primary_pair_judgments=len(candidates),maximum_single_primary_and_positive_reverse_logical=2*len(candidates),
        ideal_max8_no_token_split_primary_and_all_positive_reverse=2*math.ceil(len(candidates)/8),
        real_batch_cost_requires_server_full_message_token_packing=True,
        model_calls=0,new_split_created=False,original_reference_modified=False,
        name_alias_candidate_risk_empty=not candidates,reference_equivalence_leakage_fully_qualified=False,
        limitation='Candidate homonyms and differently scoped senses must be rejected; no automatic unscoped merge')
    return dict(summary=summary,candidates=candidates,potential_fact_group_risks=risks)
