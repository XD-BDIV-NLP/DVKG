"""独立reference身份资格；仅来源定义，同名异义拒绝，不从图反推组件。"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from ..manifest import digest
from ..splits import norm
from . import scope_identity as source


POLICY=dict(version='reference_source_identity_qualification_v1',seed=42,development_target=24,maximum_per_course=8,
    development='fixed_course_hash_selection_all_reference_candidates_no_graph_fold_rank_or_judgment_inputs',
    minimum_exact_label_agreement=23,maximum_negative_to_positive=0,valid_two_sided_citations=24,
    same_prompt='frozen_scope_identity_batch8_source_definition_scope_homonym_guards',
    reversal='every_primary_positive_in_single_batch_and_full',
    components='complete_positive_cliques_only_any_negative_insufficient_or_missing_pair_reject_whole_component',
    original_reference='never_remove_facts_existing_freeze_identity_split_only',
    maximum_actual_local_HTTP=500,global_budget='same_formal30000_48h_4slots_never_reset',human_audited=False)


def prepare_pair(course,left,right,evidence):
    if not course or any(row.get('course_id',course)!=course for row in (left,right)):
        raise ValueError('Reference identity candidate cannot cross courses')
    if not source.names(left)&source.names(right):raise ValueError('Reference identity needs an existing exact name or explicit alias')
    packet=dict(course_id=course,policy=POLICY,name_match='explicit_name_or_alias',
        left=source._side(left,evidence,reference=True),right=source._side(right,evidence,reference=True),
        scope_category=source.scope_category(left.get('scope',''),right.get('scope','')),
        both_sides_have_source=True,status='pending_machine_reference_identity',human_audited=False)
    if not packet['left']['source_quotes'] or not packet['right']['source_quotes']:
        raise ValueError('Both reference identity sides need exact original source citations')
    return dict(packet,candidate_id=digest(packet))


def candidate_pairs(entities):
    names=defaultdict(set);rows={r['entity_id']:r for r in entities}
    for eid,row in rows.items():
        for name in [row['canonical_name'],*row.get('aliases',[])]:names[norm(name)].add(eid)
    pairs={pair for ids in names.values() for pair in combinations(sorted(ids),2)}
    return sorted(pair for pair in pairs if str(rows[pair[0]].get('scope',''))!=str(rows[pair[1]].get('scope','')))


def development_plan(packets):
    if len({p['candidate_id'] for p in packets})!=len(packets):raise ValueError('Duplicate reference candidate IDs')
    by_course=defaultdict(list)
    for row in packets:by_course[row['course_id']].append(row)
    selected=[];counts={}
    for course,rows in sorted(by_course.items()):
        chosen=sorted(rows,key=lambda r:digest([42,POLICY['version'],course,r['candidate_id']]))[:8]
        selected.extend(chosen);counts[course]=len(chosen)
    value=dict(policy=POLICY,population_digest=digest(sorted(p['candidate_id'] for p in packets)),
        candidate_ids=sorted(p['candidate_id'] for p in selected),selected_per_course=counts,
        sufficient_for_declared24=len(selected)==24 and len(counts)==3,
        minimum_exact_decision_agreement=23,confirmation_or_rank_inputs=False)
    return dict(value,development_plan_id=digest(value))


def compare_development(plan,single,batched):
    complete=valid=agree=unsafe=0;reverse_ok=True
    for cid in plan['candidate_ids']:
        left=single.get(cid,{});right=batched.get(cid,{})
        complete+=left.get('status')=='completed' and right.get('status')=='completed'
        valid+=left.get('citation_valid') is True and right.get('citation_valid') is True
        a=(left.get('primary') or {}).get('decision');b=(right.get('primary') or {}).get('decision')
        agree+=a in {'same_sense','different_sense','insufficient'} and a==b
        unsafe+=bool(right.get('eligible_identity_edge') and not left.get('eligible_identity_edge'))
        for row in (left,right):
            if row.get('eligible_identity_edge') and not row.get('reversed_positive_confirmed'):reverse_ok=False
    return dict(development_plan_id=plan['development_plan_id'],fixed_denominator=24,
        completed_pairs=complete,valid_both_side_citation_pairs=valid,exact_decision_agreement=agree,
        single_negative_to_batch_positive=unsafe,every_accepted_positive_reverse_confirmed=reverse_ok,
        passes_declared_gate=plan['sufficient_for_declared24'] and complete==valid==24 and agree>=23 and unsafe==0 and reverse_ok,
        semantic_truth_claim=False,human_audited=False)


def freeze_components(origins,decisions):
    if {r['candidate_id'] for r in origins}!=set(decisions) or any(r.get('status')!='completed' or
            r.get('eligible_identity_edge') and (r.get('citation_valid') is not True or r.get('reversed_positive_confirmed') is not True)
            for r in decisions.values()):
        raise ValueError('All fixed reference candidates must complete before components freeze')
    by_course=defaultdict(list)
    for row in origins:by_course[row['course_id']].append(row)
    components={};rejected=[]
    for course,rows in sorted(by_course.items()):
        parent={e:e for r in rows for e in (r['left_reference_entity_id'],r['right_reference_entity_id'])}
        def find(key):
            while parent[key]!=key:
                parent[key]=parent[parent[key]];key=parent[key]
            return key
        eligible={}
        for row in rows:
            pair=tuple(sorted([row['left_reference_entity_id'],row['right_reference_entity_id']]))
            good=decisions[row['candidate_id']].get('eligible_identity_edge') is True
            if pair in eligible and eligible[pair]!=good:raise ValueError('Reference pair has conflicting independent judgments')
            eligible[pair]=good
            if good:
                a,b=find(pair[0]),find(pair[1]);parent[max(a,b)]=min(a,b)
        clusters=defaultdict(list)
        for key in parent:clusters[find(key)].append(key)
        for ids in sorted((sorted(v) for v in clusters.values()),key=digest):
            if len(ids)<2:continue
            if all(eligible.get(pair) is True for pair in combinations(ids,2)):
                components.setdefault(course,[]).append(ids)
            else:rejected.append(dict(course_id=course,reference_entity_ids=ids,reason='negative_insufficient_or_missing_internal_pair'))
    result=dict(policy=POLICY,components=components,rejected_components=rejected,
        fixed_candidate_count=len(decisions),original_facts_removed=0,graph_inputs=False,human_audited=False)
    return dict(result,reference_component_id=digest(result))
