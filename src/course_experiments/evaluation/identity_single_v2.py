"""独立逐对fail-closed身份协议；不放行或改写旧批量门禁。"""
from __future__ import annotations

from collections import Counter,defaultdict
from contextlib import contextmanager
import copy
from itertools import combinations
import json
from pathlib import Path
from urllib.error import URLError

from ..contributions import write_immutable
from ..manifest import digest,file_record,read_document
from ..splits import canonical_reference,freeze_split
from . import scope_identity as source,scope_identity_batch as core,scope_identity_packing as packing
from .identity_response import normalize_flags
from .run_native_full import _checkpoint,_read_checkpoint,_error,ReceiptTracker,_no_scope
from .run_scope_identity_gate import identity_projection,freeze_identity_split


POLICY=dict(version='strict_single_fail_closed_v2',source_judge=digest(source.JUDGE),
    prior_and_reference_judge=digest(core.JUDGE),requests='frozen_original_long_ID_single_pair_messages_only',
    batch_optimization=False,short_labels=False,old_failed_gates='preserved_not_passed_not_comparable_as_same_protocol',
    accepted='exact_two_sided_quotes_original_scope_sense_guards_and_positive_reverse_confirmation',
    unknown='invalid_exhausted_unresolved_kept_UNKNOWN_unmapped_never_false_negative',
    denominator='every_original_candidate_processed_all_reference_facts_fixed_test_denominator_zero_uncovered',
    conflict='reject_all_conflicting_targets_unless_independently_confirmed_reference_component',
    reference='independent_strict_positive_cliques_only_unknown_internal_pair_rejects_entire_component',
    leakage='unchanged_conservative_lexical_fact_cohold_and_equivalent_inverse_condition_holdout',
    budget='existing_shared_actual_HTTP_80k_original_48h_start_4_slots_never_reset',maximum_actual_HTTP_per_exact_body=2,human_audited=False)


class RequestAttemptsExhausted(RuntimeError):pass
class SavedIdentityMismatch(RuntimeError):pass
class StageHTTPBudgetExhausted(RuntimeError):pass


@contextmanager
def actual_attempt_cap(provider,maximum,*,total_http_cap=None):
    if type(maximum) is not int or maximum!=2:raise ValueError('Strict single v2 requires cumulative maximum two actual attempts')
    if total_http_cap is not None and (type(total_http_cap) is not int or not 1<=total_http_cap<=32438):raise ValueError('Jointly reserved v2 actual HTTP cap required')
    original=provider._write_transaction
    previous=getattr(provider,'_identity_single_attempt_cap',None);provider._identity_single_attempt_cap=maximum
    previous_total=getattr(provider,'_identity_single_total_cap',None);provider._identity_single_total_cap=total_http_cap
    def bounded(fn,*,operation,request_id=None):
        def transaction(db):
            if operation=='event_request_started' and total_http_cap is not None and db.execute(
                    "SELECT COUNT(*) FROM events WHERE status='request_started' AND stage LIKE 'identity_single_v2.%'").fetchone()[0]>=total_http_cap:
                raise StageHTTPBudgetExhausted('Cumulative jointly reserved actual HTTP cap for strict single v2 exhausted')
            if operation=='event_request_started' and db.execute(
                    "SELECT COUNT(*) FROM events WHERE status='request_started' AND key=?",(request_id,)).fetchone()[0]>=maximum:
                raise RequestAttemptsExhausted('Cumulative actual attempts for this exact original request exhausted')
            return fn(db)
        return original(transaction,operation=operation,request_id=request_id)
    provider._write_transaction=bounded
    try:yield
    finally:
        provider._write_transaction=original;provider._identity_single_attempt_cap=previous;provider._identity_single_total_cap=previous_total


def request_body(packet,branch,*,reverse=False):
    if branch not in {'source','prior','reference'}:raise ValueError('Unknown single identity branch')
    if (branch=='prior')!=(core._kind(packet)=='prior_definition_identity'):
        raise ValueError('Original source/prior provenance branch differs')
    if branch=='source':
        messages=[dict(role='system',content=source.JUDGE),dict(role='user',content=json.dumps(
            source.anonymous_packet(packet,reverse=reverse),ensure_ascii=False))];maximum=1024
    else:messages=packing.messages_for([packet],reverse=reverse);maximum=8192
    return dict(model='deepseek-ai/DeepSeek-V4-Flash',messages=messages,temperature=0,seed=42,
        enable_thinking=False,max_tokens=maximum,response_format={'type':'json_object'})


def validate_raw(packet,branch,raw,*,reverse=False):
    if raw['choices'][0].get('finish_reason')!='stop':raise ValueError('Original single identity output incomplete')
    value=json.loads(raw['choices'][0]['message']['content'])
    if branch=='source':
        wire=source.anonymous_packet(packet,reverse=reverse)
        judgment=source.validate_judgment(dict(wire,both_sides_have_source=packet['both_sides_have_source']),
            normalize_flags(value,'two_sided_source'))
        if not judgment['left_citations'] or not judgment['right_citations']:
            raise ValueError('Strict v2 completed judgment requires two-sided exact citations')
        return dict(status='validated',judgment=judgment,citation_valid=True,
            eligible_identity_edge=judgment['eligible_identity_edge'],reverse_input=reverse)
    wire=json.loads(request_body(packet,branch,reverse=reverse)['messages'][-1]['content'])
    if value.get('batch_id')!=wire['batch_id'] or not isinstance(value.get('results'),list) or len(value['results'])!=1:
        raise ValueError('Frozen single-pair envelope or candidate population changed')
    return core._validate(packet,core._anonymous(packet),normalize_flags(value['results'][0],core._kind(packet)),reverse)


def sealed_unknown(failed_decision,repair_attempts):
    for record in (failed_decision,repair_attempts):
        if file_record(record['path'])!=record:raise ValueError('Original exhausted-case receipt changed')
    failed=read_document(failed_decision['path']);_read_checkpoint(failed_decision['path'],failed['job_identity'])
    attempts=read_document(repair_attempts['path'])
    if (failed.get('status')!='failed' or failed.get('eligible_identity_edge') is not False or
            attempts.get('candidate_id')!=failed.get('candidate_id') or set(attempts.get('attempts',{}))!={'1','2'} or
            any(r.get('status')!='failed' for r in attempts['attempts'].values())):
        raise ValueError('Only the documented twice-exhausted failed candidate can remain sealed UNKNOWN')
    result=dict(candidate_id=failed['candidate_id'],failed_decision=failed_decision,repair_attempts=repair_attempts,
        classification='UNKNOWN',reason='original_strict_validation_and_two_repairs_exhausted_no_new_attempt')
    return dict(result,sealed_id=digest(result))


def run_population(packets,branch,provider,root,*,sealed=None,counter=None,worker_index=0,worker_count=1):
    actual=hasattr(provider,'runtime')
    if actual and (counter is None or getattr(provider,'_identity_single_attempt_cap',None)!=2 or type(getattr(provider,'_identity_single_total_cap',None)) is not int):
        raise ValueError('Official actual-message token counter and explicit cumulative HTTP attempt cap required')
    tracker=ReceiptTracker(provider) if actual else None
    try:return _run_population(packets,branch,provider,root,sealed=sealed,counter=counter,tracker=tracker,worker_index=worker_index,worker_count=worker_count)
    finally:
        if tracker:tracker.close()


def _run_population(packets,branch,provider,root,*,sealed=None,counter=None,tracker=None,worker_index=0,worker_count=1):
    root=Path(root);sealed=sealed or {};ids=[p['candidate_id'] for p in packets]
    if type(worker_count) is not int or not 1<=worker_count<=4 or type(worker_index) is not int or not 0<=worker_index<worker_count:
        raise ValueError('Fixed disjoint worker partition must be within original four-slot limit')
    if len(set(ids))!=len(ids) or set(sealed)-set(ids):raise ValueError('Fixed unique complete candidate population required')
    for packet in packets:
        if packet['candidate_id']!=digest({k:v for k,v in packet.items() if k!='candidate_id'}):raise ValueError('Frozen candidate hash changed')
        request_body(packet,branch)
    if sealed and branch!='source':raise ValueError('Existing exhausted source case cannot be reused in another branch')
    contract=dict(protocol=POLICY,branch=branch,packets_digest=digest(packets),sealed=sealed,
        implementation=file_record(Path(__file__)),runtime_config_digest=digest(provider.runtime.config) if hasattr(provider,'runtime') else None,
        maximum_actual_HTTP_per_exact_body=getattr(provider,'_identity_single_attempt_cap',None),token_assets=counter.assets if counter else None,
        maximum_cumulative_v2_actual_HTTP=getattr(provider,'_identity_single_total_cap',None),worker_count=worker_count,
        partition='frozen_candidate_order_index_mod_count_no_score_or_coverage')
    job=digest(contract);write_immutable(root/'contract.json',dict(contract,job_identity=job));decisions={}
    for index,packet in enumerate(packets):
        if index%worker_count!=worker_index:continue
        cid=packet['candidate_id'];folder=root/'decisions'/cid;path=folder/'decision.json';provider.raise_if_fatal()
        if path.exists():decisions[cid]=_read_checkpoint(path,job);continue
        row=dict(candidate_id=cid,processed=True,classification='UNKNOWN',eligible_identity_edge=False,
            citation_valid=False,reversed_positive_confirmed=False)
        if cid in sealed:
            evidence=sealed[cid];expected=sealed_unknown(evidence['failed_decision'],evidence['repair_attempts'])
            if evidence!=expected or evidence['candidate_id']!=cid:raise ValueError('Sealed UNKNOWN identity differs')
            row.update(sealed_unknown=evidence,reason=evidence['reason'])
        else:
            def direction(reverse):
                body=request_body(packet,branch,reverse=reverse);saved=folder/('reverse.json' if reverse else 'primary.json')
                if saved.exists():
                    try:stored=_read_checkpoint(saved,job)
                    except ValueError as exc:raise SavedIdentityMismatch('Saved original response identity changed') from exc
                    if stored['body_digest']!=digest(body):raise SavedIdentityMismatch('Saved original single request body changed')
                    raw=stored['raw_response']
                else:
                    audit=None
                    if counter:
                        formatted=copy.deepcopy(body['messages']);formatted[0]['response_format']=body['response_format']
                        tokens=max(counter.count(body['messages']),counter.count(formatted));total=tokens+body['max_tokens']+1024
                        if total>262144:raise ValueError('Complete original single messages exceed frozen working context')
                        audit=dict(actual_messages_tokens=tokens,total_with_output_and1024_reserve=total,working_context=262144,truncation=False)
                    with tracker.scope(folder/'requests.jsonl') if tracker else _no_scope():
                        raw,hit=provider.request_executor('/chat/completions',body,'identity_single_v2.'+branch+'.'+('reverse' if reverse else 'primary'))
                    _checkpoint(saved,dict(body_digest=digest(body),raw_response=raw,cache_hit=hit,raw_server_only=True,token_audit=audit),job)
                return validate_raw(packet,branch,raw,reverse=reverse)
            try:
                primary=direction(False);row.update(primary_record=primary,primary=primary['judgment'],citation_valid=True)
                if primary['eligible_identity_edge']:
                    reverse=direction(True);row['reverse']=reverse
                    if reverse['eligible_identity_edge']:row.update(classification='accepted',eligible_identity_edge=True,reversed_positive_confirmed=True)
                    else:row.update(reason='positive_primary_not_confirmed_by_reverse')
                elif primary['judgment']['decision']=='different_sense':row['classification']='confirmed_negative'
                elif primary['judgment']['decision']=='insufficient':row['classification']='insufficient'
                else:row['reason']='same_sense_without_all_original_semantic_guards'
            except (ValueError,TypeError,KeyError,IndexError,RequestAttemptsExhausted,URLError,TimeoutError,ConnectionError) as exc:
                provider.raise_if_fatal();row.update(reason='strict_validation_or_bounded_transport_unresolved',**_error(folder,exc))
        decisions[cid]=_checkpoint(path,row,job)
    if worker_count>1:
        return _checkpoint(root/'workers'/str(worker_index)/'state.json',dict(status='partition_completed_await_main_finalize',
            worker_index=worker_index,worker_count=worker_count,partition_processed=len(decisions),fixed_population=len(ids),all_candidates_processed=False),job,immutable=False)
    return finalize_population(packets,branch,root)


def population_result(decisions):
    counts=Counter(r['classification'] for r in decisions.values())
    result=dict(protocol=POLICY['version'],status='processed_strict_single_fail_closed_v2',candidate_count=len(decisions),decisions=decisions,
        all_candidates_processed=True,accepted_count=counts['accepted'],negative_count=counts['confirmed_negative'],
        insufficient_count=counts['insufficient'],unknown_count=counts['UNKNOWN'],old_batch_gate_passed=False,
        usable_for_KGE=False,original_graphs_modified=False,original_reference_denominator_preserved=True,human_audited=False)
    return result


def validate_saved_population(packets,branch,root):
    root=Path(root);contract=read_document(root/'contract.json')
    return _validate_saved_population(packets,branch,root,_read_checkpoint(root/'result.json',contract['job_identity']))


def finalize_population(packets,branch,root):
    root=Path(root);contract=read_document(root/'contract.json');job=contract['job_identity'];decisions={}
    if {p.parent.name for p in (root/'decisions').glob('*/decision.json')}!={p['candidate_id'] for p in packets}:
        raise ValueError('All and only the full fixed candidate checkpoints must exist before main-process finalization')
    for packet in packets:
        cid=packet['candidate_id'];decisions[cid]=_read_checkpoint(root/'decisions'/cid/'decision.json',job)
    result=population_result(decisions);_validate_saved_population(packets,branch,root,result)
    return _checkpoint(root/'result.json',result,job)


def _validate_saved_population(packets,branch,root,result):
    root=Path(root);contract=read_document(root/'contract.json');job=contract['job_identity']
    if (job!=digest({k:v for k,v in contract.items() if k!='job_identity'}) or
            contract['branch']!=branch or contract['packets_digest']!=digest(packets) or contract['protocol']!=POLICY or
            file_record(contract['implementation']['path'])!=contract['implementation']):
        raise ValueError('Frozen single protocol implementation or population changed')
    if set(result['decisions'])!={p['candidate_id'] for p in packets}:raise ValueError('Full saved candidate population changed')
    for packet in packets:
        cid=packet['candidate_id'];folder=root/'decisions'/cid;row=_read_checkpoint(folder/'decision.json',job)
        if row!=result['decisions'][cid] or row.get('candidate_id')!=cid:raise ValueError('Population decision differs from its checkpoint')
        if row.get('sealed_unknown'):
            original=row['sealed_unknown'];expected=sealed_unknown(original['failed_decision'],original['repair_attempts'])
            if original!=expected or row['classification']!='UNKNOWN':raise ValueError('Exhausted UNKNOWN was reclassified')
            continue
        validated={}
        for reverse,name in [(False,'primary'),(True,'reverse')]:
            path=folder/(name+'.json')
            if not path.exists():continue
            stored=_read_checkpoint(path,job)
            if stored['body_digest']!=digest(request_body(packet,branch,reverse=reverse)):raise ValueError('Original request changed')
            try:validated[name]=validate_raw(packet,branch,stored['raw_response'],reverse=reverse)
            except (ValueError,TypeError,KeyError,IndexError):
                if row['classification']!='UNKNOWN':raise ValueError('Invalid original response became a known identity decision')
        if row['classification']!='UNKNOWN':
            primary=validated.get('primary')
            if not primary or primary!=row.get('primary_record') or primary['judgment']!=row.get('primary'):
                raise ValueError('Stored original primary validation changed')
            if row['classification']=='accepted':
                reverse=validated.get('reverse')
                if not primary['eligible_identity_edge'] or not reverse or not reverse['eligible_identity_edge'] or reverse!=row.get('reverse'):
                    raise ValueError('Accepted single equivalence lacks two strict positive original responses')
            elif row['primary']['decision']!=('different_sense' if row['classification']=='confirmed_negative' else 'insufficient'):
                raise ValueError('Negative and insufficient classifications changed')
        elif not row.get('error_record'):
            primary=validated.get('primary');reverse=validated.get('reverse')
            if not primary:raise ValueError('UNKNOWN lacks an actual validated response or exhausted-failure proof')
            if primary['eligible_identity_edge']:
                if not reverse or reverse['eligible_identity_edge'] or reverse!=row.get('reverse'):
                    raise ValueError('UNKNOWN reverse disagreement lacks its original response')
            elif primary['judgment']['decision']!='same_sense':raise ValueError('Known negative or insufficient was silently reclassified UNKNOWN')
        if row.get('error_record') and file_record(row['error_record']['path'])!=row['error_record']:
            raise ValueError('Original unresolved failure receipt changed')
    counts=Counter(r['classification'] for r in result['decisions'].values())
    for field,label in [('accepted_count','accepted'),('negative_count','confirmed_negative'),('insufficient_count','insufficient'),('unknown_count','UNKNOWN')]:
        if result[field]!=counts[label]:raise ValueError('Fixed denominator classification counts changed')
    return result


def validate_population(origins,result):
    decisions=result.get('decisions',{})
    if (result.get('protocol')!=POLICY['version'] or result.get('all_candidates_processed') is not True or
            result.get('candidate_count')!=len(decisions) or set(decisions)!={r['candidate_id'] for r in origins}):
        raise ValueError('The full fixed candidate population has not been processed')
    for row in decisions.values():
        if row.get('processed') is not True:raise ValueError('Unprocessed candidate cannot become UNKNOWN or negative')
        if row.get('classification') not in {'accepted','confirmed_negative','insufficient','UNKNOWN'}:raise ValueError('Unknown result classification')
        if row.get('eligible_identity_edge') is not (row['classification']=='accepted'):raise ValueError('Unvalidated or UNKNOWN candidate cannot become an identity')
        if row['classification']=='accepted' and not (row.get('citation_valid') is True and row.get('reversed_positive_confirmed') is True):
            raise ValueError('Identity requires exact citations and positive reverse confirmation')


def mapping_projection(origins,result):
    validate_population(origins,result);projection=identity_projection(origins,result['decisions'])
    projection.update(protocol=POLICY['version'],all_candidates_processed=True,unknown_is_negative=False,
        accepted_after_conflict_veto=len({r['candidate_id'] for r in projection['mapping']}))
    projection['identity_projection_id']=digest({k:v for k,v in projection.items() if k!='identity_projection_id'})
    return projection


def reference_components(origins,result):
    validate_population(origins,result);courses=defaultdict(list);components={};rejected=[]
    for origin in origins:courses[origin['course_id']].append(origin)
    for course,rows in sorted(courses.items()):
        parent={e:e for r in rows for e in (r['left_reference_entity_id'],r['right_reference_entity_id'])};eligible={}
        def find(key):
            while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
            return key
        for row in rows:
            pair=tuple(sorted([row['left_reference_entity_id'],row['right_reference_entity_id']]))
            good=result['decisions'][row['candidate_id']]['eligible_identity_edge']
            if pair in eligible and eligible[pair]!=good:raise ValueError('Independent reference pair has conflicting judgments')
            eligible[pair]=good
            if good:
                a,b=find(pair[0]),find(pair[1]);parent[max(a,b)]=min(a,b)
        clusters=defaultdict(list)
        for key in parent:clusters[find(key)].append(key)
        for ids in sorted((sorted(v) for v in clusters.values()),key=digest):
            if len(ids)<2:continue
            if all(eligible.get(pair) is True for pair in combinations(ids,2)):components.setdefault(course,[]).append(ids)
            else:rejected.append(dict(course_id=course,reference_entity_ids=ids,reason='negative_insufficient_UNKNOWN_or_missing_internal_pair'))
    value=dict(protocol=POLICY['version'],components=components,rejected_components=rejected,
        fixed_candidate_count=len(result['decisions']),all_candidates_processed=True,original_facts_removed=0,graph_inputs=False)
    return dict(value,qualification_id=digest(value))


def freeze_reference(reference,qualification):
    if (qualification.get('protocol')!=POLICY['version'] or qualification.get('all_candidates_processed') is not True or
            qualification.get('qualification_id')!=digest({k:v for k,v in qualification.items() if k!='qualification_id'}) or
            qualification.get('original_reference_digests',{}).get(reference['course_id'])!=digest(reference)):
        raise ValueError('Independent full single qualification for this exact reference is required')
    original=copy.deepcopy(reference);frozen,_=freeze_identity_split(canonical_reference(reference),qualification['components'].get(reference['course_id'],[]))
    frozen['identity_kge_protocol']=dict(protocol=POLICY['version'],qualification_id=qualification['qualification_id'],
        original_reference_digest=digest(original),all_original_fact_ids=[f['fact_id'] for f in original['facts']],
        old_batch_gates_passed=False,UNKNOWN_reference_pairs_are_unmerged=True,zero_uncovered_in_fixed_denominator=True)
    frozen['reference_id']=digest({k:v for k,v in frozen.items() if k!='reference_id'})
    from .identity_kge import cohold_lexical_fact_risks
    split=cohold_lexical_fact_risks(frozen,freeze_split(frozen))
    if sorted(f['fact_id'] for f in original['facts'])!=sorted(fid for g in split['groups'].values() for fid in g['fact_ids']):
        raise ValueError('The full original reference fact denominator changed')
    return frozen,split


def qualify_all(mapping_preparation,reference_preparation,output):
    mapping_preparation=Path(mapping_preparation);reference_preparation=Path(reference_preparation);output=Path(output)
    def manifest(folder):
        value=read_document(folder/'manifest.json')
        if value.get('preparation_id')!=digest({k:v for k,v in value.items() if k!='preparation_id'}):raise ValueError('Frozen original preparation changed')
        for record in value['files'].values():
            if file_record(record['path'])!=record:raise ValueError('Frozen original candidate input changed')
        return value
    original=manifest(mapping_preparation);independent=manifest(reference_preparation)
    inputs=read_document(mapping_preparation/'input_records.json')
    for record in [*inputs.values(),*independent['references'].values(),*independent['source_records'].values()]:
        if file_record(record['path'])!=record:raise ValueError('Original graph/reference/source file changed')
    results={};origins={}
    for branch,folder,stem in [('source',mapping_preparation,'source_full'),('prior',mapping_preparation,'prior_full'),('reference',reference_preparation,'full')]:
        packets=read_document(folder/(stem+'_packets_server_only.json'));origins[branch]=read_document(folder/(stem+'_origins_server_only.json'))
        expected=independent['full_count'] if branch=='reference' else original['requests'][branch+'-full']['actual_pairs']
        if len(packets)!=expected:raise ValueError('Full original candidate denominator changed')
        by_id={p['candidate_id']:p for p in packets}
        for row in origins[branch]:
            packet=by_id[row['candidate_id']]
            if (row['course_id']!=packet['course_id'] or row['graph_entity_id']!=packet['right']['entity_id'] or
                    row['reference_entity_id']!=packet['left']['entity_id']):raise ValueError('Frozen origin endpoints changed')
            if branch!='reference' and inputs.get(row['graph_record']['path'])!=row['graph_record']:
                raise ValueError('Graph origin outside frozen complete inventory')
        results[branch]=validate_saved_population(packets,branch,output/branch)
        validate_population(origins[branch],results[branch])
    independent_components=reference_components(origins['reference'],results['reference']);components=independent_components['components']
    component_map={course:{eid:sorted(ids)[0] for ids in groups for eid in ids} for course,groups in components.items()}
    combined={};all_origins=[]
    for branch,basis in [('source','two_sided_textbook_source'),('prior','prior_generated_definition')]:
        if combined.keys()&results[branch]['decisions'].keys():raise ValueError('Source and prior candidate identities overlap')
        combined.update(results[branch]['decisions'])
        for origin in origins[branch]:
            row=copy.deepcopy(origin);old=row['reference_entity_id'];row.update(original_reference_entity_id=old,
                reference_entity_id=component_map.get(row['course_id'],{}).get(old,old),identity_basis=basis);all_origins.append(row)
    joint=identity_projection(all_origins,combined)
    value=dict(protocol=POLICY['version'],policy=POLICY,status='qualified_strict_single_fail_closed_v2',usable_for_KGE=True,
        all_candidates_processed=True,components=components,reference_component_proof=independent_components,
        mapping=joint['mapping'],rejections=joint['rejections'],identity_projection_id=joint['identity_projection_id'],
        branch_counts={branch:{key:r[key] for key in ('candidate_count','accepted_count','negative_count','insufficient_count','unknown_count')} for branch,r in results.items()},
        stage_receipts={branch:file_record(output/branch/'result.json') for branch in results},
        original_mapping_preparation=file_record(mapping_preparation/'manifest.json'),original_reference_preparation=file_record(reference_preparation/'manifest.json'),
        reference_records=independent['references'],original_reference_digests={course:digest(read_document(rec['path'])) for course,rec in independent['references'].items()},
        old_batch_gates_passed=False,old_source32_calibration_passed=False,reference_components_inferred_from_graphs=False,
        original_graphs_modified=False,all_reference_facts_preserved=True,zero_uncovered_fixed_denominator=True,human_audited=False)
    value['qualification_id']=digest(value);write_immutable(output/'qualification_v2.json',value);return value
