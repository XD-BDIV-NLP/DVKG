"""身份响应的无损形状适配；保留原raw，不修写ID、标签或引句。"""
from __future__ import annotations

from contextlib import contextmanager
import copy


POLICY=dict(version='identity_nested_boolean_flags_lossless_v1',
    accepted='complete_exact_three_boolean_guards_under_flags_or_existing_flat_fields',
    conflicts='reject_missing_extra_nonboolean_or_conflicting_duplicate_guards',
    forbidden='never_change_candidate_id_decision_citation_evidence_id_quote_or_prompt',
    raw_response='unchanged_original_model_response_preserved')


def normalize_flags(value,kind):
    result=copy.deepcopy(value)
    if 'flags' not in result:return result
    expected={'necessary_scope_consistent','definition_identity_sufficient','homonym_excluded'} if kind=='prior_definition_identity' else {
        'scope_consistent','source_definition_sufficient','homonym_excluded'}
    nested=result['flags']
    if not isinstance(nested,dict) or set(nested)!=expected or any(type(v) is not bool for v in nested.values()):
        raise ValueError('Nested identity flags must contain exactly the frozen three boolean guards')
    for key,value in nested.items():
        if key in result and (type(result[key]) is not bool or result[key] is not value):
            raise ValueError('Nested and flat identity guards conflict')
        result[key]=value
    result['response_shape_normalization']=POLICY['version']
    return result


@contextmanager
def parser_bridge():
    from . import scope_identity as source
    from . import prior_definition_identity as prior
    original_source=source.validate_judgment;original_prior=prior.validate_judgment
    source.validate_judgment=lambda packet,value:original_source(packet,normalize_flags(value,'two_sided_source'))
    prior.validate_judgment=lambda packet,value:original_prior(packet,normalize_flags(value,'prior_definition_identity'))
    try:yield
    finally:
        source.validate_judgment=original_source;prior.validate_judgment=original_prior
