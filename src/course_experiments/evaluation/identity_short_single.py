"""逐对原裁判的短标签请求构造；精确cache不携带packet身份。"""
from __future__ import annotations

import json

from . import scope_identity as source
from . import scope_identity_batch as core
from . import scope_identity_packing as packing
from . import identity_wire_labels as labels


POLICY=dict(version='identity_original_single_short_wire_v1',
    source='original_source119_single_prompt_1024_tokens',
    prior='original_prior24_single_batch_prompt_exactly_one_pair_8192_tokens',
    cache='complete_actual_messages_body_and_frozen_runtime_config_no_packet_hash_metadata',
    text='no_normalization_reordering_truncation_or_text_changes',
    reversal='every_new_positive_single_requires_its_own_reverse',wire=labels.POLICY)


def request_body(packet,*,reverse=False):
    if core._kind(packet)=='prior_definition_identity':
        original=packing.messages_for([packet],reverse=reverse)
        wire=json.loads(original[1]['content']);max_tokens=8192
    else:
        wire=source.anonymous_packet(packet,reverse=reverse)
        original=[dict(role='system',content=source.JUDGE)];max_tokens=1024
    short,table=labels.encode_wire(wire)
    if labels.restore_wire(short,table)!=wire:raise ValueError('Short wire changed original input')
    messages=[original[0],dict(role='system',content=labels.INSTRUCTION),
        dict(role='user',content=json.dumps(short,ensure_ascii=False))]
    body=dict(model='deepseek-ai/DeepSeek-V4-Flash',messages=messages,temperature=0,seed=42,
        enable_thinking=False,max_tokens=max_tokens,response_format={'type':'json_object'})
    return body,table
