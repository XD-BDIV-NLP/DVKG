"""失败实体对的有界短标签方向检查；旧语义与源端校验保持不变。"""
from __future__ import annotations

import json
from pathlib import Path

from ..manifest import digest, read_document
from . import identity_short_single as single, identity_wire_labels as labels
from . import run_scope_identity_gate as gate, scope_identity as source
from .run_native_full import _checkpoint, _read_checkpoint


def choose_target(rows):
    failed = [key for key,value in rows.items() if value.get('status') == 'failed']
    if not failed:
        raise ValueError('An original failed identity case is required')
    return min(failed, key=lambda key:digest([42,key]))


def direction_body(packet, reverse, locks):
    body, table = single.request_body(packet, reverse=reverse)
    if locks:
        body['messages'].insert(-1,dict(role='system',content=(
            '仅修复标识和逐字引用；已有合法语义字段必须保持为以下值，不得为通过校验改判：'
            +json.dumps(locks,ensure_ascii=False))))
    return body, table


def validate_response(packet, raw, table, reverse, locks):
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Identity completion incomplete; preserve response')
    value = labels.decode_response(json.loads(choice['message']['content']),table)
    for key, expected in locks.items():
        if type(value.get(key)) is not type(expected) or value.get(key) != expected:
            raise ValueError('Short-label repair changed a frozen semantic field')
    wire = source.anonymous_packet(packet, reverse=reverse)
    return source.validate_judgment(dict(wire,both_sides_have_source=packet['both_sides_have_source']),value)


def execute_direction(provider, body, root, job, allowance):
    root = Path(root)
    key = digest(dict(endpoint='/chat/completions',body=body,config=provider.runtime.config))
    response = root/'response.json'
    if response.exists():
        result = _read_checkpoint(response,job)
        if result['request_id'] != key:
            raise ValueError('Direction response belongs to another exact request')
        return result
    logical = root/'logical_request.json'
    if logical.exists():
        stored = _read_checkpoint(logical,job)
        if stored['request_id'] != key or stored['body'] != body:
            raise ValueError('Direction logical reservation changed')
        if hasattr(provider,'_cached'):
            raw = provider._cached(key,'/chat/completions',body)
        else:
            path = provider.root/'receipts'/(key+'.json')
            value = read_document(path) if path.exists() else None
            if value and (value['request_id'] != key or value['endpoint'] != '/chat/completions'
                    or value['request'] != body or value['runtime_config'] != provider.runtime.config):
                raise ValueError('Exact direction receipt differs')
            raw = value['response'] if value else None
        if raw is None:
            raise RuntimeError('Reserved logical request has no verified response; do not reissue')
        hit = True
    else:
        _checkpoint(logical,dict(request_id=key,body=body),job)
        with gate.local_http_cap(provider,allowance,300):
            raw, hit = provider.request_executor('/chat/completions',body,'scope_identity_short_single.direction')
    return _checkpoint(response,dict(request_id=key,raw_response=raw,cache_hit=hit),job)
