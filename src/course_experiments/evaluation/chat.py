from __future__ import annotations
import json
import time
from urllib.error import HTTPError
from ..manifest import digest
from ..contributions import write_immutable
from .protocol import check_llm


def account_status(exc):
    if isinstance(exc, HTTPError):
        return exc.code if exc.code in {401, 402, 403} else None
    if type(exc) is RuntimeError:
        return next((code for code in (401, 402, 403)
                     if str(exc) == f'Model service returned HTTP {code}'), None)
    return None

def call_json(runtime,prompt,payload,max_tokens,root,attempt=0):
    opts=runtime.config['llm']
    check_llm(opts)
    body=dict(opts.get('extra_body') or {})
    body.update(model=opts['model'],temperature=0,max_tokens=max_tokens,seed=42,enable_thinking=False,response_format={'type':'json_object'},messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    key=digest(dict(body=body,runtime_config=runtime.config,attempt=attempt,root=str(root)));start=time.monotonic()
    saved=root/(key+'.json')
    if saved.exists():
        record=json.loads(saved.read_text('utf-8'));raw=record['response']
        if record['request']!=body:raise ValueError('Cached request identity mismatch')
    else:
        raw=None
    with runtime.gate:
        raw=raw or runtime._raw_cached(key)
        if raw is None:
            try:raw=runtime._post('/chat/completions',body,opts,key,'course_evaluation')
            except RuntimeError as exc:
                # 运行时已保留 HTTP/网络详情，轻量标记保存身份但不复制异常中的未知数据。
                write_immutable(root/(key+'.error.json'),dict(error_type=type(exc).__name__,request_id=key,stage='course_evaluation',status_code=account_status(exc)))
                raise
    if not saved.exists():
        record=dict(request=body,response=raw,latency_seconds=time.monotonic()-start)
        write_immutable(saved,record)
    choice=raw['choices'][0]
    if choice.get('finish_reason')=='length':raise ValueError('Evaluation model output truncated')
    result=json.loads(choice['message']['content'])
    if not isinstance(result,dict):raise ValueError('Evaluation model returned non-object')
    return result,record
