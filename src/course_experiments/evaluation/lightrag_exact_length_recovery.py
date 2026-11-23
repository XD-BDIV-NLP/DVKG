"""仅已审计的8192截断请求可原样再试16384；其他请求沿用旧策略。"""
from __future__ import annotations
from .lightrag_length_recovery import callbacks as original_callbacks
from ..manifest import digest

POLICY=dict(version='lightrag_exact_known_8192_to16384_once_v1',
    maximum_additional_attempts=1,trigger='exact authorized request_id and extract8192 length',
    source_and_messages='unchanged',second_length='failure',official_limit_unchanged_claimed=False)

def callbacks(provider, allowed_request_ids):
    allowed=frozenset(allowed_request_ids)
    class Proxy:
        def chat_response(self,messages,**options):
            raw=provider.chat_response(messages,**options)
            if (options.get('max_tokens')!=8192 or not options.get('stage','').endswith('.extract')
                    or raw['choices'][0].get('finish_reason')!='length'):
                return raw
            body=dict(model=provider.runtime.config['llm']['model'],messages=messages,
                temperature=0,seed=42,enable_thinking=False,max_tokens=8192)
            if options.get('response_format') is not None:body['response_format']=options['response_format']
            identity=digest(dict(endpoint='/chat/completions',body=body,config=provider.runtime.config))
            if identity not in allowed:return raw
            return provider.chat_response(messages,**dict(options,max_tokens=16384))
        async def aembed(self,*args,**kwargs):return await provider.aembed(*args,**kwargs)
    return original_callbacks(Proxy())
