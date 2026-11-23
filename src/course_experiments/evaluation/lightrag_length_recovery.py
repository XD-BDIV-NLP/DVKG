"""独立续跑变体：仅抽取length增加一次输出预算，旧收据完整保留。"""
from __future__ import annotations

import asyncio

POLICY = dict(version='lightrag_extract_length4096_to8192_once_v1',
    initial_tokens=4096, recovery_tokens=8192, maximum_recovery_calls=1,
    trigger='extract role and actual raw finish_reason=length at4096',
    text='exact same source, prompt and history; no input truncation',
    completed_native_documents='retain original processed documents and all source IDs',
    budget='old attempts and new attempts remain charged to cumulative formal allowance',
    second_length='fail; never claim complete or drop failed source')


def callbacks(provider):
    async def chat(prompt, *, system_prompt=None, history_messages=None, purpose, **kwargs):
        kwargs.pop('enable_cot', None)
        limit = kwargs.pop('max_tokens', 4096)
        response_format = kwargs.pop('response_format', None)
        if kwargs:
            raise ValueError('Unsupported native LightRAG kwargs: '+','.join(sorted(kwargs)))
        messages = ([dict(role='system', content=system_prompt)] if system_prompt else [])
        messages += list(history_messages or [])+[dict(role='user', content=prompt)]
        async def request(tokens):
            return await asyncio.to_thread(provider.chat_response, messages,
                max_tokens=tokens, response_format=response_format, stage=purpose)
        raw = await request(limit)
        if (raw['choices'][0].get('finish_reason') == 'length' and limit == 4096
                and purpose.endswith('.extract')):
            raw = await request(8192)
        choice = raw['choices'][0]
        text = choice['message']['content']
        if choice.get('finish_reason') != 'stop' or not isinstance(text, str) or not text.strip():
            raise ValueError('Native LightRAG completion incomplete; preserved raw receipt')
        return text

    async def embed(texts, *, purpose):
        return await provider.aembed(texts, stage=purpose)
    return chat, embed
