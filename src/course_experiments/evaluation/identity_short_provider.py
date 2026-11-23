"""短标签传输适配；原始响应由正式provider保留，恢复后仍走原严格裁判。"""
from __future__ import annotations

import copy
import json

from ..manifest import digest
from . import identity_wire_labels as labels


POLICY=dict(version='identity_short_provider_v1',wire=labels.POLICY,
    judge='unchanged_original_single_or_batch_judge',old_success='reuse_strict_original_checkpoints',
    old_failures='preserved_original_files_new_request_configuration',
    returned_raw='restored_copy_original_raw_preserved_in_provider_and_response')


class ShortLabelProvider:
    def __init__(self,provider):self.provider=provider

    def __getattr__(self,name):return getattr(self.provider,name)

    @property
    def request_executor(self):return self.provider.request_executor

    @request_executor.setter
    def request_executor(self,value):self.provider.request_executor=value

    def chat_response(self,messages,**kwargs):
        wire=json.loads(messages[-1]['content']);short,table=labels.encode_wire(wire)
        if labels.restore_wire(short,table)!=wire:raise ValueError('Original identity wire changed')
        outgoing=copy.deepcopy(messages[:-1])+[dict(role='system',content=labels.INSTRUCTION),
            dict(role='user',content=json.dumps(short,ensure_ascii=False))]
        raw=self.provider.chat_response(outgoing,**kwargs)
        result=copy.deepcopy(raw)
        if result['choices'][0].get('finish_reason')=='stop':
            value=json.loads(result['choices'][0]['message']['content'])
            result['choices'][0]['message']['content']=json.dumps(labels.decode_response(value,table),ensure_ascii=False)
        result['identity_short_wire']=dict(policy=POLICY,mapping=table,original_response_digest=digest(raw),
            original_raw_response=raw)
        return result
