"""原生QA单次长度恢复；原始2048结果仍由既有请求收据保存。"""
from __future__ import annotations
from .native_hipporag import AuditedLLM

POLICY = dict(version='native_hippo_qa_length2048_to4096_once_v1',
    role='qa_llm', initial_tokens=2048, recovery_tokens=4096,
    trigger='actual finish_reason=length at2048', same_messages=True,
    second_length='incomplete_fixed_denominator', maximum_recovery_calls=1)

class BoundedQaLengthRecovery(AuditedLLM):
    def infer(self, chat=None, **kwargs):
        result = super().infer(chat, **kwargs)
        aliases = ('max_new_tokens','max_tokens','max_completion_tokens','max_output_tokens')
        limit = next((kwargs[k] for k in aliases if k in kwargs), self.global_config.max_new_tokens)
        if self.role != 'qa_llm' or limit != 2048 or result[1].get('finish_reason') != 'length':
            return result
        retry = {k:v for k,v in kwargs.items() if k not in aliases}
        text, metadata, hit = super().infer(chat, **retry, max_tokens=4096)
        metadata = dict(metadata, bounded_length_recovery=dict(initial_max_tokens=2048,
            initial_finish_reason='length', initial_cache_hit=result[2], retry_max_tokens=4096,
            original_usage=result[1]))
        return text, metadata, hit
