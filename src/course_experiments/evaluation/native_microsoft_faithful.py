"""官方默认窗口和原生 content 传递契约；旧严格变体另存。"""
from __future__ import annotations

import json

from . import native_microsoft_graphrag as original


def settings():
    return dict(response_type="Single Paragraph",
                index_configuration_revision="official_default1200_native_content_schema_repair_once_v2",
                chunks=dict(size=1200, overlap=100, group_by_columns=["id"],
                            strategy="tokens", encoding_model="cl100k_base",
                            prepend_metadata=False, chunk_size_includes_metadata=False),
                declared_tracks=["local"],
                fail_fast="transport or official parser failure stops; finish_reason alone is not a parser failure",
                content_contract="official FNLLM forwards content to GraphExtractor and its original one-glean parser",
                json_schema_repair=dict(max_attempts=1, trigger="official model ValidationError",
                    same_token_limit=True, raw_response_preserved=True, failure_remains_failure=True))


class FaithfulGraphChat(original.AuditedGraphChat):
    def _raw_content(self, messages, options):
        raw = self.provider.chat_response(messages, **options)
        choice = raw["choices"][0]
        self.truncation_recoveries.append(dict(stage=options["stage"],
            finish_reason=choice.get("finish_reason"),
            max_tokens=options["max_tokens"], cache_hit=raw.get("_native_cache_hit"),
            usage=raw.get("usage"), extra_strict_retry=False,
            all_facts_complete_claimed=False))
        return choice["message"]["content"]

    def _response(self, content, messages, kwargs):
        if not isinstance(content, str):
            raise ValueError("Official native content must be text")
        model = kwargs.get("json_model")
        parsed = None
        if model is not None:
            from pydantic import ValidationError
            try:
                parsed = model.model_validate_json(content)
            except ValidationError as exc:
                # 只修复官方JSON模式；原响应、错误字段和新增请求均留在既有收据中。
                fields = [dict(loc=list(e['loc']), type=e['type']) for e in exc.errors()]
                repair = dict(task='Return the complete JSON object conforming to the required schema. '
                    'Keep source-grounded content; repair missing or malformed fields. No commentary.',
                    schema=model.model_json_schema(), validation_errors=fields)
                repair_messages = messages + [dict(role='assistant', content=content),
                    dict(role='user', content=json.dumps(repair, ensure_ascii=False))]
                _, options = self._request('', None, kwargs)
                options['stage'] += '.schema_repair.1'
                self.truncation_recoveries.append(dict(stage=options['stage'],
                    schema_repair_attempt=1, validation_errors=fields,
                    original_content_sha256=original.hashlib.sha256(content.encode('utf8')).hexdigest(),
                    max_tokens=options['max_tokens'], all_facts_complete_claimed=False))
                content = self._raw_content(repair_messages, options)
                parsed = model.model_validate_json(content)
                messages = repair_messages
        return original.ChatResponse(original.ChatOutput(content),
            messages + [dict(role="assistant", content=content)], parsed)


def install():
    """仅在单独冻结的执行进程使用；不修改任何旧 APP 或索引。"""
    original.NATIVE_SETTINGS = settings()
    original.AuditedGraphChat = FaithfulGraphChat
    original.DISCLOSURE = dict(original.DISCLOSURE,
        chunk_revision="official default1200 tokens and100 overlap; unchanged full source documents",
        truncation_recovery="official content/one-glean/parser contract; no extra4096-to8192 retry; finish/usage retained, no exhaustive-facts claim",
        native_track="official Local search primary; Global integration examples retained separately, no full Global result claimed")
    return original.NATIVE_SETTINGS
