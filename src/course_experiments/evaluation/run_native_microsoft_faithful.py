"""默认1200窗口、完整标准索引和Local测评；旧600窗口实验保持独立。"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import json

from ..manifest import digest, file_record, read_document
from . import native_microsoft_faithful, native_microsoft_query_mapping
from . import run_native_microsoft_full as original
from .native_microsoft_query_mapping import remap_native_result
from .native_provider import AuditedNativeProvider, NativeBudgetExceeded
from .protocol import runtime_for
from .run_repaired_full import credentials

METHOD_HTTP_CAP = 20000


def configure():
    settings = native_microsoft_faithful.install()
    original.NATIVE_SETTINGS = settings
    original.TRACKS = ('local',)
    original.POLICY = dict(original.POLICY,
        version='microsoft_default1200_faithful_local_full_v1',
        native_settings=settings,
        microsoft_global='not_run_in_full_benchmark_integration_examples_only',
        source_grounding='complete_original_characters_in_native_token_byte_windows',
        effective_cumulative_HTTP=80000, method_cumulative_HTTP_cap=METHOD_HTTP_CAP)


def prepare_plan(run_root, book, **kwargs):
    configure()
    plan = original.prepare_plan(run_root, book, **kwargs)
    files = [Path(__file__), Path(native_microsoft_faithful.__file__),
             Path(native_microsoft_query_mapping.__file__),
             Path(native_microsoft_query_mapping.native_microsoft_spans.__file__)]
    for path in files:
        record = file_record(path)
        plan['contract']['code'][path.name] = record
        plan['index_contract']['implementation'][path.name] = record['sha256']
    plan['contract'].update(index_identity=digest(plan['index_contract']),
        tracks=['local'], effective_cumulative_HTTP=80000,
        method_cumulative_HTTP_cap=METHOD_HTTP_CAP,
        extension_authorization_call='call_ec6a116c31f8441cb621cded7ca170b9')
    return plan


async def query(session, plan, question, **kwargs):
    native = await original.query_microsoft(session, plan, question, **kwargs)
    return remap_native_result(session, native, size=1200, overlap=100)


class BoundedGraphProvider(AuditedNativeProvider):
    def chat_response(self, messages, **kwargs):
        if kwargs.get('stage') == 'native_blind_judge':
            kwargs['stage'] = 'microsoft_graphrag.native_blind_judge'
        return super().chat_response(messages, **kwargs)

    def _write_transaction(self, operation_fn, *, operation, request_id=None):
        def bounded(db):
            if operation == 'event_request_started':
                count = db.execute("SELECT COUNT(*) FROM events WHERE status='request_started' AND stage LIKE 'microsoft_graphrag.%'").fetchone()[0]
                if count >= METHOD_HTTP_CAP:
                    raise NativeBudgetExceeded('GraphRAG cumulative method HTTP allowance exhausted; keep checkpoints')
            return operation_fn(db)
        return super()._write_transaction(bounded, operation=operation, request_id=request_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--book', required=True)
    args = parser.parse_args()
    plan = prepare_plan(args.run_root, args.book)
    root = args.run_root/'formal_provider'
    if read_document(root/'limits.json')['limits'] != original.shared.LIMITS:
        raise ValueError('Original limits and deadline must remain unchanged')
    identity = digest(plan['contract'])
    transport = args.run_root/'native_full/graphrag'/args.book/identity[:16]/'transport'
    credentials()
    provider = BoundedGraphProvider(runtime_for(plan['construction'], transport), root,
                                    **original.shared.LIMITS)
    if provider.effective_http_cap != 80000:
        provider.close()
        raise ValueError('Human-authorized cumulative allowance is required')
    result = asyncio.run(original.run_book(plan, args.run_root, provider, query_fn=query))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
