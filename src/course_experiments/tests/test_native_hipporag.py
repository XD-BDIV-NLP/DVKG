from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from course_experiments.evaluation import native_hipporag as native


def config():
    return SimpleNamespace(llm_name=native.LLM_MODEL, llm_base_url=native.BASE_URL,
        embedding_model_name=native.EMBEDDING_MODEL, embedding_base_url=native.BASE_URL,
        embedding_return_as_normalized=True, max_new_tokens=2048)


def completion(content='Answer: synthetic', finish='stop'):
    return dict(choices=[dict(message=dict(content=content), finish_reason=finish)],
        usage=dict(prompt_tokens=7, completion_tokens=3, total_tokens=10))


class NativeAdapterTests(unittest.TestCase):
    def test_native_empty_entity_string_reaches_same_model_without_substitution(self):
        requests = []
        def execute(endpoint, body, stage):
            requests.append(body)
            return dict(data=[dict(index=0, embedding=[1.0]*1024)],
                        usage=dict(prompt_tokens=1,total_tokens=1)), False
        with tempfile.TemporaryDirectory() as root:
            model = native.AuditedEmbedding(config(), execute, Path(root)/'audit.jsonl')
            result = model.batch_encode([''])
        self.assertEqual(requests[0]['input'], [''])
        self.assertEqual(requests[0]['model'], native.EMBEDDING_MODEL)
        self.assertEqual(result.shape, (1,1024))
        self.assertAlmostEqual(float(np.linalg.norm(result[0])), 1.0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_llm_preserves_native_messages_format_and_token_alias(self):
        requests = []
        def execute(endpoint, body, stage):
            requests.append((endpoint, body, stage))
            return completion(), True
        model = native.AuditedLLM(config(), execute, self.root / 'calls.jsonl')
        messages = [dict(role='user', content='synthetic prompt')]
        text, metadata, cached = model.infer(messages=messages, max_new_tokens=512)
        self.assertEqual(text, 'Answer: synthetic')
        self.assertTrue(cached)
        self.assertEqual(metadata['prompt_tokens'], 7)
        self.assertEqual(requests[0][1]['messages'], messages)
        self.assertEqual(requests[0][1]['max_tokens'], 512)
        self.assertNotIn('response_format', requests[0][1])
        self.assertFalse(requests[0][1]['enable_thinking'])
        model.infer(messages, response_format={'type': 'json_object'})
        self.assertEqual(requests[1][1]['response_format'], {'type': 'json_object'})
        audit = (self.root / 'calls.jsonl').read_text('utf-8')
        self.assertNotIn('synthetic prompt', audit)
        self.assertEqual(len(audit.splitlines()), 2)

    def test_failure_is_sticky_even_if_upstream_swallows_it(self):
        count = []
        def execute(*args):
            count.append(1)
            raise RuntimeError('budget exhausted')
        model = native.AuditedLLM(config(), execute, self.root / 'calls.jsonl')
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                model.infer([dict(role='user', content='synthetic')])
        self.assertEqual(len(count), 1)
        with self.assertRaises(RuntimeError):
            model.check_error()

    def test_preserves_native_truncation_metadata_and_rejects_missing_usage(self):
        model = native.AuditedLLM(config(), lambda *args: (completion(finish='length'), False), self.root / 'calls.jsonl')
        text, metadata, _ = model.infer([dict(role='user', content='synthetic')])
        self.assertEqual(metadata['finish_reason'], 'length')
        self.assertEqual(text, 'Answer: synthetic')
        for raw in [completion(finish='content_filter'), dict(choices=completion()['choices'])]:
            model = native.AuditedLLM(config(), lambda *args: (raw, False), self.root / 'calls.jsonl')
            with self.assertRaises(ValueError):
                model.infer([dict(role='user', content='synthetic')])

    def test_rejects_provider_or_model_changes_before_executor(self):
        for field, value in [('llm_base_url', 'https://example.com/v1'), ('llm_name', 'other')]:
            cfg = config()
            setattr(cfg, field, value)
            with self.assertRaises(ValueError):
                native.AuditedLLM(cfg, lambda *args: self.fail('Network boundary crossed'), self.root / 'calls.jsonl')

    def test_embedding_reorders_provider_indices_and_normalizes(self):
        requests = []
        def execute(endpoint, body, stage):
            requests.append((endpoint, body, stage))
            return dict(data=[dict(index=1, embedding=[0., 4.] + [0.] * 1022),
                              dict(index=0, embedding=[3., 0.] + [0.] * 1022)],
                        usage=dict(prompt_tokens=4, total_tokens=4)), False
        model = native.AuditedEmbedding(config(), execute, self.root / 'calls.jsonl')
        vectors = model.batch_encode(['a', 'b'], norm=True)
        np.testing.assert_array_equal(vectors[:, :2], np.eye(2))
        self.assertEqual(vectors.shape, (2, 1024))
        self.assertEqual(requests[0][0], '/embeddings')
        self.assertEqual(requests[0][1]['input'], ['a', 'b'])
        self.assertEqual(model.batch_encode([]).shape, (0, 1024))

    def test_embedding_rejects_missing_rows_and_invalid_dimensions(self):
        for data in [[], [dict(index=0, embedding=[1., 2.])],
                     [dict(index=0, embedding=[0.] * 1024)]]:
            model = native.AuditedEmbedding(config(), lambda *args: (dict(data=data,
                usage=dict(prompt_tokens=1, total_tokens=1)), False), self.root / 'calls.jsonl')
            with self.assertRaises(ValueError):
                model.batch_encode(['a'])

    def test_embedding_single_string_remains_a_batch_and_respects_batch_size(self):
        batches = []
        def execute(endpoint, body, stage):
            batches.append(list(body['input']))
            return dict(data=[dict(index=i, embedding=[1.] + [0.] * 1023)
                for i, _ in enumerate(body['input'])], usage=dict(prompt_tokens=1, total_tokens=1)), False
        cfg = config()
        cfg.embedding_batch_size = 2
        model = native.AuditedEmbedding(cfg, execute, self.root / 'calls.jsonl')
        self.assertEqual(model.batch_encode('a').shape, (1, 1024))
        self.assertEqual(model.batch_encode(['a', 'b', 'c']).shape, (3, 1024))
        self.assertEqual(batches, [['a'], ['a', 'b'], ['c']])

    def test_rows_reject_foreign_book_and_aggregate_duplicate_text_metadata(self):
        rows = [dict(book_id='book_a', text='synthetic', evidence_ids=['e1'], chapter_id='c1'),
                dict(book_id='book_a', text='synthetic', evidence_ids=['e2'], chapter_id='c2')]
        docs, identity = native.prepare_documents(rows, 'book_a', {'book_a'})
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['metadata']['evidence_ids'], ['e1', 'e2'])
        self.assertEqual(identity['row_count'], 2)
        with self.assertRaises(ValueError):
            native.prepare_documents(rows, 'book_b', {'book_a', 'book_b'})

    def test_source_must_match_frozen_commit(self):
        with self.assertRaises(RuntimeError):
            native.verify_source(self.root)


@unittest.skipUnless(os.environ.get('HIPPORAG_NATIVE_SOURCE'), 'Pinned upstream dependency integration is opt-in')
class NativeUpstreamIntegrationTests(unittest.TestCase):
    def test_real_upstream_index_ppr_and_qa_with_synthetic_model_responses(self):
        calls = []
        def execute(endpoint, body, stage):
            calls.append((endpoint, body, stage))
            if endpoint == '/embeddings':
                rows = []
                for index, text in enumerate(body['input']):
                    vector = [0.] * 1024
                    vector[0] = 1.
                    vector[1] = .1 if 'beta' in text.lower() else .2
                    rows.append(dict(index=index, embedding=vector))
                return dict(data=rows, usage=dict(prompt_tokens=2, total_tokens=2)), False
            joined = '\n'.join(m['content'] for m in body['messages'])
            if 'fact_after_filter' in joined:
                answer = '[[ ## fact_after_filter ## ]]\n{"fact":[["alpha","connects","beta"]]}\n[[ ## completed ## ]]'
            elif body['max_tokens'] == 4096:
                answer = '{"triples":[["alpha","connects","beta"]]}'
            elif 'named_entities' in joined:
                answer = '{"named_entities":["alpha","beta"]}'
            else:
                answer = 'Answer: beta'
            return completion(answer), False
        with tempfile.TemporaryDirectory() as folder:
            bridge = native.NativeHippoRAG(source_root=os.environ['HIPPORAG_NATIVE_SOURCE'],
                save_dir=folder, target_book_id='book_a', allowed_book_ids={'book_a'},
                request_executor=execute)
            self.addCleanup(bridge.close)
            self.assertEqual(bridge.config.response_format, {'type': 'json_object'})
            bridge.index([dict(book_id='book_a', text='Alpha connects beta.', evidence_ids=['e1'])])
            solutions = bridge.retrieve(['What does alpha connect?'])
            self.assertTrue(solutions[0].graph_seeds)
            self.assertEqual(solutions[0].docs, ['Alpha connects beta.'])
            result = bridge.rag_qa(solutions)
            self.assertEqual(result[0][0].answer, 'beta')
            self.assertEqual(bridge.metadata()['upstream_commit'], native.SOURCE_PIN)
            self.assertEqual(bridge.metadata()['native_retrieval_top_k'], 200)
            self.assertTrue(any(c[1].get('max_tokens') == 512 for c in calls))
            for endpoint, body, stage in calls:
                if endpoint != '/chat/completions':
                    continue
                if stage == 'native_hipporag/extraction_llm' and body['max_tokens'] in {2048, 4096}:
                    self.assertEqual(body.get('response_format'), {'type': 'json_object'})
                else:
                    self.assertNotIn('response_format', body)
            solution = solutions[0]
            solution.doc_metadata[0]['evidence_ids'] = ['forged']
            before = len(calls)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                bridge.rag_qa(solutions)
            self.assertEqual(len(calls), before)


if __name__ == '__main__':
    unittest.main()
