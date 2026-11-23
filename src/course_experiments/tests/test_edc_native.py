from __future__ import annotations

import ast
import importlib
import importlib.util
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from email.message import Message
from unittest.mock import patch
from types import SimpleNamespace
import urllib.request
import urllib.response


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / '.runtime_cache/expanded_baseline_sources_20261016/edc'
DEPS = ROOT / '.runtime_cache/edc_native_test_dependencies'
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

REAL_SD = (
    '包含：主体实体包含客体实体，客体实体是主体实体的组成部分或成员。  \n执行：主体实体执行或实施客体实体所表示的动作、任务或操作。  \n协调：主体实体对客体实体所表示的过程、活动或任务进行协调、安排或指挥。  \n从：主体实体以客体实体为来源、起点或依据，进行某种动作或获取信息；客体表示动作的来源或出发点。',
    '保存：主语实体保存或存储宾语实体所指的内容或数据。\n\n执行：主语实体执行宾语实体所表示的操作、指令或任务。\n\n包含：主语实体包含或含有宾语实体作为其一部分。\n\n别名：宾语实体是主语实体的另一个名称，即主语实体也称为宾语实体。',
)
REAL_SD_HASHES = ('586ebf762c441b8f1fa7a486894f3bfad36317a645e65e697eb4d40852620563',
                  '30f0e4cc9acc2757887cbedbbcbdde6ae6949e77839d092b6ef7522a9058efa6')
REAL_OIE = (
    "[['处理器', '包含', '运算器'], ['处理器', '包含', '控制器'], ['处理器', '执行', '指令'], ['运算器', '执行', '算术运算'], ['控制器', '协调', '指令执行'], ['处理器', '从', '存储器读取指令']]",
    "[['中央处理器', '别名', '处理器'], ['处理器', '执行', '指令'], ['控制器', '包含', '指令译码器'], ['存储器', '保存', '指令'], ['存储器', '保存', '数据']]",
)


class EDCNativeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('course_experiments.baselines.edc_native'),
                             'independent native EDC bridge is missing')
        self.edc = importlib.import_module('course_experiments.baselines.edc_native')
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.runtime_cache')
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)

    def prepare(self, logical=30, attempts=90):
        return self.edc.prepare(REPO, self.run, self.edc.synthetic_units(),
                                max_logical_calls=logical, max_http_attempts=attempts)

    def transport(self, sender, logical=30, attempts=90):
        plan = self.prepare(logical, attempts)
        return self.edc.Transport(self.run, plan, 'fixture-secret', sender=sender)

    def native_function(self, relative, name):
        path = self.run / 'native_source' / relative
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = {'logging': logging, 'logger': logging.getLogger('edc_native_isolated_parser_test')}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace[name]

    def test_real_v11_sd_reproduces_original_parser_and_native_keyerror(self):
        self.prepare()
        parser = self.native_function('edc/utils/llm_utils.py', 'parse_relation_definition')
        canonicalize = self.native_function('edc/schema_canonicalization.py', 'canonicalize')
        for raw, expected_hash in zip(REAL_SD, REAL_SD_HASHES):
            self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), expected_hash)
            parsed = parser(raw)
            self.assertEqual(parsed, {})
            with self.assertRaisesRegex(KeyError, '包含'):
                canonicalize(SimpleNamespace(schema_dict={}), 'synthetic', ['处理器', '包含', '运算器'], parsed, '', enrich=True)

    def test_sd_separator_compatibility_is_header_only_and_preserves_actual_definitions(self):
        self.assertTrue(hasattr(self.edc, '_sd_separator_compat'), 'SD response separator compatibility missing')
        self.prepare()
        expected_sets = ({'包含', '执行', '协调', '从'}, {'保存', '执行', '包含', '别名'})
        parser = self.native_function('edc/utils/llm_utils.py', 'parse_relation_definition')
        for raw, relations in zip(REAL_SD, expected_sets):
            transformed, changes = self.edc._sd_separator_compat(raw, relations)
            self.assertEqual(set(parser(transformed)), relations)
            self.assertEqual(len(changes), 4)
            for before, after in zip(raw.splitlines(), transformed.splitlines()):
                self.assertEqual(before.replace('：', ':', 1), after)
        raw = '包含：正文：这个冒号保留: ASCII也保留。\n备注：不是已抽出的关系。\n执行:原ASCII:正文。\n**包含**：不兼容markdown。'
        transformed, changes = self.edc._sd_separator_compat(raw, {'包含', '执行'})
        self.assertEqual(transformed, '包含:正文：这个冒号保留: ASCII也保留。\n备注：不是已抽出的关系。\n执行:原ASCII:正文。\n**包含**：不兼容markdown。')
        self.assertEqual(changes, [{'line': 1, 'relation': '包含', 'from': '：', 'to': ':'}])

    def test_real_sd_response_compatibility_runs_native_without_inventing_missing_definitions(self):
        self.assertTrue(hasattr(self.edc, '_sd_separator_compat'), 'SD response separator compatibility missing')
        transport = self.transport(lambda *args: self.fail('offline parser test must not call HTTP'))
        parser = self.native_function('edc/utils/llm_utils.py', 'parse_relation_definition')
        for raw, oie in zip(REAL_SD, REAL_OIE):
            triples = ast.literal_eval(oie)
            expected = {triple[1] for triple in triples}
            text = transport.sd_response_compatibility(raw, expected, parser, {'fixture': oie})
            self.assertEqual(set(parser(text)), expected)
        events = [json.loads(line) for line in (self.run / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        transformed = [e for e in events if e['event'] == 'sd_separator_compatibility']
        self.assertEqual(len(transformed), 2)
        for event, raw in zip(transformed, REAL_SD):
            self.assertEqual(event['original_text'], raw)
            self.assertEqual(event['transformed_text'], raw.replace('：', ':'))
        self.assertFalse(any(e['event'] == 'attempt_started' for e in events))

    def test_missing_sd_definitions_still_fail_with_compatibility(self):
        self.assertTrue(hasattr(self.edc, '_sd_separator_compat'), 'SD response separator compatibility missing')
        transport = self.transport(lambda *args: self.fail('offline parser test must not call HTTP'))
        parser = self.native_function('edc/utils/llm_utils.py', 'parse_relation_definition')
        with self.assertRaisesRegex(self.edc.NativeRunError, 'missing SD definitions'):
            transport.sd_response_compatibility('包含：仅提供包含定义。', {'包含', '执行'}, parser, {'fixture': 'missing'})
        self.assertEqual(transport.events[-1]['event'], 'sd_separator_compatibility')

    def test_send_rejects_cross_and_same_host_redirect_without_forward_or_post_replay(self):
        real_builder = urllib.request.build_opener
        for status in (301, 302, 303, 307, 308):
            for target in ('https://other.invalid/stolen', self.edc.BASE_URL + '/redirected'):
                with self.subTest(status=status, target=target):
                    sent = []
                    class FixtureHTTPS(urllib.request.HTTPSHandler):
                        def https_open(self, request):
                            sent.append((request.full_url, request.get_method(), dict(request.header_items())))
                            headers = Message()
                            code = 200 if len(sent) > 1 else status
                            if code != 200:
                                headers['Location'] = target
                            response = urllib.response.addinfourl(io.BytesIO(b'{"redirect":"blocked"}'), headers,
                                                                  request.full_url, code)
                            response.msg = 'fixture'
                            return response
                    handler = FixtureHTTPS()
                    legacy = real_builder(urllib.request.ProxyHandler({}), handler)
                    request = urllib.request.Request(self.edc.BASE_URL + '/chat/completions', data=b'{}',
                                                     headers={'Authorization': 'Bearer fixture-secret'})
                    with patch.object(urllib.request, 'build_opener', side_effect=lambda *h: real_builder(*h, handler)), \
                            patch.object(urllib.request, 'urlopen', side_effect=legacy.open):
                        code, raw = self.edc.Transport._send(request, 240)
                    self.assertEqual(code, status)
                    self.assertEqual(raw, '{"redirect":"blocked"}')
                    self.assertEqual(len(sent), 1, 'redirect must not forward headers or replay POST')
                    self.assertEqual(sent[0][0], self.edc.BASE_URL + '/chat/completions')
                    self.assertEqual(sent[0][1], 'POST')

    def test_send_ignores_environment_proxy_and_journals_exactly_one_opener_open(self):
        real_builder = urllib.request.build_opener
        real_open = urllib.request.OpenerDirector.open
        sent = []
        opens = []
        class FixtureHTTPS(urllib.request.HTTPSHandler):
            def https_open(self, request):
                sent.append((request.full_url, request.host))
                response = urllib.response.addinfourl(io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}'),
                                                      Message(), request.full_url, 200)
                response.msg = 'fixture'
                return response
        handler = FixtureHTTPS()
        def counted_open(opener, *args, **kwargs):
            opens.append(1)
            return real_open(opener, *args, **kwargs)
        def legacy_urlopen(*args, **kwargs):
            return real_builder(handler).open(*args, **kwargs)
        proxy = 'http://unexpected-proxy.invalid:8080'
        with patch.dict(os.environ, {'HTTPS_PROXY': proxy, 'https_proxy': proxy, 'NO_PROXY': '', 'no_proxy': ''}), \
                patch.object(urllib.request, 'build_opener', side_effect=lambda *h: real_builder(*h, handler)), \
                patch.object(urllib.request, 'urlopen', side_effect=legacy_urlopen), \
                patch.object(urllib.request.OpenerDirector, 'open', new=counted_open):
            transport = self.transport(self.edc.Transport._send)
            result = transport.request('chat/completions', {'model': self.edc.MODEL}, 'oie')
        self.assertEqual(result['choices'][0]['message']['content'], 'ok')
        self.assertEqual(sent, [(self.edc.BASE_URL + '/chat/completions', 'api.siliconflow.cn')])
        self.assertEqual(len(opens), 1)
        self.assertEqual(sum(e['event'] == 'attempt_started' for e in transport.events), 1)
        self.assertEqual(sum(e['event'] == 'attempt_finished' for e in transport.events), 1)

    def test_prepare_is_zero_model_and_freezes_exact_configuration(self):
        plan = self.prepare()
        self.assertEqual(plan['source_pin'], '02aad17e2fc1e9845c84746b384e7fd3160edc1a')
        self.assertEqual(plan['native_max_tokens'], {'oie': 512, 'sd': 512, 'sc': 1})
        self.assertEqual(plan['model'], 'deepseek-ai/DeepSeek-V4-Flash')
        self.assertEqual(plan['embedding_model'], 'BAAI/bge-m3')
        self.assertEqual(plan['model_execution_status'], 'not_executed')
        self.assertFalse((self.run / 'attempts.jsonl').exists())
        self.assertEqual(plan, self.prepare())
        with self.assertRaises(ValueError):
            self.prepare(logical=31)

    def test_verified_source_bundle_prepares_without_a_git_checkout(self):
        plan = self.prepare()
        other = self.run / 'new_run'
        copy = self.edc.prepare(self.run / 'native_source', other, self.edc.synthetic_units(),
                                max_logical_calls=30, max_http_attempts=90)
        self.assertEqual(copy['source_files'], plan['source_files'])

    def test_bounded_retry_and_no_failed_budget_reset(self):
        calls = []
        def unavailable(req, timeout):
            calls.append(timeout)
            return 503, '{"error":"busy"}'
        transport = self.transport(unavailable)
        body = {'model': self.edc.MODEL, 'messages': [], 'max_tokens': 512}
        with self.assertRaises(self.edc.NativeRunError):
            transport.request('chat/completions', body, 'oie')
        self.assertEqual(calls, [240, 240, 240])
        resumed = self.edc.Transport(self.run, self.prepare(), 'fixture-secret', sender=unavailable)
        with self.assertRaises(self.edc.NativeRunError):
            resumed.request('chat/completions', body, 'oie')
        self.assertEqual(len(calls), 3)

    def test_402_stops_immediately_and_survives_restart(self):
        calls = []
        def exhausted(req, timeout):
            calls.append(1)
            return 402, '{"error":"fixture-secret balance"}'
        transport = self.transport(exhausted)
        with self.assertRaises(self.edc.NativeRunError):
            transport.request('chat/completions', {'model': self.edc.MODEL}, 'sd')
        with self.assertRaises(self.edc.NativeRunError):
            self.edc.Transport(self.run, self.prepare(), 'fixture-secret', sender=exhausted).request(
                'embeddings', {'model': self.edc.EMBEDDING_MODEL}, 'sc_embedding')
        self.assertEqual(len(calls), 1)
        self.assertNotIn('fixture-secret', (self.run / 'attempts.jsonl').read_text())

    def test_exact_cache_and_budget_replay_have_no_new_http(self):
        calls = []
        def good(req, timeout):
            calls.append(json.loads(req.data))
            return 200, '{"choices":[{"message":{"content":"答案"}}]}'
        transport = self.transport(good, logical=1, attempts=3)
        body = {'model': self.edc.MODEL, 'messages': [], 'max_tokens': 512}
        self.assertEqual(transport.request('chat/completions', body, 'oie')['choices'][0]['message']['content'], '答案')
        replay = self.edc.Transport(self.run, self.prepare(1, 3), 'fixture-secret', sender=good)
        self.assertEqual(replay.request('chat/completions', body, 'oie'), transport.request('chat/completions', body, 'oie'))
        self.assertEqual(len(calls), 1)
        with self.assertRaises(self.edc.NativeRunError):
            replay.request('chat/completions', body, 'sd')
        self.assertEqual(len(calls), 1)

    def test_durable_200_response_is_recovered_without_recharge_after_cache_write_crash(self):
        calls = []
        def good(req, timeout):
            calls.append(1)
            return 200, '{"choices":[{"message":{"content":"恢复"}}]}'
        transport = self.transport(good)
        original_append = transport._append
        def crash_on_success(event):
            if event['event'] == 'success':
                raise OSError('simulated crash after durable response')
            original_append(event)
        transport._append = crash_on_success
        body = {'model': self.edc.MODEL, 'messages': [], 'max_tokens': 512}
        with self.assertRaises(OSError):
            transport.request('chat/completions', body, 'oie')
        replay = self.edc.Transport(self.run, self.prepare(), 'fixture-secret', sender=good)
        self.assertEqual(replay.request('chat/completions', body, 'oie')['choices'][0]['message']['content'], '恢复')
        self.assertEqual(len(calls), 1)

    def test_plan_configuration_tamper_blocks_execute_before_http(self):
        self.prepare()
        path = self.run / 'prepare.json'
        plan = json.loads(path.read_text(encoding='utf-8'))
        plan['base_url'] = 'https://invalid.example/v1'
        path.write_text(json.dumps(plan), encoding='utf-8')
        with self.assertRaisesRegex(self.edc.NativeRunError, 'freeze'):
            self.edc.execute(self.run, api_key='fixture-secret', sender=lambda *args: self.fail('HTTP must not run'))

    def test_pending_attempt_and_offline_cache_cannot_be_recharged_as_live(self):
        calls = []
        def crash(req, timeout):
            calls.append(1)
            raise RuntimeError('process interrupted before response')
        transport = self.transport(crash)
        body = {'model': self.edc.MODEL}
        with self.assertRaises(RuntimeError):
            transport.request('chat/completions', body, 'oie')
        with self.assertRaisesRegex(self.edc.NativeRunError, 'in-flight'):
            self.edc.Transport(self.run, self.prepare(), 'fixture-secret', sender=crash).request('chat/completions', body, 'oie')
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(self.edc.NativeRunError, 'mix'):
            self.edc.Transport(self.run, self.prepare(), 'fixture-secret')

    def test_timeout_is_bounded_and_malformed_success_is_not_retried(self):
        calls = []
        def timeout(req, seconds):
            calls.append(seconds)
            raise TimeoutError()
        transport = self.transport(timeout)
        with self.assertRaises(self.edc.NativeRunError):
            transport.request('chat/completions', {'model': self.edc.MODEL}, 'oie')
        self.assertEqual(calls, [240, 240, 240])
        def malformed(req, seconds):
            calls.append(seconds)
            return 200, '{}'
        with self.assertRaises(self.edc.NativeRunError):
            self.edc.Transport(self.run, self.prepare(), 'fixture-secret', sender=malformed).request(
                'chat/completions', {'model': self.edc.MODEL, 'messages': []}, 'sd')
        self.assertEqual(len(calls), 4)
        with self.assertRaises(self.edc.NativeRunError):
            self.edc.Transport(self.run, self.prepare(), 'fixture-secret', sender=malformed).request(
                'chat/completions', {'model': self.edc.MODEL, 'messages': []}, 'sd')
        self.assertEqual(len(calls), 4)
    def test_embedding_shape_and_normalization_and_bad_dimensions(self):
        import numpy as np
        def vectors(req, timeout):
            body = json.loads(req.data)
            return 200, json.dumps({'data': [{'index': i, 'embedding': [3, 4, 0]} for i, _ in enumerate(body['input'])]})
        embedder = self.edc.BGEEmbedder(self.transport(vectors))
        self.assertEqual(embedder.prompts, {})
        self.assertEqual(embedder.encode('中文').shape, (3,))
        batch = embedder.encode(['中文', '谓词'])
        self.assertEqual(batch.shape, (2, 3))
        np.testing.assert_allclose(np.linalg.norm(batch, axis=1), [1, 1])
        with self.assertRaises(ValueError):
            embedder.encode('中文', prompt_name='sts_query')

    def test_embedding_invalid_shape_is_rejected(self):
        def wrong(req, timeout):
            return 200, '{"data":[{"index":1,"embedding":[0,0]}]}'
        with self.assertRaises(self.edc.NativeRunError):
            self.edc.BGEEmbedder(self.transport(wrong)).encode('中文')

    def test_preflight_separates_find_import_and_model_execution(self):
        self.prepare()
        result = self.edc.dependency_preflight(self.run / 'native_source', actual_import=False)
        self.assertEqual(result['native_import_status'], 'not_attempted')
        self.assertEqual(result['model_execution_status'], 'not_executed')

    def test_actual_native_pipeline_prompts_parser_sc_math_and_replay(self):
        plan = self.prepare()
        preflight = self.edc.dependency_preflight(self.run / 'native_source', actual_import=True)
        if preflight['native_import_status'] != 'passed':
            self.skipTest('actual native dependencies unavailable: ' + str(preflight))
        calls = []
        def fixture(req, timeout):
            body = json.loads(req.data)
            calls.append((req.full_url, body))
            if req.full_url.endswith('/embeddings'):
                return 200, json.dumps({'data': [{'index': i, 'embedding': [1, 0, 0]} for i, _ in enumerate(body['input'])]})
            prompt = body['messages'][-1]['content']
            if body['max_tokens'] == 1:
                answer = 'B'
            elif 'Relations:' in prompt or 'relations in the text' in prompt:
                answer = '包含: 表示组成。\n执行: 表示运行。'
            else:
                answer = "[['处理器', '包含', '运算器'], ['处理器', '执行', '指令']]"
            return 200, json.dumps({'choices': [{'message': {'content': answer}}], 'usage': {'total_tokens': 7}})
        proof = self.edc.execute(self.run, api_key='fixture-secret', sender=fixture)
        self.assertEqual(proof['execution_kind'], 'offline_http_fixture')
        self.assertEqual(proof['model_execution_status'], 'not_executed')
        self.assertEqual(proof['native_artifact_status'], 'passed')
        self.assertEqual(proof['unit_count'], 2)
        self.assertEqual(proof['distinct_predicates'], 2)
        self.assertEqual(proof['http_attempts'], 7)
        source = self.run / 'native_source'
        expected_oie = [(source / 'prompt_templates/oie_template.txt').read_text(encoding='utf-8').format_map({
            'few_shot_examples': (source / 'few_shot_examples/example/oie_few_shot_examples.txt').read_text(encoding='utf-8'),
            'input_text': unit['text'], 'entities_hint': None, 'relations_hint': None}) for unit in plan['units']]
        self.assertEqual([body['messages'][0]['content'] for _, body in calls[:2]], expected_oie)
        self.assertEqual(sum(body.get('max_tokens') == 512 for _, body in calls), 4)
        self.assertEqual(sum(body.get('max_tokens') == 1 for _, body in calls), 1)
        self.assertTrue(any(body.get('max_tokens') == 1 for _, body in calls))
        for url, body in calls:
            self.assertTrue(url.startswith('https://api.siliconflow.cn/v1/'))
            if url.endswith('/chat/completions'):
                self.assertEqual(body['model'], self.edc.MODEL)
                self.assertEqual(body['temperature'], 0)
                self.assertEqual(body['seed'], 42)
                self.assertIn(body['max_tokens'], (512, 1))
        count = len(calls)
        second = self.edc.execute(self.run, api_key='fixture-secret', sender=fixture)
        self.assertEqual(len(calls), count)
        self.assertEqual(second['native_artifact_status'], 'passed')
        self.assertFalse(self.edc.gate(self.run)['live_model_passed'])
        rows = json.loads((Path(second['native_output_dir']) / 'iter0/result_at_each_stage.json').read_text())
        self.assertEqual({t[1] for row in rows for t in row['schema_canonicalizaiton']}, {'包含', '执行'})
        self.assertTrue(all(t[0] == '处理器' for row in rows for t in row['schema_canonicalizaiton']))
        self.assertEqual(self.edc.read_source_hashes(self.run / 'native_source'), plan['source_files'])
        transport = self.edc.Transport(self.run, plan, 'fixture-secret', sender=fixture)
        with self.edc._native_import(source) as framework:
            llm = importlib.import_module('edc.utils.llm_utils')
            original = (llm.is_model_openai, llm.openai_chat_completion, framework.SentenceTransformer)
            self.assertFalse(llm.is_model_openai(self.edc.MODEL))
            with self.edc.native_bridge(framework, transport):
                self.assertTrue(llm.is_model_openai(self.edc.MODEL))
                with self.assertRaises(self.edc.NativeRunError):
                    llm.is_model_openai('fake-gpt-route')
            self.assertEqual((llm.is_model_openai, llm.openai_chat_completion, framework.SentenceTransformer), original)
        evidence = os.environ.get('EDC_NATIVE_TEST_EVIDENCE_DIR')
        if evidence:
            shutil.copytree(self.run, evidence, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


if __name__ == '__main__':
    unittest.main()
