"""离线 GPU 交叉编码器：仅评分固定问题与原文窗口，原子保存推理回执。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import time


MODEL = 'BAAI/bge-reranker-v2-m3'
REVISION = 'e099d4b9cdbd291b1569d416f19aaa6523570bc3'
MODEL_PARENT = Path(os.environ.get("DVKG_RERANKER_ROOT", "models/echa_reranker_v2_m3"))
MODEL_DIR = MODEL_PARENT / 'modelscope_official'
MANIFEST = MODEL_PARENT / 'modelscope_manifest.json'
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = 'cuda:0'
FROZEN_HASHES = {
    'config.json': '13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a',
    'configuration.json': '90dccac936e44dcb4567017a5dbb807b583604032fd53bfe2372a15881ed47e1',
    'model.safetensors': 'd9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286',
    'sentencepiece.bpe.model': 'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865',
    'special_tokens_map.json': '8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835',
    'tokenizer.json': '69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15',
    'tokenizer_config.json': '7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _keys(value, required, label):
    _require(isinstance(value, dict) and set(value) == set(required),
             'Unexpected or missing fields in ' + label)


def _identity(key):
    _require(isinstance(key, list) and len(key) == 2
             and all(isinstance(s, str) and s.strip() for s in key),
             'A namespaced two-part candidate key is required')
    return tuple(key)


def validate_packet(packet):
    """拒绝答案、标签或任何未冻结输入字段；完整原文哈希由上游绑定。"""
    _keys(packet, ['version', 'answers_included', 'model', 'revision', 'max_length', 'packets'], 'packet')
    _require(packet['version'] == 'question_window_score_packet_v1'
             and packet['answers_included'] is False and packet['model'] == MODEL
             and packet['revision'] == REVISION and type(packet['max_length']) is int
             and packet['max_length'] == MAX_LENGTH, 'Input model or protocol differs from frozen settings')
    _require(isinstance(packet['packets'], list) and packet['packets'], 'No input tasks')
    seen = set()
    for task in packet['packets']:
        _keys(task, ['task_id', 'question', 'candidates'], 'task')
        tid = task['task_id']
        _require(isinstance(tid, str) and tid.strip() and tid not in seen, 'Invalid or repeated task identity')
        seen.add(tid)
        _require(isinstance(task['question'], str) and task['question'].strip(), 'Empty question')
        rows = task['candidates']
        _require(isinstance(rows, list) and 1 <= len(rows) <= 64, 'One to 64 candidates required per task')
        keys = set()
        for row in rows:
            _keys(row, ['key', 'text', 'char_start', 'char_end', 'input_sha256', 'source_sha256'], 'candidate')
            key = _identity(row['key'])
            _require(key not in keys, 'Repeated candidate identity')
            keys.add(key)
            text = row['text']; start = row['char_start']; end = row['char_end']
            _require(isinstance(text, str) and text.strip(), 'Empty candidate text')
            _require(type(start) is int and type(end) is int and start >= 0
                     and end - start == len(text), 'Invalid original character offsets')
            _require(_sha(row['source_sha256']) and _sha(row['input_sha256'])
                     and hashlib.sha256(text.encode('utf-8')).hexdigest() == row['input_sha256'],
                     'Changed candidate text or invalid source digest')
    digest(packet)


def validate_receipt(packet, receipt):
    validate_packet(packet)
    _require(isinstance(receipt, dict) and receipt.get('input_digest') == digest(packet)
             and receipt.get('model') == MODEL and receipt.get('revision') == REVISION
             and type(receipt.get('max_length')) is int and receipt['max_length'] == MAX_LENGTH
             and receipt.get('actual_inference') is True, 'Receipt does not match input and frozen model')
    expected = {p['task_id']: p for p in packet['packets']}; seen = set()
    _require(isinstance(receipt.get('items'), list), 'Missing receipt items')
    for item in receipt['items']:
        _require(isinstance(item, dict), 'Invalid receipt item')
        tid = item.get('task_id')
        _require(isinstance(tid, str) and tid in expected and tid not in seen, 'Unknown or repeated task receipt')
        seen.add(tid)
        sources = {tuple(c['key']): c for c in expected[tid]['candidates']}; found = set()
        _require(isinstance(item.get('scores'), list), 'Missing scores')
        for scored in item['scores']:
            _require(isinstance(scored, dict), 'Invalid score entry')
            key = _identity(scored.get('key')); value = scored.get('score')
            _require(key in sources and key not in found
                     and scored.get('input_sha256') == sources[key]['input_sha256']
                     and not isinstance(value, bool) and isinstance(value, (int, float))
                     and math.isfinite(value), 'Missing, changed, duplicated or nonfinite candidate score')
            found.add(key)
        _require(found == set(sources), 'Incomplete task scores')
    _require(seen == set(expected), 'Incomplete receipt tasks')


def _file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _unique_json_fields(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON field: ' + key)
        result[key] = value
    return result


def _load_json(path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_unique_json_fields,
                      parse_constant=reject_constant)


def verify_model_assets(model_root=MODEL_DIR, manifest_path=MANIFEST, expected_hashes=None):
    """分块校验权重，不把大文件整体装入内存。"""
    expected_hashes = FROZEN_HASHES if expected_hashes is None else expected_hashes
    manifest_path = Path(manifest_path); model_root = Path(model_root)
    started = time.monotonic(); manifest = _load_json(manifest_path)
    _require(manifest.get('model') == MODEL and manifest.get('revision') == REVISION,
             'Model manifest identity differs from frozen settings')
    files = manifest.get('files')
    _require(isinstance(files, list), 'Model manifest files are missing')
    entries = {}
    for row in files:
        _require(isinstance(row, dict) and isinstance(row.get('Path'), str)
                 and row['Path'] not in entries, 'Invalid or repeated model manifest file')
        entries[row['Path']] = row
    verified = []
    for name, frozen in expected_hashes.items():
        _require(Path(name).name == name and _sha(frozen) and name in entries,
                 'Invalid frozen model file or incomplete manifest')
        entry = entries[name]; path = model_root / name
        _require(entry.get('Sha256') == frozen and entry.get('Revision', REVISION) == REVISION,
                 'Model manifest file differs from frozen digest')
        _require(path.is_file(), 'Cached model file is missing: ' + name)
        actual = _file_sha(path); size = path.stat().st_size
        _require(actual == frozen and ('Size' not in entry or entry['Size'] == size),
                 'Cached model file failed integrity check: ' + name)
        verified.append(dict(name=name, sha256=actual, bytes=size))
    return dict(model_files=verified, model_manifest_sha256=_file_sha(manifest_path),
                model_manifest_path=str(manifest_path), model_path=str(model_root),
                model_hash_seconds=time.monotonic() - started)


def truncation_statistics(before, after):
    _require(len(before) == len(after)
             and all(type(a) is int and type(b) is int and 0 < b <= a
                     and b <= MAX_LENGTH for a, b in zip(before, after)), 'Invalid actual tokenizer lengths')
    return dict(pairs=len(before), pairs_truncated=sum(a > b for a, b in zip(before, after)),
                tokens_before_truncation=sum(before), tokens_after_truncation=sum(after),
                tokens_removed_by_truncation=sum(a - b for a, b in zip(before, after)))


def publish_receipt(output_path, packet, receipt):
    """用同目录硬链接原子创建，已存在的有效回执永不覆盖。"""
    validate_receipt(packet, receipt)
    output_path = Path(output_path)
    if output_path.exists():
        validate_receipt(packet, _load_json(output_path))
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.tmp',
                prefix='.' + output_path.name + '.', dir=output_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
            stream.flush(); os.fsync(stream.fileno())
        try:
            if os.name == 'nt':
                # Windows rename 本身拒绝已存在的目标，也无需硬链接权限。
                os.rename(temporary, output_path)
            else:
                os.link(temporary, output_path)
        except FileExistsError:
            validate_receipt(packet, _load_json(output_path))
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _offline_only():
    """依赖导入、模型加载和推理阶段禁止建立任何套接字连接。"""
    def blocked(*args, **kwargs):
        raise RuntimeError('Network connections are disabled in this offline score worker')
    methods = [(socket, 'create_connection'), (socket.socket, 'connect'), (socket.socket, 'connect_ex')]
    previous = [(owner, name, getattr(owner, name)) for owner, name in methods]
    try:
        for owner, name in methods:
            setattr(owner, name, blocked)
        yield
    finally:
        for owner, name, method in previous:
            setattr(owner, name, method)


def infer(packet):
    """仅在真实 CUDA 推理完成后设置 actual_inference=True。"""
    validate_packet(packet)
    started = time.monotonic(); assets = verify_model_assets()
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; os.environ['DO_NOT_TRACK'] = '1'
    sys.path.insert(0, str(MODEL_PARENT / 'deps'))
    with _offline_only():
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _require(torch.cuda.is_available(), 'CUDA is not available; CPU fallback is forbidden')
        load_start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True,
                    trust_remote_code=False, torch_dtype=torch.float16).to(DEVICE).eval()
        torch.cuda.synchronize(); load_seconds = time.monotonic() - load_start
        _require(model.training is False and all(p.dtype == torch.float16 for p in model.parameters()),
                 'Model is not in frozen float16 evaluation mode')
        items = []; before_all = []; after_all = []; inference_seconds = 0.; batches = 0
        tokenize_seconds = 0.
        for task in packet['packets']:
            scores = []
            for offset in range(0, len(task['candidates']), BATCH_SIZE):
                rows = task['candidates'][offset:offset + BATCH_SIZE]
                queries = [task['question']] * len(rows); passages = [r['text'] for r in rows]
                token_start = time.monotonic()
                full = tokenizer(queries, passages, add_special_tokens=True, padding=False, truncation=False)
                before = [len(ids) for ids in full['input_ids']]
                batch = tokenizer(queries, passages, add_special_tokens=True, padding=True,
                    truncation='longest_first', max_length=MAX_LENGTH, return_tensors='pt')
                after = [int(x) for x in batch['attention_mask'].sum(dim=1).tolist()]
                truncation_statistics(before, after)
                tokenize_seconds += time.monotonic() - token_start
                batch = batch.to(DEVICE); torch.cuda.synchronize(); inference_start = time.monotonic()
                with torch.inference_mode():
                    logits = model(**batch).logits
                    _require(logits.numel() == len(rows), 'Expected one raw logit per query/passage pair')
                    values = logits.reshape(-1).float().cpu().tolist()
                torch.cuda.synchronize(); inference_seconds += time.monotonic() - inference_start
                _require(all(math.isfinite(v) for v in values), 'Model produced a nonfinite logit')
                batches += 1; before_all.extend(before); after_all.extend(after)
                for row, value, a, b in zip(rows, values, before, after):
                    scores.append(dict(key=row['key'], input_sha256=row['input_sha256'],
                        source_sha256=row['source_sha256'], char_start=row['char_start'],
                        char_end=row['char_end'], score=float(value), tokens_before_truncation=a,
                        tokens_after_truncation=b, truncated_tokens=a - b))
            items.append(dict(task_id=task['task_id'], scores=scores))
        receipt = dict(version='question_window_score_receipt_v1', input_digest=digest(packet),
            model=MODEL, revision=REVISION, max_length=MAX_LENGTH, actual_inference=True, items=items,
            batch_size=BATCH_SIZE, device=DEVICE, device_name=torch.cuda.get_device_name(0),
            dtype='float16', score_kind='raw_logit', normalization='none', eval_mode=True,
            inference_mode=True, local_files_only=True, trust_remote_code=False, network_disabled=True,
            torch_version=torch.__version__, transformers_version=transformers.__version__,
            cuda_version=torch.version.cuda, tokenizer_class=type(tokenizer).__name__,
            truncation_strategy='longest_first', truncation=truncation_statistics(before_all, after_all),
            batches=batches, model_load_seconds=load_seconds, inference_seconds=inference_seconds,
            tokenization_seconds=tokenize_seconds, elapsed_seconds=time.monotonic() - started,
            completed_unix=time.time(), worker_sha256=_file_sha(__file__), **assets)
    validate_receipt(packet, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    _require(args.input.resolve() != args.output.resolve(), 'Input and output must be different files')
    packet = _load_json(args.input); validate_packet(packet)
    if args.output.exists():
        validate_receipt(packet, _load_json(args.output))
        print(json.dumps(dict(status='compatible_receipt_reused', input_digest=digest(packet))))
        return 0
    receipt = infer(packet)
    receipt['input_file_sha256'] = _file_sha(args.input)
    created = publish_receipt(args.output, packet, receipt)
    print(json.dumps(dict(status='receipt_created' if created else 'compatible_receipt_reused',
                          input_digest=digest(packet), pairs=receipt['truncation']['pairs'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
