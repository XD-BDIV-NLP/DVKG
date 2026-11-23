"""原服务器使用固定官方tokenizer计数，按完整pair确定性打包。"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

from ..manifest import digest,file_record,read_document
from . import scope_identity_batch as core


ASSET_HASHES={'tokenizer.json':'8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf',
    'tokenizer_config.json':'6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547',
    'config.json':'6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023',
    'encoding/encoding_dsv4.py':'abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6'}
POLICY=dict(version='official_flash_token_packing_v2',official_repository='deepseek-ai/DeepSeek-V4-Flash-0731',
    official_revision='7872f01b1d1fe23eabc4c98b48bffcef5a386062',
    requested_model_unchanged='deepseek-ai/DeepSeek-V4-Flash',official_model_context_tokens=1048576,
    working_context_tokens=262144,max_output_tokens=8192,service_wrapper_reserve_tokens=1024,
    count='official_local_tokenizer_full_chat_messages_max_with_and_without_official_json_format_injection',
    thinking_mode='chat',maximum_pairs=8,truncate=False,oversize_single_pair='reject_before_API_report_hash_and_token_count',
    grouping='same_course_branch_hash_groups_up_to8_full_count_first_greedy_split_only_if_oversize',
    reversal='preserve_primary_group_membership_filter_positive_then_apply_same_token_packing',
    official_context_source='https://cloud-rd.siliconflow.cn/news/zaw07rwkzh2ddssh2ib8rjzj',
    tokenizer_provenance='official_main_and0731_tokenizer_json_identical_SHA_no_weights_downloaded',
    no_corpus_sent_to_tokenizer_host=True)


class OversizeIdentityPair(ValueError):pass


def messages_for(packets,*,reverse=False):
    pairs=[core._anonymous(packet) for packet in packets]
    if reverse:
        for pair in pairs:pair['left'],pair['right']=pair['right'],pair['left']
    wire=dict(version=core.POLICY['version'],kind=core._kind(packets[0]),course_id=packets[0]['course_id'],pairs=pairs)
    wire['batch_id']=digest(wire)
    return [dict(role='system',content=core.JUDGE),dict(role='user',content=json.dumps(wire,ensure_ascii=False))]


class OfficialFlashCounter:
    def __init__(self,root):
        root=Path(root);self.assets={}
        for name,sha in ASSET_HASHES.items():
            self.assets[name]=file_record(root/name)
            if self.assets[name]['sha256']!=sha:raise ValueError('Official tokenizer asset changed: '+name)
        if read_document(root/'config.json')['max_position_embeddings']!=1048576:
            raise ValueError('Official model context configuration changed')
        from tokenizers import Tokenizer
        self.tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
        spec=importlib.util.spec_from_file_location('verified_official_encoding_dsv4',root/'encoding/encoding_dsv4.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);self.encode_messages=module.encode_messages
        self.cache={}

    def count(self,messages):
        key=digest(messages)
        if key not in self.cache:
            prompt=self.encode_messages(messages,thinking_mode='chat')
            self.cache[key]=len(self.tokenizer.encode(prompt,add_special_tokens=False).ids)
        return self.cache[key]

    def audit(self,rows):
        counts=[]
        for reverse in [False,True]:
            messages=messages_for(rows,reverse=reverse);plain=self.count(messages)
            formatted=copy.deepcopy(messages);formatted[0]['response_format']={'type':'json_object'}
            counts.append(dict(reverse=reverse,messages_tokens=plain,
                messages_with_official_format_tokens=self.count(formatted)))
        tokens=max(max(r['messages_tokens'],r['messages_with_official_format_tokens']) for r in counts)
        total=tokens+8192+1024
        return dict(prompt_tokens=tokens,orientation_counts=counts,total_with_reserves=total,
            working_context_tokens=262144,official_context_tokens=1048576,fits=total<=262144,
            max_tokens=8192,service_wrapper_reserve_tokens=1024,packing_policy=POLICY['version'],
            tokenizer_usage='offline_on_original_server_known_messages_exact_host_hidden_wrapper_reserved_not_observed')


def plan_batches(packets,counter,*,single=False,progress=None):
    result=[];groups=core.plan_batches(packets);processed=0
    for number,group in enumerate(groups,1):
        if not single and counter.audit(group)['fits']:
            result.append(group);processed+=len(group)
            if progress:progress(dict(completed_original_groups=number,total_original_groups=len(groups),
                packed_groups=len(result),processed_candidates=processed))
            continue
        current=[]
        for packet in group:
            trial=current+[packet]
            if (single and current) or not counter.audit(trial)['fits']:
                if current:result.append(current);current=[]
                if not counter.audit([packet])['fits']:
                    raise OversizeIdentityPair('Complete identity pair exceeds frozen context: '+packet['candidate_id'])
            current.append(packet)
        if current:result.append(current)
        processed+=len(group)
        if progress:progress(dict(completed_original_groups=number,total_original_groups=len(groups),
            packed_groups=len(result),processed_candidates=processed))
    return result
