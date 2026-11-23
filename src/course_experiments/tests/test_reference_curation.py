from __future__ import annotations
import copy
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from course_experiments.evaluation import reference as ref
from course_experiments.evaluation.protocol import freeze_phase
from course_experiments.manifest import file_record, read_document, digest


def fixture():
    text='A includes B. B includes C when ready. D exists. X exists.'
    unit=dict(book_id='B',chunk_id='c',text=text,evidence=[dict(id='e',text=text)])
    cite=lambda quote:[dict(evidence_id='e',quote=quote)]
    entities=[dict(local_id=n,canonical_name=n,scope='s',aliases=[],citations=cite(n)) for n in 'ABCDX']
    entities[-1]['aliases']=['invented']
    facts=[dict(local_id='f1',head='A',tail='B',relation='includes',conditions='',citations=cite('A includes B')),
           dict(local_id='f2',head='B',tail='C',relation='includes',conditions='when ready',citations=cite('B includes C when ready')),
           dict(local_id='bad',head='A',tail='B',relation='includes',conditions='never stated',citations=cite('A includes B')),
           dict(local_id='own',head='A',tail='B',relation='includes',citations=cite('A')),
           dict(local_id='dep',head='X',tail='A',relation='includes',citations=cite(text)),
           dict(local_id='f3',head='D',tail='D',relation='includes',citations=cite('D'))]
    questions=[dict(type='multihop',question='A to C?',answer='via B',fact_path=['f1','f2'],citations=cite(text)),
               dict(type='conditions',question='bad condition?',answer='never stated',fact_path=['bad'],citations=cite(text)),
               dict(type='multihop',question='disconnected?',answer='no',fact_path=['f1','f3'],citations=cite(text)),
               dict(type='evidence',question='unsupported?',answer='no',fact_path=[],citations=cite('invented'))]
    return unit,dict(entities=entities,facts=facts,questions=questions),dict(relation_types={'includes':''})


class Runtime:
    config=dict(llm=dict(base_url='https://api.siliconflow.cn/v1',model='deepseek-ai/DeepSeek-V4-Flash',extra_body={'enable_thinking':False}),embedding=dict(base_url='https://api.siliconflow.cn/v1',model='BAAI/bge-m3'))
    gate=threading.BoundedSemaphore(1)
    def __init__(self,value):self.value=value;self.requests=[]
    def _raw_cached(self,key):return None
    def _post(self,endpoint,body,*args):
        self.requests.append(json.loads(body['messages'][1]['content']))
        return dict(choices=[dict(message=dict(content=json.dumps(self.value)),finish_reason='stop')],usage={'total_tokens':1})


class RetryRuntime(Runtime):
    def __init__(self,first,second):super().__init__(second);self.first=first;self.second=second
    def _post(self,endpoint,body,*args):
        self.value=self.first if not self.requests else self.second
        return super()._post(endpoint,body,*args)


class ReferenceCurationTests(unittest.TestCase):
    def setup_run(self,root,count=1):
        unit,_,schema=fixture()
        corpus=dict(chunks=[dict(id='c'+str(i),book_id='B',evidence_ids=['e']) for i in range(count)],evidence=unit['evidence'])
        for name,value in [('corpus',corpus),('schema',schema),('runtime',Runtime.config)]:
            (root/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
        construction=dict(courses={'courses':[dict(course_id='c',description='source only',books=[dict(book_id='B')])]},files=dict(assets={'schema':file_record(root/'schema.json'),'runtime_config':file_record(root/'runtime.json')},sources={'B':file_record(root/'corpus.json')}))
        cp=root/'construction.json';cp.write_text(json.dumps(construction),encoding='utf-8')
        phase=root/'phase.json';freeze_phase({'construction':cp},dict(ref.PARAMETERS,smoke_chunks=count),phase)
        return construction,cp,phase

    def test_mixed_packet_retains_exact_source_validated_candidates_with_bound_audit(self):
        unit,value,schema=fixture();original=copy.deepcopy(value)
        with self.assertRaises(ValueError):ref.validate_batch(value,unit,schema)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(value)
            with patch.object(ref,'validate',return_value=construction):
                dataset,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
            self.assertEqual(len(dataset['facts']),3)
            self.assertEqual(len(dataset['questions']),1)
            self.assertEqual(len(runtime.requests),1)
            checkpoint=read_document(next((out/'checkpoints').glob('*.json')))
            chosen={k:checkpoint[k] for k in ('entities','facts','questions')}
            ref.validate_batch(chosen,unit,schema)
            self.assertEqual(chosen['facts'],[value['facts'][i] for i in (0,1,5)])
            self.assertEqual(chosen['questions'],value['questions'][:1])
            self.assertEqual(value,original)
            audit=read_document(out/checkpoint['curation']['audit_ref'])
            self.assertEqual(audit['phase_id'],read_document(phase)['phase_id'])
            self.assertEqual(audit['source_hash'],digest(runtime.requests[0]['unit']))
            self.assertEqual(audit['raw_response_hash'],digest(value))
            self.assertTrue((out/audit['request_ref']).exists())
            self.assertEqual(audit['retained_counts'],dict(entities=4,facts=3,questions=1))
            self.assertEqual(audit['rejected_counts'],dict(entities=1,facts=3,questions=3))
            self.assertEqual(len(audit['rejections']),7)
            self.assertFalse(audit['whole_response_valid'])
            self.assertEqual(dataset['curation']['policy'],'source_validated_subset_v1')
            self.assertIn(checkpoint['curation']['audit_ref'],dataset['curation']['audit_refs'])

    def test_smoke_visits_sixteen_units_without_exceeding_twenty_eight_attempts(self):
        value=dict(entities=[],facts=[],questions=[dict(type='definition',question='?',answer='?',citations=[dict(evidence_id='e',quote='invented')])])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);construction,cp,phase=self.setup_run(root,16);runtime=Runtime(value)
            with patch.object(ref,'validate',return_value=construction):
                dataset,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=16)
            self.assertEqual(len(runtime.requests),28)
            self.assertEqual(len({r['unit']['chunk_id'] for r in runtime.requests}),16)
            self.assertEqual(dataset['status'],'partial')
            self.assertEqual(len(dataset['failed_chunks']),16)
            self.assertEqual(dataset['facts'],[]);self.assertEqual(dataset['questions'],[])
            self.assertEqual(dataset['generation_budget']['logical_calls'],28)
            self.assertFalse(list((out/'checkpoints').glob('*.json')))
            self.assertEqual(dataset['targets'],dict(questions=8,facts=24))
            self.assertEqual(ref.reference_targets(None),dict(questions=100,facts=400))
            with self.assertRaises(ValueError):ref.reference_targets(17)

    def test_duplicate_ids_reject_every_collision_and_its_dependencies(self):
        unit,value,schema=fixture()
        for field,index in [('entities',0),('facts',0)]:
            changed=copy.deepcopy(value);changed[field].append(copy.deepcopy(changed[field][index]))
            with self.subTest(field=field):
                chosen,audit=ref.curate_batch(changed,unit,schema)
                ref.validate_batch(chosen,unit,schema)
                self.assertEqual([f['local_id'] for f in chosen['facts']],['f2','f3'])
                self.assertEqual(chosen['questions'],[])
                collisions=[r for r in audit['rejections'] if r['reason']=='Ambiguous duplicate local ID']
                self.assertEqual(len(collisions),2)
        clean,_=ref.curate_batch(value,unit,schema)
        clean['questions'][0]['local_id']='same';clean['questions'].append(copy.deepcopy(clean['questions'][0]))
        chosen,audit=ref.curate_batch(clean,unit,schema)
        self.assertEqual(chosen['questions'],[])
        self.assertEqual(audit['rejected_counts']['questions'],2)

    def test_bad_name_alias_citation_and_conditions_are_removed_without_rewriting(self):
        unit,value,schema=fixture()
        for field,changes in [('entities',{'canonical_name':'invented'}),('entities',{'citations':[dict(evidence_id='foreign',quote='A')]}),('facts',{'conditions':'invented'})]:
            changed=copy.deepcopy(value);changed[field][0].update(changes);before=copy.deepcopy(changed)
            with self.subTest(field=field,changes=changes):
                chosen,audit=ref.curate_batch(changed,unit,schema)
                self.assertNotIn(changed[field][0],chosen[field])
                self.assertEqual(changed,before)
                self.assertEqual(chosen['facts'][0]['conditions'],'when ready')
                ref.validate_batch(chosen,unit,schema)
                self.assertTrue(audit['rejections'])

    def test_malformed_whole_packets_never_become_accepted_subsets(self):
        unit,value,schema=fixture()
        cases=[None,[],dict(value,entities={}),dict(value,facts=value['facts']*3),dict(value,questions=[None]),dict(value,entities=[dict(local_id=['bad'])]),dict(value,questions=[dict(citations='bad')])]
        for bad in cases:
            with self.subTest(bad=bad),self.assertRaises(ValueError):ref.curate_batch(bad,unit,schema)

    def test_valid_whole_and_legitimate_empty_batches_keep_success_path(self):
        unit,value,schema=fixture();valid,_=ref.curate_batch(value,unit,schema)
        for raw in (valid,dict(entities=[],facts=[],questions=[])):
            with self.subTest(raw=raw),tempfile.TemporaryDirectory() as d:
                root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(raw)
                with patch.object(ref,'validate',return_value=construction):
                    dataset,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(len(runtime.requests),1)
                self.assertEqual(dataset['failed_chunks'],[])
                checkpoint=read_document(next((out/'checkpoints').glob('*.json')))
                self.assertTrue(checkpoint['curation']['whole_response_valid'])
                self.assertEqual({k:checkpoint[k] for k in raw},raw)
                self.assertEqual(checkpoint['curation']['rejected_counts'],dict(entities=0,facts=0,questions=0))

    def test_replay_checkpoint_and_raw_cache_preserve_identity_but_new_phase_is_isolated(self):
        _,value,_=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(value)
            with patch.object(ref,'validate',return_value=construction):
                first,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                files={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*.json')}
                second,out2=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(first,second);self.assertEqual(out,out2)
                self.assertEqual(files,{str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*.json')})
                next((out/'checkpoints').glob('*.json')).unlink()
                cached,out3=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(first,cached);self.assertEqual(out,out3);self.assertEqual(len(runtime.requests),1)
                marker=root/'new_phase_marker.json';marker.write_text('{}',encoding='utf-8')
                next_phase=root/'next_phase.json';freeze_phase({'construction':cp,'marker':marker},dict(ref.PARAMETERS,smoke_chunks=1),next_phase)
                third,out4=ref.generate(cp,next_phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertNotEqual(first['input_id'],third['input_id']);self.assertNotEqual(out,out4)
                self.assertEqual(len(runtime.requests),2)
                old_phase=root/'old_phase.json';params=dict(ref.PARAMETERS,smoke_chunks=1);params.pop('curation_policy')
                freeze_phase({'construction':cp},params,old_phase)
                with self.assertRaisesRegex(ValueError,'parameters not frozen'):ref.generate(cp,old_phase,'c',root/'out',runtime=runtime,smoke=1)

    def test_interleaving_and_all_six_types_are_required_for_completion(self):
        corpus=lambda b:dict(evidence=[dict(id=b,text=b)],chunks=[dict(book_id=b,id=b+str(i),evidence_ids=[b]) for i in range(8)])
        corpora=[corpus(b) for b in 'DCBA']
        units=ref.selected_units(corpora,smoke=16)
        self.assertEqual([u['book_id'] for u in units],list('ABCD')*4)
        self.assertEqual(units,ref.selected_units(list(reversed(corpora)),smoke=16))
        unit,value,schema=fixture();valid,_=ref.curate_batch(value,unit,schema);records=[]
        for chunk in range(3):
            batch=copy.deepcopy(valid)
            batch['facts']=[dict(valid['facts'][i%2],local_id='f'+str(i),scope=str(i)) for i in range(8)]
            batch['questions']=[dict(valid['questions'][0],question=str(chunk*4+i),type='definition',fact_path=['f0','f1']) for i in range(4 if chunk<2 else 0)]
            ref.validate_batch(batch,unit,schema)
            records.append((dict(unit,chunk_id='c'+str(chunk)),batch))
        partial=ref.assemble('c',records,'smoke','id',dict(questions=8,facts=24))
        self.assertEqual(partial['status'],'partial')
        for i,kind in enumerate(ref.TYPES):records[i//4][1]['questions'][i%4]['type']=kind
        for source,batch in records:ref.validate_batch(batch,source,schema)
        complete=ref.assemble('c',records,'smoke','id',dict(questions=8,facts=24))
        self.assertEqual(complete['status'],'complete')

    def test_replay_rejects_changed_audit_or_checkpoint_policy(self):
        _,value,_=fixture()
        for target in ('policy','audit'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as d:
                root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(value)
                with patch.object(ref,'validate',return_value=construction):
                    _,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    checkpoint_path=next((out/'checkpoints').glob('*.json'));checkpoint=read_document(checkpoint_path)
                    if target=='policy':
                        checkpoint['curation']['policy']='unrecorded_policy'
                        checkpoint_path.write_text(json.dumps(checkpoint),encoding='utf-8')
                    else:
                        audit_path=out/checkpoint['curation']['audit_ref'];audit=read_document(audit_path)
                        audit['source_hash']='other_source';audit_path.write_text(json.dumps(audit),encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'curation audit'):
                        ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)

    def test_replay_requires_original_accepted_request_body_and_response(self):
        _,value,_=fixture()
        for change in ('missing','response','empty_response','null_response','body'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(value)
                with patch.object(ref,'validate',return_value=construction):
                    _,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    checkpoint=read_document(next((out/'checkpoints').glob('*.json')))
                    audit=read_document(out/checkpoint['curation']['audit_ref']);request_path=out/audit['request_ref']
                    if change=='missing':request_path.unlink()
                    else:
                        request=read_document(request_path)
                        if change=='response':request['response']['choices'][0]['message']['content']=json.dumps(dict(entities=[],facts=[],questions=[]))
                        elif change=='empty_response':request['response']={}
                        elif change=='null_response':request['response']=None
                        else:request['request']['messages'][0]['content']='Changed system prompt'
                        request_path.write_text(json.dumps(request),encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'curation audit'):
                        ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(len(runtime.requests),1)

    def test_replay_rejects_missing_or_changed_earlier_attempt_audit_and_raw(self):
        _,value,_=fixture();bad=dict(entities=[],facts=[],questions=[copy.deepcopy(value['questions'][-1])])
        for change in ('missing_audit','changed_audit','missing_raw','changed_raw'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);construction,cp,phase=self.setup_run(root);runtime=RetryRuntime(bad,value)
                with patch.object(ref,'validate',return_value=construction):
                    first,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(first['curation']['rejected_candidate_counts']['questions'],4)
                    audit_path=next((out/'audits').glob('*-0.json'));audit=read_document(audit_path)
                    if change=='missing_audit':audit_path.unlink()
                    elif change=='changed_audit':
                        audit['phase_id']='foreign';audit['rejected_counts']['questions']=999
                        audit_path.write_text(json.dumps(audit),encoding='utf-8')
                    else:
                        request_path=out/audit['request_ref']
                        if change=='missing_raw':request_path.unlink()
                        else:
                            request=read_document(request_path);request['response']['choices'][0]['message']['content']=json.dumps(value)
                            request_path.write_text(json.dumps(request),encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'curation audit'):
                        ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(len(runtime.requests),2)

    def test_replay_rejects_foreign_extra_audit_instead_of_counting_it(self):
        _,value,_=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(value)
            with patch.object(ref,'validate',return_value=construction):
                _,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                foreign=read_document(next((out/'audits').glob('*.json')));foreign['phase_id']='foreign'
                (out/'audits/foreign.json').write_text(json.dumps(foreign),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'curation audit'):
                    ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(len(runtime.requests),1)

    def test_malformed_attempt_replay_requires_raw_and_error_without_inventing_audit(self):
        _,value,_=fixture()
        for change in ('none','missing_raw','missing_error','changed_error'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);construction,cp,phase=self.setup_run(root);runtime=RetryRuntime([],value)
                with patch.object(ref,'validate',return_value=construction):
                    first,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(len(list((out/'audits').glob('*.json'))),1)
                    error_path=out/'errors/0-0.json'
                    if change=='missing_error':error_path.unlink()
                    elif change=='changed_error':
                        error=read_document(error_path);error['reason']='different error'
                        error_path.write_text(json.dumps(error),encoding='utf-8')
                    elif change=='missing_raw':
                        for request_path in (out/'requests').glob('*.json'):
                            if json.loads(read_document(request_path)['response']['choices'][0]['message']['content'])==[]:
                                request_path.unlink();break
                    if change=='none':
                        replay,_=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                        self.assertEqual(first,replay)
                    else:
                        with self.assertRaisesRegex(ValueError,'curation audit'):
                            ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(len(runtime.requests),2)

    def test_failed_unit_replay_requires_existing_rejection_audit(self):
        _,value,_=fixture();bad=dict(entities=[],facts=[],questions=[copy.deepcopy(value['questions'][-1])])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(bad)
            with patch.object(ref,'validate',return_value=construction):
                first,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(first['status'],'partial');self.assertEqual(len(runtime.requests),7)
                next((out/'audits').glob('*-0.json')).unlink()
                with self.assertRaisesRegex(ValueError,'curation audit'):
                    ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(len(runtime.requests),7)

    def test_transport_truncation_and_malformed_response_histories_replay_offline(self):
        _,value,_=fixture()
        for mode in ('http','truncated','invalid_json','empty_response'):
            class FirstFailure(Runtime):
                def _post(self,endpoint,body,*args):
                    response=super()._post(endpoint,body,*args)
                    if len(self.requests)==1:
                        if mode=='http':raise ref.HTTPError('',500,'Test server error',None,None)
                        if mode=='truncated':response['choices'][0]['finish_reason']='length'
                        if mode=='invalid_json':response['choices'][0]['message']['content']='{'
                        if mode=='empty_response':return {}
                    return response
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);construction,cp,phase=self.setup_run(root);runtime=FirstFailure(value)
                with patch.object(ref,'validate',return_value=construction):
                    first,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(first['generation_budget']['logical_calls'],2)
                    self.assertEqual(len(list((out/'audits').glob('*.json'))),1)
                    replay,_=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(first,replay);self.assertEqual(len(runtime.requests),2)
                    if mode=='http':
                        error_path=out/'errors/0-0.json';error=read_document(error_path);error['status_code']=503
                        error_path.write_text(json.dumps(error),encoding='utf-8')
                        with self.assertRaisesRegex(ValueError,'curation audit'):
                            ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                        self.assertEqual(len(runtime.requests),2)

    def test_malformed_text_fields_reject_whole_packet_with_local_string_diagnostic(self):
        unit,value,schema=fixture();good,_=ref.curate_batch(value,unit,schema)
        cases=[('facts',1,'conditions',[]),('facts',1,'conditions',None),('facts',1,'conditions',['when ready']),('entities',0,'scope',None),('entities',0,'canonical_name',[]),('facts',0,'head',None),('facts',0,'tail',[])]
        for collection,index,field,bad_value in cases:
            with self.subTest(collection=collection,field=field,value=bad_value):
                bad=copy.deepcopy(good);bad[collection][index][field]=bad_value;before=copy.deepcopy(bad)
                with self.assertRaises(ValueError) as raised:ref.curate_batch(bad,unit,schema)
                self.assertIsInstance(raised.exception,ref.ReferenceValidationError)
                diagnostic=raised.exception.diagnostic
                self.assertEqual(diagnostic['field'],field)
                self.assertEqual(diagnostic['local_id'],bad[collection][index]['local_id'])
                self.assertEqual(diagnostic['rule_id'],'json_string_field')
                self.assertEqual(diagnostic['origin'],'untrusted_previous_output_identifiers_not_source_evidence')
                self.assertEqual(bad,before)

    def test_conditions_type_feedback_repairs_from_unchanged_source_and_replays(self):
        unit,value,schema=fixture();good,_=ref.curate_batch(value,unit,schema)
        for index,bad_value in [(0,[]),(0,None),(1,['when ready']),(1,None)]:
            with self.subTest(index=index,value=bad_value),tempfile.TemporaryDirectory() as d:
                bad=copy.deepcopy(good);bad['facts'][index]['conditions']=bad_value;original=copy.deepcopy(bad)
                root=Path(d);construction,cp,phase=self.setup_run(root);runtime=RetryRuntime(bad,good)
                with patch.object(ref,'validate',return_value=construction):
                    first,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(len(runtime.requests),2)
                    first_request,repair=runtime.requests
                    self.assertEqual(first_request['unit'],repair['unit']);self.assertEqual(first_request['schema'],repair['schema'])
                    feedback=repair['validation_feedback']
                    self.assertIsNotNone(feedback.get('diagnostic'))
                    self.assertEqual(feedback['diagnostic']['field'],'conditions')
                    self.assertEqual(feedback['diagnostic']['local_id'],bad['facts'][index]['local_id'])
                    self.assertEqual(feedback['diagnostic']['rule_id'],'json_string_field')
                    for term in ['JSON string','conditions','array','null','unchanged source']:
                        self.assertIn(term,feedback['correction'])
                    self.assertLess(len(feedback['correction']),800)
                    checkpoint=read_document(next((out/'checkpoints').glob('*.json')))
                    self.assertEqual(checkpoint['facts'],good['facts'])
                    self.assertEqual(checkpoint['facts'][1]['conditions'],'when ready')
                    self.assertEqual(checkpoint['curation']['attempts'],2)
                    self.assertTrue(checkpoint['curation']['whole_response_valid'])
                    self.assertEqual(len(list((out/'audits').glob('*.json'))),1)
                    error=read_document(out/'errors/0-0.json')
                    self.assertEqual(error['diagnostic'],feedback['diagnostic'])
                    first_raw=read_document(out/error['request_ref'])
                    self.assertEqual(json.loads(first_raw['response']['choices'][0]['message']['content']),original)
                    ref.validate_batch(checkpoint,unit,schema)
                    replay,_=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                    self.assertEqual(first,replay);self.assertEqual(len(runtime.requests),2)
                    self.assertEqual(bad,original)

    def test_repeated_list_or_null_conditions_remain_failed_without_curation(self):
        unit,value,schema=fixture();good,_=ref.curate_batch(value,unit,schema)
        for bad_value in ([],None):
            with self.subTest(value=bad_value),tempfile.TemporaryDirectory() as d:
                bad=copy.deepcopy(good);bad['facts'][1]['conditions']=bad_value
                root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(bad)
                with patch.object(ref,'validate',return_value=construction):
                    result,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
                self.assertEqual(result['status'],'partial');self.assertEqual(result['failed_chunks'],['c0'])
                self.assertEqual(result['facts'],[]);self.assertEqual(result['questions'],[])
                self.assertEqual(len(runtime.requests),7)
                self.assertFalse(list((out/'checkpoints').glob('*.json')))
                self.assertFalse(list((out/'audits').glob('*.json')))
                for path in (out/'errors').glob('*.json'):
                    self.assertEqual(read_document(path)['diagnostic']['field'],'conditions')

    def test_delivered_prompt_type_examples_match_offline_source_validation(self):
        unit,value,schema=fixture();good,_=ref.curate_batch(value,unit,schema)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);construction,cp,phase=self.setup_run(root);runtime=Runtime(good)
            with patch.object(ref,'validate',return_value=construction):
                _,out=ref.generate(cp,phase,'c',root/'out',runtime=runtime,smoke=1)
            request=read_document(next((out/'requests').glob('*.json')))
            prompt=request['request']['messages'][0]['content'];examples=[]
            for text in re.findall(r'\{[^{}\n]*\}',prompt):
                try:example=json.loads(text)
                except json.JSONDecodeError:continue
                if 'conditions' in example:examples.append(example)
            self.assertEqual(len(examples),2)
            empty,conditional=examples
            for key in ['local_id','canonical_name','scope','head','tail','conditions']:
                self.assertIsInstance(empty[key],str)
            self.assertEqual(empty['conditions'],'');self.assertEqual(empty['aliases'],[]);self.assertEqual(empty['fact_path'],[])
            self.assertEqual(conditional['conditions'],'when ready')
            probe=copy.deepcopy(good)
            probe['facts'][0]['conditions']=empty['conditions'];probe['facts'][1]['conditions']=conditional['conditions']
            chosen,audit=ref.curate_batch(probe,unit,schema)
            self.assertTrue(audit['whole_response_valid']);self.assertEqual(chosen,probe)
            for malformed in ([],None,['when ready']):
                probe['facts'][1]['conditions']=malformed
                with self.assertRaises(ref.ReferenceValidationError):ref.curate_batch(probe,unit,schema)


if __name__=='__main__':unittest.main()
