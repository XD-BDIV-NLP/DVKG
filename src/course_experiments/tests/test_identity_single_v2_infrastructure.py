import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from course_experiments.manifest import digest,file_record,read_document
from course_experiments.tests.test_identity_single_v2 import Provider
from course_experiments.tests.test_scope_identity_batch import packet
from course_experiments.evaluation import identity_single_v2 as identity


class Counter:
    assets={'official_tokenizer':'synthetic-fixed-assets'}
    def count(self,messages):return 10


class ActualProvider(Provider):
    def __init__(self,root,*,timeout=False,bad_quote=False):
        super().__init__();self.root=Path(root);self.root.mkdir(exist_ok=True)
        self.database=self.root/'audit.sqlite';self.runtime=SimpleNamespace(config={});self.timeout=timeout;self.bad_quote=bad_quote
        self._identity_single_attempt_cap=2;self._identity_single_total_cap=3000
        with closing(sqlite3.connect(self.database)) as db,db:
            db.execute('CREATE TABLE IF NOT EXISTS events(key TEXT,status TEXT)');db.execute('CREATE TABLE IF NOT EXISTS receipts(key TEXT PRIMARY KEY,record TEXT)')
    def request_executor(self,endpoint,body,stage):
        if self.timeout:raise TimeoutError('Native global request slots remained occupied for 600 seconds')
        key=digest(dict(endpoint=endpoint,body=body,config=self.runtime.config))
        with closing(sqlite3.connect(self.database)) as db,db:db.execute('INSERT INTO events VALUES(?,?)',(key,'request_started'))
        raw,hit=super().request_executor(endpoint,body,stage)
        if self.bad_quote:
            value=json.loads(raw['choices'][0]['message']['content']);value['results'][0]['right_citations'][0]['quote']='fabricated';raw['choices'][0]['message']['content']=json.dumps(value)
        receipts=self.root/'receipts';receipts.mkdir(exist_ok=True);path=receipts/(key+'.json');path.write_text(json.dumps(dict(request_id=key,endpoint=endpoint,request=body,response=raw,runtime_config=self.runtime.config)))
        with closing(sqlite3.connect(self.database)) as db,db:db.execute('INSERT OR REPLACE INTO receipts VALUES(?,?)',(key,json.dumps(file_record(path))))
        return raw,hit


class InfrastructureCompletionTests(unittest.TestCase):
    def api(self):
        from course_experiments.evaluation import identity_single_v2_infrastructure
        return identity_single_v2_infrastructure
    def fixture(self,tmp):
        root=Path(tmp);p=packet(1,is_prior=True);provider=ActualProvider(root/'formal',timeout=True)
        identity.run_population([p],'prior',provider,root/'prior',counter=Counter())
        return root,p,ActualProvider(root/'formal')

    def test_zero_http_slot_timeout_first_execution_keeps_original_unknown_and_contract(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root,p,provider=self.fixture(tmp);folder=root/'prior/decisions'/p['candidate_id'];before=file_record(folder/'decision.json');contract=file_record(root/'prior/contract.json')
            receipt=api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            self.assertEqual(len(provider.calls),1);self.assertEqual(file_record(folder/'decision.json'),before);self.assertEqual(file_record(root/'prior/contract.json'),contract)
            self.assertEqual(receipt['classification'],'UNKNOWN');self.assertFalse(receipt['eligible_identity_edge']);self.assertEqual(receipt['actual_HTTP_before'],0);self.assertEqual(receipt['actual_HTTP_after'],1)
            self.assertTrue((folder/'primary.json').exists());self.assertFalse((folder/'reverse.json').exists())
            self.assertEqual(identity.validate_saved_population([p],'prior',root/'prior')['unknown_count'],1)
            self.assertEqual(api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter()),receipt)
            self.assertEqual(len(provider.calls),1)

    def test_any_existing_actual_model_attempt_cannot_be_reasked(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root,p,provider=self.fixture(tmp);body=identity.request_body(p,'prior');key=digest(dict(endpoint='/chat/completions',body=body,config={}))
            with closing(sqlite3.connect(provider.database)) as db,db:db.execute('INSERT INTO events VALUES(?,?)',(key,'request_started'))
            with self.assertRaises(ValueError):api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            self.assertEqual(provider.calls,[])

    def test_second_zero_http_slot_wait_is_not_false_completion(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root,p,provider=self.fixture(tmp);provider.timeout=True;folder=root/'prior/decisions'/p['candidate_id'];before=file_record(folder/'decision.json')
            with self.assertRaises(TimeoutError):api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            self.assertEqual(file_record(folder/'decision.json'),before);self.assertFalse((folder/'primary.json').exists());self.assertFalse((folder/'infrastructure_completion.json').exists())

    def test_invalid_first_actual_response_stays_unknown_without_semantic_retry(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root,p,provider=self.fixture(tmp);provider.bad_quote=True
            receipt=api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            self.assertEqual(receipt['strict_validation'],'failed_kept_UNKNOWN');self.assertEqual(len(provider.calls),1)
            api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter());self.assertEqual(len(provider.calls),1)

    def test_changed_error_or_full_population_cannot_be_completed(self):
        api=self.api()
        for change in ['error','population']:
            with tempfile.TemporaryDirectory() as tmp:
                root,p,provider=self.fixture(tmp);folder=root/'prior/decisions'/p['candidate_id']
                if change=='error':Path(read_document(folder/'decision.json')['error_record']['path']).write_text('changed')
                population=[p,packet(2,is_prior=True)] if change=='population' else [p]
                with self.assertRaises(ValueError):api.complete_unattempted_as_unknown(population,'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
                self.assertEqual(provider.calls,[])

    def test_saved_raw_tamper_blocks_completion_reuse(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root,p,provider=self.fixture(tmp);api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            path=root/'prior/decisions'/p['candidate_id']/'primary.json';value=read_document(path);value['raw_response']={};path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            self.assertEqual(len(provider.calls),1)

    def test_check_only_never_executes_or_adds_completion_evidence(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root,p,provider=self.fixture(tmp);folder=root/'prior/decisions'/p['candidate_id']
            value=api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter(),execute=False)
            self.assertEqual(value['status'],'check_only_zero_actual_slot_timeout');self.assertEqual(provider.calls,[])
            self.assertFalse((folder/'primary.json').exists());self.assertFalse((folder/'infrastructure_first_execution_proof.json').exists())

    def test_global_halt_stops_before_first_actual_execution(self):
        api=self.api()
        from course_experiments.evaluation.native_provider import NativeBudgetExceeded
        with tempfile.TemporaryDirectory() as tmp:
            root,p,provider=self.fixture(tmp)
            def halt():raise NativeBudgetExceeded('original global budget exhausted')
            provider.raise_if_fatal=halt
            with self.assertRaises(NativeBudgetExceeded):api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            self.assertEqual(provider.calls,[])

    def test_changed_pre_execution_proof_blocks_reuse_without_reasking(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root,p,provider=self.fixture(tmp);receipt=api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            Path(receipt['pre_execution_proof']['path']).write_text('{}')
            with self.assertRaises(ValueError):api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
            self.assertEqual(len(provider.calls),1)

    def test_paid_response_crash_recovers_original_receipt_without_network(self):
        api=self.api()
        for after_primary in [False,True]:
            with tempfile.TemporaryDirectory() as tmp:
                root,p,provider=self.fixture(tmp);real=api._checkpoint
                def crash(path,*args,**kwargs):
                    if after_primary:real(path,*args,**kwargs)
                    raise OSError('simulated interruption after native receipt')
                with patch.object(api,'_checkpoint',side_effect=crash):
                    with self.assertRaises(OSError):api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
                self.assertEqual(len(provider.calls),1)
                recovered=api.complete_unattempted_as_unknown([p],'prior',root/'prior',p['candidate_id'],provider,counter=Counter())
                self.assertTrue(recovered['recovered_without_HTTP']);self.assertEqual(recovered['classification'],'UNKNOWN');self.assertEqual(len(provider.calls),1)


if __name__=='__main__':unittest.main()
