import copy,unittest,tempfile,json,asyncio,hashlib
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from course_experiments.manifest import digest
from course_experiments.evaluation.run_native_final_qa import bind_final_plan,validate_preseal_template,verify_file,load_blind64,load_legacy123,verify_preseal_assets
class FinalQATests(unittest.TestCase):
 def base(self,method='hipporag'):
  index={'method':method,'book':'b','source':'same'}
  return dict(method=method,book='b',allowed_books=['b'],index_contract=index,contract=dict(index_identity=digest(index),datasets={'old':{}},policy={},code={},global_question_counts={'b':{'old':4}}))
 def test_fixed_dataset_changes_no_index_and_keeps_local_global_distinct(self):
  base=self.base('graphrag');data={'b':{'questions':[{'question_id':'q','type':'definition'}]}};dep={'index_identity':digest(base['index_contract'])}
  local=bind_final_plan(base,data,{'b':{'path':'dataset','sha256':'x'}},dep,split='legacy123',track='local',dataset_metadata={'accepted_count':123})
  glob=bind_final_plan(base,data,{'b':{'path':'dataset','sha256':'x'}},dep,split='legacy123',track='global',dataset_metadata={'accepted_count':123})
  self.assertEqual(local['index_contract'],base['index_contract']);self.assertEqual(local['contract']['tracks'],['local']);self.assertEqual(glob['contract']['tracks'],['global']);self.assertNotEqual(digest(local['contract']),digest(glob['contract']));self.assertEqual(base['contract']['datasets'],{'old':{}})
 def test_foreign_book_missing_or_duplicate_question_refused(self):
  base=self.base();dep={'index_identity':digest(base['index_contract'])}
  for data in [{'foreign':{'questions':[]}}, {'b':{'questions':[{'question_id':'q'},{'question_id':'q'}]}}]:
   with self.assertRaises(ValueError):bind_final_plan(base,data,{},dep,split='legacy123',track='native',dataset_metadata={})
 def test_index_change_and_wrong_track_refused(self):
  b=self.base();d={'b':{'questions':[{'question_id':'q'}]}}
  for dep,track in [({'index_identity':'different'},'native'),({'index_identity':digest(b['index_contract'])},'local')]:
   with self.assertRaises(ValueError):bind_final_plan(b,d,{},dep,split='legacy123',track=track,dataset_metadata={})
 def test_preseal_has_no_future_seal_or_reverse_freeze_dependency(self):
  s=dict(version='native_final_QA_preseal_v1',method='graphrag',book='b',track='global',split='blind64',app='/immutable',python='/python',recipe={},worker={},parameters={},index_pin={})
  validate_preseal_template(s)
  for key in ['sealed','algorithm_freeze','questions','dataset_index']:
   invalid=dict(s,**{key:{}})
   with self.assertRaises(ValueError):validate_preseal_template(invalid)
 def test_code_record_verification_does_not_parse_code_as_JSON(self):
  from course_experiments.manifest import file_record
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'worker.py';p.write_text('print("code")');record=file_record(p);self.assertEqual(verify_file(record),record);p.write_text('changed')
   with self.assertRaises(ValueError):verify_file(record)
 def test_new64_requires_full_seal_and_exact_quota(self):
  from course_experiments.manifest import file_record
  from course_experiments.evaluation.run_native_holdout import DependencyPending
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'sealed.json';p.write_text(json.dumps(dict(status='sealed_insufficient',target_count=64,accepted_count=63,shortage=1)))
   with patch('course_experiments.evaluation.blind64_pipeline.require_freeze',return_value=({},{})):
    with self.assertRaises(DependencyPending):load_blind64(file_record(p),dict(path='frozen',sha256='x'))
 def test_legacy_seal_or_selection_byte_change_refused_before_load(self):
  from course_experiments.manifest import file_record
  with tempfile.TemporaryDirectory() as d:
   for changed in ['sealed','selection']:
    p={k:Path(d)/(k+'.json') for k in ['sealed','selection']}
    for q in p.values():q.write_text('{}')
    records={k:file_record(q) for k,q in p.items()};p[changed].write_text('{"changed":true}')
    with patch('course_experiments.evaluation.run_native_holdout.load_holdout') as loader:
     with self.assertRaises(ValueError):load_legacy123(records['sealed'],records['selection'])
     loader.assert_not_called()
 def test_existing_light_adapter_reuses_full_processed_index_without_insert(self):
  from course_experiments.evaluation import run_native_holdout as existing
  from course_experiments.evaluation.run_native_final_qa import run_existing
  doc=SimpleNamespace(document_id='d',text='original source');records=[dict(document_id='d',text_sha256=hashlib.sha256(doc.text.encode()).hexdigest(),text_utf8_bytes=len(doc.text.encode()))]
  class Owner:
   def request_executor(self,*a):raise AssertionError('No model or index rebuild allowed')
  class Fake:
   def _manifest(self):return {'records':records}
   async def index_book(self,*a):raise AssertionError('Native insertion forbidden')
   async def close(self):pass
  adapter=Fake()
  async def status(ids):return [{'status':'processed'} for _ in ids]
  adapter.core=SimpleNamespace(doc_status=SimpleNamespace(get_by_ids=status))
  async def opener(*a):return adapter
  async def exercise(plan,root,owner,adapter_factory):
   bound=await adapter_factory(plan,Path('/index'),owner);return await bound.index_book('b',[doc])
  with patch.object(existing.shared,'open_adapter',opener),patch.object(existing.shared,'run_book',exercise):
   result=asyncio.run(run_existing(dict(method='lightrag',book='b'),'/run',Owner(),dict(index_root='/index',native_index={'records':records})))
  self.assertEqual(result,{'records':records})
 def test_graph_rows_choose_official_query_and_restore_track_configuration(self):
  from course_experiments.evaluation import run_native_holdout as existing,run_native_microsoft_full as official
  from course_experiments.evaluation.run_native_microsoft_faithful import query
  from course_experiments.evaluation.run_native_final_qa import run_existing
  previous=official.TRACKS
  async def exercise(plan,*a):
   self.assertEqual(official.TRACKS,('global',));self.assertIs(existing.query_with_source_mapping,query);return {'status':'fixture_only_no_API'}
  with patch.object(existing,'run_existing',exercise):
   result=asyncio.run(run_existing(dict(method='graphrag',contract={'tracks':['global']}),'/run',object(),{}))
  self.assertEqual(official.TRACKS,previous);self.assertEqual(result['status'],'fixture_only_no_API')
 def test_code_parameter_and_target_method_membership_before_any_sealed_read(self):
  from course_experiments.manifest import file_record
  with tempfile.TemporaryDirectory() as d:
   code=Path(d)/'worker.py';config=Path(d)/'parameters.json';template_path=Path(d)/'template.json';code.write_text('pass');config.write_text('{}');template_path.write_text('{}')
   cr,pr,tr=map(file_record,[code,config,template_path]);template=dict(registered_method_id='native_hipporag',worker=cr,code_assets=[cr],parameters={'config':pr});frozen={'methods':[dict(method_id='native_hipporag',code=[tr,cr],parameters=[pr])]}
   verify_preseal_assets(template,template_record=tr,frozen=frozen)
   for changed in [code,config]:
    previous=changed.read_text();changed.write_text('tampered')
    with self.assertRaises(ValueError):verify_preseal_assets(template,template_record=tr,frozen=frozen)
    changed.write_text(previous)
   for bad in [{'methods':[dict(method_id='foreign',code=[tr,cr],parameters=[pr])]}, {'methods':[dict(method_id='native_hipporag',code=[cr],parameters=[pr])]}, {'methods':[dict(method_id='native_hipporag',code=[tr],parameters=[pr])]}]:
    with self.assertRaises(ValueError):verify_preseal_assets(template,template_record=tr,frozen=bad)
 def test_unregistered_or_tampered_template_cannot_open_new_sealed_dataset(self):
  from course_experiments.manifest import file_record
  from course_experiments.evaluation.run_native_final_qa import prepare_from_spec
  with tempfile.TemporaryDirectory() as d:
   code=Path(d)/'worker.py';config=Path(d)/'params.json';tp=Path(d)/'template.json';code.write_text('pass');config.write_text('{}');cr,pr=map(file_record,[code,config])
   t=dict(version='native_final_QA_preseal_v1',registered_method_id='native_hipporag',method='hipporag',book='b',track='native',split='blind64',app='/immutable',python='/python',recipe={},worker=cr,parameters={'config':pr},code_assets=[cr],index_pin={})
   tp.write_text(json.dumps(t));spec={'template':file_record(tp),'algorithm_freeze':{'path':'freeze','sha256':'f'}}
   for changed in [None,code,config]:
    prior=changed.read_text() if changed else None
    if changed:changed.write_text('tamper')
    with patch('course_experiments.evaluation.blind64_pipeline.require_freeze',return_value=({'methods':[dict(method_id='foreign',code=[],parameters=[])]},{})),patch('course_experiments.evaluation.run_native_final_qa.load_blind64') as sealed_loader:
     with self.assertRaises(ValueError):prepare_from_spec(spec)
     sealed_loader.assert_not_called()
    if changed:changed.write_text(prior)
 def test_old313_is_only_graph_global_and_original_confirmation_split(self):
  b=self.base('graphrag');books={'b':{'questions':[{'question_id':'q'}]}};dep={'index_identity':digest(b['index_contract'])}
  plan=bind_final_plan(b,books,{'b':{}},dep,split='regression313',track='global',dataset_metadata={'accepted_count':313})
  self.assertEqual(set(plan['datasets']),{'confirmation'});self.assertEqual(plan['contract']['tracks'],['global'])
  with self.assertRaises(ValueError):bind_final_plan(b,books,{'b':{}},dep,split='regression313',track='local',dataset_metadata={})
  template=dict(version='native_final_QA_preseal_v1',method='graphrag',book='b',track='global',split='regression313',app='/immutable',python='/python',recipe={},worker={},parameters={},index_pin={})
  validate_preseal_template(template)
  with self.assertRaises(ValueError):validate_preseal_template(dict(template,track='local'))
if __name__=='__main__':unittest.main()
