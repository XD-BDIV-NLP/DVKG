import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from course_experiments.manifest import digest,file_record
from course_experiments.evaluation import followup_stages as stage


class SupplementPipelineTests(unittest.TestCase):
    def test_source_only_generation_holdout_and_idempotent_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);audit=root/'audit';calls=[]
            def save(p,v):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False),'utf-8');return file_record(p)
            def source(book,key,text):return dict(book_id=book,chunk_id=key,evidence_ids=[key],evidence=[dict(id=key,text=text)],text=text,vector=[1.,0.])
            jobs=[dict(name='compAB_F',active_books=['A'])]
            for cohort in ['compABCD','osAB','netAB']:
                rows=[source('A',cohort+'a','命中条件访问主存。'*100),source('C',cohort+'c','未命中条件访问主存。'*100),source('A',cohort+'old','OLD'*300)]
                index=save(root/cohort/'index.json',dict(rows=rows))
                reference=save(root/cohort/'old_reference.json',dict(course_id=cohort,questions=[dict(evidence_ids=[cohort+'old'],question='问题conditions'+cohort+'a')]))
                phase=save(root/cohort/'phase.json',dict(files=dict(index=index,reference=reference)))
                jobs.append(dict(name=cohort+'_F',parent_phase=phase))
                save(audit/'repair_validation_v3/reference'/('compAB' if cohort=='compABCD' else cohort)/'reference.json',dict(questions=[]))
            def model(runtime,prompt,payload,max_tokens,path,attempt=0):
                calls.append(payload)
                self.assertNotIn('OLD',json.dumps(payload,ensure_ascii=False))
                sources=payload['sources'];cites=[dict(evidence_id=r['evidence_id'],quote=r['text'][:8]) for r in sources]
                if prompt==stage.GENERATE_PROMPT:
                    return dict(question='问题'+payload['task_type']+sources[0]['evidence_id'],answer='条件下访问主存。',citations=cites,
                        condition='命中条件',applicable_case='命中',inapplicable_case='未命中'),{}
                if prompt==stage.ANSWER_AUDIT_PROMPT:return dict(supported=True,coherent=True,explicit_conditions=True,quotes=cites),{}
                return dict(answerable=len(sources)>1,quotes=cites),{}
            with patch.object(stage,'AUDIT',audit),patch('course_experiments.evaluation.chat.call_json',side_effect=model):
                results=stage.supplement_stage(dict(jobs=jobs),object(),root/'new')
                count=len(calls)
                repeated=stage.supplement_stage(dict(jobs=jobs),object(),root/'new')
            self.assertEqual(len(calls),count)
            self.assertEqual([r['reference'] for r in results],[r['reference'] for r in repeated])
            self.assertEqual(len(results),3)
            for r in results:
                value=json.loads(Path(r['reference']['path']).read_text('utf-8'))
                self.assertEqual(value['reference_id'],digest({k:v for k,v in value.items() if k!='reference_id'}))
                self.assertEqual(value['status'],'frozen_actual_valid_subset_target_shortfall')
                self.assertGreater(value['counts']['cross_source'],0)
                self.assertGreater(value['counts']['conditions'],0)
                self.assertNotIn('问题conditions'+r['cohort']+'a',[q['question'] for q in value['questions']])
                self.assertTrue(value['selection_rejections'])

if __name__=='__main__':unittest.main()
