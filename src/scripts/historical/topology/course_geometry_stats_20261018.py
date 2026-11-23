# -*- coding: utf-8 -*-
"""PRIMARY-only offline geometry, paired uncertainty, and blank blinded audit supplement."""
from __future__ import annotations
import argparse,ast,collections,csv,hashlib,html,json,math,os,random,socket,sys
from pathlib import Path

VERSION='geometry_stats_20261018_v1'
SEED=42
REPS=10000
CONCEPT_TYPES={'Concept','Method','Principle','Formula','Algorithm','Structure','Process','Metric','Tool','Phenomenon'}
SOURCES={
 'forman_1d':'https://arxiv.org/abs/1603.00386',
 'global_efficiency':'https://arxiv.org/abs/cond-mat/0101396',
 'effective_diameter':'https://snap.stanford.edu/class/cs224w-readings/leskovec07evolution.pdf',
 'paired_bootstrap':'https://aclanthology.org/D12-1091/',
 'finite_monte_carlo_p':'https://doi.org/10.2202/1544-6115.1585',
 'holm':'https://www.jstor.org/stable/4615733'}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def record(path):
 path=Path(path).resolve(strict=True)
 return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def checked(r):
 if record(r['path'])!=r:raise ValueError('Frozen source record changed')
 return Path(r['path'])
def immutable(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def offline():
 def denied(*a,**k):raise RuntimeError('Supplement forbids network and model calls')
 socket.socket.connect=denied;socket.create_connection=denied
 for key in list(os.environ):
  if key.upper().endswith('_API_KEY') or key.upper().endswith('_PROXY'):os.environ.pop(key)
def semantic(graph):
 entities=graph.get('entities',[dict(e,entity_id=e.get('entity_id',e.get('id')),canonical_name=e.get('canonical_name',e.get('name'))) for e in graph.get('nodes',[])])
 selected=[e for e in entities if e.get('type') in CONCEPT_TYPES and e.get('type_status')!='native_untyped_candidate']
 ids={e['entity_id'] for e in selected}
 if len(ids)!=len(selected):raise ValueError('Duplicate semantic entity identity')
 relations=graph.get('relations',[dict(e,head_entity_id=e['source'],tail_entity_id=e['target'],relation='native_adjacency') for e in graph.get('edges',[])])
 kept=[r for r in relations if r.get('head_entity_id') in ids and r.get('tail_entity_id') in ids]
 return selected,kept,dict(input_entities=len(entities),excluded_entities=len(entities)-len(selected),input_relations=len(relations),excluded_endpoint_relations=len(relations)-len(kept))
def percentile(values,q):
 if not values:return None
 values=sorted(values);p=(len(values)-1)*q;lo=math.floor(p);hi=math.ceil(p)
 return values[lo]+(values[hi]-values[lo])*(p-lo)
def geometry(graph):
 entities,relations,audit=semantic(graph);ids=sorted(e['entity_id'] for e in entities);adj={i:set() for i in ids};edges=set();loops=0;parallel=0
 for r in relations:
  u,v=r['head_entity_id'],r['tail_entity_id']
  if u==v:loops+=1;continue
  edge=tuple(sorted((u,v)))
  if edge in edges:parallel+=1;continue
  edges.add(edge);adj[u].add(v);adj[v].add(u)
 audit.update(removed_loops=loops,collapsed_parallel_or_reverse_edges=parallel,projection='typed concept semantic endpoints, undirected simple unweighted, all retained isolates')
 curvature=[4-len(adj[u])-len(adj[v]) for u,v in sorted(edges)];n=len(ids)
 sources=ids if n<=256 else sorted(random.Random(SEED).sample(ids,256));hist=collections.Counter()
 for source in sources:
  distances={source:0};queue=collections.deque([source])
  while queue:
   u=queue.popleft()
   for v in adj[u]:
    if v not in distances:distances[v]=distances[u]+1;queue.append(v)
  hist.update(d for v,d in distances.items() if v!=source)
 possible=len(sources)*(n-1);reachable=sum(hist.values());target=math.ceil(.9*reachable);effective=None;running=0
 for d,count in sorted(hist.items()):
  running+=count
  if effective is None and running>=target:effective=d
 return dict(version=VERSION,status='completed' if n else 'insufficient_no_typed_semantic_entities',nodes=n,simple_edges=len(edges),projection=audit,
  forman_1d=dict(formula='4 - degree(u) - degree(v); no 2-cells/triangle augmentation',edge_count=len(curvature),mean=sum(curvature)/len(curvature) if curvature else None,minimum=min(curvature) if curvature else None,median=percentile(curvature,.5),maximum=max(curvature) if curvature else None,negative_fraction=sum(x<0 for x in curvature)/len(curvature) if curvature else None,histogram=dict(sorted(collections.Counter(curvature).items()))),
  distances=dict(mode='exact_all_sources' if n<=256 else 'uniform_without_replacement_source_BFS',seed=SEED,source_count=len(sources),source_ids_digest=digest(sources),total_nodes=n,ordered_pairs_observed=possible,reachable_ordered_pairs_observed=reachable,unreachable_ordered_pairs_observed=possible-reachable,reachable_pair_fraction=reachable/possible if possible else None,finite_pair_mean_distance=sum(d*c for d,c in hist.items())/reachable if reachable else None,global_efficiency=sum(c/d for d,c in hist.items())/possible if possible else None,unreachable_efficiency_contribution=0,effective_diameter_90_finite_pairs=effective,percentile_convention='smallest integer d with cumulative finite nonself pair fraction >=0.90; source-sample estimate when sampled',maximum_observed_finite_distance=max(hist) if hist else None,maximum_distance_interpretation='exact_finite_diameter' if n<=256 else 'lower_bound_on_finite_diameter',finite_distance_histogram=dict(sorted(hist.items()))))
def fixed_scores(reference,rows):
 ids=[q['question_id'] for q in reference['questions']]
 if len(set(ids))!=len(ids) or set(ids)!=set(rows):raise ValueError('Full fixed question identity mismatch')
 values=[]
 for qid in ids:
  row=rows[qid]
  if row.get('question_id',qid)!=qid:raise ValueError('Per-question ID mismatch')
  if row['status']=='completed':
   value=row['scores']['f1']
   if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid completed F1')
  elif row['status']=='unanswerable_active_sources':value=0.0
  else:raise ValueError('Unfinished/unknown question status; do not silently impute missing scores')
  values.append(value)
 return values
def paired(left,right):
 if len(left)!=len(right) or not left:raise ValueError('Paired arrays must share nonempty question denominator')
 differences=[a-b for a,b in zip(left,right)];n=len(differences);observed=sum(differences)/n
 rng=random.Random(SEED);bootstrap=[]
 for _ in range(REPS):bootstrap.append(sum(differences[rng.randrange(n)] for _ in range(n))/n)
 rng=random.Random(SEED);exceed=0
 for _ in range(REPS):
  permuted=sum(d if rng.getrandbits(1) else -d for d in differences)/n
  exceed+=abs(permuted)>=abs(observed)-1e-15
 return dict(metric='F1',question_count=n,left_mean=sum(left)/n,right_mean=sum(right)/n,mean_difference=observed,paired_bootstrap_percentile_ci95=[percentile(bootstrap,.025),percentile(bootstrap,.975)],permutation_two_sided_p=(exceed+1)/(REPS+1),permutation_exceedances=exceed,replicates=REPS,seed=SEED,unit='paired fixed-reference question; conditional on this frozen question set and these outputs',permutation_assumption='Within-question method labels/signs exchangeable under sharp null; related questions may violate independent sign-flip assumption',inference_limit='Question-level uncertainty only; no new LLM seeds, no corpus/course population generalization; dependent questions may make intervals optimistic')
def holm(values):
 result=[None]*len(values);maximum=0
 for rank,i in enumerate(sorted(range(len(values)),key=lambda i:values[i])):
  maximum=max(maximum,min(1,values[i]*(len(values)-rank)));result[i]=maximum
 return result
def stratified(rows,key,count):
 groups=collections.defaultdict(list)
 for row in sorted(rows,key=lambda r:digest(r)):groups[row[key]].append(row)
 rng=random.Random(SEED)
 for group in groups.values():rng.shuffle(group)
 output=[]
 while len(output)<count and any(groups.values()):
  for kind in sorted(groups):
   if groups[kind] and len(output)<count:output.append(groups[kind].pop())
 return output

def phase_job(result_record,phase_record):
 result=read(checked(result_record));phase=read(checked(phase_record))
 if phase['phase_id']!=digest({k:v for k,v in phase.items() if k!='phase_id'}):raise ValueError('Phase content digest mismatch')
 if result['status']!='completed' or result['input_digest']!=phase['phase_id']:raise ValueError('Completed result/phase identity mismatch')
 reference_record=phase['files']['reference'];reference=read(checked(reference_record))
 if result['split_id']!=reference['reference_id'] or result['question_count']!=len(reference['questions']):raise ValueError('Reference identity/count mismatch')
 if result['failed_questions']:raise ValueError('Failed questions require a separate prespecified analysis')
 questions=[]
 for q in reference['questions']:
  p=Path(result_record['path']).parent/'questions'/(q['question_id']+'.json');questions.append(record(p))
 rows={q['question_id']:read(r['path']) for q,r in zip(reference['questions'],questions)};fixed_scores(reference,rows)
 completed=[r for r in rows.values() if r['status']=='completed'];unanswerable=len(rows)-len(completed)
 if len(completed)!=result['completed_count'] or unanswerable!=result['unanswerable_count']:raise ValueError('Original completion counts disagree with question files')
 if completed and not math.isclose(sum(r['scores']['f1'] for r in completed)/len(completed),result['metrics']['f1'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Original aggregate F1 disagrees with question files')
 if result['graph_id']!=digest(read(checked(phase['files']['graph']))):raise ValueError('Original result graph identity mismatch')
 return dict(result=result_record,phase=phase_record,reference=reference_record,reference_id=reference['reference_id'],phase_id=phase['phase_id'],graph=phase['files']['graph'],index=phase['files']['index'],parameters=phase['parameters'],question_files=questions,question_ids_digest=digest([q['question_id'] for q in reference['questions']]))
def prepare(spec_path,output):
 offline();spec=read(spec_path);base=Path(output)
 if base.exists():raise FileExistsError('Immutable supplement namespace already exists')
 projection=checked(spec['projection_source']);tree=ast.parse(projection.read_text('utf-8-sig'))
 types=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONCEPT_TYPES' for t in n.targets))
 if types!=CONCEPT_TYPES:raise ValueError('Native semantic concept type convention changed')
 jobs={name:phase_job(row['result'],row['phase']) for name,row in spec['jobs'].items()}
 graphs={}
 for row in spec['graphs']:
  if 'source_phase' in row:
   phase=read(checked(row['source_phase']))
   if phase['phase_id']!=row['source_phase_id'] or phase['files']['graph']!=row['graph']:raise ValueError('Graph/source-phase binding mismatch')
  path=checked(row['graph']);graph=read(path);graphs.setdefault(row['graph']['sha256'],dict(row,semantic_nodes=len(semantic(graph)[0])))
 for job in jobs.values():
  path=checked(job['graph']);graphs.setdefault(job['graph']['sha256'],dict(graph=job['graph'],label='RAG_graph_'+job['graph']['sha256'][:12],semantic_nodes=len(semantic(read(path))[0])))
 for contrast in spec['contrasts']:
  a,b=[jobs[contrast[k]] for k in ['left','right']]
  if a['reference']!=b['reference'] or a['question_ids_digest']!=b['question_ids_digest']:raise ValueError('Contrasts cannot pool references')
  parameters=[{k:v for k,v in j['parameters'].items() if k not in {'control','active_books'}} for j in [a,b]]
  if parameters[0]!=parameters[1]:raise ValueError('Model/retrieval budgets differ')
 if len(spec['contrasts'])!=9 or len({c['contrast_id'] for c in spec['contrasts']})!=9:raise ValueError('Declared contrast family must remain 9 unique tests')
 manifest=dict(version=VERSION,seed=SEED,bootstrap_replicates=REPS,permutation_replicates=REPS,declared_contrast_family=spec['contrasts'],multiple_testing='Holm FWER 0.05 across all 9 declared contrasts; CIs marginal not simultaneous',score_policy='Fixed full reference; completed original F1, native explicitly unanswerable=0 sidecar, missing/failed rejected, no native outputs rewritten',jobs=jobs,graphs=list(graphs.values()),audit_courses=spec['audit_courses'],definitions=SOURCES,source_code=record(__file__),spec=record(spec_path),projection_source=spec['projection_source'],native_evaluator=spec['native_evaluator'],raw_text_exported_outside_primary=False,human_audited=False)
 for r in [manifest['projection_source'],manifest['native_evaluator']]:checked(r)
 manifest['manifest_id']=digest(manifest);base.mkdir();immutable(base/'manifest.json',manifest)
 return dict(status='prepared_zero_API',manifest=record(base/'manifest.json'),jobs=len(jobs),unique_graphs=len(graphs),contrasts=len(spec['contrasts']),human_audited=False,raw_text_exported_outside_primary=False)
def load_manifest(base):
 manifest=read(base/'manifest.json')
 if manifest['manifest_id']!=digest({k:v for k,v in manifest.items() if k!='manifest_id'}):raise ValueError('Manifest digest changed')
 checked(manifest['source_code']);checked(manifest['spec']);checked(manifest['projection_source']);checked(manifest['native_evaluator'])
 for job in manifest['jobs'].values():
  for key in ['result','phase','reference','graph','index']:checked(job[key])
  for row in job['question_files']:checked(row)
 for row in manifest['graphs']:checked(row['graph'])
 return manifest
def run(base):
 offline();base=Path(base);manifest=load_manifest(base)
 geometries=[]
 for row in manifest['graphs']:
  target=base/'geometry'/(row['graph']['sha256']+'.json')
  if target.exists():
   previous=read(target)
   if previous['manifest_id']!=manifest['manifest_id']:raise ValueError('Geometry resume manifest mismatch')
   geometries.append(previous);continue
  value=dict(input=row,manifest_id=manifest['manifest_id'],metrics=geometry(read(row['graph']['path'])));immutable(target,value);geometries.append(value)
 values={}
 for name,job in manifest['jobs'].items():
  reference=read(job['reference']['path']);rows={r['question_id']:r for r in [read(f['path']) for f in job['question_files']]};values[name]=fixed_scores(reference,rows)
 contrasts=[]
 for contrast in manifest['declared_contrast_family']:
  a,b=[manifest['jobs'][contrast[k]] for k in ['left','right']]
  result=paired(values[contrast['left']],values[contrast['right']]);contrasts.append(dict(contrast,reference_id=a['reference_id'],left_phase_id=a['phase_id'],right_phase_id=b['phase_id'],**result))
 for row,adjusted in zip(contrasts,holm([r['permutation_two_sided_p'] for r in contrasts])):row.update(holm_adjusted_p=adjusted,holm_reject_0_05=adjusted<=.05)
 if not (base/'paired_statistics.json').exists():immutable(base/'paired_statistics.json',dict(version=VERSION,manifest_id=manifest['manifest_id'],contrasts=contrasts,definitions=SOURCES))
 return dict(status='completed_geometry_and_paired_statistics',manifest_id=manifest['manifest_id'],unique_graphs=len(geometries),contrasts=contrasts,human_audited=False,raw_text_exported_outside_primary=False)

def bounded_citations(rows):
 return [dict(evidence_id=c.get('evidence_id'),quote=str(c.get('quote',''))[:600],quote_truncated=len(str(c.get('quote','')))>600) for c in rows[:5]]
def blind_qa(question,rows,number):
 names=sorted(rows);random.Random(SEED+number).shuffle(names);labels=dict(zip('ABC',names))
 candidates=[]
 for label,name in labels.items():
  row=rows[name];candidate=row.get('candidate',dict(answer='[原运行：当前来源不可答，无生成答案]',citations=[]))
  contexts=[]
  for c in row.get('context',[])[:5]:
   contexts.append(dict(evidence_ids=c.get('evidence_ids',[]),text=str(c.get('text',''))[:1200],text_truncated=len(str(c.get('text','')))>1200))
  candidates.append(dict(label=label,answer=candidate['answer'],citations=bounded_citations(candidate['citations']),context=contexts))
 return dict(unit_id='Q%03d'%number,type=question['type'],question=question['question'],machine_reference_answer=question['answer'],reference_citations=bounded_citations(question['citations']),candidates=candidates),dict(question_id=question['question_id'],labels=labels)
def audit(base):
 offline();base=Path(base);manifest=load_manifest(base);folder=base/'human_audit'
 if folder.exists():raise FileExistsError('Preserve original blinded package and blank forms')
 folder.mkdir(mode=0o700);keyfolder=folder/'coordinator_only';keyfolder.mkdir(mode=0o700)
 pack=[];key=[];forms=[];selection=[];number=0;edge_number=0
 for course,job_names in manifest['audit_courses'].items():
  jobs={name:manifest['jobs'][name] for name in job_names};first=next(iter(jobs.values()));reference=read(first['reference']['path']);sample=stratified(reference['questions'],'type',12)
  by_name={name:{r['question_id']:r for r in [read(f['path']) for f in job['question_files']]} for name,job in jobs.items()}
  selection.append(dict(course=course,reference_id=first['reference_id'],question_type_available=dict(collections.Counter(q['type'] for q in reference['questions'])),question_type_selected=dict(collections.Counter(q['type'] for q in sample))))
  for q in sample:
   number+=1;unit,mapping=blind_qa(q,{name:rows[q['question_id']] for name,rows in by_name.items()},number);unit['course']=course;pack.append(unit);key.append(dict(unit_id=unit['unit_id'],course=course,**mapping))
   for candidate in unit['candidates']:forms.append(dict(unit_id=unit['unit_id'],candidate=candidate['label'],task='QA',correctness='',evidence_support='',conditions_completeness='',notes=''))
  graph_jobs=[name for name in job_names if name.endswith('_F') or name.endswith('_G0')];edge_sample=[]
  for name in graph_jobs:
   entities,relations,_=semantic(read(jobs[name]['graph']['path']));names={e['entity_id']:e.get('canonical_name',e['entity_id']) for e in entities}
   unique={digest([r['head_entity_id'],r['relation'],r['tail_entity_id'],r.get('conditions','')]):r for r in relations if r['head_entity_id']!=r['tail_entity_id']}
   selected=stratified(list(unique.values()),'relation',6)
   selection[-1].setdefault('edge_strata',[]).append(dict(job=name,available=len(unique),selected=len(selected),available_relation_types=dict(collections.Counter(r['relation'] for r in unique.values())),selected_relation_types=dict(collections.Counter(r['relation'] for r in selected))))
   for relation in selected:edge_sample.append((name,relation,names))
  random.Random(SEED).shuffle(edge_sample)
  for name,r,names in edge_sample:
   edge_number+=1;uid='E%03d'%edge_number;evidence=[]
   for item in r.get('evidence',[])[:5]:
    if isinstance(item,dict):evidence.append(dict(evidence_id=item.get('id',item.get('evidence_id')),text=str(item.get('text',item.get('quote','')))[:1200],text_truncated=len(str(item.get('text',item.get('quote',''))))>1200))
    elif isinstance(item,str):evidence.append(dict(text=item[:1200],text_truncated=len(item)>1200))
   unit=dict(unit_id=uid,course=course,head=names[r['head_entity_id']],relation=r['relation'],tail=names[r['tail_entity_id']],conditions=r.get('conditions',''),evidence=evidence)
   pack.append(unit);key.append(dict(unit_id=uid,course=course,job=name,relation_id=r.get('relation_id'),graph=jobs[name]['graph']))
   forms.append(dict(unit_id=uid,candidate='',task='edge',correctness='',evidence_support='',conditions_completeness='',notes=''))
 instructions='''独立盲审说明：两名领域评审分别评分，首次评分前不得查看 coordinator_only 或互相讨论。候选 A/B/C 在每题内以固定 seed42 独立打乱，标签不对应固定方法；不得推测方法后打分。机器参考答案尚未人工验证，原文证据优先。每维分别填 0/1/2，缺乏证据可在 notes 写“无法判断”，保留空分待仲裁，不将缺失记零。correctness：0 实质错误/无答案，1 部分正确或有显著遗漏，2 正确且无实质错误；evidence_support：0 无证据/不支持/引用错误，1 支持部分断言，2 原引文充分支持主要断言；conditions_completeness：0 遗漏或误述关键条件，1 部分条件或内容遗漏，2 关键条件与内容完整（真正无条件者可为2）。边题按头实体—关系—尾实体及成立条件逐项核查；仅先验而无证据时不得凭空确认教材支持。截断证据明确标注，必要时由协调员在 PRIMARY 原路径查看完整原件。完成后由第三方协调员比较两表，对任意维度不一致或空分仲裁并记录理由，原评分保持原样。报告实际人数、完成数量、逐维一致率和加权 Cohen kappa；不得把已制包称为已完成审查。12题/课程尽量六型各2题，12边/课程分别从G0和F各6边按关系类型轮转抽样；抽样不足或关系类型未覆盖如实披露。此包是有限抽查，不代表整图准确率。所有原文/题答/表格只保留PRIMARY，禁止复制到本地或外部服务。'''
 immutable(folder/'review_items.json',dict(instructions=instructions,items=pack,human_audited=False))
 immutable(keyfolder/'unblinding_key.json',dict(manifest_id=manifest['manifest_id'],mapping=key,selection=selection,never_distribute_to_raters=True))
 os.chmod(keyfolder/'unblinding_key.json',0o600)
 for name in ['rater_A.csv','rater_B.csv','adjudication.csv']:
  columns=list(forms[0])
  if name=='adjudication.csv':columns+=['rater_A_correctness','rater_B_correctness','rater_A_evidence_support','rater_B_evidence_support','rater_A_conditions_completeness','rater_B_conditions_completeness','adjudicator','reason']
  with (folder/name).open('x',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(forms)
 page='<html lang="zh"><meta charset="UTF-8"><title>双人独立盲审包（待评分）</title><style>body{max-width:1000px;margin:auto;font:16px/1.6 sans-serif}article{border-top:2px solid #456;padding:1em}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:26px}</style><h1>双人独立盲审包（待评分）</h1><p>'+html.escape(instructions)+'</p>'
 for unit in pack:page+='<article><h2>'+html.escape(unit['unit_id'])+'</h2><pre>'+html.escape(json.dumps(unit,ensure_ascii=False,indent=2))+'</pre></article>'
 page+='</html>'
 with (folder/'review.html').open('x',encoding='utf-8') as f:f.write(page)
 result=dict(status='prepared_blank_two_rater_pack_not_human_reviewed',manifest_id=manifest['manifest_id'],question_units=number,edge_units=edge_number,rater_scoring_rows=len(forms),selection=selection,files=[record(p) for p in sorted(folder.rglob('*')) if p.is_file()],human_audited=False,raw_text_exported_outside_primary=False)
 immutable(base/'audit_verified.json',result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('step',choices=['prepare','run','audit','status']);parser.add_argument('--output',required=True);parser.add_argument('--spec');args=parser.parse_args();base=Path(args.output)
 if args.step=='prepare':value=prepare(args.spec,base)
 elif args.step=='run':value=run(base);immutable(base/'verified.json',value)
 elif args.step=='audit':value=audit(base)
 else:value=dict(prepared=(base/'manifest.json').exists(),verified=read(base/'verified.json') if (base/'verified.json').exists() else None,audit=read(base/'audit_verified.json') if (base/'audit_verified.json').exists() else None)
 print(json.dumps(value,ensure_ascii=True))
if __name__=='__main__':main()
