from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from ..manifest import read_document,validate,file_record
from ..contributions import write_immutable
from ..splits import freeze_split,project_graph,common_test_mask
from .protocol import freeze_phase,validate_phase,bound_input,runtime_for,bind_runtime

def parser():
    p=argparse.ArgumentParser(description='Frozen course evaluation thin adapters');sub=p.add_subparsers(dest='stage',required=True)
    phase=sub.add_parser('freeze-phase');phase.add_argument('--paths',required=True);phase.add_argument('--parameters',required=True);phase.add_argument('--output',required=True)
    gen=sub.add_parser('generate-reference');gen.add_argument('--construction',required=True);gen.add_argument('--phase',required=True);gen.add_argument('--course-id',required=True);gen.add_argument('--output',required=True);gen.add_argument('--smoke-chunks',type=int)
    ref=sub.add_parser('freeze-reference');ref.add_argument('--reference',required=True);ref.add_argument('--output',required=True);ref.add_argument('--split-output',required=True)
    project=sub.add_parser('project-kge')
    for flag in ('reference','split','graph','output'):project.add_argument('--'+flag,required=True)
    export=sub.add_parser('export-kge')
    for flag in ('projection','split','phase','output'):export.add_argument('--'+flag,required=True)
    mask=sub.add_parser('common-mask');mask.add_argument('--projection',action='append',required=True);mask.add_argument('--output',required=True)
    patch=sub.add_parser('patch-evaluators');patch.add_argument('--evaluator',required=True);patch.add_argument('--sweep',required=True);patch.add_argument('--output',required=True)
    protocol=sub.add_parser('prepare-kge')
    for flag in ('projection','split','phase','output'):protocol.add_argument('--'+flag,required=True)
    protocol.add_argument('--common-mask');protocol.add_argument('--smoke',action='store_true')
    index=sub.add_parser('build-index');index.add_argument('--construction',required=True);index.add_argument('--phase',required=True);index.add_argument('--course-id',required=True);index.add_argument('--output',required=True)
    rag=sub.add_parser('rag')
    for flag in ('construction','reference','index','graph','phase','output'):rag.add_argument('--'+flag,required=True)
    rag.add_argument('--active-book',action='append',required=True);rag.add_argument('--control',choices=['graph','vector_only','no_retrieval'],default='graph')
    topo=sub.add_parser('topology')
    for flag in ('graph','evaluator','phase','output'):topo.add_argument('--'+flag,required=True)
    topo.add_argument('--native-undirected',action='store_true')
    kge=sub.add_parser('kge')
    for flag in ('graph','protocol','patched-evaluator','phase','output'):kge.add_argument('--'+flag,required=True)
    kge.add_argument('--epochs',type=int,required=True);kge.add_argument('--device',default='cuda:0')
    return p

def main(argv=None):
    args=parser().parse_args(argv);result=None
    if args.stage=='freeze-phase':result=freeze_phase(read_document(args.paths),read_document(args.parameters),args.output)
    elif args.stage=='generate-reference':
        from .reference import generate
        result,root=generate(args.construction,args.phase,args.course_id,args.output,smoke=args.smoke_chunks)
        print(json.dumps(dict(status=result['status'],reference_id=result['reference_id'],root=str(root),questions=len(result['questions']),facts=len(result['facts']))));return
    elif args.stage=='freeze-reference':
        from .reference import freeze_reference
        result=freeze_reference(read_document(args.reference),args.output);write_immutable(args.split_output,freeze_split(result))
    elif args.stage=='project-kge':
        result=project_graph(read_document(args.graph),read_document(args.reference),read_document(args.split));write_immutable(args.output,result)
    elif args.stage=='export-kge':
        from .kge import export_kge_package
        result=export_kge_package(read_document(args.projection),read_document(args.split),args.phase,args.output)
    elif args.stage=='common-mask':
        projections=[read_document(p) for p in args.projection]
        result=dict(projection_ids=[p['projection_id'] for p in projections],split_id=projections[0]['split_id'],mask=common_test_mask(projections));write_immutable(args.output,result)
    elif args.stage=='patch-evaluators':
        from .kge import patch_evaluators
        result=patch_evaluators(args.evaluator,args.sweep,args.output)
    elif args.stage=='prepare-kge':
        from .kge import prepare_protocol
        phase=validate_phase(args.phase);phase['path']=str(Path(args.phase).resolve());mask=read_document(args.common_mask)['mask'] if args.common_mask else None
        result=prepare_protocol(read_document(args.projection),read_document(args.split),phase,args.output,smoke=args.smoke,common_mask=mask)
    elif args.stage=='build-index':
        from .rag import build_index
        construction=validate(args.construction);phase=validate_phase(args.phase);bound_input(phase,'construction',construction)
        if phase['parameters']!=dict(seed=42,embedding_model='BAAI/bge-m3',stage='build_index'):raise ValueError('Index parameters not frozen')
        course=next(c for c in construction['courses']['courses'] if c['course_id']==args.course_id)
        corpora=[read_document(construction['files']['sources'][b['book_id']]['path']) for b in course['books']]
        runtime=runtime_for(construction,Path(args.output)/'runtime')
        try:
            bind_runtime(phase,runtime);result=build_index(corpora,runtime,args.output)
        finally:runtime.db.close()
    elif args.stage=='rag':
        from .rag import evaluate
        construction=validate(args.construction);phase=validate_phase(args.phase);bound_input(phase,'construction',construction);runtime=runtime_for(construction,Path(args.output)/'runtime')
        try:result=evaluate(read_document(args.reference),read_document(args.index),read_document(args.graph),args.active_book,args.control,args.phase,args.output,runtime)
        finally:runtime.db.close()
    elif args.stage=='topology':
        from .topology import evaluate
        phase=validate_phase(args.phase);graph=read_document(args.graph);bound_input(phase,'graph',graph)
        if phase['files'].get('evaluator_original')!=file_record(args.evaluator):raise ValueError('Topology evaluator not frozen')
        result=evaluate(graph,args.evaluator,native_undirected=args.native_undirected)
        if phase['files'].get('src_utils')!=result['utils_dependency']:raise ValueError('Actual topology src.utils differs from frozen dependency')
        result['input_digest']=phase['phase_id'];write_immutable(args.output,result)
    elif args.stage=='kge':
        from .kge import run_external
        from .topology import load_evaluator
        phase=validate_phase(args.phase);graph=read_document(args.graph);bound_input(phase,'graph',graph)
        record=file_record(args.patched_evaluator)
        if phase['files'].get('evaluator_patched')!=record:raise ValueError('Patched evaluator not frozen')
        evaluator=load_evaluator(args.patched_evaluator,record['sha256'])
        ns=SimpleNamespace(seed=42,pykeen_model='TransE',pykeen_epochs=args.epochs,pykeen_device=args.device,external_protocol=args.protocol)
        inputs=SimpleNamespace(concept_kg=graph)
        result=evaluator.run_pykeen(inputs,ns,Path(args.output));write_immutable(Path(args.output)/'external_kge_result.json',result)
    print(json.dumps({k:v for k,v in (result or {}).items() if k not in {'entities','facts','questions','groups','files','rows','projection'}},ensure_ascii=False))

if __name__=='__main__':main()
