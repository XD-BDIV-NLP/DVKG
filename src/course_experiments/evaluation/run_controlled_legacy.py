"""旧313/123全40行的薄预算包装；原共同检索/回答/裁判入口保持原样。"""
from __future__ import annotations
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3,sys
from unittest.mock import patch
from ..manifest import file_record,read_document
from . import run_baseline_rag_bridge as original
from .baseline_rag_bridge import METHODS,POLICY
from .blind64_pipeline import validate_provider,budget_snapshot,ORIGINAL_STARTED
from .native_provider import AuditedNativeProvider
from .run_independent_holdout import LocalHTTPBudget,HoldoutBudgetExceeded


def legacy_count(database,at_event=None):
    with closing(sqlite3.connect('file:'+str(Path(database))+'?mode=ro',uri=True,timeout=30)) as db:
        query="SELECT COUNT(*) FROM events WHERE status='request_started' AND substr(stage,1,20)='controlled_baseline/' AND (instr(stage,'/regression/')>0 OR instr(stage,'/independent/')>0)"
        return db.execute(query+(' AND id<=?' if at_event is not None else ''),[] if at_event is None else [at_event]).fetchone()[0]


def validate_allocation(value,database):
    if (value.get('version')!='controlled_old40_allocation_v1' or value.get('additional_actual_HTTP_cap')!=8000 or value.get('native_group_cap')!=45000
            or value.get('original_started_unix')!=ORIGINAL_STARTED or type(value.get('anchor_event_id')) is not int or value['anchor_event_id']<0
            or type(value.get('legacy_actual_at_anchor')) is not int or value['legacy_actual_at_anchor']<0):raise ValueError('Explicit original-start controlled old40 allowance required')
    with closing(sqlite3.connect('file:'+str(Path(database))+'?mode=ro',uri=True,timeout=30)) as db:
        last=db.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
    if last<value['anchor_event_id'] or legacy_count(database,value['anchor_event_id'])!=value['legacy_actual_at_anchor']:raise ValueError('Controlled original cumulative accounting anchor differs')
    return value['legacy_actual_at_anchor']+8000


def obligations():
    return [dict(method=m,cohort=c,split=s) for m in METHODS for c in POLICY['cohorts'] for s in ['regression','independent']]


class LegacyBudget(LocalHTTPBudget):
    def __init__(self,provider,run_root,cap):
        super().__init__(provider,Path(run_root)/'controlled_old40_allowance','controlled_old40_v1',cap=cap)
        self.prefix='controlled_baseline/'
    def count(self):return legacy_count(self.provider.database)


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--allocation-sha',required=True);args,remaining=p.parse_known_args()
    record=file_record(args.allocation)
    if record['sha256']!=args.allocation_sha:raise ValueError('Controlled old40 approved allocation changed')
    allocation=read_document(args.allocation)
    root=Path(remaining[remaining.index('--run-root')+1]);budget_snapshot(root/'formal_provider')
    validate_allocation(allocation,root/'formal_provider/audit.sqlite')
    class Provider(AuditedNativeProvider):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.legacy_budget=None
            try:
                validate_provider(self);cap=validate_allocation(allocation,self.database)
                self.legacy_budget=LegacyBudget(self,self.root.parent,cap);self.legacy_budget.__enter__()
            except Exception:
                super().close();raise
        def close(self):
            if self.legacy_budget is not None:self.legacy_budget.__exit__(None,None,None);self.legacy_budget=None
            super().close()
    with patch.object(original,'AuditedNativeProvider',Provider),patch.object(sys,'argv',[sys.argv[0],*remaining]):return original.main()


if __name__=='__main__':raise SystemExit(main())
