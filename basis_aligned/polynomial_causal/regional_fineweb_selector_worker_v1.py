"""Isolated selector worker: validate/flush outputs before bypassing broken C shutdown.

Known FineWeb streaming finalization aborts after valid outputs even with explicit
iterator cleanup. This is a process-exit workaround, not a fix to that library bug.
Never import run in a long-lived agent process: successful run deliberately exits.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
import regional_fineweb_selector_v3 as selector

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stem',required=True);parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
 selector.P=args.output_dir;selector.STEM=args.stem;selector.main()
 row_path=selector.P/(args.stem+'_ROWS.json');control_path=selector.P/(args.stem+'_CPU_CONTROL.json')
 rows=json.loads(row_path.read_text());control=json.loads(control_path.read_text())
 assert control['pred_a'] and control['selection_model_calls']==0
 assert len(rows['contexts'])==20 and len(rows['rows'])==240 and len({tuple(r['ids']) for r in rows['rows']})==40
 assert hashlib.sha256(row_path.read_bytes()).hexdigest()==control['row_sha256']
 # Path.write_text closed both files; fsync before intentionally skipping finalizers.
 for path in [row_path,control_path]:
  with path.open('rb') as f:os.fsync(f.fileno())
 print('Validated and flushed selector artifacts; isolated-worker exit workaround.',flush=True)
 sys.stdout.flush();sys.stderr.flush();os._exit(0)
if __name__=='__main__':main()
