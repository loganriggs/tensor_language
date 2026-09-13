"""Read the published control strength before registering a new newline screen.

This helper does not choose or relax a scientific threshold. Passing a proposed
minimum checks whether the unchanged historical fixture could meet it at all.
"""
from pathlib import Path
import argparse,json,torch
P=Path(__file__).resolve().parent

def summarize():
    old=torch.load(P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ARTIFACT.pt',weights_only=True)
    rows=json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ROWS.json').read_text())['rows']
    result=[]
    for family in (0,1):
        indices=[i for i,r in enumerate(rows) if r['pool']=='fineweb' and r['family']==family]
        z=old['ce'][indices];m=old['margins'][indices,0]
        result.append(dict(family=family,count=len(indices),native_positive=int((m>0).sum()),native_mean_margin=float(m.mean()),zero_meanabs_CE=float((z[:,4]-z[:,0]).abs().mean()),mean_replacement_meanabs_CE=float((z[:,5]-z[:,0]).abs().mean())))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--minimum-zero-control',type=float);args=parser.parse_args();torch.set_num_threads(2);rows=summarize();print(json.dumps(rows,indent=2))
    if args.minimum_zero_control is not None and any(r['zero_meanabs_CE']<args.minimum_zero_control for r in rows):
        parser.exit(2,'Proposed control-strength bar exceeds published fixture strength; reconsider design before registration, not after results.\n')
