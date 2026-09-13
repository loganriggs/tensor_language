from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 a=torch.load(P/'CROSSFIRST_LAST_ATTENTION_TERMS_V1_ARTIFACT.pt',weights_only=True)['readouts'];rows=[]
 for k in range(8):
  lo,hi=(24*k,24*(k+1)) if k<4 else (96+16*(k-4),112+16*(k-4));z=a[lo:hi,:,0];e=z-z[:,0:1];ref=e[:,1];records={}
  for j,name in [(2,'formula'),(3,'direct'),(4,'direct_cross'),(5,'direct_cross_norm')]:
   diff=e[:,j]-ref;records[name]=dict(same_effect_sign=int(((e[:,j]*ref)>0).sum()),reference_exact_zero=int((ref==0).sum()),meanabs_prediction_error=float(diff.abs().mean()),maxabs_prediction_error=float(diff.abs().max()),meanabs_effect=float(e[:,j].abs().mean()))
  rows.append(dict(group=k,count=hi-lo,meanabs_reference_effect=float(ref.abs().mean()),records=records))
 r=dict(rows=rows,scope='Per-prefix outcome and absolute error audit on existing diagnostic rows. Zero signs explicitly reported. Does not supply new holdout evidence; relative and absolute fidelity remain distinct.')
 (P/'CROSSFIRST_LAST_ATTENTION_TERMS_V1_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
