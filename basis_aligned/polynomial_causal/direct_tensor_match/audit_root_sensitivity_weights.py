"""Sensitivity concentration and in-sample affine oracle, not a fitted candidate."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
 rows=json.loads((P/'ROOT_REMOVAL_GEOMETRY_V1.json').read_text())['scalar_rows'];out=[]
 for domain in ['fineweb','stdlib']:
  for newline in [True,False]:
   rr=[r for r in rows if r['domain']==domain and r['newline']==newline];t=np.array([r['native'] for r in rr]);h=np.array([r['candidate'] for r in rr]);w=np.array([r['sensitivity'] for r in rr]);w=w/w.mean();error=(h-t)**2*w;reference=t*t*w
   order=np.argsort(w)[::-1];top=order[:max(1,int(np.ceil(.1*len(w))))];scale=np.sqrt(np.mean(t*t));a=np.column_stack([h/scale,np.ones(len(h))]);coef=np.linalg.lstsq(a*np.sqrt(w)[:,None],t/scale*np.sqrt(w),rcond=None)[0];pred=a@coef*scale
   out.append(dict(domain=domain,condition='newline' if newline else 'other',n=len(w),weight_effective_n=float(w.sum()**2/(w*w).sum()),highest_sensitivity_decile_reference_share=float(reference[top].sum()/reference.sum()),highest_sensitivity_decile_error_share=float(error[top].sum()/error.sum()),weighted_error=float(np.sqrt(error.sum()/reference.sum())),in_sample_affine_error=float(np.sqrt(((pred-t)**2*w).sum()/reference.sum())),oracle_slope=float(coef[0]),oracle_intercept_in_target_rms=float(coef[1])))
 (P/'ROOT_SENSITIVITY_WEIGHT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
