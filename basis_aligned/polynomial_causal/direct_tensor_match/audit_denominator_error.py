"""Exact scalar-stage reweighting diagnosis on an existing native capture."""
from pathlib import Path
import hashlib,json,math
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 source=json.loads((P/'REMOVAL_STAGE_GEOMETRY_V1.json').read_text());path=P/'REMOVAL_STAGE_STATES_V1.pt'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==source['cache_sha256']
 cache=torch.load(path,weights_only=True);rows=[]
 for cell in source['summary']:
  if cell['kind']!='lean' or cell['errors'][-1]<=.1:continue
  aa=[];tt=[];dd=[]
  for d in cache['rows']:
   if d['domain']!=cell['domain']:continue
   mask=d['newline'] if cell['condition']=='newline' else ~d['newline']
   aa.append(d['candidates'][f"lean_{cell['seed']}"][mask].double());tt.append(d['exact'][mask].double());dd.append(d['denominator'][mask].double())
  a,t,den=[torch.cat(v) for v in [aa,tt,dd]];assert (den>0).all()
  e=(a-t).square();ref=t.square();we=e/den.square();wr=ref/den.square()
  raw=float((e.sum()/ref.sum()).sqrt());weighted=float((we.sum()/wr.sum()).sqrt())
  assert abs(raw-cell['errors'][0])<1e-6 and abs(weighted-cell['errors'][1])<1e-6
  k=math.ceil(len(den)*.1);idx=den.argsort()[:k];worst=we.topk(k).indices
  row=dict(seed=cell['seed'],domain=cell['domain'],alpha=cell['alpha'],condition=cell['condition'],states=len(den),raw_error=raw,weighted_error=weighted,lowest_denominator_decile_error_share=float(we[idx].sum()/we.sum()),lowest_denominator_decile_reference_share=float(wr[idx].sum()/wr.sum()),worst_weighted_error_decile_share=float(we[worst].sum()/we.sum()),reference_share_on_worst_error_decile=float(wr[worst].sum()/wr.sum()),denominator_quantiles=torch.quantile(den,torch.tensor([0.,.1,.5,.9,1.],dtype=den.dtype)).tolist())
  rows.append(row);print(json.dumps(row))
 (P/'DENOMINATOR_ERROR_V1.json').write_text(json.dumps(dict(rows=rows,prediction=all(r['lowest_denominator_decile_error_share']>.5 for r in rows),cache_sha256=source['cache_sha256'],scope='Opened native states; exact scalar metric reweighting, alpha repetitions not independent. No fitting or semantic/causal identification.'),indent=2)+'\n')
if __name__=='__main__':main()
