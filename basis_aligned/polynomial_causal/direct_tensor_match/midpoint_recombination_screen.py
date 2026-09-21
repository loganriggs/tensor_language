"""CPU exact-weight test of source-only recombination in normalized input slots."""
from pathlib import Path
import json,time,torch

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
 p=Path(__file__).resolve().parent;out=p/'MIDPOINT_RECOMBINATION_V2.json';assert not out.exists()
 ck=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
 state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
 print('checkpoint loaded',flush=True)
 ru=torch.linalg.qr(state['lm_head.weight'].double(),mode='r').R
 L=state['transformer.h.17.mlp.Left.weight'].double();R=state['transformer.h.17.mlp.Right.weight'].double();C=ru@state['transformer.h.17.mlp.Down.weight'].double()
 rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
 n,m,y=[rows[k].flatten(0,1).double() for k in ['n','m','y']]
 S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
 nl,nr,ml,mr=n@L.T,n@R.T,m@L.T,m@R.T
 teacher=lambda idx:((nl*mr[idx]+nr*ml[idx])@C.T)
 base=teacher(torch.arange(len(n)));replay=float((base-y).norm()/y.norm());print('native replay',replay,flush=True);assert replay<1e-5
 original=torch.load(p/'MIDPOINT_GRAPH_FROZEN_V1.pt',weights_only=True)['rank8']
 pruned=torch.load(p/'MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.pt',weights_only=True)['products512']
 models={}
 for name,e in [('original1024',original),('pruned512',pruned)]:
  e={k:v.double() for k,v in e.items()}
  A=e['Pn']@e['Tn'] if 'Pn' in e else e['A'];B=e['Pm']@e['Tm'] if 'Pm' in e else e['B'];W=e['output_basis']@e['output_core'] if 'output_basis' in e else e['reduced_writers']
  models[name]=(n@A,m@B,W,e['full_mean']-e['product_mean']@W.T)
 records=[];controls=[]
 for shift in [0,1,7,16]:
  idx=torch.arange(len(n)).reshape(rows['n'].shape[:2]).roll(shift,0).flatten();truth=teacher(idx);delta=truth-base
  average_context=((nl.mean(0)*(mr[idx]-mr)+nr.mean(0)*(ml[idx]-ml))@C.T)
  interaction=delta-average_context
  if shift: controls.append(dict(shift_documents=shift,average_context_effect_error=float((interaction@S).norm()/(delta@S).norm())))
  for name,(aa,bb,W,bias) in models.items():
   pred=(aa*bb[idx])@W.T+bias;baseline=(aa*bb)@W.T+bias
   centered=truth-y.mean(0)
   pred_interaction=((aa-aa.mean(0))*(bb[idx]-bb))@W.T
   if shift: controls.append(dict(shift_documents=shift,program=name,context_interaction_relative_error=float(((pred_interaction-interaction)@S).norm()/(interaction@S).norm())))
   records.append(dict(shift_documents=shift,program=name,replacement_error=float(((pred-truth)@S).norm()/(centered@S).norm()),source_swap_effect_error=None if shift==0 else float((((pred-baseline)-delta)@S).norm()/(delta@S).norm()),effect_energy_over_paired_variation=None if shift==0 else float((delta@S).norm()/((y-y.mean(0))@S).norm())))
 result=dict(replay=replay,records=records,context_controls=controls,seconds=time.perf_counter()-start,scope='Exact original folded weights on cached calibration normalized n,m. Cross-document same-position cyclic source-only recombination, fixed n. No new marginal inputs, no normalization recomputation, no native final logits. Program constants frozen; delta cancels constants. New combinations are a reuse screen, not fresh-text OOD or semantic intervention identification.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
