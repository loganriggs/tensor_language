"""Weight-only complete joint-key product path bank, no retained-rank reduction.
Uniform32relative-position influence; four rank64 bands span all256key directions.
A trace/PSD/reconstruction<=1e-10. Bands are frozen probe spaces, not circuits.
"""
import json,time,hashlib
from pathlib import Path
import torch
from joint_qk_source_influence_v1 import influence
from joint_qk_source_ports_v1 import source_ports
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);tic=time.perf_counter();torch.set_default_dtype(torch.float64)
 out=P/'SCALAR_JOINT_KEY_BANK_V1_RESULT.json';assert not out.exists()
 cp=P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt';p=torch.load(cp,weights_only=True,map_location='cpu');banks=[];rows=[];errors=[]
 for h in range(2):
  q1,q2,k1,k2=[p[n][h].double() for n in ('q1','q2','k1','k2')]
  basis=torch.linalg.qr(torch.cat((k1,k2)).T,mode='reduced').Q
  errors.append(float((basis.T@basis-torch.eye(256)).norm()))
  matrices=[];totals=[]
  for pos in range(32):
   rotation=rotary(32,128).T@rotary(pos,128);ka,kb=rotation@k1@basis,rotation@k2@basis
   S=influence(q1,q2,ka,kb);total=source_ports(q1,q2,ka,kb,torch.eye(256))['total']
   errors.append(abs(float(S.trace()/total)-1));matrices.append(S);totals.append(float(total))
  S=torch.stack(matrices).mean(0);ev,U=torch.linalg.eigh(S);order=torch.arange(255,-1,-1);ev=ev[order];U=U[:,order];bank=(basis@U).reshape(1152,4,64).permute(1,0,2).contiguous();banks.append(bank)
  recon=torch.cat([b for b in bank],1);errors.extend([float((k1-k1@recon@recon.T).norm()/k1.norm()),float((k2-k2@recon@recon.T).norm()/k2.norm()),max(0.,float(-ev[-1]/ev.sum()))])
  rows.append(dict(head=['8.2','9.8'][h],band_influence_fractions=[float(ev[j:j+64].sum()/ev.sum()) for j in range(0,256,64)],relative_boundary_gaps=[float((ev[j-1]-ev[j])/ev[0]) for j in (64,128,192)],position_top64_overlap=[float((torch.linalg.eigh(matrices[pos])[1][:,-64:].T@U[:,:64]).square().sum()/64) for pos in (0,15,31)],mean_coefficient_norm_squared=sum(totals)/32))
 result=dict(pred_a=max(errors)<=1e-10,max_instrument_error=max(errors),rows=rows,seconds=time.perf_counter()-tic,source_sha=hashlib.sha256(cp.read_bytes()).hexdigest(),scope='Weights-only complete joint-key source basis for heads8.2/9.8; ten unordered source-band pair paths per scalar value sector. Native full denominators required. No low-rank approximation, fitting text, task-specific input identification, or causal result. Boundary gaps/position overlaps are descriptive, not stability certification.')
 torch.save(dict(bands=torch.stack(banks)),P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
if __name__=='__main__':main()
