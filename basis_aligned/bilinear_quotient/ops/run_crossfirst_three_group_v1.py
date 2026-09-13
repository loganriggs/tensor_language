#!/usr/bin/env python3
# BQGATE:0bodyforwards;480finalreadouts;160cachedprefixes;60seconds.
"""pred_a cachedbase/fullport readouts replay<=1e-4 relative eachpanel.
pred_b three-group effect error<=.10 relative fullport effect eachregionalcell.
pred_c effect signs match>=23/24 in eachregionalcell.
Null: omitted score-score/value-defect terms matter despite small aggregated write norms.
Price0transformerforwards,480finalreadouts,60seconds; onlynativeunembedding GPUloaded.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
STEM='CROSSFIRST_THREE_GROUP_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('0transformerforwards480finalreadouts;cachedthree-group approximation');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(60);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
 checkpoint=next(k for k in files if k.endswith('pytorch_model.bin'));sd=torch.load(checkpoint,weights_only=True,mmap=True);U=sd['lm_head.weight'].cuda();states=torch.load(P/(STEM+'_INPUT.pt'),weights_only=True)['states'].cuda();measure=torch.zeros(160,3,2,dtype=torch.float64);count=0
 for i,row in enumerate(rows):
  for arm in range(3):
   logits=(30*torch.tanh(F.linear(F.rms_norm(states[i:i+1,arm],(1152,)),U)/30))[0];count+=1
   if i<96:measure[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measure[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
   else:measure[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measure[i,arm,1]=(logits[198]-logits[11]).cpu()
 assert count==480
 prior=torch.load(P/'CROSSFIRST_ATTENTION17_PORTS_V1_ARTIFACT.pt',weights_only=True)['readouts'][:,[0,2]];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[rel(measure[lo:hi,:2],prior[lo:hi]) for lo,hi in [(0,96),(96,160)]];cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=measure[lo:hi,:,0];ref=z[:,1]-z[:,0];pred=z[:,2]-z[:,0];cells.append(dict(cell=label,effect_error=rel(pred,ref),same_sign=int(((pred*ref)>0).sum()),reference_zeros=int((ref==0).sum()),maxabs_prediction_error=float((pred-ref).abs().max()),meanabs_reference=float(ref.abs().mean()),meanabs_prediction=float(pred.abs().mean())))
 result={'pred_a':max(replay)<=1e-4,'pred_b':all(c['effect_error']<=.1 for c in cells[:4]),'pred_c':all(c['same_sign']>=23 for c in cells[:4]),'anchor_replay':replay,'cells':cells,'body_forwards':0,'final_readouts':count,'seconds':time.perf_counter()-tic,'scope':'Fixedthree-contraction approximation selectedfromknown-panel geometric audit. Nativecachedreadout screen, no newholdout/OOD or autonomousstates. FineWebdiagnostic only.'}
 torch.save(dict(measures=measure),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
