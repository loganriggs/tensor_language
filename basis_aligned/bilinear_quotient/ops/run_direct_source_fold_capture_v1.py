#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_rows pred_b_source pred_c_product
"""Capture existing calibration input ports for original-weight source folding.
pred_a_rows normalized n,m replay<1e-5; pred_b_source exact Q source reads<1e-4;
pred_c_product reconstructed downstream scalar product replay<1e-4.
32original calibration captures,64tokens each; no new discovery/validation panel.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=32,context=64,fit=False,calibration_only=True)));return
 import torch
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=P/'MIDPOINT_SOURCE_FOLD_CAPTURE_V1.json';assert not out.exists();start=time.perf_counter()
 previous=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);e={k:v.cuda().double() for k,v in torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True).items()};fold=torch.load(P/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.pt',weights_only=True);Qa=fold['a']['matrix'].cuda();Qb=fold['b']['matrix'].cuda();a=e['A'][:,0];b=e['B'][:,0]
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];records=[];checks=[];target=[];fields={k:[] for k in ['z','recipient_scale','h_reader','source_reads']};num=den=0.
 for doc,row in enumerate(previous['tokens']):
  c=capture(model,row[None,:64].cuda());z=c['x16'].flatten(0,1).double();h=c['h17'].flatten(0,1).double();source=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).flatten(0,1).double();scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-source/2)/scale;m=source/scale
  for name,v in [('n',n),('m',m)]:checks.append(float((v-previous[name][doc].cuda().double()).norm()/v.norm()))
  exact=torch.stack([source@a,source@b],1);quadratic=torch.stack([((z@Qa)*z).sum(1),((z@Qb)*z).sum(1)],1);records.append(float((quadratic-exact).norm()/exact.norm()))
  base=(n@a-e['mean_n']@a)*(m@b-e['mean_m']@b)
  pred=((h@a-.5*quadratic[:,0])/scale[:,0]-e['mean_n']@a)*(quadratic[:,1]/scale[:,0]-e['mean_m']@b)
  num+=float((pred-base).square().sum());den+=float(base.square().sum())
  for key,value in [('z',z.float()),('recipient_scale',scale),('h_reader',h@a),('source_reads',exact)]:fields[key].append(value.cpu())
 torch.save({**{k:torch.stack(v) for k,v in fields.items()},'tokens':previous['tokens']},P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt')
 result=dict(predictions=dict(pred_a_rows=max(checks)<1e-5,pred_b_source=max(records)<1e-4,pred_c_product=(num/den)**.5<1e-4),normalized_row_replay=max(checks),source_form_replay=max(records),product_replay=(num/den)**.5,seconds=time.perf_counter()-start,scope='Existing32calibration documents, native normalized MLP16 input z plus original final-recipient scale and h-reader port. Input states remain native; exact source forms verified against native float32 operations before fitting any approximation.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
if __name__=='__main__':main()
