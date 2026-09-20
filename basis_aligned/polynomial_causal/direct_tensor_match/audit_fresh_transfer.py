"""Document-level uncertainty and position-stratified audit of frozen programs."""
import json
from pathlib import Path
import torch
from audit_centered_compact import evaluate
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);data=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];q=torch.load(P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt',weights_only=True)['programs']['centered'];h={k:v.double() for k,v in torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True)['programs'][8].items()};records=[];gen=torch.Generator().manual_seed(2036);indices=torch.randint(32,(2000,32),generator=gen)
 for panel in data:
  x=panel['rows'].double();y=panel['targets'].double();length=panel['context'];yp=evaluate(q,x);bank=((x@h['U'].flatten(0,1).T)*(x@h['V'].flatten(0,1).T)).reshape(len(x),4,4).sum(2);i,j=torch.triu_indices(4,4);yh=((bank[:,i]*bank[:,j])@h['Z'].T)@h['W'].T+h['constant'];den=y.square().sum(1).reshape(32,length);errors={n:(p-y).square().sum(1).reshape(32,length) for n,p in [('quadratic',yp),('quartic',yh)]};ds=den.sum(1);boot={n:(v.sum(1)[indices].sum(1)/ds[indices].sum(1)).sqrt() for n,v in errors.items()};rows=[]
  for begin in range(0,length,64):rows.append(dict(start_position=begin,end_position=begin+64,errors={n:float((v[:,begin:begin+64].sum()/den[:,begin:begin+64].sum()).sqrt()) for n,v in errors.items()}))
  records.append(dict(context=length,position_blocks=rows,document_bootstrap_intervals={n:torch.quantile(v,torch.tensor([.025,.975],dtype=torch.float64)).tolist() for n,v in boot.items()},paired_quadratic_minus_quartic_interval=torch.quantile(boot['quadratic']-boot['quartic'],torch.tensor([.025,.975],dtype=torch.float64)).tolist()))
 short=data[0]['rows'].reshape(32,64,1152).double();long=data[1]['rows'].reshape(32,256,1152)[:,:64].double();prefix=float((short-long).norm()/short.norm());out=dict(records=records,prefix_input_relative_disagreement=prefix,scope='Document-resampled descriptive intervals for frozen candidates,32documents only. Position blocks audit averages. Cached targets roundedFP32. Not broad OOD or semantic identification.');(P/'FRESH_TRANSFER_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
