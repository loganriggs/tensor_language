"""Same contraction/gradient in full vs exact two-head supports; CPU benchmark.
A value/gradient<=1e-8; B median five-trial speedup>=1.5; C target<=1e-10.
"""
from pathlib import Path
import json,time,torch,statistics
from folded_producer_cubic_weights_v1 import weights
from folded_normalized_router_v1 import rotary
from cubic_secant_block_v1 import capture
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(413);out=P/'CUBIC_SUPPORT_KERNEL_V1_BENCHMARK.json';assert not out.exists()
 binding=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(qa,ka,qb,kb,v,o),_,_=weights(sd,C,'cpu');ix=[2,17];qa,ka,qb,kb,v= [t[ix] for t in (qa,ka,qb,kb,v)];o=o[:,ix]
 ka=torch.cat((ka,torch.zeros_like(ka)),-1);kb=torch.cat((kb,torch.zeros_like(kb)),-1)
 read=torch.einsum('ohk,hks->hos',o,v);source=torch.cat((ka.flatten(0,1),kb.flatten(0,1),read.flatten(0,1)));bs=torch.linalg.qr(source.T,mode='reduced').Q;bq=torch.linalg.qr(torch.cat((qa.flatten(0,1),qb.flatten(0,1))).T,mode='reduced').Q
 r=rotary(8,128).T@rotary(7,128);ka=torch.einsum('ab,hbs->has',r,ka);kb=torch.einsum('ab,hbs->has',r,kb);full=(qa,ka,qb,kb,v,o);small=(qa@bq,ka@bs,qb@bq,kb@bs,v@bs,o)
 theta=torch.randn(16,3,bs.shape[1]);theta=theta/theta.norm(dim=-1,keepdim=True);mix=torch.eye(16)
 def evaluate(mode):
  z=theta.clone().requires_grad_();tic=time.perf_counter();value=capture(z@bs.T if mode=='full' else z,mix,full if mode=='full' else small);gradient=torch.autograd.grad(value,z)[0];return value.detach(),gradient,time.perf_counter()-tic
 # Warm both paths, then alternate their order to reduce ordering bias.
 evaluate('full');evaluate('small');times={'full':[],'small':[]};latest={}
 for trial in range(5):
  for mode in (['full','small'] if trial%2==0 else ['small','full']):
   value,g,seconds=evaluate(mode);times[mode].append(seconds);latest[mode]=(value,g)
 vf,gf=latest['full'];vs,gs=latest['small'];ve=float(abs(vf-vs)/vf.abs().clamp_min(1e-30));ge=float((gf-gs).norm()/gf.norm().clamp_min(1e-30));replay=float((read-(read@bs)@bs.T).norm()/read.norm());ratio=statistics.median(times['full'])/statistics.median(times['small'])
 result=dict(pred_a=float(vf.abs())>1e-14 and float(gf.norm())>1e-12 and max(ve,ge)<=1e-8,pred_b=ratio>=1.5,pred_c=replay<=1e-10,capture=float(vf),gradient_norm=float(gf.norm()),value_relative_error=ve,gradient_relative_error=ge,folded_value_reconstruction_error=replay,source_coordinates=bs.shape[1],query_coordinates=bq.shape[1],seconds=times,median_speedup=ratio,scope='CPU2threads five alternating gradient evaluations, same seeded rank16 candidate in exact union support for heads8.2/9.8. Fullpath includes lifting latent readers. Reducedpath excludes one-time basis build; no GPU timing, fit convergence or circuit effect evidence.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
