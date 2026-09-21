"""Actual pairwise reuse executor: derivatives and floating-point stability.

No new fitting, native forward, or claim of fresh validation. Test the saved
conditional interface on opened rows. Compiler failures remain outside the
exported set and are explicitly retained from experiment metadata.
"""
from pathlib import Path
import sys,json,torch
from pairwise_reader_graph import expand,source_reads
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2)

def cast(x):
 if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
 return x.float() if isinstance(x,torch.Tensor) and x.is_floating_point() else x

def components(z,h,p):
 reads=source_reads(z,p);s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
 return torch.stack([((h@p['pairs'][str(j)]['h_reader']-.5*reads[:,2*j])/s-p['pairs'][str(j)]['alpha'])*(reads[:,2*j+1]/s-p['pairs'][str(j)]['beta']) for j in range(3)],1)

def main(prefix):
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/f'{prefix}_NATIVE_PROGRAMS_V1.pt',weights_only=True);meta=json.loads((P/f'{prefix}_NATIVE_V1.json').read_text());ids=data['indices'];z=data['z'][ids];h=data['h'][ids];rows=[]
 for key,p in programs.items():
  with torch.no_grad():
   values=components(z,h,p);reads=source_reads(z,p);p32=cast(p);v32=components(z.float(),h.float(),p32).double();r32=source_reads(z.float(),p32).double()
   scalar=((v32-values).norm(dim=0)/(values-values.mean(0)).norm(dim=0)).tolist();source=((r32-reads).norm(dim=0)/(reads-reads.mean(0)).norm(dim=0)).tolist()
  zg=z[:16].clone().requires_grad_();hg=h[:16];phi=components(zg,hg,p);bundle=expand(p);Q=torch.cat([decode(bundle[str(j)]) for j in range(3)])
  lin=torch.stack([bundle[str(j)][k+'_linear'] for j in range(3) for k in ('a','b')],1);bias=torch.stack([bundle[str(j)][k+'_bias'] for j in range(3) for k in ('a','b')])
  with torch.no_grad():
   q=torch.einsum('ni,oij,nj->no',zg,Q,zg)+zg@lin+bias;grad=2*torch.einsum('oij,nj->noi',Q,zg)+lin.T[None];s=(hg.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
  gradient_replays=[]
  for j in range(3):
   actual=torch.autograd.grad(phi[:,j].sum(),zg,retain_graph=j<2)[0]
   with torch.no_grad():
    block=bundle[str(j)];a=(hg@block['h_reader']-.5*q[:,2*j])/s-block['alpha'];b=q[:,2*j+1]/s-block['beta'];expected=-.5*(b/s)[:,None]*grad[:,2*j]+(a/s)[:,None]*grad[:,2*j+1]
    gradient_replays.append(float((actual-expected).norm()/expected.norm()))
  rows.append(dict(key=key,fp32_component_drift=scalar,fp32_source_drift=source,actual_executor_gradient_replay=gradient_replays,precision_pass=max(scalar+source)<1e-4,gradient_pass=max(gradient_replays)<1e-8))
 failures=[dict(key=r['key'],failure=r.get('compiler_failure')) for r in meta['records'] if not r.get('instrument',False)]
 out=dict(records=rows,unexported_instrument_failures=failures,scope='448 previously opened states for FP32 stability, first16 for actual graph autodiff versus dense analytic derivatives holding later h fixed. Drift normalized by FP64 variation. Does not validate h response, OOD or semantic identity.')
 (P/f'{prefix}_EXECUTION_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'ORTHOGONAL_PRIVATE')
