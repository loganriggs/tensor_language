#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences; exact prior-parent audit,no optimization,300sec.
"""pred_a newcapture/metric/starenergy replay<=1e-8;
pred_b oldcenteredparent with optimal newmetric partners retains>=90%new capture;
pred_c its fullstar functioncos>=.9 withnew star. No semantic alias guarantee.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from fullu_input_blocks_v1 import sandwich
from shared_input_factor_v1 import value_gradient,native_partner
STEM='COMPOSED_PARENT_REUSE_V1'

def inner(a,b,c,d):
    return .5*((a@c)*(b*d).sum()+(b@c)@(d@a))

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,old_parents=2,fits=0)));return
    out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(300);start=time.perf_counter()
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    prior=torch.load('/dev/shm/bilin18_shared_input_factor_native_v1.pt',weights_only=True)
    current=torch.load(P/'COMPOSED_SHARED_PARENT_V1_PROGRAM.pt',weights_only=True)
    result0=json.loads((P/'COMPOSED_SHARED_PARENT_V1_RESULT.json').read_text())
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=current['producer_scale'];h=scale**2*d0@atom_gram(l0,r0)@d0.T;he,hv=torch.linalg.eigh(h);hs=(hv*he.sqrt())@hv.T
    u=sd['lm_head.weight'].double().cuda();u-=u.mean(0);root=torch.linalg.cholesky(u.T@u);g=d1.T@(u.T@u)@d1
    l,r=l1@hs,r1@hs;k=sandwich(l,r,g,torch.eye(1152,device='cuda'));total=float(k.trace())
    a=current['readers'][current['best']].cuda();b=root.T@current['physical_partner'].cuda()@hs
    energy=inner(a,b,a,b);value,_=value_gradient(a,l,r,g,k)
    errors=[abs(float(energy/total)-result0['best_capture']),abs(float(energy/value)-1)]
    reports=[]
    for name,old in prior.items():
        raw=old['readers'][old['best']].cuda();c=hs@raw;c/=c.norm()
        partner=root.T@native_partner(c,l,r,d1);e=inner(c,partner,c,partner)
        direct,_=value_gradient(c,l,r,g,k);errors.append(abs(float(e/direct)-1))
        original_parent=hs@raw;original_partner=root.T@old['partner'].cuda()@hs
        original_energy=inner(original_parent,original_partner,original_parent,original_partner)
        reports.append(dict(prior=name,parent_cosine=float(a@c),matched_partner_capture=float(e/total),capture_fraction_of_new=float(e/energy),matched_star_cosine=float(inner(a,b,c,partner)/(energy*e).sqrt()),original_folded_star_cosine=float(inner(a,b,original_parent,original_partner)/(energy*original_energy).sqrt()),original_folded_star_energy_fraction=float(original_energy/total)))
    center=next(x for x in reports if x['prior']=='centered')
    result={'pred_a':max(errors)<=1e-8,'pred_b':center['capture_fraction_of_new']>=.9,'pred_c':center['matched_star_cosine']>=.9}
    result.update(reports=reports,identity_errors=errors,execution_seconds=time.perf_counter()-start,source_shas=binding,scope='Same producer-path weight functions; partner matching and original star distinguished. No semantic circuit or OOD identification.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
