#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;128cachedendpoints,oneindependentSVD;300sec.
"""pred_a exact reference/replay; pred_b function/output/partner cos>=.95;
pred_c swaps<=.1/sign>=.9/live>=4/removal<=.02; pred_d writes<=.05.
Null: stable parent does not identify the entire branch. No fitting to text.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from shared_input_factor_v1 import native_partner
from quartic_frozen_native_score_v2 import score
STEM='MATCHED_PARTNER_CROSS_START_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=128,independent_arm=1,node=251,mode=8)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not artifact.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{n}.weight'].double().cuda() for layer in (16,17) for n in ('Left','Right','Down')]
    scale=float(sd['transformer.h.17.lambdas'][0]);g=atom_gram(l0,r0);h=scale**2*d0@g@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0;hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T
    u=sd['lm_head.weight'].double().cuda();uc=u-u.mean(0);root=torch.linalg.cholesky(uc.T@uc).T;del u,uc
    frames=torch.load(P/'AMORTIZED_SPARSE_FRAME_NATIVE_V1_FRAMES.pt',weights_only=True)
    a=frames['frames'][0][:,0].cuda();c=frames['frames'][1][:,251].cuda()
    prior=torch.load(P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt',weights_only=True);b=scale*hi@d0@(g@prior['compiled_producer_readers'][:,8].cuda());w=prior['output_writers'][:,8].cuda()
    partner=native_partner(c,l1@hs,r1@hs,d1);perp=partner-(partner@c)[:,None]*c[None,:]
    left,singular,right=torch.linalg.svd(root@perp,full_matrices=False);d=right[7];v=torch.linalg.solve_triangular(root,(left[:,7]*singular[7])[:,None],upper=True)[:,0]
    def inner(a,b,w,c,d,v):return .5*((a@c)*(b@d)+(a@d)*(b@c))*((root@w)@(root@v))
    cosine=float(inner(a,b,w,c,d,v)/(inner(a,b,w,a,b,w)*inner(c,d,v,c,d,v)).sqrt())
    output_cos=float(abs((root@w)@(root@v))/((root@w).norm()*(root@v).norm()));partner_cos=float(abs(b@d)/(b.norm()*d.norm()))
    readers=scale*d0.T@hi@torch.stack([c,d],1)
    cache=torch.load(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_PORTS.pt',weights_only=True);ports=cache['ports'];x=ports['input16'].double().cuda()
    hidden=(x@l0.T)*(x@r0.T);den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    oldvalues=hidden@prior['compiled_producer_readers'][:,[0,8]].cuda();reference=oldvalues.prod(-1)[:,None]*w[None,:]/den[:,None]
    values=hidden@readers;candidate=values.prod(-1)[:,None]*v[None,:]/den[:,None]
    z=scale*hidden@d0.T@hi;reconstructed=(z@a)*(z@b);coefficient_reference=oldvalues.prod(-1)
    checks=dict(reference=float((reference.cpu()-cache['writes']['8']).norm()/cache['writes']['8'].norm()),partner_reconstruction=float((reconstructed-coefficient_reference).norm()/coefficient_reference.norm()),
                unit_partner=abs(float(b@b)-1),perpendicular=abs(float(a@b)))
    rows=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json').read_text())['rows']
    previous=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_RESULT.json').read_text())['reports'][1]
    torch.set_default_dtype(torch.float32)
    effects=score([reference.cpu(),reference.cpu(),candidate.cpu()],ports['pre']+ports['native_output'],rows,sd['lm_head.weight'].float(),reference_effects=dict(zero_ce=[previous['zero_ce']],swaps=[previous['swaps']]))
    families=[]
    for family in sorted({row['family'] for row in rows}):
        ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family],device='cuda');ep=(2*ids[:,None]+torch.tensor([0,1],device='cuda')).flatten()
        families.append(dict(family=family,relative_write_error=float((candidate[ep]-reference[ep]).norm()/reference[ep].norm())))
    result={'pred_a':max(checks.values())<=1e-8 and effects['pred_a'],'pred_b':cosine>=.95 and output_cos>=.95 and partner_cos>=.95,
            'pred_c':effects['pred_b'] and effects['pred_c'],'pred_d':all(f['relative_write_error']<=.05 for f in families)}
    torch.save(dict(compiled_readers=readers.cpu(),writer=v.cpu(),arm=1,node=251,mode=8,conditional_floats=10627200),artifact)
    result.update(function_cosine=cosine,output_cosine=output_cos,partner_cosine=partner_cos,parent_cosine=float(abs(a@c)),
                  neighboring_squared_singular_gaps=[float(1-singular[7].square()/singular[6].square()),float(1-singular[8].square()/singular[7].square())],
                  checks=checks,effects=effects,families=families,artifact_sha=digest(artifact),source_shas=binding,
                  execution_seconds=time.perf_counter()-tic,scope='Fixed independent-start same-ordinal branch; paired coefficient recurrence and fresh conditional intervention preservation, not global recovery or corpus OOD.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','effects')}),flush=True)

if __name__=='__main__':main()
