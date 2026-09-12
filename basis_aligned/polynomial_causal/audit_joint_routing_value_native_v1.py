"""Native-weight exact routing/value polynomial and full-U decoder replay."""
from pathlib import Path
import itertools,json,time,hashlib
import torch
import torch.nn.functional as F
from folded_normalized_router_v1 import rotary,EPS
from joint_routing_value_polynomial_v1 import contract
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def main():
    torch.set_num_threads(2);torch.manual_seed(73280);tic=time.perf_counter();dt=torch.float64
    control=json.loads((P/'JOINT_ROUTING_VALUE_POLYNOMIAL_V1_CONTROL.json').read_text());assert max([v for k,v in control.items() if k!='gradient_errors']+control['gradient_errors'])<1e-10
    sd=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
    def w(k):return sd[k].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    o=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128)
    mix=float(sd['transformer.h.17.attn.lamb']);v=torch.cat([(1-mix)*w('transformer.h.17.attn.c_v.weight'),mix*w('transformer.h.0.attn.c_v.weight')],dim=1).reshape(9,128,2304)
    q=F.rms_norm(torch.randn(12,1152,dtype=dt),(1152,),eps=EPS);qr=rotary(8,128)
    direct=[];folded=[];errors=[];permutation=[]
    for pos in (0,7):
        s=torch.cat([F.rms_norm(torch.randn_like(q),(1152,),eps=EPS) for _ in range(2)],dim=1);sr=rotary(pos,128);rotation=qr.T@sr
        ka=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],dim=-1)
        kb=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],dim=-1)
        scores=[];factors=[]
        for qm,km in ((q1,k1),(q2,k2)):
            qa=torch.einsum('ni,hki->nhk',q,qm);ks=torch.einsum('ni,hki->nhk',s[:,:1152],km)
            factors.append(((qa.square().mean(-1)+EPS)*(ks.square().mean(-1)+EPS)).sqrt())
            nq=F.rms_norm(qa,(128,),eps=EPS)@qr.T;nk=F.rms_norm(ks,(128,),eps=EPS)@sr.T;scores.append((nq*nk).sum(-1)/128)
        value=torch.einsum('ni,hki->nhk',s,v);ref=torch.einsum('nh,nhk,ohk->no',scores[0]*scores[1],value,o)
        pred=contract(q[:,None].expand(-1,2,-1),s[:,None].expand(-1,3,-1),q1,ka,q2,kb,v,o,1/(128**2*factors[0]*factors[1]))
        direct.append(ref);folded.append(pred);errors.append(float((ref-pred).norm()/ref.norm()))
        qq=torch.randn(3,2,1152,dtype=dt);ss=torch.randn(3,3,2304,dtype=dt);base=contract(qq,ss,q1,ka,q2,kb,v,o)
        for perm in itertools.permutations(range(3)):
            z=contract(qq.flip(1),ss[:,perm],q1,ka,q2,kb,v,o);permutation.append(float((base-z).norm()/base.norm()))
    y=sum(folded);reference=sum(direct);errors.append(float((y-reference).norm()/reference.norm()))
    residual=torch.randn_like(q);z=residual+reference;den=z.square().mean(-1)+EPS
    l=w('transformer.h.17.mlp.Left.weight');r=w('transformer.h.17.mlp.Right.weight');d=w('transformer.h.17.mlp.Down.weight');bias=w('transformer.h.17.mlp.Down_bias')
    # Exact unembedding is applied after the residual write, avoiding U@D materialization.
    u=w('lm_head.weight');zn=F.rms_norm(z,(1152,),eps=EPS)
    target=(((zn@l.T)*(zn@r.T))@d.T+bias)@u.T
    lr=residual@l.T;rr=residual@r.T;ly=y@l.T;ry=y@r.T
    products=[lr*rr,lr*ry+ly*rr,ly*ry]
    composed=sum((prod@d.T)@u.T for prod in products)/den[:,None]+bias@u.T
    decoder_error=float((target-composed).norm()/target.norm());seconds=time.perf_counter()-tic
    result={'pred_a':max(errors)<=1e-10,'pred_b':decoder_error<=1e-10,'pred_c':max(permutation)<=1e-10 and seconds<=120,'attention_errors':errors,'full_unembedding_decoder_error':decoder_error,'permutation_error':max(permutation),'execution_seconds':seconds,'value_mix':mix,'body_forwards':0,'scope':'Native-weight replay on synthetic normalized ports, not native text or learned factorization. Final RMS and capped logits beyond this unembedded MLP write are not eliminated.'}
    result['source_shas']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),P/'joint_routing_value_polynomial_v1.py',P/'folded_normalized_router_v1.py',P/'JOINT_ROUTING_VALUE_NATIVE_V1_PREREGISTRATION.md']}
    out=P/'JOINT_ROUTING_VALUE_NATIVE_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
