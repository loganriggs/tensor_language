"""Full current-value remainder as independently swappable routing/value ports."""
from pathlib import Path
from datetime import datetime,timezone
import json,torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph


class CurrentRemainder:
    def __init__(self,program):
        self.p=program
        self.reference=SharedGraph(program)

    def routing(self,current):
        p=self.p;x=current.double();n=x.shape[1]
        inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
        angles=torch.outer(torch.arange(n,dtype=torch.float32),inv)
        co,si=angles.cos().bfloat16().to(x.device),angles.sin().bfloat16().to(x.device)
        def rotate(z):
            a,b=z.chunk(2,-1)
            return torch.cat((a*co+b*si,-a*si+b*co),-1)
        nums=[x@p[k].double().T for k in ['k1','k2']]
        shared=torch.cat(nums,-1)@p['key_coordinates'];full=[];reflected=[]
        for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
            q=rotate(F.rms_norm(F.linear(current,p[qn].to(current.dtype)),(128,),eps=torch.finfo(torch.float32).eps)).double()
            key=F.linear(current,p[kn].to(current.dtype))
            den=(key.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
            inside=shared@self.reference.adapters[j].T
            full.append(q@rotate(nums[j]/den).transpose(-1,-2)/128)
            reflected.append(q@rotate((nums[j]-2*inside)/den).transpose(-1,-2)/128)
        even=(full[0]*full[1]+reflected[0]*reflected[1])/2
        return even.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=x.device).tril(),0)

    def values(self,current):
        p=self.p;v=current.double()@p['current_value'].double().T
        return (1-p['mixture'].double())*v-(v@p['scalar_value_coordinates'])[...,None]*p['scalar_output_coordinates']

    def write(self,channels):
        return channels@self.p['output'].double().T

    @staticmethod
    def changes(g0,z0,g1,z1):
        dg,dz=g1-g0,z1-z0
        return dict(routing=dg@z0,values=g0@dz,mixed=dg@dz,joint=g1@z1-g0@z0)


@torch.no_grad()
def main():
    p=Path(__file__).resolve().parent;out=p/'CURRENT_REMAINDER_CROSSED_V1_CPU_CONTROL.json'
    assert not out.exists();torch.set_num_threads(2);torch.manual_seed(1409261310)
    graph=CurrentRemainder(torch.load(p/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True))
    a=F.rms_norm(torch.randn(2,19,1152),(1152,));b=F.rms_norm(torch.randn_like(a),(1152,))
    g0,z0,g1,z1=graph.routing(a),graph.values(a),graph.routing(b),graph.values(b)
    reference=graph.reference.state(a,first_values=torch.zeros(2,19,128))[1]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    parts=graph.changes(g0,z0,g1,z1)
    errors=dict(self_replay=rel(g0@z0,reference),
                independent_mixed_identity=rel(parts['routing']+parts['values']+parts['mixed'],parts['joint']))
    ratio=float(parts['mixed'].norm()/parts['joint'].norm())
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=max(errors.values())<=1e-10,
                pred_b=ratio>=1e-4,errors=errors,mixed_to_full_change=ratio,
                additional_model_scalars=0,ports_per_token='T routing coefficients plus128 value channels, FP64; original program and current-state generators retained',
                scope='Native weights with synthetic normalized contexts; independently contracted mixed term and self-reference, no native causal or semantic claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
