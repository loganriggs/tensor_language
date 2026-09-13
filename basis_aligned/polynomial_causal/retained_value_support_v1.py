"""Exact-zero source support and packed contraction replay on frozen ports."""
import json,time
from pathlib import Path
import torch
from three_group_shared_dag_v1 import execute
P=Path(__file__).resolve().parent

def packed(gate,value):
    shape=value.shape;source=shape[-2];width=shape[-1]
    g=gate.reshape(-1);v=value.reshape(-1,width)
    active=(g!=0)&(v!=0).any(-1)
    owner=torch.arange(g.numel())//source
    out=value.new_zeros(g.numel()//source,width)
    out.index_add_(0,owner[active],g[active,None]*v[active])
    return out.reshape(*shape[:-2],width),int(active.sum()),g.numel(),int((v!=0).any(-1).sum())

@torch.no_grad()
def main():
    torch.set_num_threads(2);start=time.perf_counter()
    saved=torch.load(P/'ADDITIVE_HEAD_RAW_PORTS_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    rows=[];numerator=denominator=0.;totals=[0,0,0];dense=[0,0,0]
    for index,allports in enumerate(saved['ports']):
        native,child,remainder,additive=[allports[k] for k in (0,1,3,4)]
        a,b,v=native;ac,bc,vc=[x-y for x,y in zip(child,native)];ar,br,vr=[x-y for x,y in zip(remainder,native)]
        abar=a+ac+ar;bbar=b+bc+br;vbar=v+vc+vr
        gates=[ar*bbar+(a+ac)*br,ac*bbar+(a+ar)*bc,(additive[0]-abar)*bbar+abar*(additive[1]-bbar)]
        parts=[packed(g,x) for g,x in zip(gates,[vc,vr,vbar])]
        out=sum(p[0] for p in parts);ref=execute(native,child,remainder,additive)
        error=float((out-ref).square().sum());norm=float(ref.square().sum());numerator+=error;denominator+=norm
        for k,p in enumerate(parts):totals[k]+=p[1];dense[k]+=p[2]
        rows.append(dict(index=index,active_sources=[p[1] for p in parts],dense_sources=[p[2] for p in parts],nonzero_value_sources=[p[3] for p in parts],relative_error=(error/max(norm,1e-300))**.5))
    replay=(numerator/denominator)**.5;assert replay<1e-10
    result=dict(pred_a=replay<=1e-10,pred_b=sum(totals)<=.8*sum(dense),active_sources=totals,dense_sources=dense,
        value_multiplication_saving_fraction=1-sum(totals)/sum(dense),aggregate_replay_error=replay,rows=rows,seconds=time.perf_counter()-start,
        scope='120cached native ports; exact zero tests only, no threshold pruning or fit. Packed index_add executes active products; support detection/index/state costs not priced. Empirical support is not a universal producer guarantee or semantic reuse.')
    (P/'RETAINED_VALUE_SUPPORT_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
