"""Text-independent paired raw/normalized functional diagnostic; frozen weights."""
from pathlib import Path
import json,torch
from shared_query_product_objective_v1 import rotation
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(7131140)
 n=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True);b=n['key_basis'][1].double();q=[n[k][1].double() for k in ['q1','q2']];k=[n[name][1].double() for name in ['k1','k2']];adapters=[ki@b for ki in k]
 saved=torch.load(P/'COMPLETE_EVEN_KEY_V1_GRAMS.pt',weights_only=True)
 bases={'svd':saved['original_basis'],'queryfold':saved['queryfold_basis'],'complete':torch.load(P/'COMPLETE_EVEN_KEY_FIT_V1_PROGRAM.pt',weights_only=True)['basis_rotation']}
 projectors={name:u@u.T for name,u in bases.items()};sums={mode:{name:[0.,0.] for name in bases} for mode in ['raw','normalized']};blocks=[];eps=torch.finfo(torch.float32).eps
 for block in range(32):
  x=torch.randn(256,1152,dtype=torch.float64);y=torch.randn_like(x);queries=[x@qi.T for qi in q];keys=[y@ki.T for ki in k];t=y@b
  for position in [1,4,16,63]:
   rot=rotation(position);queries_rot=[qi@rot.T for qi in queries]
   full=[(qi*ki).sum(-1)/128 for qi,ki in zip(queries_rot,keys)]
   inside=[(qi*(t@ai.T)).sum(-1)/128 for qi,ai in zip(queries_rot,adapters)]
   true=full[0]*full[1]-full[0]*inside[1]-inside[0]*full[1]+2*inside[0]*inside[1]
   den=torch.ones(256,dtype=torch.float64)
   for qi,ki in zip(queries,keys):den*=((qi.square().mean(-1)+eps)*(ki.square().mean(-1)+eps)).sqrt()
   for name,proj in projectors.items():
    ins=[(qi*((t@proj)@ai.T)).sum(-1)/128 for qi,ai in zip(queries_rot,adapters)]
    pred=full[0]*full[1]-full[0]*ins[1]-ins[0]*full[1]+2*ins[0]*ins[1]
    for mode,scale in [('raw',torch.ones_like(den)),('normalized',den)]:
     err=((pred-true)/scale).square().sum();reference=(true/scale).square().sum();sums[mode][name][0]+=float(err);sums[mode][name][1]+=float(reference)
     blocks.append(dict(block=block,position=position,name=name,mode=mode,error2=float(err),reference2=float(reference)))
 errors={mode:{name:(a/b)**.5 for name,(a,b) in values.items()} for mode,values in sums.items()}
 change={name:errors['normalized'][name]/errors['raw'][name]-1 for name in bases}
 result={'pred_a':any(abs(v)>=.1 for v in change.values()),'relative_errors':errors,'normalization_relative_change':change,'pairs':8192,'positions':[1,4,16,63],'blocks':blocks,'scope':'Independent Gaussian residual query/key pairs with sharedquerysource androundedrotation. Completeeven score; normalizersmatch native QK epsilon, input residualRMS not imposed (would alter epsilon only). No value/suffix, text fit or native semantic/OOD claim.'}
 (P/'EVEN_KEY_NORMALIZED_RANDOM_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='blocks'},indent=2))
if __name__=='__main__':main()
