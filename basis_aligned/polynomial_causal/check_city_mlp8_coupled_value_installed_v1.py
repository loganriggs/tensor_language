"""Installed full-suffix replay of the coupled direct-plus-MLP8 value package."""
import hashlib,json,os,signal,time,sys
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(P/'extracted_circuits/city_mlp8_coupled_value_v1')]
from regional_endpoint_batching_v1 import group_rows
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
import execute
STEM='CITY_MLP8_COUPLED_VALUE_INSTALLED_V1'

@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(300)
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    program=torch.load(P/'extracted_circuits/city_mlp8_coupled_value_v1/program.pt',weights_only=True);runtime=execute.prepare(program)
    from fastload import load_model_fast
    model=load_model_fast().eval(); records=[];calls=0
    for stem,rowstem in [('CITY_MLP8_NORM_CLOSED_FRESH_V1_SCREEN','CITY_MLP8_NORM_CLOSED_FRESH_V1'),('CITY_VALUE_PATH_FRESH_V2','CITY_VALUE_PATH_FRESH_V2')]:
        rows=json.loads((P/(rowstem+'_ROWS.json')).read_text())['rows'];groups,mapping=group_rows(rows);screen=torch.load(P/(stem+'_ARTIFACT.pt'),weights_only=True);ids=torch.tensor([r['ids'] for r in groups]);edit=torch.cat([f['inputs']['delta'] for f in screen['fixtures']]);corr=[]
        for i,f in enumerate(screen['fixtures']):corr.append(execute.execute(runtime,z=f['inputs']['z'],delta=f['inputs']['delta'],token_ids=torch.tensor([rows[i*6]['ids']])))
        correction=torch.cat(corr).float();state={'arm':0};attn8=model.transformer.h[8].attn;old8=attn8.squared_attention
        def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
        h8=attn8.register_forward_hook(post8);attn9=model.transformer.h[9].attn;old9=attn9.squared_attention
        def changed(module,q,k,v,q2,k2):
            if state['arm']:
                v=v.clone();v[:,:,8]-=correction
            return old9(q,k,v,q2,k2)
        attn9.squared_attention=MethodType(changed,attn9);got=[]
        try:
            for arm in range(2):
                state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
                for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
                scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30);a=torch.zeros(40,10,dtype=torch.float64)
                for i,row in enumerate(groups):
                    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):a[i,j]=scores[i,left]-scores[i,right]
                got.append(a)
        finally:h8.remove();attn8.squared_attention=old8;attn9.squared_attention=old9
        got_exp=__import__('regional_endpoint_batching_v1').expand(torch.stack(got),mapping,6);expected=screen['values'][4].double();diff=got_exp[1]-expected;outside=float(correction[edit.abs().sum(-1)==0].abs().max());records.append({'panel':stem,'sequences':40,'max_abs_error':float(diff.abs().max()),'relative_error':float(diff.norm()/expected.norm()),'outside':outside,'passes':float(diff.abs().max())<=1e-4 and float(diff.norm()/expected.norm())<=1e-5 and outside==0})
    r={'pred_a':all(x['passes'] for x in records),'panels':records,'body_forwards':160,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Installed coupled operator under native attention8 edit and full suffix; V1/V2 screen arm4 is the fixed behavioral reference. Native z8/upstream delta/token IDs remain external.','source_shas':{'check_city_mlp8_coupled_value_installed_v1.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));signal.alarm(0)
if __name__=='__main__':main()
