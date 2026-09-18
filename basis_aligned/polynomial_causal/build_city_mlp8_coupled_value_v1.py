"""Build the exact coupled direct-plus-MLP8 value operator from checkpoint weights."""
from pathlib import Path
import hashlib,json,os,time,torch
import torch.nn.functional as F
from mlp8_coupled_value_v1 import prepare,execute
P=Path(__file__).resolve().parent;STEM='CITY_MLP8_COUPLED_VALUE_V1'

@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter()
    out=P/(STEM+'_CPU_RESULT.json');assert not out.exists()
    binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'));sd=torch.load(checkpoint,weights_only=True,mmap=True)
    row_files=[P/'CITY_MLP8_NORM_CLOSED_FRESH_V1_ROWS.json',P/'CITY_VALUE_PATH_FRESH_V2_ROWS.json'];rows=[]
    for path in row_files: rows.extend(json.loads(path.read_text())['rows'])
    tokens=torch.tensor(sorted({int(t) for row in rows for t in row['ids']}))
    program={'left':sd['transformer.h.8.mlp.Left.weight'].clone(),'right':sd['transformer.h.8.mlp.Right.weight'].clone(),'down':sd['transformer.h.8.mlp.Down.weight'].clone(),'bias':sd['transformer.h.8.mlp.Down_bias'].clone(),'value_reader':sd['transformer.h.9.attn.c_v.weight'][1024:1152].clone(),'lambdas9':sd['transformer.h.9.lambdas'].clone(),'mixture':sd['transformer.h.9.attn.lamb'].clone(),'token_ids':tokens,'initial_table':F.rms_norm(sd['transformer.wte.weight'][tokens].float(),(1152,)).clone()}
    runtime=prepare(program);checks=[]
    for stem in ['CITY_MLP8_NORM_CLOSED_FRESH_V1_SCREEN','CITY_VALUE_PATH_FRESH_V2']:
        artifact=torch.load(P/(stem+'_ARTIFACT.pt'),weights_only=True);rs=json.loads((P/(stem.replace('_SCREEN','')+'_ROWS.json')).read_text())['rows'];groups=[]
        # Artifacts are ordered by group; each group has ten endpoint/control probes.
        for i in range(40): groups.append(rs[i*6*1:i*6*1+6])
        errors=[];rho_errors=[]
        for i,f in enumerate(artifact['fixtures']):
            ids=torch.tensor([rs[i*6]['ids']]);inputs=f['inputs'];got,rho=execute(runtime,z=inputs['z'],delta=inputs['delta'],token_ids=ids,return_rho=True);expected=f['native_joint_value_delta'];errors.append(float((got-expected).norm()/expected.norm()))
            native_rho=(inputs['mixed9_edited'].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt() if 'mixed9_edited' in inputs else None
            if native_rho is not None:rho_errors.append(float((rho-native_rho).norm()/native_rho.norm()))
        checks.append({'stem':stem,'fixtures':len(artifact['fixtures']),'max_joint_relative_error':max(errors),'max_rho_relative_error':max(rho_errors) if rho_errors else None,'passes':max(errors)<=1e-4})
    floats=sum(v.numel() for v in program.values() if v.is_floating_point());r={'pred_a':all(x['passes'] for x in checks),'checks':checks,'tokens':len(tokens),'stored_float_scalars':floats,'stored_float_bytes':4*floats,'stored_integer_bytes':8*len(tokens),'derived_folded_down_scalars':runtime['folded_down'].numel(),'derived_folded_down_bytes':runtime['folded_down'].numel()*8,'native_input_scalars_T32':36864,'intervention_scalars_T32':36864,'native_scalar_norm_inputs':0,'full_model_forwards':0,'seconds':time.perf_counter()-start,'scope':'Exact coupled direct attention8 plus MLP8-mediated head9.8 value correction; native z8/upstream delta/token IDs inputs and later suffix external. Independent subterms are not promoted.','source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['mlp8_coupled_value_v1.py','build_city_mlp8_coupled_value_v1.py','CITY_MLP8_NORM_CLOSED_FRESH_V1_ROWS.json','CITY_VALUE_PATH_FRESH_V2_ROWS.json']}}
    torch.save(program,P/(STEM+'_PROGRAM.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
