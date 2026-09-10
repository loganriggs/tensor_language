"""Frozen-best fit evaluation on validation rows only; never opens test rows."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main(checkpoint_name):
    start=time.perf_counter();torch.set_num_threads(2)
    cp=P/checkpoint_name;state=torch.load(cp,map_location='cpu',weights_only=False)
    model=QuadraticModel(state['config']['kind'],1152)
    model.load_state_dict(state['best']['model'])
    w=state['best']['writer']
    with torch.no_grad():a,b,c=model.components()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double();metric=u.T@u;del u
    data=torch.load(P/'UNSUPERVISED_DATA_V2_STATES.pt',map_location='cpu',weights_only=True,mmap=True)
    rows=data['validation_rows']
    x=(data['x'][rows].float()*data['x_scale'][rows]).reshape(-1,1152).double()
    y=(data['y'][rows].float()*data['y_scale'][rows]).reshape(-1,1152).double()-sd['transformer.h.17.mlp.Down_bias'].double()
    error=total=0.
    for first in range(0,len(x),1024):
        xx,yy=x[first:first+1024],y[first:first+1024]
        predicted=(((xx@a.T)*(xx@b.T))@c)@w.T
        delta=predicted-yy
        error+=float(((delta@metric)*delta).sum());total+=float(((yy@metric)*yy).sum())
    baseline=json.loads((P/'NATURAL_STATE_BASELINES_V1_AUDIT.json').read_text())['results']['validation']
    relative=error/total
    result=dict(schema='structured.fit.validation.v1',checkpoint=checkpoint_name,
        checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),config=state['config'],
        chunk=state['next_chunk']-1,converged=state['converged'],fit_diagnostics=state['best']['diagnostics'],
        validation_states=len(x),validation_squared_relative_error=relative,
        error_relative_to_train_mean_baseline=relative/baseline['constant_squared_relative_error'],
        error_relative_to_affine_baseline=relative/baseline['affine_squared_relative_error'],
        parameter_numbers=model.storage_numbers()+w.numel(),wall_seconds=time.perf_counter()-start,
        scope='Fixed validation rows, historically opened corpus. Neither fresh/OOD nor causal. Unfinished fits remain unfinished; no final method ranking from this evaluation.')
    out=P/(cp.stem.replace('_CHECKPOINT','')+f'_CHUNK_{state["next_chunk"]-1:02d}_VALIDATION.json')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main(sys.argv[1])
