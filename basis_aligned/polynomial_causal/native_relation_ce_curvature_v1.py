"""Exact CE composition accounting, no correction fitted into the program.
A prior GPU CE-cross replay<=1e-4 nats, additive identity<=1e-9.
B tail-nonlinearity meanabs<=.01 eachtask/panel.
C tail-nonlinearity RMS / full CE-cross RMS<=.1 eachtask/panel.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from native_relation_split_v1 import evaluate


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    p=Path(__file__).parent;output=p/'NATIVE_RELATION_CE_CURVATURE_V1.json';assert not output.exists()
    result_path=p/'NATIVE_RELATION_COMPOSITION_V1_RESULT.json'
    previous=json.loads(result_path.read_text());assert previous['pred_a']
    binding=json.loads((p/'NATIVE_RELATION_COMPOSITION_V1_BINDING.json').read_text())['files']
    checkpoint=next(path for path in binding if path.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    unembedding=weights['lm_head.weight'].float()
    source=p/'NATIVE_RELATION_SPLIT_V1.pt';assert hashlib.sha256(source.read_bytes()).hexdigest()==binding[str(source)]
    program=torch.load(source,weights_only=True,map_location='cpu')
    errors=[];identities=[];panels={};sources=[source,result_path]
    for label,stem in [('original','FROZEN_BRANCH_MORPHOLOGY_V1'),('holdout','NATIVE_RELATION_HOLDOUT_V1')]:
        cp=p/(stem+'_ENDPOINTS.pt');rp=p/(stem+'_ROWS.json');sources.extend([cp,rp])
        assert all(hashlib.sha256(path.read_bytes()).hexdigest()==binding[str(path)] for path in (cp,rp))
        cache=torch.load(cp,weights_only=True,map_location='cpu');rows=json.loads(rp.read_text())['rows']
        lead,rest=evaluate(program,cache['ports']['input'].double())
        lc,rc=lead@program['writers'].T,rest@program['writers'].T
        h=cache['ports']['pre']+cache['ports']['native_output'];records=[]
        for start in range(0,64,8):
            ids=list(range(start,min(64,start+8)));idx=torch.tensor(ids)
            base=h[2*idx];dl=lc[2*idx+1]-lc[2*idx];dr=rc[2*idx+1]-rc[2*idx]
            states=torch.stack([base,base+dl.float(),base+dr.float(),base+(dl+dr).float()],1)
            logits=(30*torch.tanh(F.linear(F.rms_norm(states.reshape(-1,1152),(1152,)),unembedding)/30)).double().reshape(len(ids),4,-1)
            targets=torch.tensor([rows[i]['base_answer_id'] for i in ids])
            ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets[:,None].expand(-1,4).reshape(-1),reduction='none').reshape(len(ids),4)
            additive=logits[:,1]+logits[:,2]-logits[:,0]
            add_ce=F.cross_entropy(additive,targets,reduction='none')
            curvature=add_ce-ce[:,1]-ce[:,2]+ce[:,0]
            tail=ce[:,3]-add_ce
            actual=ce[:,3]-ce[:,1]-ce[:,2]+ce[:,0]
            identities.append(float((actual-curvature-tail).abs().max()))
            for j,i in enumerate(ids):
                old=previous['panels'][label]['records'][i];assert old['row_id']==rows[i]['row_id']
                errors.append(abs(float(actual[j])-old['ce_nonadditivity']))
                records.append(dict(row_id=rows[i]['row_id'],family=rows[i]['family'],
                    ce_cross=float(actual[j]),additive_logit_loss_curvature=float(curvature[j]),
                    tail_nonlinearity=float(tail[j])))
        families={}
        for family in ('A1','A2','P','C'):
            local=[r for r in records if r['family']==family]
            cross=torch.tensor([r['ce_cross'] for r in local],dtype=torch.float64)
            tail=torch.tensor([r['tail_nonlinearity'] for r in local],dtype=torch.float64)
            curvature=torch.tensor([r['additive_logit_loss_curvature'] for r in local],dtype=torch.float64)
            families[family]=dict(mean_ce_cross=float(cross.mean()),mean_abs_ce_cross=float(cross.abs().mean()),
                mean_loss_curvature=float(curvature.mean()),mean_abs_tail_nonlinearity=float(tail.abs().mean()),
                relative_tail_rms=float(tail.norm()/cross.norm()))
        panels[label]=dict(families=families,records=records)
    a=max(errors)<=1e-4 and max(identities)<=1e-9
    result=dict(pred_a=a,pred_b=a and all(panel['families'][f]['mean_abs_tail_nonlinearity']<=.01 for panel in panels.values() for f in ('A1','A2')),
        pred_c=a and all(panel['families'][f]['relative_tail_rms']<=.1 for panel in panels.values() for f in ('A1','A2')),
        maximum_gpu_ce_cross_replay_error=max(errors),maximum_accounting_error=max(identities),panels=panels,
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        checkpoint_sha256=binding[checkpoint],scope='Exact known CE/logsumexp accounting on frozen intervention logits. '
        'Additive-logit CE is a diagnostic, not native tail execution or a fitted loss. Original composition D failure remains. '
        'No new circuit identification, selective control or ordinary replacement claim.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('panels','sources')},indent=2))
    print(json.dumps({label:panel['families'] for label,panel in panels.items()},indent=2))


if __name__=='__main__':main()
