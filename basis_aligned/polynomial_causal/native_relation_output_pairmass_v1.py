"""Explain ambient-private control CE changes with an exact probability identity.
A CE=binaryCE+pairMassCost identity<=1e-9 and prior GPU CE replay<=1e-4 nats.
B binary CE meanabschange<=.02 eachfamily.
C pairMassCost change RMS/full CE change RMS>=.75 eachfamily.
No fitting or repair of failed output-split preservation bars.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from native_relation_split_v1 import evaluate


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'NATIVE_RELATION_OUTPUT_PAIRMASS_V1.json';assert not out.exists()
    priorpath=p/'NATIVE_RELATION_OUTPUT_SPLIT_PHYSICAL_V1_RESULT.json';prior=json.loads(priorpath.read_text());assert prior['pred_a']
    binding=json.loads((p/'NATIVE_RELATION_OUTPUT_SPLIT_PHYSICAL_V1_BINDING.json').read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((p/'NATIVE_RELATION_NEIGHBOR_V1_ROWS.json').read_text())['rows']
    cache=torch.load(p/'NATIVE_RELATION_NEIGHBOR_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')
    program=torch.load(p/'NATIVE_RELATION_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    split=torch.load(p/'NATIVE_RELATION_OUTPUT_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].float();ports=cache['ports'];h=ports['pre']+ports['native_output']
    lead,rest=evaluate(program,ports['input'].double());w=(lead+rest)@split['splits']['ambient']['private_writers'].T
    old={r['row_id']:r for r in prior['records'] if r['panel']=='neighbor'}
    records=[];errors=[];replay=[]
    for start in range(0,64,8):
        indices=list(range(start,start+8));states=torch.cat([h[start:start+8],h[start:start+8]-w[start:start+8].float()])
        z=30*torch.tanh(F.linear(F.rms_norm(states,(1152,)),u)/30)
        answers=torch.tensor([rows[j//2][('base' if j%2==0 else 'donor')+'_answer_id'] for j in indices]*2)
        foils=torch.tensor([rows[j//2][('base' if j%2==0 else 'donor')+'_foil_id'] for j in indices]*2)
        logits=z.double();za=logits.gather(1,answers[:,None]).squeeze(1);zf=logits.gather(1,foils[:,None]).squeeze(1)
        pairlogsum=torch.logaddexp(za,zf)
        binary=F.softplus(zf-za);mass=torch.logsumexp(logits,-1)-pairlogsum
        ce=F.cross_entropy(logits,answers,reduction='none')
        errors.append(float((ce-binary-mass).abs().max()))
        for k,j in enumerate(indices):
            row=rows[j//2];side=j%2
            delta=float(ce[k+8]-ce[k]);bd=float(binary[k+8]-binary[k]);md=float(mass[k+8]-mass[k])
            replay.append(abs(delta-old[row['row_id']]['arms'][3]['zero_ce'][side]))
            records.append(dict(family=row['family'],verb=row['verb'],side='donor' if side else 'base',
                ce_change=delta,binary_ce_change=bd,pair_mass_cost_change=md,
                native_binary_ce=float(binary[k]),native_pair_probability=float((-mass[k]).exp())))
    cells=[]
    for family in ('past','progressive'):
        local=[r for r in records if r['family']==family]
        a=torch.tensor([r['ce_change'] for r in local],dtype=torch.float64)
        b=torch.tensor([r['binary_ce_change'] for r in local],dtype=torch.float64)
        m=torch.tensor([r['pair_mass_cost_change'] for r in local],dtype=torch.float64)
        cells.append(dict(family=family,full_meanabs=float(a.abs().mean()),binary_meanabs=float(b.abs().mean()),
            pair_mass_meanabs=float(m.abs().mean()),pair_mass_relative_rms=float(m.norm()/a.norm()),
            binary_relative_rms=float(b.norm()/a.norm()),
            pair_mass_and_binary_cosine=float((m@b)/(m.norm()*b.norm()))))
    result=dict(pred_a=max(errors)<=1e-9 and max(replay)<=1e-4,pred_b=all(c['binary_meanabs']<=.02 for c in cells),
        pred_c=all(c['pair_mass_relative_rms']>=.75 for c in cells),max_identity_error=max(errors),max_gpu_replay_nats=max(replay),
        cells=cells,records=records,source_result_sha256=digest(priorpath),
        scope='Exact probability accounting for frozen ambient-private component. Known developmental control panel; no verdict repair or data fitting.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
