"""Compile frozen first-stream producer reads from tokens, with no contextual body."""
from pathlib import Path
import json,hashlib
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
@torch.no_grad()
def main():
    out=P/'REGIONAL_FIRST_TOKEN_V1_RESULT.json';art=P/'REGIONAL_FIRST_TOKEN_V1_ARTIFACT.pt';assert not out.exists() and not art.exists()
    torch.set_num_threads(2)
    sets={k:json.loads((P/f'REGIONAL_SOURCE_BLOCK_{v}_ROWS.json').read_text())['rows'] for k,v in [('original','V2'),('heldout','OOD_V1')]}
    token_ids=sorted(set(t for rows in sets.values() for row in rows for t in row['ids']))
    sd=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');E=sd['transformer.wte.weight'][token_ids].float();lam=sd['transformer.h.0.lambdas'].float()
    x0=F.rms_norm(E,(1152,));r=lam[0]*x0+lam[1]*x0;first=F.rms_norm(r,(1152,))
    v0=F.linear(first,sd['transformer.h.0.attn.c_v.weight'].float()).reshape(len(token_ids),9,128).double()
    programs=torch.load(P/'REGIONAL_PRODUCER_OV_FOLD_V1_ARTIFACT.pt',weights_only=True);reads={};errors={}
    for layer,fold in programs.items():
        direct=torch.einsum('nd,ahd->nha',first.double(),fold['first_value_readers'])
        reference=float(fold['value_mix'])*torch.einsum('nhk,ahk->nha',v0,fold['projected_output'])
        errors[layer]=float((direct-reference).norm()/reference.norm());reads[layer]=direct
    assert max(errors.values())<1e-5
    index={t:i for i,t in enumerate(token_ids)};pairs=[]
    for name,rows in sets.items():
        for i in range(0,len(rows),2):
            left,right=rows[i:i+2];assert len(left['ids'])==len(right['ids'])
            changed=[s for s,(a,b) in enumerate(zip(left['ids'],right['ids'])) if a!=b]
            off=[]
            for layer,v in reads.items():
                delta=v[[index[t] for t in left['ids']]]-v[[index[t] for t in right['ids']]]
                mask=torch.ones(len(delta),dtype=torch.bool);mask[changed]=False
                off.append(float(delta[mask].abs().max()) if mask.any() else 0.)
            assert max(off)==0
            pairs.append(dict(panel=name,family=left['family'],pair_index=i//2,changed_positions=changed,left_changed_ids=[left['ids'][s] for s in changed],right_changed_ids=[right['ids'][s] for s in changed]))
    torch.save(dict(token_ids=torch.tensor(token_ids),first_value_reads=reads),art)
    result=dict(native_first_value_relative_errors=errors,unique_tokens=len(token_ids),pairs=pairs,stored_read_entries=sum(v.numel() for v in reads.values()),scope='Exact token-only first-value payload at three selected producer layers. Table covers only supplied token IDs; general executor requires original embedding table plus native normalization and H maps. Routing and current-stream values remain external. No corpus fitting or OOD behavioral claim.',artifact_sha=hashlib.sha256(art.read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='pairs'}));print('changed position count set',sorted(set(len(v['changed_positions']) for v in pairs)))
if __name__=='__main__':main()
