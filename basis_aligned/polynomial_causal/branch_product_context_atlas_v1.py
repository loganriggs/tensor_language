"""Descriptive context atlas of frozen products; no fitting or causal scoring."""
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent;output=root/'BRANCH_PRODUCT_CONTEXT_ATLAS_V1.json'
    assert not output.exists()
    source=root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    node=torch.load(source,weights_only=True,map_location='cpu')['nodes'][1]
    scalar_path=root/'BRANCH_ALL_DONOR_ALIGNMENT_V1_SCALARS.pt'
    scalars=torch.load(scalar_path,weights_only=True,map_location='cpu')
    tokenizer_path=Path('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    sources=[source,scalar_path,tokenizer_path];panels={};errors=[]
    for label,suffix in [('fineweb','SUPPRESSION_V1'),('corpus_shift','CORPUS_SHIFT_V1')]:
        panel_path=root/f'SHARED_NODE_PARENT1_{suffix}_ROWS.pt'
        cache_path=root/f'SHARED_NODE_PARENT1_{suffix}_ENDPOINTS.pt';sources += [panel_path,cache_path]
        panel=torch.load(panel_path,weights_only=True,map_location='cpu')
        cache=torch.load(cache_path,weights_only=True,map_location='cpu')
        x=cache['ports']['input'].double();u=x@node['reader'].double();p=x@node['partners'].double()
        a=u[:,None]*p
        errors.append(float((a-scalars[label]['amplitudes'].reshape(384,2)).norm()/a.norm()))
        rows=panel['rows'];scores=cache['scores'].reshape(384,4);ranks=cache['ranks'].reshape(384)
        assert panel['documents']==scalars[label]['documents']
        result={}
        for branch in range(2):
            result[str(branch)]={}
            for mode in ('positive','negative','near_zero'):
                values=a[:,branch] if mode!='near_zero' else a[:,branch].abs()
                order=sorted(range(384),key=lambda i:((-float(values[i]) if mode=='positive' else float(values[i])),i))
                selected=[];seen=set()
                for i in order:
                    document=panel['documents'][i%128]
                    if document in seen:continue
                    seen.add(document);meta=panel['metadata'][i]
                    assert meta['document']==document and meta['family']==i//128
                    assert meta['target_id']==int(rows[i,-1])
                    selected.append(dict(index=i,document=document,family=i//128,
                        domain=panel.get('domain_labels',['fineweb']*128)[i%128],
                        context=tokenizer.decode(rows[i,-49:-1].tolist()),target=tokenizer.decode([int(rows[i,-1])]),
                        amplitude=float(a[i,branch]),shared_value=float(u[i]),private_value=float(p[i,branch]),
                        native_ce=float(scores[i,0]),native_target_rank=int(ranks[i]),
                        branch_removal_ce_added=float(scores[i,branch+1])))
                    if len(selected)==6:break
                result[str(branch)][mode]=selected
        panels[label]=result
    receipt=dict(pred_a=max(errors)<=1e-9,amplitude_errors=errors,panels=panels,
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        scope='Descriptive extremes selected after prior validation; target families were chosen from output weights. '
              'No semantic hypothesis, independence, fresh data, fitted factors, or causal identification established.')
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    for label,result in panels.items():
        for branch,modes in result.items():
            for mode in ('positive','negative'):
                for row in modes[mode][:3]:
                    print(json.dumps(dict(panel=label,branch=branch,mode=mode,**row)))


if __name__=='__main__':main()
