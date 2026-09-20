"""Shared frozen two-site source setup; native replay required by consumers."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'basis_aligned/polynomial_causal'
def build(model,context,modal,capture_fn=None):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(ROOT/'basis_aligned/bilinear_quotient/ops'))
    import subject_number_sparse_graph_token_extraction_v1 as graph
    capture_fn=graph._capture if capture_fn is None else capture_fn
    decoder=json.loads((P/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(decoder['threshold'])/float(axis.norm())
    panel,template=context['panel'],context['template']
    rows=json.loads((P/context.get('rows_file',f'SOURCE_OOD_V2_{panel.upper()}_ROWS.json')).read_text());entries=[r for r in rows if r['template']==template]
    tokens=torch.tensor([r['token_ids'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda');read=torch.tensor([r['readout_position'] for r in entries],device='cuda')
    answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda');pairs=torch.cat([answers[:,None,:],modal[None].expand(len(entries),len(modal),2)],dim=1)
    initial=F.rms_norm(model.transformer.wte(tokens),(model.config.n_embd,)).float();raw,x0,first,pb,_=capture_fn(model,initial,torch,F)
    source_components={}
    directions=torch.zeros(*raw.shape,2,device='cuda',dtype=torch.float64)
    for index,(role,position) in enumerate([('subject','subject_position'),('attractor','control_position')]):
        pos=torch.tensor([r[position] for r in entries],device='cuda');v=initial[batch,pos].double();orth=v-(v@unit)[:,None]*unit
        removed=initial.clone();removed[batch,pos]=(threshold*unit+((v.square().sum(1)-threshold**2)/orth.square().sum(1)).sqrt()[:,None]*orth).float()
        changed,_,_,pe,_=capture_fn(model,removed,torch,F)
        original=torch.stack([(pe[i]-pb[i])[batch,pos].float().double() for i in range(5)],dim=1)
        delta=(changed.double()-raw.double())[batch,pos];six=torch.cat([original,(delta-original.sum(1))[:,None,:]],dim=1)
        source_components[role]=(pos,six)
        amplitudes=torch.tensor(context['amplitudes'][role],device='cuda',dtype=torch.float64);assert bool((amplitudes[:,1]==0).all())
        directions[batch,pos,:,index]=torch.einsum('bi,bid->bd',amplitudes,six)
    return dict(entries=entries,raw=raw,x0=x0,first=first,directions=directions,read=read,pairs=pairs,batch=batch,prefix_calls=3,source_components=source_components)
