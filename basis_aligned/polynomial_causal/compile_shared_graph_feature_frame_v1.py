"""Compile a frozen shared-parent union; preserve pair terms exactly once."""
import hashlib,json
from pathlib import Path
import torch
from shared_parent_intervention_v1 import banks,concatenate,value,disable
from ll1_joint_parent_graph_v1 import execute as graph_value
from closed_feature_program_v1 import encode,execute
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    path=P/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt';graph=torch.load(path,weights_only=True,map_location='cpu')
    mixed,pair,_=banks(graph);left,right,metric_writers=concatenate(mixed,pair)
    left=left.double();right=right.double()
    writers=torch.linalg.solve_triangular(graph['output_whitener'].double(),metric_writers.double(),upper=True)
    mixed_ids=torch.cat([g['parent_ids'] for g in graph['groups']]);dependencies=[(int(k),) for k in mixed_ids]+[tuple(map(int,p)) for p in graph['pairs']]
    offdiag=[i for i,p in enumerate(graph['pairs']) if p[0]!=p[1]];assert offdiag
    pair_index=max(offdiag,key=lambda i:float(writers[:,len(mixed_ids)+i].norm()))
    parents=tuple(map(int,graph['pairs'][pair_index]));masks=[(),(parents[0],),(parents[1],),parents]
    raw=torch.cat([left.T,right.T,writers],1);norms=raw.norm(dim=0);raw=raw[:,norms>0]/norms[norms>0]
    u,s,_=torch.linalg.svd(raw,full_matrices=False);rank=int((s>s[0]*1e-12).sum());b=u[:,:rank]
    factors=[left,right,writers.T]
    frame_error=max(float((a-(a@b)@b.T).norm()/a.norm()) for a in factors)
    block=dict(left=left@b,right=right@b,down=b.T@writers,bias=torch.zeros(rank,dtype=torch.float64),reentry=torch.tensor([1.,0.],dtype=torch.float64))
    torch.manual_seed(6134601);probe=torch.randn(64,1152,dtype=torch.float64)
    all_parents=list(range(len(graph['readers'])));private=disable(graph,probe,all_parents)
    graph_errors=[];term_edits=[]
    for mask in masks:
        edit=[i for i,dep in enumerate(dependencies) if set(dep)&set(mask)];term_edits.append(tuple((0,i) for i in edit))
        coeff=writers.clone();coeff[:,edit]=0
        direct=graph_value(graph,probe) if not mask else disable(graph,probe,mask)
        physical=torch.linalg.solve_triangular(graph['output_whitener'].double(),(direct-private).T,upper=True).T
        projected=value((left,right,coeff),probe)
        graph_errors.append(float((physical-projected).norm()/physical.norm()))
    ports=torch.load(P/'SHARED_NODE_PARENT1_FINEWEB_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports']
    x0=ports['pre'].double();assert x0.shape==(72,1152)
    binding=json.loads((P/'REGIONAL_RECURSIVE_KEY_V1_BINDING.json').read_text())['files'];ck=next(k for k in binding if k.endswith('pytorch_model.bin'))
    sd=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');readout=sd['lm_head.weight'].double();assert readout.shape==(50304,1152)
    encoded=encode(x0,b,readout);eps=torch.finfo(torch.float32).eps
    direct_logits=[];compiled_logits=[];cells=[]
    def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
    normed=x0/(x0.square().mean(-1,keepdim=True)+eps).sqrt()
    for mask,edits in zip(masks,term_edits):
        coeff=writers.clone();coeff[:,[i for _,i in edits]]=0
        h=x0+value((left,right,coeff),normed)
        direct=30*torch.tanh((h@readout.T)/(h.square().mean(-1,keepdim=True)+eps).sqrt()/30)
        compiled=execute(encoded,[block],edits)
        direct_logits.append(direct);compiled_logits.append(compiled['logits'])
        cells.append(dict(parents_removed=list(mask),logit_error=rel(compiled['logits'],direct),feature_error=rel(compiled['features'],h@b),norm_error=rel(compiled['norm2'],h.square().sum(-1))))
    effects=[rel(compiled_logits[i]-compiled_logits[0],direct_logits[i]-direct_logits[0]) for i in (1,2,3)]
    norms=[float((direct_logits[i]-direct_logits[0]).norm()) for i in (1,2,3)]
    interaction=direct_logits[3]-direct_logits[2]-direct_logits[1]+direct_logits[0]
    predicted=compiled_logits[3]-compiled_logits[2]-compiled_logits[1]+compiled_logits[0]
    interaction_error=rel(predicted,interaction)
    active=[g for g in graph['groups'] if len(g['parent_ids'])]
    compact=graph['readers'].numel()+sum(len(g['parent_ids'])*1152+g['writer'].numel()+g['shared_coeff'].numel() for g in active)
    program_scalars=b.numel()+sum(t.numel() for t in block.values())
    price=dict(ambient=1152,frame_rank=rank,cp_terms=len(left),compact_shared_dag_scalars=compact,expanded_cp_scalars=left.numel()+right.numel()+writers.numel(),frame_program_scalars=program_scalars,
               common_full_unembedding_scalars=readout.numel(),compiled_feature_unembedding_scalars=50304*rank,per_input_encoded_scalars=rank+1+50304,
               scope='Compact DAG counts folded private partners and each active group writer, with shared pairs computed once; integer indices excluded. Frame includes dense adapter. FullU/initial-complement readout remain charged, so compare all columns before claiming savings.')
    a=max([frame_error]+graph_errors)<=1e-10;bb=a and all(max(c['logit_error'],c['feature_error'],c['norm_error'])<=1e-10 for c in cells)
    result={'pred_a':a,'pred_b':bb,'pred_c':bb and max(effects)<=1e-8 and min(norms)>1e-8 and interaction.norm()>1e-8 and interaction_error<=1e-6}
    result['pred_c']=bool(result['pred_c']);result.update(selected_parents=list(parents),frame_error=frame_error,graph_errors=graph_errors,cells=cells,effect_errors=effects,effect_norms=norms,interaction_norm=float(interaction.norm()),interaction_relative_error=interaction_error,price=price,
                source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),graph_sha=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Exact extracted shared union of an existing unconverged weight-fit graph. Native72preMLPinputs used as boundary inputs; private-onlyMLP/bias/background excluded. FullU replay is equivalence of subprograms, not native behavioral fidelity, producer closure or new linguistic OOD.')
    torch.save(dict(basis=b,block=block,dependencies=dependencies,parents=parents),P/'SHARED_GRAPH_FEATURE_FRAME_V1_ARTIFACT.pt')
    (P/'SHARED_GRAPH_FEATURE_FRAME_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
