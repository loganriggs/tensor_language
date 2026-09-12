"""Exact writer-span state for the previously compiled weight-derived union."""
import hashlib,json
from pathlib import Path
import torch
from ll1_joint_parent_graph_v1 import execute as graph_value
from shared_parent_intervention_v1 import disable
from writer_state_program_v1 import encode,execute
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    graph=torch.load(P/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt',weights_only=True,map_location='cpu')
    frame=json.loads((P/'SHARED_GRAPH_FEATURE_FRAME_V1_RESULT.json').read_text());parents=tuple(frame['selected_parents'])
    active=[g for g in graph['groups'] if len(g['parent_ids'])];k=len(active);assert k==19
    metric_w=torch.stack([g['writer'] for g in active],1).double()
    w=torch.linalg.solve_triangular(graph['output_whitener'].double(),metric_w,upper=True)
    partners=torch.cat([g['cross']@g['private'] for g in active]).double()
    f=torch.cat([graph['readers'].double(),partners]);nparent=len(graph['readers']);nmixed=len(partners)
    products=[];dependencies=[];mixing=torch.zeros(k,nmixed+len(graph['pairs']),dtype=torch.float64);cursor=0
    for gi,g in enumerate(active):
        for parent in g['parent_ids']:
            products.append((int(parent),nparent+cursor));dependencies.append((int(parent),));mixing[gi,cursor]=1;cursor+=1
        mixing[gi,nmixed+g['pair_ids']]=g['shared_coeff'].double()
    for pair in graph['pairs']:
        p=tuple(map(int,pair));products.append(p);dependencies.append(p)
    products=torch.tensor(products,dtype=torch.long);assert len(products)==42 and f.shape==(37,1152)
    block=dict(products=products,mixing=mixing,bias=torch.zeros(k,dtype=torch.float64),reentry=torch.tensor([1.,0.],dtype=torch.float64))
    x0=torch.load(P/'SHARED_NODE_PARENT1_FINEWEB_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports']['pre'].double()
    binding=json.loads((P/'REGIONAL_RECURSIVE_KEY_V1_BINDING.json').read_text())['files'];ck=next(p for p in binding if p.endswith('pytorch_model.bin'))
    sd=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');readout=sd['lm_head.weight'].double()
    encoded=encode(x0,w,[f],readout);eps=torch.finfo(torch.float32).eps
    inp=x0/(x0.square().mean(-1,keepdim=True)+eps).sqrt();private=disable(graph,inp,list(range(nparent)))
    masks=[(),(parents[0],),(parents[1],),parents];actual=[];predicted=[];cells=[]
    def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
    for mask in masks:
        edits=tuple((0,i) for i,dep in enumerate(dependencies) if set(dep)&set(mask))
        score=graph_value(graph,inp) if not mask else disable(graph,inp,mask)
        write=torch.linalg.solve_triangular(graph['output_whitener'].double(),(score-private).T,upper=True).T
        h=x0+write;logits=30*torch.tanh((h@readout.T)/(h.square().mean(-1,keepdim=True)+eps).sqrt()/30)
        result=execute(encoded,[block],edits);reconstructed=result['alpha']*x0+result['coefficients']@w.T
        cells.append(dict(parents_removed=list(mask),state_error=rel(reconstructed,h),norm_error=rel(result['norm2'],h.square().sum(-1)),logit_error=rel(result['logits'],logits)))
        actual.append(logits);predicted.append(result['logits'])
    errors=[rel(predicted[i]-predicted[0],actual[i]-actual[0]) for i in (1,2,3)]
    inter=actual[3]-actual[2]-actual[1]+actual[0];pinter=predicted[3]-predicted[2]-predicted[1]+predicted[0]
    intererror=rel(pinter,inter)
    core=f.numel()+w.numel()+encoded['cross'][0].numel()+encoded['gram'].numel()+mixing.numel()+block['bias'].numel()+2
    price=dict(core_scalars=core,output_adapter_scalars=encoded['output_w'].numel(),changing_state=k,
               encoded_scalars_per_input=len(f)+k+1+readout.shape[0],fullU_common_scalars=readout.numel(),dense_mixing_scalars=mixing.numel(),product_index_int64=products.numel(),
               compact_DAG_reference=frame['price']['compact_shared_dag_scalars'],frame_core_reference=frame['price']['frame_program_scalars'],frame_output_adapter_reference=frame['price']['compiled_feature_unembedding_scalars'])
    a=all(max(c['state_error'],c['norm_error'],c['logit_error'])<=1e-10 for c in cells)
    bb=a and max(errors)<=1e-8 and intererror<=1e-6 and inter.norm()>1e-8
    cc=bb and core+price['output_adapter_scalars']<price['frame_core_reference']+price['frame_output_adapter_reference'] and price['encoded_scalars_per_input']<=frame['price']['per_input_encoded_scalars']
    result={'pred_a':a,'pred_b':bool(bb),'pred_c':bool(cc)}
    result.update(cells=cells,effect_errors=errors,interaction_relative_error=intererror,interaction_norm=float(inter.norm()),price=price,
                  source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Exact execution of the same extracted shared union from an unconverged graph fit. Reduced dynamic writer state and adapter cost do not establish native fullMLP fidelity or upstream extraction; fullU and50304 initial readout values remain explicitly charged.')
    torch.save(dict(readers=f,writers=w,block=block,dependencies=dependencies),P/'SHARED_GRAPH_WRITER_STATE_V1_ARTIFACT.pt')
    (P/'SHARED_GRAPH_WRITER_STATE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
