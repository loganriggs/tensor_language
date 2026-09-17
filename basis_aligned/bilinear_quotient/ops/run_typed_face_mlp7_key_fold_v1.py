#!/usr/bin/env python3
# BQGATE:8bodyforwards;8prefixes;300seconds;no fitting.
"""pred_a native key-source closure<=1e-5; pred_b float64 fold<=1e-12/native<=1e-5.
pred_c ordered Gram source closure<=1e-12. Native states/norms remain external.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows
STEM='TYPED_FACE_MLP7_KEY_FOLD_V1'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1_ROWS.json').read_text())['rows']
    groups,_=group_rows(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        torch.manual_seed(17092121)
        x=torch.randn(3,7,dtype=torch.float64);l=torch.randn(11,7,dtype=torch.float64);r=torch.randn_like(l)
        d=torch.randn(7,11,dtype=torch.float64);k=torch.randn(5,7,dtype=torch.float64);h=(x@l.T)*(x@r.T)
        error=rel((h@d.T)@k.T,h@(k@d).T);assert error<=1e-12
        print('8bodyforwards;8prefixes;CPU composed native product identity',error);return
    output=P/(STEM+'_RESULT.json');assert not output.exists()
    start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();b7=model.transformer.h[7];b8=model.transformer.h[8]
    L=b7.mlp.Left.weight.double();R=b7.mlp.Right.weight.double();D=b7.mlp.Down.weight.double();bias=b7.mlp.Down_bias.double()
    keys=torch.stack([b8.attn.c_k.weight[256:384],b8.attn.c_k2.weight[256:384]]).double()
    folded=keys@D;bias_reads=keys@bias
    captures=[];state={}
    def block7_pre(module,args):
        x,_,x0=args;state['u']=module.lambdas[0]*x+module.lambdas[1]*x0;state['e']=x0
    def attn7_out(module,args,result):state['g']=state['u']+result[0]
    def mlp7_pre(module,args):state['z']=args[0]
    def mlp7_out(module,args,result):state['m']=result
    def block8_pre(module,args):
        x,_,x0=args;state['r8']=module.lambdas[0]*x+module.lambdas[1]*x0
    def attn8_pre(module,args):
        current=args[0];city=state['city']
        data={name:value[:,city].detach().clone() for name,value in state.items() if isinstance(value,torch.Tensor)}
        data['native_keys']=torch.stack([module.c_k(current)[:,:,256:384][:,city],module.c_k2(current)[:,:,256:384][:,city]],1)
        captures.append(data)
    handles=[b7.register_forward_pre_hook(block7_pre),b7.attn.register_forward_hook(attn7_out),b7.mlp.register_forward_pre_hook(mlp7_pre),b7.mlp.register_forward_hook(mlp7_out),b8.register_forward_pre_hook(block8_pre),b8.attn.register_forward_pre_hook(attn8_pre)]
    try:
        for row in groups:
            state['city']=row['city_position'];ids=torch.tensor([row['ids']],device='cuda')
            x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
            for block in model.transformer.h:x,first=block(x,first,x0)
    finally:
        for handle in handles:handle.remove()
    native_errors=[];exact_errors=[];native_mlp_errors=[];gram_errors=[];sources=[];grams=[]
    l0,l1=b8.lambdas.double()
    for item in captures:
        z=item['z'].double();h=(z@L.T)*(z@R.T)
        direct_m=h@D.T+bias
        direct=torch.einsum('bd,jkd->bjk',direct_m,keys)
        folded_value=torch.einsum('bh,jkh->bjk',h,folded)+bias_reads
        exact_errors.append(rel(folded_value,direct))
        native_m=torch.einsum('bd,jkd->bjk',item['m'].double(),keys)
        native_mlp_errors.append(rel(folded_value,native_m))
        projected=torch.stack([l0*torch.einsum('bd,jkd->bjk',item['g'].double(),keys),l0*native_m,l1*torch.einsum('bd,jkd->bjk',item['e'].double(),keys)],2)
        rho=(item['r8'].float().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
        native_errors.append(rel(projected.sum(2)/rho[:,:,None],item['native_keys'].double()))
        gram=torch.einsum('bjsd,bjtd->bjst',projected,projected)/128
        target=projected.sum(2).square().mean(-1)
        gram_errors.append(rel(gram.sum((-1,-2)),target))
        sources.append(projected.cpu());grams.append(gram.cpu())
    source=torch.cat(sources);gram=torch.cat(grams);paired=source[::2]-source[1::2];parent=paired.sum(2)
    records={}
    for j,name in enumerate(['K1','K2']):
        target=parent[:,j].flatten();stats={}
        for index,label in enumerate(['carry','MLP7','initial']):
            term=paired[:,j,index].flatten();den=target.norm()
            stats[label]={'paired_change_norm_ratio':float(term.norm()/den),'aligned_fraction':float(term@target/(target@target)),'cosine':float(term@target/(term.norm()*den).clamp_min(1e-30))}
        records[name]=stats
    result={'pred_a':max(native_errors)<=1e-5,'pred_b':max(exact_errors)<=1e-12 and max(native_mlp_errors)<=1e-5,'pred_c':max(gram_errors)<=1e-12,
      'native_key_errors':native_errors,'exact_fold_errors':exact_errors,'native_mlp_projected_errors':native_mlp_errors,'gram_errors':gram_errors,'paired_source_changes':records,
      'folded_reader_shape':list(folded.shape),'folded_scalars':folded.numel()+bias_reads.numel()+2,'external_input_weight_scalars':L.numel()+R.numel(),'retained_norm_generator_D_scalars':D.numel(),
      'body_forwards':len(groups),'seconds':time.perf_counter()-start,'source_shas':binding,'panel_status':'opened four-context native fold',
      'scope':'Exact native key numerator/source Gram fold; not causal dominance, closed state generation, omitted normalization, fresh prediction or storage adoption.'}
    torch.save({'sources':source,'grams':gram,'folded_readers':folded.cpu(),'bias_reads':bias_reads.cpu(),'lambda8':b8.lambdas.detach().cpu()},P/(STEM+'_ARTIFACT.pt'))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
