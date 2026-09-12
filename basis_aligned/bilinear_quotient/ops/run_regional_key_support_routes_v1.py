#!/usr/bin/env python3
# BQGATE:112 existing rows;18 batches through14blocks;3 frozen key supports;180sec;JSON only;no fitting.
"""pred_a all-port reconstructed cue-score replay <=2e-5 across all panels/layers.
pred_b pred_a and forward13 original-panel routing relative error <=.1 all3layers.
pred_c pred_a/pred_b and forward13 geographic routing error <=.1 all3layers.
Null: query-visible routing still requires omitted contextual updates.
Backward13 and fresh-query contexts are diagnostic. No support changes.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from jacclust.tt_model import apply_rotary_emb
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='REGIONAL_KEY_SUPPORT_ROUTES_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=[]
    for panel,stem in enumerate(['REGIONAL_SOURCE_BLOCK_V2','REGIONAL_SOURCE_BLOCK_OOD_V1','REGIONAL_BEHAVIOR_CONTROLS_V2']):
        rs=json.loads((P/(stem+'_ROWS.json')).read_text())['rows'];validate(rs)
        for i in range(0,len(rs),2):
            a,b=rs[i]['ids'],rs[i+1]['ids'];change=[j for j in range(len(a)) if a[j]!=b[j]];assert len(change)==1
            for ids in (a,b):rows.append(dict(ids=ids,cue=change[0],prefix=tuple(ids[:change[0]+1]),panel=panel))
    buckets={}
    for row in rows:buckets.setdefault(len(row['ids']),[]).append(row)
    batches=[v[o:o+8] for v in buckets.values() for o in range(0,len(v),8)];assert len(rows)==112 and len(batches)==18
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('18batches112rows14blocks;nativequeries;frozen13supports;180sec');return
    out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval()
    supports=json.loads((P/'REGIONAL_KEY_SUPPORT_V1_FROZEN.json').read_text())['supports'];supports['all']=list(range(26))
    records=torch.load(P/'REGIONAL_KEY_PREFIX_V3_ARTIFACT.pt',map_location='cpu',weights_only=True)
    states={}
    for r in records:
        for name,support in supports.items():
            keep=[k for k in support if k<len(r['parts'])]
            raw=r['anchor'].double()+r['parts'][keep].double().sum(0)
            states[(tuple(r['prefix']),r['layer'],name)]=F.rms_norm(raw.float(),(1152,)).cuda()
    sums={};last_sums={}
    for batch in batches:
        tokens=torch.tensor([r['ids'] for r in batch],device='cuda');x0=F.rms_norm(model.transformer.wte(tokens),(1152,));x=x0;v=None
        for j,block in enumerate(model.transformer.h[:14]):
            x=block.lambdas[0]*x+block.lambdas[1]*x0;inp=F.rms_norm(x,(1152,))
            if j in (8,9,13):
                module=block.attn;n,t,_=inp.shape;cos,sin=module.rotary(inp.reshape(n,t,9,128))
                def project(z,name):return apply_rotary_emb(F.rms_norm(getattr(module,name)(z).reshape(n,t,9,128),(128,)),cos,sin).double()
                q1=project(inp,'c_q');q2=project(inp,'c_q2');k1=project(inp,'c_k');k2=project(inp,'c_k2')
                def scores(ka,kb):
                    return torch.stack([(q1[i]*ka[i,r['cue']]).sum(-1)*(q2[i]*kb[i,r['cue']]).sum(-1)/(128**2) for i,r in enumerate(batch)])
                reference=scores(k1,k2)
                for name in supports:
                    approx=inp.clone()
                    for i,r in enumerate(batch):approx[i,r['cue']]=states[(r['prefix'],j,name)]
                    predicted=scores(project(approx,'c_k'),project(approx,'c_k2'))
                    for i,r in enumerate(batch):
                        key=(r['panel'],j,name); ref=reference[i,r['cue']:]; delta=predicted[i,r['cue']:]-ref
                        for target,diff,base in ((sums,delta,ref),(last_sums,delta[-1],ref[-1])):
                            old=target.setdefault(key,[0.,0.,0]);old[0]+=float(diff.square().sum());old[1]+=float(base.square().sum());old[2]+=diff.numel()
            a,v=block.attn(inp,v);x=x+a;x=x+block.mlp(F.rms_norm(x,(1152,)))
    cells=[]
    for key,(num,den,count) in sorted(sums.items()):
        last=last_sums[key];assert den>0 and last[1]>0
        cells.append(dict(panel=key[0],layer=key[1],arm=key[2],relative_error=(num/den)**.5,last_position_error=(last[0]/last[1])**.5,reference_rms=(den/count)**.5))
    replay=max(c['relative_error'] for c in cells if c['arm']=='all')
    orig=max(c['relative_error'] for c in cells if c['arm']=='forward' and c['panel']==0)
    ood=max(c['relative_error'] for c in cells if c['arm']=='forward' and c['panel']==1)
    a=replay<=2e-5;b=a and orig<=.1
    result={'pred_a':a,'pred_b':b,'pred_c':b and ood<=.1}
    result.update(cells=cells,max_replay=replay,original_forward_max=orig,geographic_forward_max=ood,seconds=time.perf_counter()-tic,body_batches=18,source_shas=binding,
                  scope='Query-visible cue routing audit of frozen conditional key supports. Native queries and port-producing updates retained. No downstream value/write weighting or behavioral sufficiency claim; aggregate errors can hide individual heads/queries.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('cells','source_shas')},indent=2))


if __name__=='__main__':main()
