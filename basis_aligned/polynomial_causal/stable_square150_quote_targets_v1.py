"""Frozen quote-target coverage and activation check on historical FineWeb."""
import hashlib
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F


@torch.no_grad()
def main():
    start=time.perf_counter();torch.set_num_threads(2)
    root=Path(__file__).parent
    path=Path('/dev/shm/bilin18_frozen_radial_fineweb_v1.pt')
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    assert source_hash=='b1107a91d55df69bc05eefade10ffb362a5c1a81c783e7dfa3c746ea2dec72de'
    data=torch.load(path,weights_only=True,map_location='cpu')
    program=torch.load(root/'STABLE_SQUARE150_NATIVE_INTERFACE_V1.pt',weights_only=True,map_location='cpu')
    quote=torch.tensor([366,705,1,6]);targets=data['rows'][:,1:129].long()
    mask=torch.isin(targets,quote);locations=mask.nonzero()
    if len(locations)>256:
        generator=torch.Generator().manual_seed(2201)
        locations=locations[torch.randperm(len(locations),generator=generator)[:256]]
    pairs=[];used=set()
    for row,pos in locations.tolist():
        candidates=[j for j in range(128) if not mask[row,j] and (row,j) not in used]
        if not candidates:continue
        control=min(candidates,key=lambda j:(abs(j-pos),j));used.add((row,control))
        pairs.append([row,pos,control])
    assert pairs, 'No quote/control pairs in cache.'
    x=torch.stack([data['ports']['input'][r,p] for r,p,c in pairs]+[data['ports']['input'][r,c] for r,p,c in pairs]).double()
    pre=torch.stack([data['ports']['pre'][r,p] for r,p,c in pairs]+[data['ports']['pre'][r,c] for r,p,c in pairs]).float()
    native=torch.stack([data['ports']['native_output'][r,p] for r,p,c in pairs]+[data['ports']['native_output'][r,c] for r,p,c in pairs]).float()
    truth=torch.tensor([int(targets[r,p]) for r,p,c in pairs]+[int(targets[r,c]) for r,p,c in pairs])
    pilot=json.loads((root/'MATCHED_SHARED_GROUPS_V1_RESULT.json').read_text())
    ck=next(k for k in pilot['binding'] if k.endswith('pytorch_model.bin'))
    sd=torch.load(ck,weights_only=True,map_location='cpu',mmap=True);unembedding=sd['lm_head.weight'].float()
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    exact=((x@l.T)*(x@r.T))@d.T+sd['transformer.h.17.mlp.Down_bias'].double()
    scalar=(x@program['reader'].double()).square();component=scalar[:,None]*program['physical_writer'].double()
    def scores(output):return 30*torch.tanh((F.rms_norm(pre+output.float(),(1152,))@unembedding.T)/30)
    base=scores(native);control=scores(exact);changed=scores(native-component)
    replay=float((base-control).norm()/base.norm())
    base_log=base.double().log_softmax(-1);changed_log=changed.double().log_softmax(-1)
    ce_added=-changed_log.gather(1,truth[:,None])[:,0]+base_log.gather(1,truth[:,None])[:,0]
    quote_prob=base_log[:,quote].exp().sum(1);delta_prob=changed_log[:,quote].exp().sum(1)-quote_prob
    n=len(pairs);a,b=scalar[:n],scalar[n:]
    activation_ratio=float(a.mean()/b.mean())
    capability=float(torch.isin(base[:n].argmax(1),quote).double().mean())
    summaries=[]
    for name,sl in [('quote_target',slice(0,n)),('matched_nonquote',slice(n,2*n))]:
        summaries.append(dict(name=name,count=n,mean_square_activation=float(scalar[sl].mean()),median_square_activation=float(scalar[sl].median()),
                              native_quote_probability_mean=float(quote_prob[sl].mean()),mean_quote_probability_change=float(delta_prob[sl].mean()),
                              mean_ce_added=float(ce_added[sl].mean()),mean_absolute_position_ce_change=float(ce_added[sl].abs().mean())))
    result=dict(pred_a=replay<=1e-5,pred_b=n>=16 and len(set(p[0] for p in pairs))>=8,
                pred_c=activation_ratio<=.5,pred_d=capability>=.5,native_tail_replay=replay,
                all_quote_target_positions=int(mask.sum()),paired_quote_positions=n,distinct_quote_rows=len(set(p[0] for p in pairs)),
                quote_vs_control_mean_activation_ratio=activation_ratio,native_quote_family_top1_fraction=capability,
                summaries=summaries,pairs=pairs,data_source_sha256=source_hash,seconds=time.perf_counter()-start,
                scope='Postselected token family, frozen existing weight atom, historical cached FineWeb. Nearest-position controls are observational and may share discourse context. No fitting, fresh/OOD evidence, or causal input-cue identification.')
    (root/'STABLE_SQUARE150_QUOTE_TARGETS_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))


if __name__=='__main__':main()
