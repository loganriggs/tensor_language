"""Energy-matched old-reader controls for common-interface agreement.

A explicit partner-matrix versus batched projected inner <=1e-9.
B each anchor cosine exceeds its matched-control mean by >=.02.
C each anchor cosine exceeds its matched-control 95th percentile.
"""
import hashlib
import json
from pathlib import Path
import torch
from ll1_joint_parent_graph_v2 import factors
from ll1_group_matching_v1 import group_inner
from shared_input_factor_v1 import native_partner
from global_reader_rebase_v1 import matrix_inner


def projected_values(parts,u):
    a,s,c=parts
    au=(u@a.flatten(0,1).T).reshape(len(u),len(a),a.shape[1])
    qu=torch.einsum('bgr,grd->bgd',au*s[None],a)
    q=torch.einsum('bgd,bd->bg',qu,u)
    return qu,q


def projected_inner(first,second,writer_gram):
    a,alpha=first;b,beta=second
    return 2*(a*torch.einsum('gh,bhd->bgd',writer_gram,b)).sum((1,2))-(alpha*(beta@writer_gram.T)).sum(1)


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;output=root/'GLOBAL_READER_AGREEMENT_CONTROLS_V1.json'
    assert not output.exists()
    paths=[root/f'SHARED_READER_JOINT_FIT_V1_{label}_GRAPH.pt' for label in ('SPECTRAL','NATIVE')]
    graphs=[torch.load(p,weights_only=True,map_location='cpu') for p in paths]
    parts=[factors(g) for g in graphs]
    bank_path=root/'WEIGHT_SQUARE_POLISH_V1_FINAL.pt'
    controls=torch.load(bank_path,weights_only=True,map_location='cpu')['model']['a'].double()
    anchors=graphs[0]['readers'][[1,9]]
    u=torch.cat((anchors,controls));u/=u.norm(dim=1,keepdim=True)
    writer_grams=[parts[0][2]@parts[0][2].T,parts[1][2]@parts[1][2].T,parts[0][2]@parts[1][2].T]
    scores=[]
    for offset in range(0,len(u),16):
        x=u[offset:offset+16]
        v,w=[projected_values(p,x) for p in parts]
        scores.append(torch.stack((projected_inner(v,v,writer_grams[0]),
            projected_inner(w,w,writer_grams[1]),projected_inner(v,w,writer_grams[2])),-1))
    scores=torch.cat(scores)
    checks=[]
    for index in (0,1,2):
        reader=u[index];matrices=[]
        for a,s,c in parts:
            aa=a.flatten(0,1)
            ww=(c[:,:,None]*s[:,None,:]).permute(1,0,2).reshape(c.shape[1],-1)
            matrices.append(native_partner(reader,aa,aa,ww))
        m,n=matrices
        explicit=torch.stack((matrix_inner(reader,m,reader,m),matrix_inner(reader,n,reader,n),matrix_inner(reader,m,reader,n)))
        checks.append(float((explicit-scores[index]).norm()/explicit.norm()))
    cosines=scores[:,2]/(scores[:,0]*scores[:,1]).sqrt()
    energies=scores[:,:2].mean(1)
    rows=[]
    for i,parent in enumerate((1,9)):
        eligible=torch.where((u[2:]@u[i]).abs()<.95)[0]
        discrepancy=(energies[2:][eligible].log()-energies[i].log()).abs()
        chosen=eligible[discrepancy.argsort()[:16]]
        values=cosines[2:][chosen]
        mean=float(values.mean());percentile=float(torch.quantile(values,.95))
        anchor=float(cosines[i])
        rows.append(dict(spectral_reader=parent,anchor_cosine=anchor,
            control_mean=mean,control_p95=percentile,excess_over_mean=anchor-mean,
            control_ids=chosen.tolist(),control_cosines=values.tolist(),
            control_energy_ratios=(energies[2:][chosen]/energies[i]).tolist(),
            eligible_controls=len(eligible),anchor_mean_projection_energy=float(energies[i])))
    aa,bb,ab=group_inner(parts[0],parts[0]).sum(),group_inner(parts[1],parts[1]).sum(),group_inner(parts[0],parts[1]).sum()
    valid=max(checks)<=1e-9 and bool(torch.isfinite(scores).all()) and bool((scores[:,:2]>0).all())
    result=dict(pred_a=valid,pred_b=valid and all(r['excess_over_mean']>=.02 for r in rows),
        pred_c=valid and all(r['anchor_cosine']>r['control_p95'] for r in rows),
        maximum_replay=max(checks),whole_function_cosine=float(ab/(aa*bb).sqrt()),rows=rows,
        all_control_cosines=cosines[2:].tolist(),all_control_mean_energies=energies[2:].tolist(),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[bank_path]},
        scope='Post-result specificity control using a pre-existing weight-derived reader bank. '
              'Energy matching may be imperfect; no behavioral or independent reader-discovery claim.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','all_control_cosines','all_control_mean_energies')},indent=2))


if __name__=='__main__':main()
