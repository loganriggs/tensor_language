"""Endpoint distribution and paired-effect metrics shared by query-source screens."""
import torch


def score(rows, outputs):
    native=outputs['native'];device=native[0].device
    ix=torch.arange(len(rows),device=device)
    ans=torch.tensor([r['donor_answer_id'] for r in rows],device=device)
    foil=torch.tensor([r['donor_foil_id'] for r in rows],device=device)
    center=lambda z:z-z.mean(-1,keepdim=True)
    contrasts=lambda zs:(zs[1][ix,ans]-zs[1][ix,foil])-(zs[0][ix,ans]-zs[0][ix,foil])
    effect=center(native[1])-center(native[0]);effect_norm=float(effect.norm())
    contrast=contrasts(native);margin_norm=float(contrast.norm());arms={}
    for arm,zs in outputs.items():
        if arm=='native':continue
        kl=torch.stack([(b.log_softmax(-1).exp()*(b.log_softmax(-1)-v.log_softmax(-1))).sum(-1) for b,v in zip(native,zs)]).flatten()
        flips=sum(int((b.argmax(-1)!=v.argmax(-1)).sum()) for b,v in zip(native,zs))
        error=float((center(zs[1])-center(zs[0])-effect).norm())
        margin_error=float((contrasts(zs)-contrast).norm())
        endpoint_norm=float(torch.stack([center(v)-center(b) for b,v in zip(native,zs)]).norm())
        arms[arm]={'mean_kl':float(kl.mean()),'p99_kl':float(torch.quantile(kl,.99)),'max_kl':float(kl.max()),'top1_changes':flips,
            'paired_effect_relative_error':error/effect_norm if effect_norm>1e-8 else None,
            'margin_contrast_relative_error':margin_error/margin_norm if margin_norm>1e-8 else None,
            'paired_effect_error_norm':error,'native_paired_effect_norm':effect_norm,'margin_error_norm':margin_error,'native_margin_contrast_norm':margin_norm,
            'endpoint_change_norm':endpoint_norm,'endpoint_change_to_native_pair_norm':endpoint_norm/max(effect_norm,1e-8),
            'distribution_pass':float(kl.mean())<=.001 and float(torch.quantile(kl,.99))<=.01 and flips==0,
            'paired_effect_pass':(error<=.1*effect_norm if effect_norm>1e-8 else error<=1e-8) and (margin_error<=.1*margin_norm if margin_norm>1e-8 else margin_error<=1e-8)}
    return {'pair_count':len(rows),'arms':arms,'rows':[{'row_id':r['row_id'],'native_contrast':float(contrast[i]),
        'arm_contrasts':{a:float(contrasts(z)[i]) for a,z in outputs.items() if a!='native'}} for i,r in enumerate(rows)]}
