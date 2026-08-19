"""Track-1 ROUND 2: three rule-compliant DISTINCTIVE explanations, scored on
the free-norm fingerprints with per-token base entropy as a new feature.

E1 (edge explanation, attn6): "attn6 transports L5's MLP-written content"
   compiles to FINGERPRINT KINSHIP: attn6's fingerprint should resemble mlp5's
   more than the median other mlp's.
E2 (mlp9 refined): "trims overshoot on CONFIDENT errors" compiles to: deletion
   benefit concentrated on high-loss & LOW-entropy tokens -- distinctive from
   plain difficulty (high-loss regardless of confidence).
E3 (attn14 refined): same confident-error compilation; should match or beat its
   plain-difficulty score (+0.183).

REGISTERED PREDICTIONS: (a) E1: corr(attn6, mlp5) exceeds the median
corr(attn6, mlp_j != 5,6) by >= 0.05; (b) E2: the confident-error predictor
beats mlp9's plain-difficulty score by >= 0.05; (c) E3: confident-error >=
plain-difficulty - 0.02 (refinement helps or ties)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, m
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bilin18_track1_round2_results.json'

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

@torch.no_grad()
def base_entropy():
    ents=[]
    for i in range(384,448,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(1152,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(1152,)))/30)).float()
        p=F.softmax(lg,dim=-1)
        ent=-(p*p.clamp_min(1e-12).log()).sum(-1)
        ents.append(ent.reshape(-1))
    return torch.cat(ents).cpu()

def main():
    t0=time.time()
    d=torch.load(PT+'bilin18_fingerprint_atlas.pt')
    base=d['base'].float()
    fps={k:v.float() for k,v in d['fingerprints'].items()}
    ent=base_entropy().float()
    # E1: kinship
    k5=abs(spearman(fps['attn6'],fps['mlp5']))
    others=[abs(spearman(fps['attn6'],fps[f'mlp{j}']))
            for j in range(18) if j not in (5,6)]
    medo=sorted(others)[len(others)//2]
    pa=(k5-medo)>=0.05
    print(f'E1 attn6~mlp5 kinship {k5:.3f} vs median-other {medo:.3f} -> '
          f'{"HELD" if pa else "FAILED"}',flush=True)
    # confident-error feature: rank(loss) - rank(entropy) (high when loss high
    # AND entropy low = confident error)
    rl=base.argsort().argsort().float(); re=ent.argsort().argsort().float()
    conf_err=(rl-re)
    for tag in ('mlp9','attn14'):
        plain=spearman(-base,fps[tag])
        refined=spearman(-conf_err,fps[tag])
        print(f'{tag}: plain {plain:+.3f} | confident-error {refined:+.3f}',
              flush=True)
        if tag=='mlp9': p9=(plain,refined)
        else: p14=(plain,refined)
    pb=(p9[1]-p9[0])>=0.05
    pc=p14[1]>=p14[0]-0.02
    out={'kinship':k5,'kinship_median_other':medo,
         'mlp9':{'plain':p9[0],'refined':p9[1]},
         'attn14':{'plain':p14[0],'refined':p14[1]},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(b) mlp9 refined beats plain by >=0.05: {'HELD' if pb else 'FAILED'}")
    print(f"(c) attn14 refined >= plain-0.02: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
