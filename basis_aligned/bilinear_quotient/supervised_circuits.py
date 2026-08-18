"""SUPERVISED FUNCTION CIRCUITS. Wave 3 exposed a metric mismatch: honest
function-level stories cannot be precise against one of 256 micro-clusters
because one function spans many clusters. Flip the unit: define ten classic
function slices mechanically (induction/copy, digit continuation,
sentence-end punctuation, newline/formatting, quote close, bracket close,
subword continuation, immediate repetition, capitalized-name continuation,
list comma), then certify each as a CIRCUIT by: (i) ownership replication --
the slice's component z-profile matches across windows A (rows 300-512) and
C (rows 120-300), cosine >= 0.8 and >= 2/3 top-owner overlap; (ii) causal
concentration -- summed raw deletion damage of the top-3 owners on-slice
exceeds that on CE-matched off-slice sites by >= 2x, in BOTH windows.

REGISTERED PREDICTIONS: (a) >= 6/10 slices ownership-replicable; (b) the
induction slice's top-3 owners include attn1 or attn2 (the model's known
early lexical attention); (c) >= 4 slices pass full causal concentration;
(d) family coverage: >= 40% of the 147 structural circuits' discovery
tokens fall in some function slice (the unsupervised clusters are largely
these functions, split by ownership)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'supervised_circuits_results.json'
DEV='cuda'

def feats_for(R0,R1):
    F=[]
    for row in range(R0,R1):
        toks=FW[row,:257].tolist()
        for pos in range(256):
            t=toks[pos+1]; p=toks[pos]
            tg=enc.decode([t]); pv=enc.decode([p])
            s=tg.strip()
            F.append({
              'ind': t in toks[:pos+1],
              'rep': t==p,
              'digit': s.isdigit() and not tg.startswith(' '),
              'sentend': tg in ('.','!','?') ,
              'newline': '\n' in tg,
              'qclose': tg in ('"',"'",'”','’') and any(
                  q in enc.decode(toks[max(0,pos-60):pos+1])
                  for q in ('"','“')),
              'bclose': s in (')',']') and any(
                  b in enc.decode(toks[max(0,pos-60):pos+1])
                  for b in ('(','[')),
              'subword': (not tg.startswith(' ')) and s.isalpha(),
              'name': tg.startswith(' ') and s[:1].isupper() and
                      pv.strip()[:1].isupper() if pv.strip() else False,
              'comma': tg==',',
            })
    return F
SL=['ind','rep','digit','sentend','newline','qclose','bclose','subword',
    'name','comma']

@torch.no_grad()
def main():
    t0=time.time()
    reg=torch.load(PT+'circuits_registry.pt',weights_only=False)
    big=torch.load(PT+'circuit_atlas_big.pt',weights_only=False)
    thr=torch.load(PT+'circuit_atlas_third.pt',weights_only=False)
    keys=reg['keys']
    MA=torch.stack([big['fingerprints'][k].float().to(DEV) for k in keys])
    MC=torch.stack([thr['fingerprints'][k].float().to(DEV) for k in keys])
    bA=big['base'].float().to(DEV); bC=thr['base'].float().to(DEV)
    wpA=bA<=bA.median(); wpC=bC<=bC.median()
    disc=reg['disc_mask'].to(DEV)
    mu=MA[:,disc].mean(1,keepdim=True)
    sd=MA[:,disc].std(1,keepdim=True).clamp_min(1e-8)
    ZA=(MA-mu)/sd; ZC=(MC-mu)/sd
    print('computing features...',flush=True)
    FA=feats_for(300,512); FC=feats_for(120,300)
    mA={s:torch.tensor([f[s] for f in FA],device=DEV)&wpA for s in SL}
    mC={s:torch.tensor([f[s] for f in FC],device=DEV)&wpC for s in SL}
    def match_ctrl(base,slice_m,wp_m,g):
        # CE-quantile-matched control sites from wp & ~slice
        pool=(wp_m&~slice_m)
        qb=torch.quantile(base[slice_m],torch.linspace(0,1,11,device=DEV))
        ctrl=torch.zeros_like(slice_m)
        pidx=pool.nonzero().squeeze(1)
        pb=base[pidx]
        for i in range(10):
            need=int(((base[slice_m]>=qb[i])&(base[slice_m]<=qb[i+1])).sum())
            cand=pidx[(pb>=qb[i])&(pb<=qb[i+1])]
            if len(cand)==0: continue
            take=cand[torch.randperm(len(cand),generator=g,device=DEV)
                      [:need]]
            ctrl[take]=True
        return ctrl
    g=torch.Generator(device=DEV).manual_seed(0)
    results={}; rep_ok=0; conc_ok=0
    for s in SL:
        nA=int(mA[s].sum()); nC=int(mC[s].sum())
        if nA<80 or nC<80:
            results[s]={'n':(nA,nC),'verdict':'underpowered'}
            print(f'{s:8s}: underpowered ({nA}/{nC})',flush=True)
            continue
        pA=ZA[:,mA[s]].abs().mean(1); pC=ZC[:,mC[s]].abs().mean(1)
        cos=float((pA/pA.norm())@(pC/pC.norm()))
        tA=set(pA.argsort(descending=True)[:3].tolist())
        tC=set(pC.argsort(descending=True)[:3].tolist())
        ov=len(tA&tC)
        owners=[keys[i] for i in sorted(tA, key=lambda i:-float(pA[i]))]
        r_ok=cos>=0.8 and ov>=2
        rep_ok+=r_ok
        conc=[]
        for (M,base,ms,wp_) in ((MA,bA,mA[s],wpA),(MC,bC,mC[s],wpC)):
            ctrl=match_ctrl(base,ms,wp_,g)
            dmg_on=float(M[list(tA)][:,ms].sum(0).mean())
            dmg_off=float(M[list(tA)][:,ctrl].sum(0).mean())
            conc.append(dmg_on/max(dmg_off,1e-6))
        c_ok=r_ok and all(c>=2.0 for c in conc)
        conc_ok+=c_ok
        results[s]={'n':(nA,nC),'owners':owners,'profile_cos':round(cos,3),
                    'owner_overlap':ov,'concentration':[round(c,1) for c in conc],
                    'ownership_replicates':bool(r_ok),
                    'causal_certified':bool(c_ok)}
        print(f'{s:8s}: n {nA}/{nC} owners {owners} cos {cos:.2f} '
              f'ov {ov} conc {[f"{c:.1f}" for c in conc]} '
              f'{"CERTIFIED" if c_ok else ("replicates" if r_ok else "no")}',
              flush=True)
    # family coverage of unsupervised clusters
    idxA=reg['disc_mask'].nonzero().squeeze(1)
    anysl=torch.zeros(len(FA),dtype=torch.bool,device=DEV)
    for s in SL: anysl|=mA[s]
    cov=float(anysl[idxA.to(DEV)].float().mean())
    ind_owners=results.get('ind',{}).get('owners',[])
    pa=rep_ok>=6
    pb=any(o in ('attn1','attn2') for o in ind_owners)
    pc=conc_ok>=4
    pd=cov>=0.40
    out={'slices':results,'rep_ok':rep_ok,'causal_ok':conc_ok,
         'cluster_coverage':round(cov,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pd)}
    print(f"\n(a) >=6 replicable: {'HELD' if pa else 'FAILED'} ({rep_ok})")
    print(f"(b) induction owned by early attn: {'HELD' if pb else 'FAILED'} "
          f"({ind_owners})")
    print(f"(c) >=4 causally certified: {'HELD' if pc else 'FAILED'} "
          f"({conc_ok})")
    print(f"(d) cluster coverage >=40%: {'HELD' if pd else 'FAILED'} "
          f"({cov:.0%})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
