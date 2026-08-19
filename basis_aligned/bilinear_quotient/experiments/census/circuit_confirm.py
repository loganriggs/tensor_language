"""Stage F: CONFIRMATION on the third window (rows 120-300, 45k tokens,
never used for discovery or either scoring wave). Third-window tokens are
assigned to clusters via the SAME discovery-half z-statistics and
centroids; each of the 39 provisional certifications' rules is applied;
same bars (precision >= max(5x base, 0.15), recall >= 0.15, fires >= 10).
What passes here is FINAL-certified; what fails reverts to provisional-
failed with the honest note that waves 1-2 overfit the replication half.

REGISTERED PREDICTIONS: (a) >= 60% of the 39 confirm (the bars were
conservative enough to survive fresh data); (b) 'sound' entries confirm at
a higher rate than 'redteam-fixed'/'refined' entries (adversarially
surviving originals generalize best); (c) median confirmed lift >= 5x."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_confirm_results.json'
DEV='cuda'

def tclass(tok):
    if '\n' in tok: return 'newline'
    s=tok.strip()
    if not s: return 'other'
    if s.isdigit(): return 'digit'
    if all(not c.isalnum() for c in s): return 'punct'
    if s[0].isupper(): return 'upper_start'
    if tok.startswith(' ') and s.isalpha(): return 'space_word'
    return 'other'

def main():
    t0=time.time()
    reg=torch.load(PT+'circuits_registry.pt',weights_only=False)
    big=torch.load(PT+'circuit_atlas_big.pt',weights_only=False)
    thr=torch.load(PT+'circuit_atlas_third.pt',weights_only=False)
    keys=reg['keys']
    Mb=torch.stack([big['fingerprints'][k].float().to(DEV) for k in keys])
    disc=reg['disc_mask'].to(DEV)
    mu=Mb[:,disc].mean(1,keepdim=True)
    sd=Mb[:,disc].std(1,keepdim=True).clamp_min(1e-8)
    M3=torch.stack([thr['fingerprints'][k].float().to(DEV) for k in keys])
    base3=thr['base'].float().to(DEV)
    wp=base3<=base3.median()
    Z3=((M3-mu)/sd)[:,wp].T
    Z3=Z3/Z3.norm(dim=1,keepdim=True).clamp_min(1e-8)
    C=reg['centroids'].to(DEV)
    a3=(Z3@C.T).argmax(1).cpu()
    idx3=wp.nonzero().squeeze(1).cpu()
    R0=120
    feats=[]
    for gi in idx3.tolist():
        row=R0+gi//256; pos=gi%256
        t=int(FW[row,pos+1]); p=int(FW[row,pos])
        p2=int(FW[row,pos-1]) if pos>=1 else -1
        tg=enc.decode([t])
        feats.append({'target':tg,'prev':enc.decode([p]),
            'prev2':enc.decode([p2]) if p2>=0 else '',
            'is_induction':t in FW[row,:pos+1].tolist(),
            'target_equals_prev':t==p,'target_class':tclass(tg),'pos':pos})
    def fires(rule,f):
        if 'target_in' in rule and f['target'] not in rule['target_in']: return False
        if 'prev_in' in rule and f['prev'] not in rule['prev_in']: return False
        if 'prev2_in' in rule and f['prev2'] not in rule['prev2_in']: return False
        if rule.get('is_induction') and not f['is_induction']: return False
        if rule.get('target_equals_prev') and not f['target_equals_prev']: return False
        if 'target_class_in' in rule and f['target_class'] not in rule['target_class_in']: return False
        if 'pos_min' in rule and f['pos']<rule['pos_min']: return False
        if 'pos_max' in rule and f['pos']>rule['pos_max']: return False
        return True
    N=len(feats)
    prov=json.load(open(PT+'circuits_semantic_wave2.json'))
    stories={}
    import glob
    for f in sorted(glob.glob(PT+'circuits_stageD/stories_*.json')):
        for s in json.load(open(f)): stories[s['circuit_id']]=s
    w2rules={}
    for f in sorted(glob.glob(PT+'circuits_wave2/refined_*.json')):
        for s in json.load(open(f)): w2rules[s['circuit_id']]=s.get('rule')
    for f in sorted(glob.glob(PT+'circuits_wave2/verdicts_*.json')):
        for s in json.load(open(f)):
            if s['verdict']!='sound': w2rules[s['circuit_id']]=s.get('rule')
    final=[]; byst={'sound':[0,0],'redteam-fixed':[0,0],'refined':[0,0]}
    lifts=[]
    for e in prov:
        j=e['circuit_id']
        rule=(stories[j]['rule'] if e['status']=='sound' else w2rules.get(j))
        if rule is None: continue
        member=(a3==j); nb=int(member.sum())
        byst[e['status']][1]+=1
        if nb<8: continue
        hit=torch.tensor([fires(rule,f) for f in feats])
        nf=int(hit.sum())
        if nf<10: continue
        prec=float((hit&member).sum())/nf
        rec=float((hit&member).sum())/nb
        br=nb/N
        if prec>=max(5*br,0.15) and rec>=0.15:
            byst[e['status']][0]+=1
            lifts.append(prec/max(br,1e-9))
            final.append({'circuit_id':j,'story':e['story'],
                'status':e['status'],'precision':round(prec,3),
                'recall':round(rec,3),'lift':round(prec/br,1)})
    lifts.sort()
    medl=lifts[len(lifts)//2] if lifts else 0
    rate=len(final)/max(len(prov),1)
    rs=byst['sound']; rf=byst['redteam-fixed']; rr=byst['refined']
    pa=rate>=0.60
    pb=(rs[0]/max(rs[1],1))>=max(rf[0]/max(rf[1],1),rr[0]/max(rr[1],1))
    pc=medl>=5
    json.dump(final,open(PT+'circuits_FINAL_certified.json','w'),indent=1)
    out={'provisional':len(prov),'confirmed':len(final),
         'by_status':{k:f'{v[0]}/{v[1]}' for k,v in byst.items()},
         'median_lift':medl,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f'CONFIRMED {len(final)}/{len(prov)} | by status '
          f'{out["by_status"]} | median lift {medl:.1f}x')
    print(f"(a) >=60% confirm: {'HELD' if pa else 'FAILED'}")
    print(f"(b) sound confirms best: {'HELD' if pb else 'FAILED'}")
    print(f"(c) median lift >=5x: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
