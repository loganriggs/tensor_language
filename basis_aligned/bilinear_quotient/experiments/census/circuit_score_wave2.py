"""Stage E v2: score wave-2 outputs. Inputs: circuits_wave2/refined_*.json
(fixed stories/rules for the 107 failures) and verdicts_*.json (red-team
judgments on the 27 wave-1 certified: sound keeps its original rule;
gerrymandered/mismatch entries are re-scored with the red-team's fixed rule
or dropped if null). DSL v2: adds prev2_in, target_equals_prev,
target_class_in. Same bars: held-out precision >= max(5x base, 0.15),
recall >= 0.15, fires >= 10.

ADAPTIVE-REUSE CAVEAT, stated in advance: wave 2 optimizes against the same
replication half wave 1 was scored on, so wave-2 passes are provisional
until confirmed on the third window (rows 120-300, circuit_atlas_third.py)
-- the scoreboard will mark them PROVISIONAL until Stage F confirms.

REGISTERED PREDICTIONS: (a) total provisional-certified (sound survivors +
refined passes) >= 55; (b) red-team marks >= 20/27 sound (the mechanical
bars mostly picked real categories); (c) refined-rule pass rate on the 107
failures >= 25%."""
import json, sys, glob, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_score_wave2_results.json'
R0=300

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
    rep=reg['rep_mask']; ar=reg['assign_rep']
    idx_rep=rep.nonzero().squeeze(1)
    feats=[]
    for gi in idx_rep.tolist():
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
    def score(rule,j):
        member=(ar==j); nb=int(member.sum())
        if nb==0 or rule is None: return None
        hit=torch.tensor([fires(rule,f) for f in feats])
        nf=int(hit.sum())
        if nf<10: return {'ok':False,'why':'no_fire'}
        prec=float((hit&member).sum())/nf
        rec=float((hit&member).sum())/nb
        br=nb/N
        return {'ok':prec>=max(5*br,0.15) and rec>=0.15,
                'precision':round(prec,3),'recall':round(rec,3),
                'lift':round(prec/max(br,1e-9),1)}
    wave1=json.load(open(PT+'circuits_semantic_certified.json'))
    w1={c['circuit_id']:c for c in wave1}
    verdicts=[]
    for f in sorted(glob.glob(PT+'circuits_wave2/verdicts_*.json')):
        verdicts+=json.load(open(f))
    sound=0; fixed_pass=0; dropped=0
    final=[]
    for v in verdicts:
        j=v['circuit_id']
        if v['verdict']=='sound':
            sound+=1
            e=dict(w1[j]); e['status']='sound'
            final.append(e)
        else:
            s=score(v.get('rule'),j)
            if s and s.get('ok'):
                fixed_pass+=1
                final.append({'circuit_id':j,'story':v['story'],**s,
                              'status':'redteam-fixed'})
            else:
                dropped+=1
    refined=[]
    for f in sorted(glob.glob(PT+'circuits_wave2/refined_*.json')):
        refined+=json.load(open(f))
    rpass=0; rtried=0
    for s_ in refined:
        if s_.get('rule') is None: continue
        rtried+=1
        sc=score(s_['rule'],s_['circuit_id'])
        if sc and sc.get('ok'):
            rpass+=1
            final.append({'circuit_id':s_['circuit_id'],'story':s_['story'],
                          **sc,'status':'refined'})
    json.dump(final,open(PT+'circuits_semantic_wave2.json','w'),indent=1)
    pa=len(final)>=55
    pb=sound>=20
    pc=rtried>0 and rpass/rtried>=0.25
    out={'sound':sound,'redteam_fixed_pass':fixed_pass,
         'redteam_dropped':dropped,'refined_tried':rtried,
         'refined_pass':rpass,'total_provisional':len(final),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'red-team: {sound} sound, {fixed_pass} fixed-pass, {dropped} '
          f'dropped | refined: {rpass}/{rtried} pass | TOTAL provisional '
          f'{len(final)}')
    print(f"(a) >=55 total: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=20 sound: {'HELD' if pb else 'FAILED'}")
    print(f"(c) refined pass >=25%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
