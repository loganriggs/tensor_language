"""Wave 2 prep: red-team + refine packs. For each FAILED circuit: wave-1
story/rule + measured failure mode + fresh positives + false-positive
contexts + missed-member contexts. For each CERTIFIED circuit: adversarial
pack (is the rule the story, or a token list?). DSL v2 adds: target_class
(newline/digit/punct/upper_start/space_word/other), prev2_in,
target_equals_prev. Mechanical prep, no registered predictions."""
import json, sys, os, glob, torch, collections
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
R0=300
reg=torch.load(PT+'circuits_registry.pt',weights_only=False)
rep=reg['rep_mask']; ar=reg['assign_rep']
disc=reg['disc_mask']; ad=reg['assign_disc']
idx_rep=rep.nonzero().squeeze(1); idx_disc=disc.nonzero().squeeze(1)
def tclass(tok):
    if '\n' in tok: return 'newline'
    s=tok.strip()
    if not s: return 'other'
    if s.isdigit(): return 'digit'
    if all(not c.isalnum() for c in s): return 'punct'
    if s[0].isupper(): return 'upper_start'
    if tok.startswith(' ') and s.isalpha(): return 'space_word'
    return 'other'
def feat(gi):
    row=R0+gi//256; pos=gi%256
    t=int(FW[row,pos+1]); p=int(FW[row,pos])
    p2=int(FW[row,pos-1]) if pos>=1 else -1
    ind=t in FW[row,:pos+1].tolist()
    tg=enc.decode([t])
    return {'target':tg,'prev':enc.decode([p]),
            'prev2':enc.decode([p2]) if p2>=0 else '',
            'is_induction':ind,'pos':pos,
            'target_class':tclass(tg),
            'target_equals_prev':t==p,'row':row}
def ctx(gi,n=30):
    row=R0+gi//256; pos=gi%256
    return (f"...{enc.decode(FW[row,max(0,pos-n):pos+1].tolist())}"
            f"[->{enc.decode([int(FW[row,pos+1])])}]")
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
featR=[feat(gi) for gi in idx_rep.tolist()]
stories={}
for f in sorted(glob.glob(PT+'circuits_stageD/stories_*.json')):
    for s in json.load(open(f)): stories[s['circuit_id']]=s
cert={c['circuit_id'] for c in json.load(open(PT+'circuits_semantic_certified.json'))}
N=len(featR)
refine=[]; redteam=[]
for c in reg['certified']:
    j=c['id']; s=stories.get(j)
    if s is None or s.get('rule') is None: continue
    member=(ar==j); nb=int(member.sum())
    if nb==0: continue
    hit=torch.tensor([fires(s['rule'],f) for f in featR])
    nf=int(hit.sum())
    prec=float((hit&member).sum())/max(nf,1)
    rec=float((hit&member).sum())/nb
    fp=[ctx(int(idx_rep[i])) for i in (hit&~member).nonzero().squeeze(1)[:8]]
    miss=[ctx(int(idx_rep[i])) for i in (~hit&member).nonzero().squeeze(1)[:8]]
    dt=idx_disc[(ad==j).nonzero().squeeze(1)]
    pos_ex=[ctx(int(g)) for g in dt[torch.randperm(len(dt))[:20]].tolist()]
    entry={'circuit_id':int(j),'owners':c['top'][:3],
           'wave1_story':s['story'],'wave1_rule':s['rule'],
           'held_out_precision':round(prec,3),'held_out_recall':round(rec,3),
           'positives':pos_ex,'false_positives_rule_fired_but_not_member':fp,
           'missed_members_rule_did_not_fire':miss}
    (redteam if j in cert else refine).append(entry)
os.makedirs(PT+'circuits_wave2',exist_ok=True)
def dump(lst,tag,per=19):
    n=0
    for i in range(0,len(lst),per):
        json.dump(lst[i:i+per],open(PT+f'circuits_wave2/{tag}_{n}.json','w'),
                  indent=1); n+=1
    return n
nr=dump(refine,'refine'); nt=dump(redteam,'redteam',per=14)
print(f'refine circuits {len(refine)} -> {nr} packs | redteam {len(redteam)} -> {nt} packs')
