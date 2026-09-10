"""Frozen lexical authorities and small algebra controls for a shared -ing reader."""
import hashlib,json
import torch

FIT=[('work','working'),('walk','walking'),('talk','talking'),('play','playing'),
     ('help','helping'),('look','looking'),('wait','waiting'),('learn','learning')]
TEST=[('read','reading'),('cook','cooking'),('sing','singing'),('watch','watching'),
      ('build','building'),('draw','drawing'),('move','moving'),('write','writing'),
      ('drive','driving'),('dance','dancing'),('jump','jumping'),('sleep','sleeping'),
      ('listen','listening'),('study','studying'),('paint','painting'),('swim','swimming')]

def build(tokenizer,control_rows):
    encode=lambda s:tokenizer.encode(s,add_special_tokens=False)
    def one(s):
        a=encode(' '+s);assert len(a)==1,(s,a)
        return a[0]
    fit_ids=[[one(a),one(b)] for a,b in FIT];panels={'A1':[],'A2':[],'P':[],'C':control_rows}
    for i,(bare,ing) in enumerate(TEST):
        for name in ('A1','A2','P'):
            lead=f'The workers often {bare}. '
            if name=='A2':lead+='According to the report, '
            base=lead+'the workers can still';donor=lead+('the workers may still' if name=='P' else 'the workers are still')
            ba,da=one(bare),one(bare if name=='P' else ing);bf,df=one(ing),one(ing if name=='P' else bare)
            bi,di=encode(base),encode(donor);assert len(bi)==len(di) and bi[-1]==di[-1]
            assert encode(base+' '+bare)==bi+[ba] and encode(base+' '+ing)==bi+[bf]
            assert encode(donor+' '+(bare if name=='P' else ing))==di+[da]
            rid=hashlib.sha256((name+'|'+base+'|'+donor).encode()).hexdigest()
            panels[name].append(dict(row_id=rid,group_id=str(i),base_ids=bi,donor_ids=di,
                base_answer_id=ba,base_foil_id=bf,donor_answer_id=da,donor_foil_id=df,
                base_semantic_position=len(bi)-1,donor_semantic_position=len(di)-1,
                base_text=base,donor_text=donor,bare=bare,gerund=ing,answer_changes=name!='P'))
    assert not set(a for pair in FIT for a in pair)&set(a for pair in TEST for a in pair)
    return {'fit_pairs':FIT,'fit_ids':fit_ids,'test_pairs':TEST,'panels':panels,
            'scope':'Weight-only fitting lexicon; new authored held-out verbs and frames, not training-distribution OOD. C reused.'}

def controls():
    gen=torch.Generator().manual_seed(9111370);rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
    l,r,d,e,u=rand(9,5),rand(9,5),rand(5,9),rand(5),rand(6,5);e=e/e.norm()
    c=e@d;q=l.T@(c[:,None]*r);q=(q+q.T)/2
    expected=((u@l.T)*(u@r.T))@d.T@e
    err=float(((u@q*u).sum(1)-expected).abs().max())
    h,m=rand(6,5),rand(6,5);dh,dm=rand(6,5),rand(6,5)
    split=(dh@e-dm@e)+(dm@e);closure=float((split-dh@e).abs().max())
    assert err<1e-10 and closure<1e-10
    return {'quadratic_max_abs':err,'skip_mlp_scalar_closure':closure,'gpu_accessed':False}
