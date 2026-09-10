"""Exact source-response bank semantics for directional cuts in linear controls."""
import numpy as np


def controls():
    results={}
    for name,families in [('a_to_m',['A','M']),('m_to_a',['M','A']),('both_directions',['A','M','A'])]:
        d=len(families)+1;maps=[]
        for i in range(len(families)):
            w=np.zeros((d,d));w[i+1,i]=1;maps.append(w)
        x=np.ones(d);delta=np.eye(d)[0]
        def run(source,clamps):
            state=source.copy();writes=[]
            for i,w in enumerate(maps):
                write=clamps[i] if i in clamps else w@state
                writes.append(write.copy());state=state+write
            return state,writes
        _,native=run(x,{})
        _,bank_a=run(x-delta,{i:native[i] for i,f in enumerate(families) if f=='M'})
        _,bank_m=run(x-delta,{i:native[i] for i,f in enumerate(families) if f=='A'})
        cube=np.zeros((2,2))
        for am in (0,1):
            for ma in (0,1):
                clamps={}
                for i,f in enumerate(families):
                    if f=='A' and not ma:clamps[i]=bank_a[i]
                    if f=='M' and not am:clamps[i]=bank_m[i]
                state,_=run(x-delta,clamps);cube[am,ma]=state[-1]
        total=cube[0,0]-cube[1,1];am=cube[0,0]-cube[1,0];ma=cube[0,0]-cube[0,1]
        result={'total':float(total),'a_to_m':float(am),'m_to_a':float(ma),'interaction':float(total-am-ma)}
        expected={'a_to_m':[1,1,0,0],'m_to_a':[1,0,1,0],'both_directions':[1,0,0,1]}[name]
        assert list(result.values())==expected
        results[name]=result
    return {'passed':True,'planted_cases':results,
            'scope':'Exact linear directed-path controls for the specified banks; nonlinear native interactions need not count paths.'}
