"""Exact command-mode interaction of two sourcewise matching scores."""
# BQGATE: LIBRARY
import command_response_modes as C


def decompose(first,second,mean_payload,causal):
    a=C.modes(first);b=C.modes(second)
    scalar={'first_joint':a['11']*b['00'],'second_joint':a['00']*b['11'],
            'first_T_second_I':a['10']*b['01'],'first_I_second_T':a['01']*b['10']}
    terms={name:(value*causal)@mean_payload for name,value in scalar.items()}
    products={ab:first[ab]*second[ab]*causal for ab in C.CELLS}
    mixed=C.modes(products)['11']@mean_payload
    return {'terms':terms,'mixed':mixed,'crossed':terms['first_T_second_I']+terms['first_I_second_T'],
            'first_modes':a,'second_modes':b}


def controls():
    import torch
    dtype=torch.float64;checks={}
    payload=torch.tensor([[2.,-3.],[1.,4.]],dtype=dtype);mask=torch.tensor([[1.,0.]],dtype=dtype)
    for label in ('crossed','first_joint','second_joint'):
        first={};second={}
        for ab in C.CELLS:
            a,b=((-1)**int(x) for x in ab)
            first[ab]=torch.tensor([[a if label=='crossed' else a*b if label=='first_joint' else 1.,100*a*b]],dtype=dtype)
            second[ab]=torch.tensor([[b if label=='crossed' else a*b if label=='second_joint' else 1.,200.]],dtype=dtype)
        d=decompose(first,second,payload,mask)
        checks[label+'_closure']=torch.equal(sum(d['terms'].values()),d['mixed']) and torch.equal(d['mixed'],payload[:1])
        selected=d['crossed'] if label=='crossed' else d['terms'][label]
        omitted=d['mixed']-selected
        checks[label+'_live_selected_and_omission']=torch.equal(selected,d['mixed']) and float((d['mixed']-omitted).norm())>1
    return {'passed':all(checks.values()),'checks':checks}
