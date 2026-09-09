"""Exact command-character convolution of source-routing and shared payload."""
# BQGATE: LIBRARY
import command_response_modes as C


def decompose(patterns,payloads):
    p=C.modes(patterns);u=C.modes(payloads)
    terms={'joint_router':p['11']@u['00'],'router_I_payload_T':p['01']@u['10'],
           'router_T_payload_I':p['10']@u['01'],'joint_payload':p['00']@u['11']}
    native={ab:patterns[ab]@payloads[ab] for ab in C.CELLS}
    return {'terms':terms,'mixed':C.modes(native)['11'],'crossed':terms['router_I_payload_T']+terms['router_T_payload_I'],
            'native':native,'payload_mixed':u['11']}


def controls():
    import torch
    dtype=torch.float64;checks={}
    for label in ('joint_router','crossed'):
        p={};u={}
        for ab in C.CELLS:
            a,b=((-1)**int(x) for x in ab)
            p[ab]=torch.tensor([[2.+(a*b if label=='joint_router' else b),-1.]],dtype=dtype)
            u[ab]=torch.tensor([[3.+(2*a if label=='crossed' else 0),-2.],[1.,4.]],dtype=dtype)
        d=decompose(p,u);mixed=d['mixed'];total=sum(d['terms'].values())
        expected=torch.tensor([[3.,-2.]] if label=='joint_router' else [[2.,0.]],dtype=dtype)
        checks[label+'_exact']=torch.equal(mixed,total) and torch.equal(mixed,expected)
        checks[label+'_no_joint_payload']=torch.equal(d['payload_mixed'],torch.zeros_like(d['payload_mixed']))
        selected=d['terms']['joint_router'] if label=='joint_router' else d['crossed']
        checks[label+'_selected_live']=torch.equal(selected,mixed) and float(mixed.norm())>0
        omitted=d['crossed'] if label=='joint_router' else d['terms']['joint_router']
        checks[label+'_omitted_term_negative']=float((mixed-omitted).norm())>1
    return {'passed':all(checks.values()),'checks':checks}
