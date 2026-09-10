"""Separate a bilinear module's own cue/context interaction from its input's.

Four corners are OBSERVED normalized inputs. Nonadditivity includes upstream
computation and normalization; it is not evidence for either one in isolation.
"""
import json
import torch


def bilinear(x,y,left,right,down):
    return ((x@left.T)*(y@right.T))@down.T


def partition(n00,n01,n10,n11,left,right,down):
    u=n10-n00;v=n01-n00;w=n11-n10-n01+n00;a=n00+u+v
    b=lambda x,y:bilinear(x,y,left,right,down)
    local=b(u,v)+b(v,u)
    inherited=b(a,w)+b(w,a)+b(w,w)
    full=b(n11,n11)-b(n10,n10)-b(n01,n01)+b(n00,n00)
    return {'local_product':local,'inherited_input':inherited,'full':full,
            'three_corner_input':a,'input_nonadditivity':w}


def controls():
    torch.set_num_threads(2);gen=torch.Generator().manual_seed(910328)
    rand=lambda *shape:torch.randn(*shape,generator=gen,dtype=torch.float64)
    ns=[rand(5,4) for _ in range(4)];left,right,down=rand(7,4),rand(7,4),rand(3,7)
    r=partition(*ns,left,right,down)
    error=lambda x,y:float((x-y).abs().max())
    errors={'four_corner_identity':error(r['full'],r['local_product']+r['inherited_input'])}
    a,b,c,d=ns
    pred=bilinear(r['three_corner_input'],r['three_corner_input'],left,right,down)
    actual=bilinear(d,d,left,right,down)
    errors['three_corner_prediction_remainder']=error(actual-pred,r['inherited_input'])
    scale=torch.tensor([2.,-4.,.5,8.,-.25,2.,-1.],dtype=torch.float64)
    rr=partition(*ns,left*scale[:,None],right,down/scale)
    errors['gauge_parts']=max(error(r[k],rr[k]) for k in ('local_product','inherited_input','full'))
    z=torch.zeros(2,dtype=torch.float64);e=torch.eye(2,dtype=torch.float64);one=torch.ones(1,1,dtype=torch.float64)
    local=partition(z,e[1],e[0],e[0]+e[1],e[0:1],e[1:2],one)
    inherited=partition(z,e[1],e[0],2*e[0]+e[1],e[0:1],e[0:1],one)
    assert local['local_product'].item()==1 and local['inherited_input'].item()==0
    assert inherited['local_product'].item()==0 and inherited['inherited_input'].item()==3
    # Even additive raw upstream inputs generally become nonadditive after RMS.
    raw=rand(3,4);norm=lambda x:torch.nn.functional.rms_norm(x,(4,))
    n00=norm(raw[0]);n01=norm(raw[0]+raw[1]);n10=norm(raw[0]+raw[2]);n11=norm(raw.sum(0))
    normalized=partition(n00,n01,n10,n11,left,right,down)
    errors['normalized_corners_identity']=error(normalized['full'],normalized['local_product']+normalized['inherited_input'])
    assert float(normalized['input_nonadditivity'].norm())>.01
    assert float(normalized['inherited_input'].norm())>.01
    assert all(v<1e-10 for v in errors.values())
    return {'passed':True,'errors':errors,'local_only':{'local':1.,'inherited':0.},'inherited_only':{'local':0.,'inherited':3.},
            'normalization_input_nonadditivity_norm':float(normalized['input_nonadditivity'].norm()),
            'normalization_inherited_output_norm':float(normalized['inherited_input'].norm()),
            'native_model_loaded':False,'gpu_accessed':False,
            'scope':'Exact local partition only; no native semantic interaction or downstream fidelity established.'}


if __name__=='__main__':print(json.dumps(controls(),indent=2))
