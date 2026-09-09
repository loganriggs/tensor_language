"""Fixed L2 join heads: can their two QK functions share one bilinear matcher?"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import torch
import exact_source_edit_reference as E

BASE=Path(__file__).resolve().parent
OUT=BASE/'JOIN_MATCHER_COMMON_FACTOR_V1.json'


def compare(a,b):
    x=a.flatten();y=b.flatten();scale=float((x@y)/(x@x))
    residual=float((y-scale*x).norm()/y.norm())
    i=int(x.abs().argmax());j=int((x[i]*y-y[i]*x).abs().argmax())
    values=[float(x[i]),float(y[i]),float(x[j]),float(y[j])]
    u,v,w,z=map(Fraction,values);minor=u*z-v*w
    return {'best_scalar_second_from_first':scale,'relative_coefficient_error':residual,
        'stored_forms_nonproportional':bool(minor),'minor_flat_indices':[i,j],
        'minor_float_hex':[v.hex() for v in values],
        'exact_minor_numerator':str(minor.numerator),'exact_minor_denominator':str(minor.denominator)}


def main():
    torch.set_num_threads(2);assert not OUT.exists()
    positive=compare(torch.eye(2,dtype=torch.float64),-2*torch.eye(2,dtype=torch.float64))
    negative=compare(torch.eye(2,dtype=torch.float64),torch.tensor([[1.,1.],[0.,1.]],dtype=torch.float64))
    assert not positive['stored_forms_nonproportional'] and positive['relative_coefficient_error']==0
    assert negative['stored_forms_nonproportional'] and negative['relative_coefficient_error']>.1
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));layer=program.background.layers[2]
    def rotated(weight,head,pos):
        z=weight.reshape(4,32,128)[head].T;a,b=z.chunk(2,-1)
        return z*layer.rotary.cos_cached[0,pos,0]+torch.cat((-b,a),-1)*layer.rotary.sin_cached[0,pos,0]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37908);fixture=torch.randn(3,4,128,dtype=torch.float64)
    results={};errors=[]
    with torch.inference_mode():
        normalized=layer.norm(fixture);native=layer.pattern(fixture)
        for head in (1,2):
            forms=[rotated(getattr(layer,'q'+str(j)).weight,head,3)@rotated(getattr(layer,'k'+str(j)).weight,head,1).T/32 for j in (1,2)]
            terms=[torch.einsum('bi,ij,bj->b',normalized[:,3],a,normalized[:,1]) for a in forms]
            error=float((terms[0]*terms[1]-native[:,head,3,1]).abs().max());errors.append(error)
            results[str(head)]={**compare(*forms),'native_product_oracle_max_abs':error,
                'first_form_frobenius':float(forms[0].norm()),'second_form_frobenius':float(forms[1].norm())}
    result={'scope':'fixed L2H1/H2,query position3/source1,full normalized-state bilinear forms; not a reachable-state or behavioral impossibility',
        'candidate':'second native QK function equals one scalar times first, permitting exact reuse as a signed square',
        'controls_passed':True,'mechanical_passed':max(errors)<=1e-9,'heads':results,
        'native_coefficients_removed':0,
        'numeric_status':'nonzero minors certify stored FP64 folded forms; native correspondence is numerical, not interval-certified',
        'limits':'Best scalar is a function-comparison diagnostic, not a trained replacement or a relative-logit error bound; no alternative head/lag selected'}
    OUT.write_text(json.dumps(result,indent=2)+'\n');assert result['mechanical_passed'];print(json.dumps(result,indent=2))


if __name__=='__main__':main()
