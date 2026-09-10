"""Exact endpoint-conditioned accounting, not separately intervened effects."""
import hashlib,json
from pathlib import Path
import torch


def main():
    p=Path(__file__).resolve().parent;sfile=p/'LEXICAL_FORM_INTERCHANGE_V1_STATES.pt';pfile=p/'LEXICAL_FORM_READOUT_FACTORS_V1_PROGRAM.pt'
    saved=torch.load(sfile,map_location='cpu',weights_only=True);program=torch.load(pfile,map_location='cpu',weights_only=True)
    e=torch.load(p/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].double()
    H=torch.tensor([[1,1,1,1],[-1,-1,1,1],[-1,1,-1,1],[1,-1,-1,1]],dtype=torch.float64)
    rho=lambda x:torch.sqrt(x.square().mean(-1)+torch.finfo(torch.float32).eps)
    lex=lambda z:torch.stack((z[:,2]-z[:,0],z[:,3]-z[:,1]),-1)
    reports={}
    for name in ['A1','A2']:
        U=torch.einsum('ba,nbd->nad',H,saved[name]['reader_components'])
        for after,before in [('FB','B'),('FX','X')]:
            label=name+'_'+after;compiled=program[label];a=compiled['a'];s0=compiled['s_base'];c0=compiled['c_base']
            h0=saved[name]['states'][before]['h'].double();h1=saved[name]['states'][after]['h'].double();r0,r1=rho(h0),rho(h1);s1=h1@e
            c1=torch.einsum('nkd,nd->nk',U,h1-s1[:,None]*e);n0=a*s0[:,None]+c0
            p0=n0/r0[:,None];p1=(a*s1[:,None]+c1)/r1[:,None];dp=p1-p0
            z0=30*torch.tanh(p0/30);z1=30*torch.tanh(p1/30);dz=z1-z0
            gain=torch.where(dp.abs()>1e-8,dz/dp,1-torch.tanh(p0/30).square())
            raw={'scalar':a*(s1-s0)[:,None]/r1[:,None],'complement':(c1-c0)/r1[:,None],'norm':n0*(1/r1-1/r0)[:,None]}
            terms={k:lex(gain*v) for k,v in raw.items()};target=lex(dz);error=sum(terms.values())-target;assert float(error.abs().max())<=1e-10
            cos=float((terms['scalar']*terms['complement']).sum()/(terms['scalar'].norm()*terms['complement'].norm()))
            reports[label]=dict(identity_max_abs=float(error.abs().max()),scalar_complement_cosine=cos,
                term_norm_over_total={k:float(v.norm()/target.norm()) for k,v in terms.items()},sum_term_norm_over_total=sum(float(v.norm()/target.norm()) for v in terms.values()),
                scope='Softcap secant gain depends on both native endpoints; these terms are exact accounting, not independent counterfactual effects or an extracted predictor.')
    out=dict(schema='lexical.form_numerator_cancellation.v1',reports=reports,states_sha256=hashlib.sha256(sfile.read_bytes()).hexdigest(),program_sha256=hashlib.sha256(pfile.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_forwards=0,gpu_accessed=False,checkpoint_loaded=False)
    with (p/'LEXICAL_FORM_NUMERATOR_CANCELLATION_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
