"""Check the actual paired sampler under a correlated, nonzero-mean Gaussian."""
import json
from pathlib import Path
import torch
from conditional_reader_bound import reader_frame,paired_inputs

def main():
 torch.set_num_threads(2);rng=torch.Generator().manual_seed(260928);d=7
 L=torch.randn(d,d,generator=rng,dtype=torch.float64);mu=torch.randn(d,generator=rng,dtype=torch.float64);readers=torch.randn(3,d,generator=rng,dtype=torch.float64)
 Q,_=reader_frame(readers,L);P=Q@Q.T;cov=L@L.T;cross=L@P@L.T
 a,b=paired_inputs(mu,L,Q,65536,rng);ac=a-mu;bc=b-mu
 rel=lambda x,y:float((x-y).norm()/y.norm())
 r=dict(marginal_covariance_errors=[rel(ac.T@ac/len(a),cov),rel(bc.T@bc/len(b),cov)],cross_covariance_error=rel(ac.T@bc/len(a),cross),reader_pair_error=float(((a-b)@readers.T).norm()/(a@readers.T).norm()),mean_errors_in_rms_units=[float((x.mean(0)-mu).norm()/cov.trace().sqrt()) for x in [a,b]],seed=260928,pairs=len(a),scope='Actual sampler in correlated Gaussian with nonzero mean; verifies marginal and pair covariance, not native-model accuracy.')
 assert max(r['marginal_covariance_errors']+[r['cross_covariance_error']]+r['mean_errors_in_rms_units'])<.04
 assert r['reader_pair_error']<1e-12
 Path(__file__).with_name('CONDITIONAL_SAMPLER_AUDIT_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
