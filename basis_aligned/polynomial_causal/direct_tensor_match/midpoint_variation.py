from pathlib import Path
import torch,json
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');out=p/'MIDPOINT_VARIATION_V1.json';assert not out.exists();torch.set_num_threads(4);torch.set_grad_enabled(False)
z=torch.load(p/'MIDPOINT_NATIVE_V1.pt',weights_only=True);audit=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True);S=audit['metric_sqrt'];M=audit['metric'];mu=z['stats']['calibration']['native']['mean'];cal=z['stats']['calibration']['native']['second']-mu[:,None]*mu[None,:];_,W=torch.linalg.eigh(S@cal@S);W=W.flip(1);results={}
for name,s in z['stats'].items():
 s=s['native'];mean=s['mean'];G=s['second'];variation=G-mean[:,None]*mu[None,:]-mu[:,None]*mean[None,:]+mu[:,None]*mu[None,:];SGS=S@variation@S;energy=SGS.trace();total=(M*G.T).sum();kept=(W*(SGS@W)).sum(0).cumsum(0)
 results[name]=dict(constant_mean_error_over_total=float((energy/total).sqrt()),error_over_calibration_mean_baseline={str(r):float(((energy-kept[r-1]).clamp_min(0)/energy).sqrt()) for r in [4,16,32,64,128,256]},error_over_total={str(r):float(((energy-kept[r-1]).clamp_min(0)/total).sqrt()) for r in [4,16,32,64,128,256]})
result=dict(results=results,scope='CPU moments-only successor audit. Vocabulary-centered pre-final-normalization outputs; mean and affine output basis fit calibration only. Denominator is error of the fixed calibration-mean predictor, not each held panel mean. No learned scalar input computations or behavioral validation.')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
