#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_individual pred_c_subspace
"""Calibration-split recovery of midpoint4outputfeatures.
pred_a_replay fullcal moment replay<1e-8/referencefeaturecos>.99999/toy distinguishesrotation;
pred_b_individual every complementaryhalf pair minimum sign-permutation cosine>.9;
pred_c_subspace everyhalf pair minprincipalcos>.95.
Null fixed-reader behavioral fidelity does not imply stable data-discovered units.
Price32nativeforwards and11output eigendecompositions; nofit/selection onconfirmation.
Contiguous,parity,seed0/1complementary16doc partitions fixed. Crossdomain modes diagnosticonly.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=32,partitions=['contiguous','parity','random0','random1'],features=4)));return
 import torch
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from feature_basis_alignment import compare,toy_check
 from disk_guard import guard_torch_save
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=P/'MIDPOINT_STABILITY_V1.json';assert not out.exists();start=time.perf_counter();toy=toy_check();model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();C=ru@b17.mlp.Down.weight.double();z=torch.load(P/'MIDPOINT_NATIVE_V1.pt',weights_only=True);S=torch.load(P/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].cuda();e=torch.load(P/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);reference=torch.linalg.solve(S,e['scalar_readers'].cuda());tokens=torch.load(BQ/'.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:32,:65];tokenhash=hashlib.sha256(tokens.numpy().tobytes()).hexdigest();assert tokenhash==z['stats']['calibration']['token_sha256'];ys=[];ns=[];ms=[]
 for row in tokens:
  c=capture(model,row[None,:64].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-m/2)/scale;m=m/scale;ys.append(((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@C.T);ns.append(n.float().cpu());ms.append(m.float().cpu())
 Y=torch.stack(ys);flat=Y.flatten(0,1);mean=flat.mean(0);gram=flat.T@flat/len(flat);target=z['stats']['calibration']['native']['second'].cuda();replay=float((gram-target).norm()/target.norm())
 def basis(rows):
  y=rows.flatten(0,1);y=y-y.mean(0);G=(y@S).T@(y@S)/len(y);eig,W=torch.linalg.eigh(G);return W.flip(1)[:,:4],eig.flip(0)[:8]
 full,fulleig=basis(Y);fullalignment=compare(reference,full);splits={'contiguous':list(range(16)),'parity':list(range(0,32,2))}
 for seed in [0,1]:splits[f'random{seed}']=torch.randperm(32,generator=torch.Generator().manual_seed(261106+seed))[:16].tolist()
 records=[];saved={}
 for name,ids in splits.items():
  other=[i for i in range(32) if i not in ids];a,ea=basis(Y[ids]);b,eb=basis(Y[other]);records.append(dict(partition=name,first_documents=ids,second_documents=other,between_halves=compare(a,b),first_vs_full=compare(full,a),second_vs_full=compare(full,b),first_eigenvalues=ea.cpu().tolist(),second_eigenvalues=eb.cpu().tolist()));saved[name]=dict(first_basis=a.cpu(),second_basis=b.cpu())
 domains={}
 for domain in ['fineweb','code']:
  st=z['stats'][domain]['native'];mu=st['mean'].cuda();cov=st['second'].cuda()-mu[:,None]*mu[None,:];eig,W=torch.linalg.eigh(S@cov@S);domains[domain]=compare(full,W.flip(1)[:,:4])
 pred=dict(pred_a_replay=replay<1e-8 and fullalignment['minimum_matched_cosine']>.99999,pred_b_individual=all(r['between_halves']['minimum_matched_cosine']>.9 for r in records),pred_c_subspace=all(r['between_halves']['minimum_principal_cosine']>.95 for r in records))
 guard_torch_save(dict(n=torch.stack(ns),m=torch.stack(ms),y=Y.float().cpu(),tokens=tokens,token_sha256=tokenhash,scope='Originalcalibration normalized midpoint/source and reduced teacher outputs. No held rows.'),str(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt'));guard_torch_save(dict(partitions=saved,full_basis=full.cpu(),full_eigenvalues=fulleig.cpu()),str(P/'MIDPOINT_STABILITY_BASES_V1.pt'))
 result=dict(predictions=pred,toy=toy,calibration_moment_replay=replay,full_reference_alignment=fullalignment,partitions=records,domain_comparisons=domains,token_sha256=tokenhash,seconds=time.perf_counter()-start,scope='Stability of covariance-defined output features, not product/input-factor identity or semantic uniqueness. Fixedcomplementarycalibrationhalves; crossdomaincomparisons diagnostic.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
