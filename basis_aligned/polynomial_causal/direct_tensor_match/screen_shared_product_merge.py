"""Cross-output product reuse screen on the frozen private512 interaction.

Registered before running: (a) exact and orthogonal toy controls pass;
(b) >=16 disjoint cross-output merges each have pair error <=0.5 deleting the
weaker original term; (c) exact summed-error accounting for the selected edits.
Metric: independent centered calibration input covariances and vocabulary-
centered output geometry. This screens a restricted continuous graph edit;
no held-out/native model claim and no refitting of the complete graph.
"""
from pathlib import Path
import json,torch
from shared_product_merge import fit_pair_cores,self_test
p=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_grad_enabled(False)
out=p/'MIDPOINT_SHARED_PRODUCT_MERGE_SCREEN_V1.json';assert not out.exists()
control=self_test()
e=torch.load(p/'MIDPOINT_PRIVATE_FROZEN_V1.pt',weights_only=True)['private512']
rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double()
n-=n.mean(0);m-=m.mean(0)
S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
A=e['A'].double();B=e['B'].double();W=e['reduced_writers'].double()
X=n@A/(len(n)**.5);Y=m@B/(len(m)**.5);Z=S@W
G=[X.T@X,Y.T@Y,Z.T@Z]
scale=[g.diag().clamp_min(1e-30).sqrt() for g in G]
C=[g/s[:,None]/s[None,:] for g,s in zip(G,scale)]
amplitude=scale[0]*scale[1]*scale[2]
ids=torch.triu_indices(len(amplitude),len(amplitude),1)
# Shared native writer groups are not cross-branch proposals.
cross=C[2][ids[0],ids[1]].abs()<.999999
ids=ids[:,cross]
similarity=(C[0][ids[0],ids[1]]*C[1][ids[0],ids[1]]).abs()
chosen=similarity.topk(min(2048,len(similarity))).indices
ids=ids[:,chosen].T
pairgrams=torch.stack([g[ids[:,:,None],ids[:,None,:]] for g in C],1)
r=fit_pair_cores(pairgrams,amplitude[ids])
delete=amplitude[ids].square().min(1).values
ratio=r['error']/delete
order=ratio.argsort(); used=set(); selected=[]
for j in order.tolist():
 a,b=ids[j].tolist()
 if ratio[j]>.5:break
 if a in used or b in used:continue
 selected.append(j);used.update([a,b])
 if len(selected)==32:break
# Turn local solutions back into combinations of the original unit factors.
# Pseudoinverse permits exactly shared readers (singular 2D spans).
coeff=torch.linalg.pinv(r['roots'])@r['factors'].unsqueeze(-1)
coeff=coeff.squeeze(-1)
for mode in range(3):coeff[:,mode]/=scale[mode][ids]
merged=[torch.einsum('dpk,pk->dp',v[:,ids],coeff[:,mode]) for mode,v in enumerate([A,B,W])]
merged[2]*=r['gain'][None,:]
# Exact coefficient residual norm of all accepted edits, including cross terms.
if selected:
 oldids=ids[selected].flatten();sel=torch.tensor(selected)
 AA=torch.cat([A[:,oldids],merged[0][:,sel]],1)
 BB=torch.cat([B[:,oldids],merged[1][:,sel]],1)
 WW=torch.cat([W[:,oldids],-merged[2][:,sel]],1)
 ga=(n@AA).T@(n@AA)/len(n);gb=(m@BB).T@(m@BB)/len(m);gw=(S@WW).T@(S@WW)
 residual_energy=float((ga*gb*gw).sum().clamp_min(0))
else:residual_energy=0.
target_energy=float((G[0]*G[1]*G[2]).sum())
records=[dict(pair=ids[j].tolist(),error_vs_deletion=float(ratio[j]),pair_relative_error=float((r['error'][j]/r['energy'][j]).sqrt())) for j in selected]
result=dict(controls=control,candidate_pairs=len(ids),selected_disjoint_merges=len(selected),selected=records,best_error_vs_deletion=float(ratio.min()),summed_local_error_energy=float(r['error'][selected].sum()),joint_residual_energy=residual_energy,graph_relative_error=(residual_energy/target_energy)**.5,predictions=dict(pred_a_controls=True,pred_b_sixteen_reusable_pairs=len(selected)>=16,pred_c_combined_error_accounting="not exercised: zero edits selected" if not selected else "computed; independent joint replay still required"),scope='Approximate pair rank-one fits; top2048 cross-output pairs by input tensor cosine, eight ALS starts. No exhaustive/global optimum, no native evaluation. Combined coefficient error includes cross-edit terms. Linear and constant branches unchanged.')
out.write_text(json.dumps(result,indent=2)+'\n')
if selected:
 torch.save(dict(pairs=ids[selected],A=merged[0][:,selected],B=merged[1][:,selected],writers=merged[2][:,selected]),p/'MIDPOINT_SHARED_PRODUCT_MERGE_PROPOSALS_V1.pt')
print(json.dumps(result,indent=2))
