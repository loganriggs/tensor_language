import itertools,json
from pathlib import Path
import torch
from shared_quadratic_bank import bank_gram
P=Path(__file__).resolve().parent

def qgram(U,V,X,Y):
 E=torch.cat([U,V],1);F=torch.cat([V,U],1)/2;A=torch.cat([X,Y],1);B=torch.cat([Y,X],1)/2;t=torch.einsum('aid,bjd->abij',F,A);s=torch.einsum('bid,ajd->baij',B,E);return torch.einsum('abij,baji->ab',t,s)

def inverse_sqrt(G):
 ev,V=torch.linalg.eigh(G);return (V*ev.clamp_min(ev.max()*1e-12).rsqrt())@V.T

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);source=torch.load(P/'NATIVE_LEARNED_SHARED_BANK_V1.pt',weights_only=True)['students'];keys=list(source);models={key:{n:a.double() for n,a in value.items()} for key,value in source.items()};pairs=list(zip(*torch.triu_indices(8,8).tolist()));first=[pairs.index((i,j)) for i in range(4) for j in range(i,4)];second=[pairs.index((i+4,j+4)) for i in range(4) for j in range(i,4)];rows=[]
 # Independently verify quadratic crossGram onsmall denseforms.
 torch.manual_seed(1816);U,V,X,Y=[torch.randn(3,2,5) for _ in range(4)];Q=(U.transpose(-1,-2)@V+V.transpose(-1,-2)@U)/2;R=(X.transpose(-1,-2)@Y+Y.transpose(-1,-2)@X)/2;error=float((qgram(U,V,X,Y)-Q.flatten(1)@R.flatten(1).T).abs().max());assert error<1e-11
 for ka,kb in itertools.combinations(keys,2):
  a,b=models[ka],models[kb];G=bank_gram(torch.cat([a['U'],b['U']]),torch.cat([a['V'],b['V']]));Ga=G[first][:,first];Gb=G[second][:,second];cross=G[first][:,second];na=((a['C'].T@a['C'])*Ga).sum();nb=((b['C'].T@b['C'])*Gb).sum();inner=((a['C'].T@b['C'])*cross).sum();cos=float(inner/(na*nb).sqrt());qa=qgram(a['U'],a['V'],a['U'],a['V']);qb=qgram(b['U'],b['V'],b['U'],b['V']);qab=qgram(a['U'],a['V'],b['U'],b['V']);principal=torch.linalg.svdvals(inverse_sqrt(qa)@qab@inverse_sqrt(qb));assert abs(cos)<=1+1e-8 and float(principal.max())<=1+1e-8;rows.append(dict(first=list(ka),second=list(kb),polynomial_cosine=cos,quadratic_span_principal_cosines=principal.tolist(),first_norm=float(na.sqrt()),second_norm=float(nb.sqrt())))
 out=dict(records=rows,quadratic_gram_dense_check=error,scope='Exact coefficient-function angles and gauge-invariant quadratic-span overlap across8nativefits. Stability of approximations only; no semantic/causalidentity claim. Poorfit/budget effects retained.')
 (P/'LEARNED_BANK_STABILITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');high=[r for r in rows if r['first'][1]==.005 and r['second'][1]==.005];print('high-rate function cosines',[(r['first'],r['second'],r['polynomial_cosine'],min(r['quadratic_span_principal_cosines'])) for r in high])
if __name__=='__main__':main()
