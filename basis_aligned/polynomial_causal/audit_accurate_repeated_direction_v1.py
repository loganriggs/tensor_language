"""Independent gradient-block signal/noise accounting; no candidate selection."""
from pathlib import Path
import hashlib,json
import torch
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);path=P/'ACCURATE_REPEATED_DIRECTION_V1_ARTIFACT.pt'
    data=torch.load(path,weights_only=True);a,b=[x.flatten(1) for x in data['blocks']];n,m=len(a),len(b)
    cross=a@b.T;rows=cross.mean(1);cols=cross.mean(0);signal=cross.mean()
    interaction=cross-rows[:,None]-cols[None,:]+signal
    interaction_var=interaction.square().sum()/((n-1)*(m-1))
    variance=rows.var(unbiased=True)/n+cols.var(unbiased=True)/m-interaction_var/(n*m)
    noise=[float((x-x.mean(0)).square().sum()/((len(x)-1)*len(x))) for x in (a,b)]
    leave=[]
    for j in range(1,4):
        deltas=[e[:,0]-e[:,j] for e in data['validation'].values()]
        total=sum(float(d.mean()) for d in deltas)/4
        # Omit the same numbered64-probe batch from every independently drawn grade.
        estimates=[sum(float((d.sum()-d[i:i+64].sum())/(len(d)-64)) for d in deltas)/4 for i in range(0,len(deltas[0]),64)]
        leave.append(dict(step_index=j,mean_improvement=total,leave_batch_out_range=[min(estimates),max(estimates)]))
    result=dict(probes_per_gradient_grade=65536,independent_blocks=[n,m],cross_replica_signal_squared_estimate=float(signal),signal_variance_estimate=float(variance),signal_standard_error_estimate=float(variance.clamp_min(0).sqrt()),mean_gradient_noise_squared=noise,mean_gradient_norm_squared=[float(x.mean(0).square().sum()) for x in (a,b)],interaction_variance_estimate=float(interaction_var),validation=leave,artifact_sha=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Cross mean is unbiased for squared mean gradient under independent identically distributed blocks. Variance uses two-way row/column/interaction accounting; its finite-sample estimate can be negative and is preserved. No Gaussian confidence guarantee, no claim of zero true gradient, and no inference about alternative stationary points or native circuit behavior.')
    out=P/'ACCURATE_REPEATED_DIRECTION_V1_BLOCK_AUDIT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
