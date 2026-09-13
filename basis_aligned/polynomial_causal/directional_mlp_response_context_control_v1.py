"""CPU numerical equivalence and kernel-only amortization receipt."""
from pathlib import Path
import json,time,torch
from directional_mlp_bridge_v1 import execute
from directional_mlp_response_context_v1 import prepare,evaluate
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(241352);program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True);z=torch.randn(32,1152,dtype=torch.float64);baseline=torch.randn_like(z);amps=[torch.full((32,1),float(a),dtype=torch.float64) for a in torch.linspace(-1,1,24)]
 # Warm both schedules, then compare identical signed amplitude requests.
 execute(z,baseline,amps[0],program);evaluate(amps[0],prepare(z,baseline,program))
 tic=time.perf_counter();old=[execute(z,baseline,a,program) for a in amps];oldtime=time.perf_counter()-tic;tic=time.perf_counter();ctx=prepare(z,baseline,program);new=[evaluate(a,ctx) for a in amps];newtime=time.perf_counter()-tic;error=float((torch.stack(new)-torch.stack(old)).norm()/torch.stack(old).norm());zero=float(evaluate(torch.zeros_like(amps[0]),ctx).abs().max());result=dict(equivalence_error=error,zero_amplitude_maxabs=zero,amplitudes=24,context_positions=32,old_kernel_seconds=oldtime,prepared_kernel_seconds=newtime,measured_kernel_speedup=oldtime/newtime,matrix_applications_old=24,matrix_applications_prepared=1,scope='CPUkernel-only FP64 response equivalence onsyntheticcontexts. No actualmodelthroughput guarantee, nativecontextresponsevalidation stillqueued. BaselineMLPoutput/Jzpreparation costs included inpreparedkernel timer; originalbaselinegeneration external.')
 assert error<1e-10 and zero==0;(P/'DIRECTIONAL_MLP_RESPONSE_CONTEXT_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
