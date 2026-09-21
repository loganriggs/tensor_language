"""Exact centered-bilinear correction compilation and literal dense weight price."""
from pathlib import Path
import torch,json

def main():
 torch.set_num_threads(2);g=torch.Generator().manual_seed(26205)
 rand=lambda *s:torch.randn(*s,generator=g,dtype=torch.float64)
 n,m,Pn,Pm,Tn,Tm,W,D,nbar,mbar,b=rand(31,7),rand(31,7),rand(7,4),rand(7,4),rand(4,6),rand(4,6),rand(6,5),rand(6,5),rand(7),rand(7),rand(5)
 sn,sm=n@Pn,m@Pm;aa,bb=sn@Tn,sm@Tm;am,bm=nbar@Pn@Tn,mbar@Pm@Tm
 direct=(aa*bb)@W+b+((aa-am)*(bb-bm))@D
 Kn=-(Tn*bm)@D;Km=-(Tm*am)@D;bias=b+(am*bm)@D
 compiled=(aa*bb)@(W+D)+sn@Kn+sm@Km+bias
 err=float((direct-compiled).norm()/direct.norm());assert err<1e-12
 p=Path(__file__).resolve().parent
 result=dict(replay=err,products=512,shared_input_coefficients=2*(1152*256+256*512),dense_combined_output_coefficients=512*1152,linear_compensation_coefficients=2*256*1152,total_weight_coefficients=2*(1152*256+256*512)+512*1152+2*256*1152,scope='Exact algebra on dense toy. Price for dense compiled native dimensions; bias separate. No lowrank correction compression established. Same512 nonlinear products, two extra linear maps from existing256dim input features.')
 out=p/'MIDPOINT_CENTERED_CORRECTION_COMPILE_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
