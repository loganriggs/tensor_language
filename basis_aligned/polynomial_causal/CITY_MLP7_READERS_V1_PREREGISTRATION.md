# Exact reader fold through MLP7, retaining the coupled response

CITY_SOURCE7_SPLIT_NULL_V1 failed composition specificity (0.346 interaction,
random median0.245, beats0/16). Retain the full coupled city-response model.
The selective MLP7-dependent source effect motivates tracing that input deeper;
it does not justify independent MLP7/complement modules.

For W=stack(K1_8.2,K2_8.2,Vcurrent_8.2), shape384x1152, MLP7 output
m=D[(Lz)*(Rz)]+b gives Wm=C[(Lz)*(Rz)]+c with C=WD, c=Wb.
L,R=[4608,1152], C=[384,4608], c=[384]. z is the native normalized MLP7
input at the city position. Source lambda8,0, initial/context additions, mixed8
RMS and key RMS, query factors, rotary, inherited values and suffix stay external.
No averaging or fitting. This folds all three useful readers through a complete
native bilinear output map rather than selecting weight-magnitude terms.

Compute C/c inFP64 and storeFP32, including L/R in the program's literal price.
CPU synthetic identity<=1e-10 relative; rounded program<=1e-4 relative. This is
weight/implementation evidence, not predictive identification on text.

Native check on all40opened Pile sequences,40body forwards/120seconds:
- a: unedited spelling/control scores replay prior native<=1e-5abs AND relative.
- b: folded reader output against native Wm has relative error<=1e-4 on EVERY
  sequence, and combined error<=1e-4 separately for each128-dimensional reader.
- c: all tensors finite, exactly40forwards/fixtures, and program contains fewer
  floating scalars and bytes than L,R,D,b plus the three unfused reader matrices.

Capture normalized MLP7 city inputs and native projected reader outputs. Export
does not close the upstream normalized-state dependency or certify replacing the
entire downstream circuit. Next native downstream replacement must be tested
separately before adopting a circuit implementation.
