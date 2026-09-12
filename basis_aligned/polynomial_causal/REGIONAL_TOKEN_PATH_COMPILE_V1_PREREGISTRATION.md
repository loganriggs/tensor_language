# Exact token-to-attention path contraction

Fresh32behavior-panelrows; changedcueposition1verified. Frozen54firstvaluereads
indexed producerlayerj in8/9/13,headh andchilda. Foldtheconditional paths:
K[j,h,a,o]=sum_t routing_j[h,t,1]*downstream[t,a,o], where downstream includes
parent17, bothQK17gatefactors, privatequerywriters and1/rho17. Then delta_write
=sum_jha K*delta_firstvalue[j,h,a]. No fitting or ranktruncation.

A:nativeproducer/attentionreplay<=1e-5 and compiledactualdonorwrite<=1e-10relative.
B:A andindependent synthetic54portperturbation versusdirecttwo-stagewrite<=1e-10.
C:A/B andcompiledsuffixcontrast replay versus savedfirstswap<=1e-5relative.
Syntheticseed61207,normalstd.1in54portspace; notclaimedreachabletoken edits.
BothQK factors remainjoint. Fixedreceiverconditions arepartoftheinterface.

Price6bodyforwards32rows,2suffixarmspluscompiledsuffix,180secmanagedcap;
Kis32x54x1152FP64inmemory (~16MB),62,208numbers percontext. Saveonlyfirsttwo
representativeKmatrices plus smallwrite/marginreceipts (~2MB). Theseoperators
are not shared acrosscontexts or freeweights: Kconstructionstillrequiresnative
upstreamrouting anddownstreamports. No smallerindependentprogram claim.
Prior normalizedresponse/attentionpathalgebrachecked; novelty isthis validated
multi-producer-to-final-attention54portcontraction,notbilinearityitself.

Queue schema repair before native execution: split existing conjunctive checks
into required A/B/C fields; all numerical bars and test data unchanged.
