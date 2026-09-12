# Current versus shared-first values

FrozenA8/A9/A13group; original32prompts2families. HoldrecipientjointQKrouting.
Fourvertices: native,currentvaluedonor,firstvaluedonor,bothvaluesdonor. Useexact
foldedE/H maps, actualsignedmixing,recipientRMS17anddownstreamports. Fullnative
lastMLP/readout evaluated afterproducer-swap write. No datafit.
A:nativeproducer/attention/recipientandpriorbothvaluedonorrelativeerrors<=1e-5.
B:A andbothvaluetransfer>=90%priorfullgrouptransfer,>=5/8positive EACHfamily.
C:A/B andcurrentonlytransfer>=90%bothvalues,unrelatedmeanabs<=.5regional EACHfamily.
Null: firststream contributes substantially, so currentdominance fails. Preserve
misses. Earlierrole/groupoutcomesknown; current/firstoutcomes uninspected.
Bothvalues must replay priorroute/valuearm2, not priorfullgroupswaparm3.
Price7bodyforwards32rows4–12tokens,4suffixarms,3producerQKrecomputations perbatch,
180sec managedGPUcap,~1.5MBreceipt, no optimization. Firststream input is normalized
attention0input; do not replace it by unnormalized embeddings.
