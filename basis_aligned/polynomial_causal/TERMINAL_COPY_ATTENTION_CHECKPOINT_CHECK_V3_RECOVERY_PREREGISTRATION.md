# Terminal-copy attention checkpoint check v3 recovery

V2 is spent with authority SHA256
`8d84f8a568fa84d5b2dbce2f17e9b3c65ab8fb1966bbe616105ca55d17b4475f` and
failure SHA256
`cc1a57303e51bc838a094adfddd202c6d278af963520da23a94c99e6d738bc72`.
The contraction-layout repair did not change the failure. A subsequent read-only
checkpoint diagnostic localized the discrepancy before opening v3:

- a direct replay written with the checkpoint's own operators is bit-identical;
- copied Q projections are bit-identical;
- rotated Q differs by relative `0.00623`;
- the final unpartitioned write differs by relative `0.00810`.

The cause is exact: `Rotary.inv_freq` is a plain float32 attribute, not a registered
buffer. `model.to(bfloat16)` therefore leaves it float32, while the old adapter's
blanket `.to(device,dtype)` cast the owned copy to bfloat16. V3 changes only adapter
construction from device-and-dtype conversion to device-only movement. Every copied
tensor thus preserves its source dtype: projection weights remain bfloat16 and rotary
frequencies remain float32. A synthetic regression test now freezes this property.

V3 otherwise reuses v2's analytically bounded relative recomposition rule and the
exact v1 checkpoint, seed, layer, shape, price, and lifecycle semantics. It binds all
spent v1/v2 authorities and failures. A pass remains an engineering result only and
does not complete E4.

