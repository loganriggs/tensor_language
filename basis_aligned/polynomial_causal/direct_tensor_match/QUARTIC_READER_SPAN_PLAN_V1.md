Measure the native quartic function's sensitivity outside the learned leaf span.

Path is pure MLP16→MLP17→QR-reduced unembedding, retaining all products but excluding residual/bias terms and normalization between them. Fixed student is the learned32x4 inherited-long bank. Its128 left plus128 right linear forms span at most256 of1152 coordinates. Any function of those readers is invariant to all orthogonal perturbations, regardless of its output writer or root graph edits.

At16 evenly spaced rows in calibration panel0, compute exact native JacobianJ by the bilinear chain rule. Q is an orthonormal basis of all learned readers. Unread response is Jperp=J-(JQ)Q^T. Measure ||Jperp||F/||J||F at each anchor and in aggregate. Sum Jperp^TJperp to identify the size of the missing sensitivity subspace. This is input-isotropic local derivative geometry, not a natural-functional or semantic lower bound.

Pred_a_integrity: explicitJacobian/autograd toy<1e-12; nativefirstanchorFP32vsFP64<1e-3; fixedstudent invariance under nullspace perturbations<1e-5relative. Pred_b_missing: aggregateunreadJacobianfraction>.10 and at least12/16anchors>.10. Pred_c_concentrated: at most32 new directions capture90%of unread Jacobian squared energy across these anchors. Null: unread response small or diffuse. If diffuse, adding a few readers cannot recover most of this local sensitivity at frozenexistingreaders; rotating the whole dictionary remains allowed.

Use deterministic projectedGaussian directions at eachanchor, norm.01||x||; evaluate actual finite teacherchange and studentchange. No outputfit or candidate adoption. Baseline fullreaderbasis haszero unread response byidentity; planted visible andinvisible quartics validate both extremes. Price remains656products903168coefficients; this is a diagnostic, not exportedadditionalfeatures. GPUmanagedonly, oneboundedjob.

Prior-art distinction: GRADIENT_READER_ADDITION_PLAN_V1 examined a selected scalar program with12readers and fitted added cross terms. This experiment diagnoses the complete1152-output purequartic target under the newer256-reader bank; it does not repeat the old scalar intervention claim.
