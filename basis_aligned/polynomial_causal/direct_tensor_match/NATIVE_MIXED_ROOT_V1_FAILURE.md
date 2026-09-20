# V1 instrument failure; V2 repair — 2026-09-20 18:52 UTC

V1 terminated with exit1 before emitting scientific results. `root_basis_search.symmetric_square` constructed its upper-triangle index and off-diagonal mask on CPU, then multiplied that mask with a CUDA tensor. CPU-only validation had not exercised this path. This is a device-placement bug, not evidence against mixed-root representations.

Repair: create indices/mask on S.device. V1 runner/log remain historical; freshV2 uses V2output paths and an explicit CPU/CUDA transform agreement tripwire before native work. Architecture, inputs, predictionbars and evaluationpanels unchanged. Helper was changed only after V1terminal status; no running/queuedjob imports thishelper except the freshV2. OriginalNATIVE_MIXED_ROOT_PLAN_V1.md remains scientific preregistration.
