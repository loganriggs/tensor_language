# Induction typed consumer group response V4 evidence serialization

V3 passed its corrected instrument and found no DISCOVERY-eligible group, but its result serialized only each candidate's vocabulary-RMS reductions. It omitted the candidate CE-damage and correct-fraction cell reports that are also part of eligibility. The terminal is therefore not independently auditable from the result artifact alone.

V4 changes only result serialization: save the complete DISCOVERY and CONFIRM cell reports for all eight frozen candidate groups in addition to the existing reductions. Runtime, 33/1,056 price, rows, interventions, candidate groups, metrics, thresholds, ranking, and no-fallback logic remain unchanged. V3 remains a runtime-valid observation but will not be registered as scientific authority; V4 is authoritative only if its stored reports reproduce selection and terminal exactly.
