# Upstream mixed-state v2: audit the registered rounded intervention correctly

V1 is preserved as invalid. Its FP64 mode removal is exact and parent/causal
replays are zero, but FP32 state/mode error .0001220703125 exceeds the fixed
absolute .0001/.00001 instrument limits. No scientific verdict is adopted
from V1. Its outcomes are opened, so this repeat is a numerical verification,
not fresh hypothesis testing.

Same boundary, cells, heads, intervention, model and B/C scientific predicates
as BILIN18_UPSTREAM_MIXED_COMMAND_STATE_V1_PREREGISTRATION.md. The specified
intervention already returns FP32; its deployed target remains nearest-FP32
rounding of the exact FP64 mixed-mode projection. Do not alter state construction,
select a head, change boundary or claim an exact zero after rounding.

Replace only V1's two fixed absolute deployed-state requirements with a
per-coordinate floating-point envelope. For ideal state x and nearest FP32
y, let b=half max(nextafter(y,+infinity)-y, y-nextafter(y,-infinity)). Require
actual=y bitwise, |actual-x|<=b at every coordinate, all finite, and preserved
mode relative RMS<=1e-5 as before. The error of ANY balanced command mode is
bounded by (b00+b01+b10+b11)/4 by the triangle inequality. Check that bound
coordinatewise, allowing only8*eps64*sum_uv|ideal_mode_uv| for evaluating the
FP64 mode comparison. This covers zero mixed mode without dividing by zero.
The maximum bounds and observed errors must be reported, not hidden.

The existing exact FP64 removal/preservation bars remain abs1e-9/relative1e-10.
Native parent replay, causal order, call counts, restoration, B mixed-read
retention<=.1, and C mixed-output retention<=.1 plus each other-mode error<=.01
remain unchanged. Model and output precision remain deployedFP32. This audit
proves correct rounding of the specified edit, not a bound on an imaginary
FP64 network's downstream behavior.

Four CPU controls already pass: large-state projection exceeds the old absolute
bar while satisfying the rounding envelope; one-ULP corruption is rejected;
exactly representable states pass. ULP is the gap between adjacent representable
floating-point values. On the planted fixture max mode error is .0001220703125.
This provides a reproducible precision diagnosis independent of the trained
model's scientific outcome.

Run64 full forwards/256sequences/384 extra local contractions through managedGPU;
no fit/update, all545902902 parameters and four-cell context charged. Parent
V1 result and rounding audit hashes are frozen. Preserve both result files.
