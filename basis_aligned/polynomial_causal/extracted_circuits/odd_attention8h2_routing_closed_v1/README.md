# Routing-closed inherited-city head8.2 edge

This package extends the rank-one edge with exact head8.2 city routing. It takes
the normalized block8 current state, the city position, and recipient/donor
normalized city-token embeddings. It computes both normalized BF16-rotated QK
dot products, multiplies them, reconstructs the inherited city value, and emits
the destination-by-residual head8.2 write.

Across96 native rows, routing replay is within1.62e-7 and the composed write is
within2.31e-7. `CONTROL.json` replays one native context per template. The
program stores884,737 scalars.

The interface price is deliberately explicit. If block8 current is transmitted
only for this edge, the runtime input is8.34--10.00% larger than the dense write
at20--24 tokens. If block8 current is already a shared state port, the rank-one
output factorization retains its94.91--95.75% incremental write-interface
saving. Block8 state generation, head9.8 O, and the suffix remain external.
