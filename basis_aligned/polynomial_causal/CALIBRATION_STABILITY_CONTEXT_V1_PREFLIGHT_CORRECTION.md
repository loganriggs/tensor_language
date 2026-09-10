# Pre-execution gate correction

The first enqueue was rejected before model-free execution or queue insertion: the static gate only recognizes quoted pred_* mapping keys, not dict keyword arguments. Changed the three prediction keys to an explicit dictionary literal; formulas, data, and scientific bars are unchanged. Prior runner bytes remain in commit bbced3733. No native run took place under that hash.
