# Output schema

The compressed NumPy archive `chains_and_posteriors.npz` contains numeric arrays only. Load it with `allow_pickle=False`.

| Key | Shape and meaning |
| --- | --- |
| `m1_chain` | `(m1_samples, 3)`; position in days, before-change rate and after-change rate in events/day |
| `k_chains` | `(n_chains, steps+1)`; model index, including the initial state |
| `integrated_rate_chains` | Same shape; integral of each sampled intensity over the observation window, in expected events |
| `k_values` | `(31,)`; model indices from 0 to 30 |
| `k_probabilities` | `(31,)`; normalized frequencies from all retained draws |
| `chain_i_values` | Flat concatenation of every state in chain `i` |
| `chain_i_offsets` | `(steps+2,)`; boundaries into the corresponding flat array |
| `rate_time_days` | `(400,)`; equally spaced times from zero to the observation duration |
| `rate_mean` | `(400,)`; posterior mean intensity in events/day |
| `rate_p5`, `rate_p25`, `rate_p75`, `rate_p95` | `(400,)`; pointwise intensity percentiles in events/day |

A variable-length state has the layout `[s1, ..., sk, h0, ..., hk]`. Reconstruct state `j` from chain zero as follows:

```python
import numpy as np

with np.load("outputs/coal/chains_and_posteriors.npz", allow_pickle=False) as archive:
    offsets = archive["chain_0_offsets"]
    values = archive["chain_0_values"]
    j = 10001
    state = values[offsets[j]:offsets[j + 1]]
    k = (len(state) - 1) // 2
    positions, rates = state[:k], state[k:]
```

The first state is the initialization. If `burn=10000`, state index 10001 is the first retained transition draw. `posterior_k.csv` uses all retained states; `rate_function.csv` uses the explicitly recorded `rate_stride` subsample. There is no further thinning for model probabilities or scalar diagnostics.

Move statistics in `summary.json` count attempts and accepts over the complete transition run, including burn-in. M1 position acceptance likewise includes its burn-in. Diagnostic JSON null values indicate an undefined statistic, rather than a zero estimate.

Trace figures display at most about 4,000 points per chain; their title records the display stride. Saved chains and computed scalar diagnostics retain every transition draw.
