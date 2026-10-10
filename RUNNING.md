# Running the analysis

Run these commands from the repository root. Python **3.11+** is required; the pinned direct dependencies were tested with Python **3.13.7**. No GPU or external dataset is needed.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

In Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The requirements file pins direct dependencies; it is not a full transitive lockfile.

## Command-line analysis

```bash
python scripts/run_analysis.py --output-dir outputs/coal
```

| Setting | Default | Meaning |
| --- | ---: | --- |
| `--data` | `data/coal_intervals_days.csv` | One-column chronological interval CSV |
| `--seed` | 1123 | SeedSequence entropy for independent NumPy PCG64 streams |
| `--steps` | 100000 | RJMCMC transitions per chain |
| `--burn` | 10000 | Initial transitions discarded per chain; also used for M1 burn-in |
| `--chains` | 4 | Independent RJMCMC chains |
| `--m1-samples` | 50000 | Retained M1 draws |
| `--rate-stride` | 20 | Subsampling interval for the rate-function summary only |

The program first samples the one-change-point model, then the reversible-jump chains, and saves figures and numerical results. The default run takes roughly a few minutes on a laptop. A shorter example is:

```bash
python scripts/run_analysis.py --steps 20000 --burn 5000 --chains 2 --m1-samples 10000 --output-dir outputs/quick
```

Short runs exercise the software and illustrate the method. Inspect the diagnostics before using their posterior frequencies as precise estimates. Changing the random seed produces another realization; changing settings or dependency versions can change the trajectory.

The first spawned random stream belongs to the M1 sampler and the remaining streams to the RJMCMC chains. The samplers use explicit generators and do not modify NumPy's global random state. Each RJMCMC chain starts at the constant-rate model with a rate drawn from its conditional posterior.

## Output files

| File | Contents |
| --- | --- |
| `summary.json` | Priors, seed, sample counts, evidence, model probabilities, scalar diagnostics and move acceptance |
| `chains_and_posteriors.npz` | M1 chain, complete RJMCMC states, scalar traces, model probabilities and rate summaries |
| `posterior_k.csv` | Probabilities for every $k=0,\ldots,30$, including unvisited models |
| `rate_function.csv` | Time grid, posterior mean and pointwise 5th, 25th, 75th and 95th rate percentiles |
| `fixed_models.png` / `.svg` | Recorded events and one-change-point posterior summaries |
| `reversible_jump.png` / `.svg` | Model frequencies and model-averaged rate |
| `trace_diagnostics.png` / `.svg` | Model-index and integrated-rate traces across chains |

Generated files go under `outputs/`, which is ignored by version control. Commands overwrite matching filenames; choose different output directories to retain multiple runs.

For array shapes and safe reconstruction of variable-length states, see [Output schema](docs/outputs.md).

## Notebooks

```bash
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name coal-changepoints --display-name "Python (coal change points)"
jupyter lab notebooks/01_fixed_dimension_models.ipynb
```

Select **Python (coal change points)** and run all cells from a fresh kernel. Both notebooks use the bundled data. The second notebook can inspect a completed command-line run by setting a results directory before opening Jupyter:

```bash
export COAL_RESULTS_DIR="$(pwd)/outputs/coal"
jupyter lab notebooks/02_reversible_jump.ipynb
```

Without `COAL_RESULTS_DIR`, it performs a shorter two-chain analysis. The notebook reports which mode and settings it uses. Unset the variable to request a fresh run.

## Reproducibility and interpretation

Every state has `k` ordered positions followed by `k+1` positive rates. Chain index zero is the initial state. The retained set is `chain[burn + 1:]`, containing exactly `steps - burn` transition draws. Model probabilities and diagnostics use all retained draws. Only the rate summary uses `--rate-stride`; this limits the memory needed for pointwise percentile calculations.

Rates are stored in events per day and plotted in events per 365.25-day year. Time plots show elapsed years from the first recorded event, not reconstructed calendar dates. Rate bands are pointwise posterior intervals, not prediction intervals for future counts or simultaneous bands for the entire curve.

The diagnostics report classical split R-hat and a scalar autocorrelation ESS estimate for model index and integrated intensity. They are not rank-normalized multivariate diagnostics and cannot alone establish mixing of every change-point configuration. Compare chains, inspect traces and extend runs when necessary. An unvisited model has an estimated frequency of zero, not proven zero posterior probability.

See [Numerical validation](docs/numerical_checks.md) for analytic references and sampler checks.

## Tests

```bash
python -m pytest -q
python -m pip check
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named coal_changepoints` | Activate the environment, install the package and select its notebook kernel. |
| Missing interval file | Run from the repository root or pass `--data` explicitly. |
| High R-hat or low ESS | Inspect trace behavior and run longer independent chains; do not rely only on acceptance rates. |
| JSON diagnostic is `null` | A constant trace has no estimable within-chain variance; this can happen in short runs. |
| Different results between runs | Compare random seeds, chain lengths, burn-in settings and dependency versions. |

The optional `scripts/export_coal_data.R` regenerates the public CSV from `boot` 1.3-32. R is not needed to run any Python analysis; see [Data provenance](docs/data.md).
