'''Run the coal-mining change-point analysis from the repository root.'''

import argparse
import json
from pathlib import Path

from coal_changepoints.pipeline import run_analysis


def main() -> None:
    '''Parse sampling settings and save the analysis outputs.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/coal_intervals_days.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/coal'))
    parser.add_argument('--seed', type=int, default=1123)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--burn', type=int, default=10000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--m1-samples', type=int, default=50000)
    parser.add_argument('--rate-stride', type=int, default=20)
    args = parser.parse_args()
    summary = run_analysis(args.data, args.output_dir, seed=args.seed, steps=args.steps,
                           burn=args.burn, n_chains=args.chains, m1_samples=args.m1_samples,
                           rate_stride=args.rate_stride, progress=lambda message: print(message, flush=True))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
