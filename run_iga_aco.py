#!/usr/bin/env python3
"""
Run IGA-ACO Hybrid algorithm via the Benchmark harness.

This runner handles the hyphenated module name by using importlib to load
`algorithms.iga-aco` and then runs the algorithm across instances using
the existing `Benchmark` class so outputs match project conventions.
"""
import argparse
import os
import importlib
import pandas as pd
from benchmark import Benchmark


def parse_args():
    p = argparse.ArgumentParser(description='Run IGA-ACO via Benchmark')

    p.add_argument('--instance', '-i', type=str, default=None,
                   help='Specific instance filename inside data dir')
    p.add_argument('--data-dir', '-d', type=str, default='data/SolomonPotvinBengio',
                   help='Directory holding instance files')
    p.add_argument('--num-runs', '-n', type=int, default=3,
                   help='Number of independent runs per instance')

    # IGA params
    p.add_argument('--num-iterations', type=int, default=500)
    p.add_argument('--iga-population', type=int, default=100)
    p.add_argument('--iga-elite-ratio', type=float, default=0.1)
    p.add_argument('--iga-vnd-probability', type=float, default=0.3)

    # ACO params
    p.add_argument('--aco-num-ants', type=int, default=50)
    p.add_argument('--aco-alpha', type=float, default=1.0)
    p.add_argument('--aco-beta', type=float, default=2.0)
    p.add_argument('--aco-rho', type=float, default=0.1)
    p.add_argument('--aco-q0', type=float, default=0.9)

    # Exchange
    p.add_argument('--exchange-interval', type=int, default=10)

    p.add_argument('--output-dir', '-o', type=str, default='results')
    p.add_argument('--verbose', '-v', action='store_true')

    return p.parse_args()


def main():
    args = parse_args()

    # Import the hyphenated module using importlib
    iga_module = importlib.import_module('algorithms.iga-aco')
    IGAACOHybrid = iga_module.IGAACOHybrid

    # Build config for algorithm (keys expected by algorithms/iga-aco.py)
    config = {
        'num_iterations': args.num_iterations,
        'iga_population_size': args.iga_population,
        'iga_elite_ratio': args.iga_elite_ratio,
        'iga_vnd_probability': args.iga_vnd_probability,
        'aco_num_ants': args.aco_num_ants,
        'aco_alpha': args.aco_alpha,
        'aco_beta': args.aco_beta,
        'aco_rho': args.aco_rho,
        'aco_q0': args.aco_q0,
        'exchange_interval': args.exchange_interval,
    }

    # Prepare problem files list
    problem_files = []
    if args.instance:
        instance_path = os.path.join(args.data_dir, args.instance)
        problem_files = [instance_path]
    else:
        if os.path.exists(args.data_dir):
            files = [f for f in os.listdir(args.data_dir) if f.endswith('.txt')]
            problem_files = [os.path.join(args.data_dir, f) for f in sorted(files)]
        else:
            problem_files = [None]

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Benchmark and run
    algorithms = {'IGAACO': IGAACOHybrid}
    benchmark = Benchmark(
        algorithms=algorithms,
        problem_paths=problem_files,
        num_runs=args.num_runs,
        algorithm_configs={'IGAACO': config}
    )

    raw_results = benchmark.run()
    stats = benchmark.get_statistics(raw_results)

    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    config_str = f"pop{args.iga_population}_iter{args.num_iterations}_ants{args.aco_num_ants}"
    raw_path = os.path.join(args.output_dir, f'raw_results_IGAACO_{config_str}_{timestamp}.csv')
    stats_path = os.path.join(args.output_dir, f'statistics_IGAACO_{config_str}_{timestamp}.csv')

    raw_results.to_csv(raw_path, index=False)
    stats.to_csv(stats_path, index=False)

    if args.verbose:
        print(f"Saved raw results to: {raw_path}")
        print(f"Saved statistics to: {stats_path}")


if __name__ == '__main__':
    main()
