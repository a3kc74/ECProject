"""
Main script for running TW-TSP optimization experiments.

Demonstrates usage of the complete framework including:
- Problem loading
- Algorithm configuration
- Benchmark execution
- Statistical analysis
- Visualization
"""
import os
import sys
import argparse
import pandas as pd

# Import problem and algorithms
from problem.tw_tsp import TWTSPProblem
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing

# Import utilities
from benchmark import Benchmark
from utils.visualizer import plot_convergence, plot_benchmark_results, plot_best_route


def parse_arguments():
    """
    Parse command-line arguments for TW-TSP optimization.
    """
    parser = argparse.ArgumentParser(
        description='TW-TSP Optimization Framework - Benchmark multiple algorithms on TSP with Time Windows',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithm', '-a',
        choices=['GA', 'SA', 'both'],
        default='both',
        help='Algorithm to run: GA (Genetic Algorithm), SA (Simulated Annealing), or both'
    )
    
    # Problem instance selection
    parser.add_argument(
        '--instance', '-i',
        type=str,
        default=None,
        help='Specific problem instance file to test (e.g., n20w20.001.txt). If not specified, runs all instances in data directory.'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data/SolomonPotvinBengio',
        help='Directory containing problem instance files'
    )
    
    parser.add_argument(
        '--num-runs', '-n',
        type=int,
        default=3,
        help='Number of independent runs per algorithm-instance combination'
    )
    
    # Genetic Algorithm parameters
    ga_group = parser.add_argument_group('Genetic Algorithm Parameters')
    ga_group.add_argument('--ga-population', type=int, default=150, help='GA population size')
    ga_group.add_argument('--ga-generations', type=int, default=2000, help='GA number of generations')
    ga_group.add_argument('--ga-mutation-rate', type=float, default=0.08, help='GA mutation rate')
    ga_group.add_argument('--ga-crossover-rate', type=float, default=0.92, help='GA crossover rate')
    ga_group.add_argument('--ga-tournament-size', type=int, default=20, help='GA tournament selection size')
    ga_group.add_argument('--ga-elitism-count', type=int, default=3, help='GA number of elite solutions preserved')
    
    # Simulated Annealing parameters
    sa_group = parser.add_argument_group('Simulated Annealing Parameters')
    sa_group.add_argument('--sa-initial-temp', type=float, default=1000.0, help='SA initial temperature')
    sa_group.add_argument('--sa-final-temp', type=float, default=0.1, help='SA final temperature')
    sa_group.add_argument('--sa-cooling-rate', type=float, default=0.96, help='SA cooling rate')
    sa_group.add_argument('--sa-iterations-per-temp', type=int, default=200, help='SA iterations per temperature')
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Directory to save result CSV files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """
    Main execution function for TW-TSP experiments.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    print("\n" + "=" * 70)
    print("TIME-WINDOWED TSP OPTIMIZATION FRAMEWORK")
    print("=" * 70 + "\n")
    
    # ========================================================================
    # Configure algorithms based on arguments
    # ========================================================================
    print("Step 1: Configuring algorithms...")
    
    ga_config = {
        'population_size': args.ga_population,
        'num_generations': args.ga_generations,
        'mutation_rate': args.ga_mutation_rate,
        'crossover_rate': args.ga_crossover_rate,
        'tournament_size': args.ga_tournament_size,
        'elitism_count': args.ga_elitism_count
    }
    
    sa_config = {
        'initial_temperature': args.sa_initial_temp,
        'final_temperature': args.sa_final_temp,
        'cooling_rate': args.sa_cooling_rate,
        'iterations_per_temp': args.sa_iterations_per_temp
    }
    
    algorithm_configs = {
        'GA': ga_config,
        'SA': sa_config
    }
    
    # Select algorithms based on argument
    algorithms = {}
    if args.algorithm in ['GA', 'both']:
        algorithms['GA'] = GeneticAlgorithm
        if args.verbose:
            print("  - Genetic Algorithm configured")
            for key, value in ga_config.items():
                print(f"      {key}: {value}")
    
    if args.algorithm in ['SA', 'both']:
        algorithms['SA'] = SimulatedAnnealing
        if args.verbose:
            print("  - Simulated Annealing configured")
            for key, value in sa_config.items():
                print(f"      {key}: {value}")
    
    print(f"  Registered {len(algorithms)} algorithm(s): {', '.join(algorithms.keys())}")
    num_runs = args.num_runs

    # ========================================================================
    # Define problem instances based on arguments
    # ========================================================================
    print("\nStep 2: Setting up problem instances...")
    
    problem_files = []
    
    if args.instance:
        # Specific instance file provided
        instance_path = os.path.join(args.data_dir, args.instance)
        if os.path.exists(instance_path):
            problem_files = [instance_path]
            print(f"  ✓ Using specific instance: {args.instance}")
        else:
            print(f"  ✗ Instance file not found: {instance_path}")
            print(f"  ⚠ Using mock data instead")
            problem_files = [None]
    else:
        # Run all instances in directory
        if os.path.exists(args.data_dir):
            all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.txt')]
            problem_files = [os.path.join(args.data_dir, f) for f in sorted(all_files)]
            
            if problem_files:
                print(f"  ✓ Found {len(problem_files)} problem instances in {args.data_dir}")
                if args.verbose:
                    for i, pf in enumerate(problem_files, 1):
                        print(f"     {i}. {os.path.basename(pf)}")
            else:
                print(f"  ⚠ No .txt files found in {args.data_dir}")
                problem_files = [None]
        else:
            print(f"  ✗ Directory not found: {args.data_dir}")
            print(f"  ⚠ Using mock data instead")
            problem_files = [None]
    
    print(f"  Total problems to test: {len(problem_files)}")
    
    # ========================================================================
    # Initialize and run benchmark
    # ========================================================================
    print("\nStep 3: Running benchmark...")
    
    benchmark = Benchmark(
        algorithms=algorithms,
        problem_paths=problem_files,
        num_runs=num_runs,
        algorithm_configs=algorithm_configs
    )
    
    # Execute benchmark
    raw_results = benchmark.run()
    
    # Calculate statistics
    stats_results = benchmark.get_statistics(raw_results)
    
    # ========================================================================
    # Display results
    # ========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK STATISTICS")
    print("=" * 70)
    print("\n", stats_results.to_string(index=False))
    print("\n" + "=" * 70)
    
    # Display detailed breakdown for best solutions
    if args.verbose:
        print("\nDETAILED SOLUTION ANALYSIS")
        print("=" * 70)
        
        # Get problem name
        if problem_files[0]:
            problem_name = os.path.basename(problem_files[0])
        else:
            problem_name = "mock_instance"
        
        # Get problem instance from benchmark
        problem = benchmark.get_problem(problem_name)
        
        for algo_name in algorithms.keys():
            best_solution = benchmark.get_best_solution(algo_name, problem_name)
            if best_solution:
                details = problem.calculate_solution_details(best_solution)
                print(f"\n{algo_name} - Best Solution Details:")
                print(f"  Total Distance (pure):    {details['total_distance']:>12.2f}")
                print(f"  Time Window Violations:   {details['num_violations']:>12d}")
                print(f"  Penalty Amount:           {details['total_penalty']:>12.2f}")
                print(f"  Final Fitness:            {details['fitness']:>12.2f}")
                if details['num_violations'] == 0:
                    print(f"  ✓ Feasible solution (no violations)")
                else:
                    print(f"  ✗ Infeasible solution ({details['num_violations']} violations)")
        
        # Display best-known solution comparison if available
        if problem.has_best_known_solution():
            print("\n" + "-" * 70)
            print("COMPARISON WITH BEST-KNOWN SOLUTION")
            print("-" * 70)
            best_known = problem.get_best_known_solution()
            print(f"\nBest-Known Solution (from literature):")
            if best_known['reported_cost'] is not None:
                print(f"  Reported Cost:            {best_known['reported_cost']:>12.2f}")
            if best_known['fitness'] is not None:
                print(f"  Calculated Fitness:       {best_known['fitness']:>12.2f}")
            if best_known['distance'] is not None:
                print(f"  Calculated Distance:      {best_known['distance']:>12.2f}")
            if best_known['violations'] is not None:
                print(f"  Calculated Violations:    {best_known['violations']:>12d}")
            
            print(f"\nGap Analysis (vs Calculated Fitness):")
            for algo_name in algorithms.keys():
                best_solution = benchmark.get_best_solution(algo_name, problem_name)
                if best_solution:
                    details = problem.calculate_solution_details(best_solution)
                    if best_known['fitness'] is not None and best_known['fitness'] > 0:
                        gap = ((details['fitness'] - best_known['fitness']) / best_known['fitness']) * 100
                        print(f"  {algo_name:8s} Gap: {gap:>8.2f}% (Fitness: {details['fitness']:.2f})")

    
    print("\n" + "=" * 70)
    
    # ========================================================================
    # Save results to CSV with algorithm config in filename
    # ========================================================================
    print("\nStep 4: Saving results...")
    
    results_dir = args.output_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Generate algorithm config string for filename
    algo_names = '_'.join(algorithms.keys())
    config_str = f"pop{ga_config['population_size']}_gen{ga_config['num_generations']}"
    
    # Create filenames with algorithm and config info
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    raw_results_filename = f'raw_results_{algo_names}_{config_str}_{timestamp}.csv'
    stats_filename = f'statistics_{algo_names}_{config_str}_{timestamp}.csv'
    
    raw_results_path = os.path.join(results_dir, raw_results_filename)
    stats_results_path = os.path.join(results_dir, stats_filename)
    
    # Save raw results with all columns
    raw_results.to_csv(raw_results_path, index=False)
    print(f"  ✓ Saved raw results: {raw_results_filename}")
    if args.verbose:
        print(f"    Columns: {', '.join(raw_results.columns)}")
    
    # Save statistics
    stats_results.to_csv(stats_results_path, index=False)
    print(f"  ✓ Saved statistics: {stats_filename}")
    if args.verbose:
        print(f"    Columns: {', '.join(stats_results.columns)}")

    
    # # ========================================================================
    # # TODO 6: Visualization
    # # ========================================================================
    # print("\nStep 5: Generating visualizations...")
    
    # plots_dir = 'plots'
    # if not os.path.exists(plots_dir):
    #     os.makedirs(plots_dir)
    
    # # Get problem name for visualization
    # if problem_files[0]:
    #     problem_name = os.path.basename(problem_files[0])
    # else:
    #     problem_name = 'mock_instance'
    
    # # 6.1: Plot convergence curves
    # print("  Generating convergence plot...")
    # history_data = {}
    # for algo_name in algorithms.keys():
    #     history = benchmark.get_convergence_history(algo_name, problem_name, run=1)
    #     if history:
    #         history_data[algo_name] = history
    
    # if history_data:
    #     convergence_path = os.path.join(plots_dir, 'convergence_comparison.png')
    #     plot_convergence(
    #         history_data=history_data,
    #         title=f'Convergence Comparison on {problem_name}',
    #         save_path=convergence_path
    #     )
    
    # # 6.2: Plot benchmark box-plot
    # print("  Generating benchmark comparison plot...")
    # benchmark_path = os.path.join(plots_dir, 'benchmark_boxplot.png')
    # plot_benchmark_results(
    #     results_df=raw_results,
    #     title='Algorithm Performance Comparison',
    #     save_path=benchmark_path
    # )
    
    # # 6.3: Plot best route for each algorithm
    # print("  Generating route visualizations...")
    
    # # Get problem instance from benchmark (already loaded)
    # if problem_files[0]:
    #     problem_name = os.path.basename(problem_files[0])
    # else:
    #     problem_name = 'mock_instance'
    
    # problem = benchmark.get_problem(problem_name)
    
    # for algo_name in algorithms.keys():
    #     best_solution = benchmark.get_best_solution(algo_name, problem_name)
    #     best_fitness = benchmark.get_best_fitness(algo_name, problem_name)
        
    #     if best_solution:
    #         route_path = os.path.join(plots_dir, f'best_route_{algo_name}.png')
    #         plot_best_route(
    #             problem=problem,
    #             solution=best_solution,
    #             title=f'Best Route - {algo_name} (Fitness: {best_fitness:.2f})',
    #             save_path=route_path
    #         )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n📊 Benchmark Summary:")
    print(f"  - Algorithms tested: {len(algorithms)} ({', '.join(algorithms.keys())})")
    print(f"  - Problems tested: {len(problem_files)}")
    print(f"  - Runs per config: {num_runs}")
    print(f"  - Total experiments: {len(algorithms) * len(problem_files) * num_runs}")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print(f"  - Raw results: {raw_results_filename}")
    print(f"  - Statistics: {stats_filename}")
    
    if args.verbose:
        print(f"\n📈 CSV File Columns:")
        print(f"  Raw Results: {', '.join(raw_results.columns)}")
        print(f"  Statistics: {', '.join(stats_results.columns)}")
    
    print("\n" + "=" * 70 + "\n")



if __name__ == "__main__":
    main()

