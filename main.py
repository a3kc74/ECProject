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
import pandas as pd

# Import problem and algorithms
from problem.tw_tsp import TWTSPProblem
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing

# Import utilities
from benchmark import Benchmark
from utils.visualizer import plot_convergence, plot_benchmark_results, plot_best_route


def main():
    """
    Main execution function for TW-TSP experiments.
    """
    print("\n" + "=" * 70)
    print("TIME-WINDOWED TSP OPTIMIZATION FRAMEWORK")
    print("=" * 70 + "\n")
    
    # ========================================================================
    # TODO 1: Define algorithm configurations
    # ========================================================================
    print("Step 1: Configuring algorithms...")
    
    ga_config = {
        'population_size': 200,
        'num_generations': 500,
        'mutation_rate': 0.15,
        'crossover_rate': 0.85,
        'tournament_size': 50,
        'elitism_count': 10
    }
    
    sa_config = {
        'initial_temperature': 1000.0,
        'final_temperature': 0.1,
        'cooling_rate': 0.95,
        'iterations_per_temp': 50
    }
    
    algorithm_configs = {
        'GA': ga_config,
        'SA': sa_config
    }

    num_runs = 1  # Number of independent runs per configuration
    
    print("  - Genetic Algorithm configured")
    for key, value in ga_config.items():
        print(f"      {key}: {value}")
    print("  - Simulated Annealing configured")
    for key, value in sa_config.items():
        print(f"      {key}: {value}")

    # ========================================================================
    # TODO 2: Define problem instances
    # ========================================================================
    print("\nStep 2: Setting up problem instances...")

    # Define the data directory
    data_dir = 'data/SolomonPotvinBengio'
    
    # Automatically crawl all .txt files in the directory
    if os.path.exists(data_dir):
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        problem_files = [os.path.join(data_dir, f) for f in sorted(all_files)]
        
        if problem_files:
            print(f"  ✓ Found {len(problem_files)} problem instances in {data_dir}")
            for i, pf in enumerate(problem_files, 1):
                print(f"     {i}. {os.path.basename(pf)}")
        else:
            print(f"  ⚠ No .txt files found in {data_dir}")
            problem_files = [None]  # Use mock data
    else:
        print(f"  ✗ Directory not found: {data_dir}")
        print(f"  ⚠ Using mock data instead")
        problem_files = [None]
    
    print(f"  Total problems to test: {len(problem_files)}")
    
    # ========================================================================
    # TODO 3: Initialize algorithms
    # ========================================================================
    print("\nStep 3: Initializing algorithms...")
    
    algorithms = {
        # 'GA': GeneticAlgorithm,
        'SA': SimulatedAnnealing
    }
    
    print(f"  Registered {len(algorithms)} algorithms")
    
    # ========================================================================
    # TODO 4: Run benchmark
    # ========================================================================
    print("\nStep 4: Running benchmark...")
    
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
    # TODO 5: Display results
    # ========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK STATISTICS")
    print("=" * 70)
    print("\n", stats_results.to_string(index=False))
    print("\n" + "=" * 70)
    
    # Display detailed breakdown for best solutions
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
    # TODO 6: Save results to CSV with algorithm config in filename
    # ========================================================================
    print("\nStep 5: Saving results...")
    
    results_dir = 'results'
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
    print(f"    Columns: {', '.join(raw_results.columns)}")
    
    # Save statistics
    stats_results.to_csv(stats_results_path, index=False)
    print(f"  ✓ Saved statistics: {stats_filename}")
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
    print(f"  - Algorithms tested: {len(algorithms)}")
    print(f"  - Problems tested: {len(problem_files)}")
    print(f"  - Runs per config: {num_runs}")
    print(f"  - Total experiments: {len(algorithms) * len(problem_files) * num_runs}")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print(f"  - Raw results: {raw_results_filename}")
    print(f"  - Statistics: {stats_filename}")
    
    print(f"\n📈 CSV File Columns:")
    print(f"  Raw Results: {', '.join(raw_results.columns)}")
    print(f"  Statistics: {', '.join(stats_results.columns)}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

