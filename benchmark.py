"""
Benchmark framework for comparing multiple algorithms on multiple problem instances.

Provides systematic testing, statistical analysis, and result aggregation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Type, Tuple
import time
import os
import sys
import math

from problem.tw_tsp import TWTSPProblem
from algorithms.base_algorithm import BaseAlgorithm
from utils.time_estimator import TimeEstimator


class Benchmark:
    """
    Benchmark framework for systematic algorithm comparison.
    
    Runs multiple algorithms on multiple problem instances for statistical analysis.
    """
    
    def __init__(self, algorithms: Dict[str, Type[BaseAlgorithm]], 
                 problem_paths: List[str], 
                 num_runs: int,
                 algorithm_configs: Dict[str, dict] = None):
        """
        Initialize benchmark framework.
        
        Args:
            algorithms: Dictionary mapping algorithm names to algorithm classes
                       Example: {'GA': GeneticAlgorithm, 'SA': SimulatedAnnealing}
            problem_paths: List of paths to problem instance files
            num_runs: Number of independent runs per algorithm per problem
            algorithm_configs: Dictionary mapping algorithm names to their configurations
        """
        self.algorithms = algorithms
        self.problem_paths = problem_paths
        self.num_runs = num_runs
        self.algorithm_configs = algorithm_configs or {}
        
        # Storage for results
        self.raw_results: List[Dict] = []
        self.best_solutions: Dict[Tuple[str, str], Tuple[List[int], float]] = {}
        self.convergence_histories: Dict[Tuple[str, str, int], List[float]] = {}
        self.problems: Dict[str, TWTSPProblem] = {}  # Store loaded problem instances
    
    def run(self) -> pd.DataFrame:
        """
        Run the complete benchmark.
        
        Executes all algorithms on all problems for the specified number of runs.
        Shows time estimates for overall completion.
        
        Returns:
            DataFrame containing all raw results with columns:
                - algorithm: Algorithm name
                - problem: Problem instance name
                - run: Run number
                - best_fitness: Best fitness found in this run
                - run_time: Execution time in seconds
        """
        print("=" * 70)
        print("STARTING BENCHMARK")
        print("=" * 70)
        print(f"Number of algorithms: {len(self.algorithms)}")
        print(f"Number of problems: {len(self.problem_paths)}")
        print(f"Runs per configuration: {self.num_runs}")
        print(f"Total experiments: {len(self.algorithms) * len(self.problem_paths) * self.num_runs}")
        print("=" * 70)
        
        # Time tracking
        total_experiments = len(self.algorithms) * len(self.problem_paths) * self.num_runs
        experiments_completed = 0
        start_time = time.time()
        run_times = []
        
        for problem_path in self.problem_paths:
            # Extract problem name from path
            problem_name = os.path.basename(problem_path)
            
            print(f"\n--- Problem: {problem_name} ---")
            
            # Load problem instance
            try:
                problem = TWTSPProblem(problem_path, 'spb')
                print(f"Loaded problem with {problem.num_customers} customers")
            except Exception as e:
                print(f"Error loading problem {problem_path}: {e}")
                print("Creating mock problem instance instead...")
                problem = TWTSPProblem()  # Use mock data
                problem_name = "mock_instance"
            
            # Store problem instance for later use
            self.problems[problem_name] = problem
            
            # Get best-known solution if available
            best_known_fitness = None
            if problem.has_best_known_solution():
                best_known = problem.get_best_known_solution()
                best_known_fitness = best_known['fitness']
                print(f"Best-known solution: {best_known_fitness:.2f}")
            
            for algo_name, algo_class in self.algorithms.items():
                print(f"\n  Algorithm: {algo_name}")
                
                # Get configuration for this algorithm
                config = self.algorithm_configs.get(algo_name, {})
                
                # Storage for best solution across all runs
                overall_best_fitness = float('inf')
                overall_best_solution = None
                
                for run in range(self.num_runs):
                    # Calculate remaining experiments
                    experiments_completed += 1
                    elapsed = time.time() - start_time
                    
                    # Estimate remaining time
                    if experiments_completed > 1:
                        avg_time_per_experiment = elapsed / experiments_completed
                        remaining_experiments = total_experiments - experiments_completed
                        estimated_remaining = avg_time_per_experiment * remaining_experiments
                        estimated_total = elapsed + estimated_remaining
                        
                        time_str = self._format_time(estimated_remaining)
                        total_str = self._format_time(estimated_total)
                        progress = (experiments_completed / total_experiments) * 100
                        
                        print(f"    Run {run + 1}/{self.num_runs}...", end=' ')
                    else:
                        print(f"    Run {run + 1}/{self.num_runs}...", end=' ')
                    
                    # Create algorithm instance (disable progress in benchmark for cleaner output)
                    config_with_no_progress = config.copy()
                    config_with_no_progress['show_progress'] = False
                    algorithm = algo_class(problem, config_with_no_progress)
                    
                    # Run algorithm and measure time
                    run_start = time.time()
                    best_solution, best_fitness, fitness_history = algorithm.solve()
                    run_end = time.time()
                    
                    run_time = run_end - run_start
                    run_times.append(run_time)
                    
                    # Calculate detailed metrics
                    solution_details = problem.calculate_solution_details(best_solution)
                    
                    # Use the recalculated fitness (with correct penalty coefficient)
                    best_fitness = solution_details['fitness']
                    
                    # Calculate gap from best-known solution
                    gap = None
                    if best_known_fitness is not None and best_known_fitness > 0:
                        gap = ((best_fitness - best_known_fitness) / best_known_fitness) * 100
                    
                    print(f"Fitness: {best_fitness:.2f}, Time: {run_time:.2f}s, "
                          f"Distance: {solution_details['total_distance']:.2f}, "
                          f"Violations: {solution_details['num_violations']}", end='')
                    
                    # Show time estimate
                    if experiments_completed > 1:
                        print(f" [Est. remaining: {time_str}]")
                    else:
                        print()
                    
                    # Store raw results
                    self.raw_results.append({
                        'algorithm': algo_name,
                        'problem': problem_name,
                        'run': run + 1,
                        'best_fitness': best_fitness,
                        'run_time': run_time,
                        'total_distance': solution_details['total_distance'],
                        'num_violations': solution_details['num_violations'],
                        'total_penalty': solution_details['total_penalty'],
                        'best_known': best_known_fitness,
                        'gap': gap
                    })
                    
                    # Store convergence history
                    self.convergence_histories[(algo_name, problem_name, run + 1)] = fitness_history
                    
                    # Update overall best solution
                    if best_fitness < overall_best_fitness:
                        overall_best_fitness = best_fitness
                        overall_best_solution = best_solution
                
                # Store best solution for this algorithm-problem combination
                self.best_solutions[(algo_name, problem_name)] = (
                    overall_best_solution, 
                    overall_best_fitness
                )
                
                print(f"  Best fitness across all runs: {overall_best_fitness:.2f}")
        
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETED")
        print("=" * 70)
        
        # Print time summary
        total_time = time.time() - start_time
        avg_run_time = np.mean(run_times) if run_times else 0
        print(f"\nTotal benchmark time: {self._format_time(total_time)}")
        print(f"Average time per experiment: {avg_run_time:.2f}s")
        print(f"Total experiments: {experiments_completed}/{total_experiments}")
        print("=" * 70)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.raw_results)
        return results_df
    
    def get_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical summary of results.
        
        Args:
            results_df: DataFrame from run() method
        
        Returns:
            DataFrame with statistics (best, mean, std, median) grouped by algorithm and problem
        """
        # Group by algorithm and problem
        grouped = results_df.groupby(['algorithm', 'problem'])
        
        # Calculate statistics
        stats = grouped['best_fitness'].agg([
            ('best', 'min'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('median', 'median'),
            ('worst', 'max')
        ]).reset_index()
        
        # Add average run time
        avg_time = grouped['run_time'].mean().reset_index()
        stats = stats.merge(avg_time, on=['algorithm', 'problem'])
        stats.rename(columns={'run_time': 'avg_run_time'}, inplace=True)
        
        # Add average distance
        avg_distance = grouped['total_distance'].mean().reset_index()
        stats = stats.merge(avg_distance, on=['algorithm', 'problem'])
        stats.rename(columns={'total_distance': 'avg_distance'}, inplace=True)
        
        # Add average violations
        avg_violations = grouped['num_violations'].mean().reset_index()
        stats = stats.merge(avg_violations, on=['algorithm', 'problem'])
        stats.rename(columns={'num_violations': 'avg_violations'}, inplace=True)
        
        # Add total violations (sum across all runs)
        total_violations = grouped['num_violations'].sum().reset_index()
        stats = stats.merge(total_violations, on=['algorithm', 'problem'])
        stats.rename(columns={'num_violations': 'total_violations'}, inplace=True)
        
        # Add best-known solution (take first value since it's same for all runs)
        best_known = grouped['best_known'].first().reset_index()
        stats = stats.merge(best_known, on=['algorithm', 'problem'])
        
        # Calculate gap for the best solution (not average gap)
        # Gap = (best - best_known) / best_known * 100
        stats['gap'] = None
        mask = (stats['best_known'].notna()) & (stats['best_known'] > 0)
        stats.loc[mask, 'gap'] = ((stats.loc[mask, 'best'] - stats.loc[mask, 'best_known']) / stats.loc[mask, 'best_known']) * 100
        
        # Add algorithm configuration parameters
        config_data = []
        for algo_name in stats['algorithm'].unique():
            if algo_name in self.algorithm_configs:
                config = self.algorithm_configs[algo_name]
                config_data.append({
                    'algorithm': algo_name,
                    'population_size': config.get('population_size', None),
                    'num_generations': config.get('num_generations', None),
                    'mutation_rate': config.get('mutation_rate', None),
                    'crossover_rate': config.get('crossover_rate', None),
                    'tournament_size': config.get('tournament_size', None),
                    'elitism_count': config.get('elitism_count', None),
                    'initial_temperature': config.get('initial_temperature', None),
                    'final_temperature': config.get('final_temperature', None),
                    'cooling_rate': config.get('cooling_rate', None),
                    'iterations_per_temp': config.get('iterations_per_temp', None)
                })
        
        if config_data:
            config_df = pd.DataFrame(config_data)
            stats = stats.merge(config_df, on='algorithm', how='left')
        
        return stats
    
    def get_best_solution(self, algorithm_name: str, problem_name: str) -> List[int]:
        """
        Retrieve the best solution found for a specific algorithm-problem combination.
        
        Args:
            algorithm_name: Name of the algorithm
            problem_name: Name of the problem instance
        
        Returns:
            Best solution as list of customer IDs, or None if not found
        """
        key = (algorithm_name, problem_name)
        if key in self.best_solutions:
            solution, fitness = self.best_solutions[key]
            return solution
        else:
            print(f"No solution found for {algorithm_name} on {problem_name}")
            return None
    
    def get_best_fitness(self, algorithm_name: str, problem_name: str) -> float:
        """
        Retrieve the best fitness found for a specific algorithm-problem combination.
        
        Args:
            algorithm_name: Name of the algorithm
            problem_name: Name of the problem instance
        
        Returns:
            Best fitness value, or inf if not found
        """
        key = (algorithm_name, problem_name)
        if key in self.best_solutions:
            solution, fitness = self.best_solutions[key]
            return fitness
        else:
            return float('inf')
    
    def get_convergence_history(self, algorithm_name: str, problem_name: str, run: int = 1) -> List[float]:
        """
        Retrieve convergence history for a specific run.
        
        Args:
            algorithm_name: Name of the algorithm
            problem_name: Name of the problem instance
            run: Run number (1-indexed)
        
        Returns:
            List of fitness values over iterations, or empty list if not found
        """
        key = (algorithm_name, problem_name, run)
        return self.convergence_histories.get(key, [])
    
    def get_problem(self, problem_name: str) -> TWTSPProblem:
        """
        Retrieve loaded problem instance.
        
        Args:
            problem_name: Name of the problem instance
        
        Returns:
            TWTSPProblem instance, or None if not found
        """
        return self.problems.get(problem_name)
    
    def save_results(self, results_df: pd.DataFrame, filepath: str) -> None:
        """
        Save results to CSV file.
        
        Args:
            results_df: DataFrame containing results
            filepath: Path where CSV will be saved
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format seconds to readable time string.
        
        Args:
            seconds: Time in seconds
        
        Returns:
            Formatted time string (HH:MM:SS or MM:SS)
        """
        if seconds is None or math.isnan(seconds) or math.isinf(seconds):
            return "??:??"
        
        seconds = max(0, seconds)
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
