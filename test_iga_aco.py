"""
Test script for IGA-ACO Hybrid Algorithm.

This script demonstrates the usage of the IGA-ACO algorithm on TW-TSP instances.
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.tw_tsp import TWTSPProblem
import importlib
import time

# Import IGA-ACO module (handle hyphenated filename)
iga_aco_module = importlib.import_module('algorithms.iga-aco')
IGAACOHybrid = iga_aco_module.IGAACOHybrid


def test_iga_aco_mock():
    """Test IGA-ACO on mock instance."""
    print("=" * 80)
    print("Testing IGA-ACO on Mock Instance")
    print("=" * 80)
    
    # Create mock problem
    problem = TWTSPProblem()
    
    print(f"\nProblem: {problem.instance_name}")
    print(f"Number of customers: {problem.num_customers}")
    
    # Configure algorithm
    config = {
        'num_iterations': 100,
        'iga_population_size': 50,
        'iga_elite_ratio': 0.1,
        'iga_vnd_probability': 0.3,
        'aco_num_ants': 30,
        'aco_alpha': 1.0,
        'aco_beta': 2.0,
        'aco_rho': 0.1,
        'aco_q0': 0.9,
        'exchange_interval': 10
    }
    
    print("\nAlgorithm Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create and run algorithm
    algorithm = IGAACOHybrid(problem, config)
    
    print("\n" + "-" * 80)
    print("Running IGA-ACO Hybrid Algorithm...")
    print("-" * 80 + "\n")
    
    start_time = time.time()
    best_solution, best_fitness, fitness_history = algorithm.solve()
    end_time = time.time()
    
    # Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    # Convert to native Python ints for clean display
    clean_solution = [int(x) for x in best_solution]
    print(f"\nBest Solution: {clean_solution}")
    print(f"Best Fitness: {best_fitness:.2f}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    
    # Calculate solution details
    details = problem.calculate_solution_details(best_solution)
    print(f"\nSolution Details:")
    print(f"  Total Distance: {details['total_distance']:.2f}")
    print(f"  Time Window Violations: {details['num_violations']}")
    print(f"  Total Penalty: {details['total_penalty']:.2f}")
    print(f"  Final Fitness: {details['fitness']:.2f}")
    
    print(f"\nFitness Improvement:")
    print(f"  Initial: {fitness_history[0]:.2f}")
    print(f"  Final: {fitness_history[-1]:.2f}")
    print(f"  Improvement: {fitness_history[0] - fitness_history[-1]:.2f} ({((fitness_history[0] - fitness_history[-1]) / fitness_history[0] * 100):.2f}%)")


def test_iga_aco_real_instance():
    """Test IGA-ACO on real instance."""
    print("\n\n" + "=" * 80)
    print("Testing IGA-ACO on Real Instance (rc_201.1)")
    print("=" * 80)
    
    # Load real instance
    instance_path = "data/OhlmannThomas/n150w120.003.txt"
    
    if not os.path.exists(instance_path):
        print(f"\nWarning: Instance file not found: {instance_path}")
        print("Skipping real instance test.")
        return
    
    problem = TWTSPProblem(instance_path, format='spb')
    
    print(f"\nProblem: {problem.instance_name}")
    print(f"Number of customers: {problem.num_customers}")
    
    # Check for best-known solution
    if problem.has_best_known_solution():
        best_known = problem.get_best_known_solution()
        print(f"\nBest-Known Solution:")
        print(f"  Fitness: {best_known['fitness']:.2f}")
        print(f"  Distance: {best_known['distance']:.2f}")
        print(f"  Violations: {best_known['violations']}")
    
    # Configure algorithm - optimized for better results
    config = {
        'num_iterations': 1000,          # Increased for better convergence
        'iga_population_size': 100,     # Larger population for diversity
        'iga_elite_ratio': 0.1,         # Standard elite ratio
        'iga_vnd_probability': 0.7,     # Higher VND probability for intensification
        'aco_num_ants': 70,             # More ants for exploration
        'aco_alpha': 1.0,               # Pheromone importance
        'aco_beta': 5.0,                # Higher heuristic importance (time-aware)
        'aco_rho': 0.1,                 # Evaporation rate
        'aco_q0': 0.9,                  # High exploitation
        'exchange_interval': 10         # Exchange every 10 iterations
    }
    
    print("\nAlgorithm Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create and run algorithm
    algorithm = IGAACOHybrid(problem, config)
    
    print("\n" + "-" * 80)
    print("Running IGA-ACO Hybrid Algorithm...")
    print("-" * 80 + "\n")
    
    start_time = time.time()
    best_solution, best_fitness, fitness_history = algorithm.solve()
    end_time = time.time()
    
    # Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    # Convert to native Python ints for clean display
    clean_solution = [int(x) for x in best_solution]
    print(f"\nBest Solution: {clean_solution}")
    print(f"Best Fitness: {best_fitness:.2f}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    
    # Calculate solution details
    details = problem.calculate_solution_details(best_solution)
    print(f"\nSolution Details:")
    print(f"  Total Distance: {details['total_distance']:.2f}")
    print(f"  Time Window Violations: {details['num_violations']}")
    print(f"  Total Penalty: {details['total_penalty']:.2f}")
    print(f"  Final Fitness: {details['fitness']:.2f}")
    
    # Compare with best-known if available
    if problem.has_best_known_solution():
        best_known = problem.get_best_known_solution()
        gap = ((best_fitness - best_known['fitness']) / best_known['fitness']) * 100
        print(f"\nComparison with Best-Known:")
        print(f"  Gap: {gap:.2f}%")
    
    print(f"\nFitness Improvement:")
    print(f"  Initial: {fitness_history[0]:.2f}")
    print(f"  Final: {fitness_history[-1]:.2f}")
    print(f"  Improvement: {fitness_history[0] - fitness_history[-1]:.2f} ({((fitness_history[0] - fitness_history[-1]) / fitness_history[0] * 100):.2f}%)")


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("IGA-ACO Hybrid Algorithm Test Suite")
    print("=" * 80)
    
    # Test on mock instance
    test_iga_aco_mock()
    
    # Test on real instance
    test_iga_aco_real_instance()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
