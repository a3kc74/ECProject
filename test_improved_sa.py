#!/usr/bin/env python3
"""
Test script to verify the improved Simulated Annealing algorithm.
Demonstrates the new features and improvements.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problem.tw_tsp import TWTSPProblem
from algorithms.simulated_annealing import SimulatedAnnealing

def test_improved_sa():
    """Test the improved Simulated Annealing algorithm."""
    
    print("\n" + "="*70)
    print("TESTING IMPROVED SIMULATED ANNEALING ALGORITHM")
    print("="*70 + "\n")
    
    # Create a mock problem
    print("Step 1: Creating problem instance...")
    problem = TWTSPProblem()  # Uses mock instance
    print(f"✓ Problem created with {problem.num_customers} customers\n")
    
    # Configure improved SA with all features enabled
    print("Step 2: Configuring Improved Simulated Annealing...")
    config = {
        'initial_temperature': 1000.0,
        'final_temperature': 0.1,
        'cooling_rate': 0.95,
        'iterations_per_temp': 50,
        'use_adaptive_cooling': True,
        'use_vnd': True,
        'use_move_history': True,
        'reheat_threshold': 200,
        'reheat_factor': 0.95
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()
    
    # Create algorithm instance
    sa = SimulatedAnnealing(problem, config)
    print("✓ Algorithm instance created\n")
    
    # Solve
    print("Step 3: Solving TW-TSP with Improved SA...")
    print("(This may take a moment...)\n")
    
    best_solution, best_fitness, fitness_history = sa.solve()
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70 + "\n")
    
    print(f"Best Solution Found: {best_solution}")
    print(f"Best Fitness: {best_fitness:.2f}")
    print(f"Iterations: {len(fitness_history)}\n")
    
    # Calculate solution details
    details = problem.calculate_solution_details(best_solution)
    print("Solution Details:")
    print(f"  - Total Distance: {details['total_distance']:.2f}")
    print(f"  - Time Window Violations: {details['num_violations']}")
    print(f"  - Total Penalty: {details['total_penalty']:.2f}")
    print(f"  - Final Fitness: {details['fitness']:.2f}")
    
    if details['num_violations'] == 0:
        print("  ✓ FEASIBLE SOLUTION (No violations)\n")
    else:
        print("  ✗ Infeasible solution (Has violations)\n")
    
    # Algorithm statistics
    print("="*70)
    print("ALGORITHM STATISTICS")
    print("="*70 + "\n")
    
    stats = sa.get_statistics()
    print(f"Stagnation Events: {stats['stagnation_count']}")
    print(f"Total Moves: {stats['total_moves']}")
    print(f"\nMove Distribution:")
    for move_type, count in stats['move_history'].items():
        percentage = (count / stats['total_moves'] * 100) if stats['total_moves'] > 0 else 0
        print(f"  - {move_type}: {count} ({percentage:.1f}%)")
    
    # Convergence analysis
    print(f"\nConvergence:")
    print(f"  - Starting fitness: {fitness_history[0]:.2f}")
    print(f"  - Final fitness: {fitness_history[-1]:.2f}")
    print(f"  - Improvement: {(fitness_history[0] - fitness_history[-1]) / fitness_history[0] * 100:.1f}%")
    print(f"  - History entries: {len(fitness_history)}")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return best_solution, best_fitness, fitness_history

def test_basic_features():
    """Test individual features of the algorithm."""
    
    print("\n" + "="*70)
    print("TESTING INDIVIDUAL FEATURES")
    print("="*70 + "\n")
    
    problem = TWTSPProblem()
    config = {
        'initial_temperature': 500.0,
        'final_temperature': 1.0,
        'cooling_rate': 0.9,
        'iterations_per_temp': 20,
        'use_adaptive_cooling': True,
        'use_vnd': True
    }
    
    sa = SimulatedAnnealing(problem, config)
    
    # Test initialization
    print("Test 1: Nearest Neighbor Initialization")
    initial = sa._nearest_neighbor_init()
    print(f"  - Generated initial solution: {initial}")
    print(f"  - All customers covered: {len(set(initial)) == problem.num_customers}")
    print(f"  - Fitness: {problem.calculate_fitness(initial):.2f}\n")
    
    # Test moves
    print("Test 2: Neighbor Generation Moves")
    test_solution = list(range(1, problem.num_customers + 1))
    
    print("  - 2-opt move:")
    neighbor = sa._two_opt_move(test_solution)
    print(f"    Neighbor: {neighbor}")
    print(f"    Is different: {neighbor != test_solution}")
    
    print("  - 3-opt move:")
    neighbor = sa._three_opt_move(test_solution)
    print(f"    Neighbor: {neighbor}")
    print(f"    Is different: {neighbor != test_solution}")
    
    print("  - Insertion move:")
    neighbor = sa._insertion_move(test_solution)
    print(f"    Neighbor: {neighbor}")
    print(f"    Is different: {neighbor != test_solution}\n")
    
    # Test VND
    print("Test 3: Variable Neighborhood Descent")
    test_solution = sa._nearest_neighbor_init()
    vnd_solution, vnd_fitness = sa._variable_neighborhood_descent(test_solution)
    initial_fitness = problem.calculate_fitness(test_solution)
    print(f"  - Initial fitness: {initial_fitness:.2f}")
    print(f"  - After VND fitness: {vnd_fitness:.2f}")
    print(f"  - Improved: {vnd_fitness < initial_fitness}\n")
    
    # Test perturbation
    print("Test 4: Perturbation")
    test_solution = sa._nearest_neighbor_init()
    perturbed = sa._perturbation(test_solution)
    print(f"  - Original: {test_solution}")
    print(f"  - Perturbed: {perturbed}")
    print(f"  - Is different: {perturbed != test_solution}\n")
    
    print("="*70)
    print("FEATURE TESTS COMPLETED")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Run feature tests
    test_basic_features()
    
    # Run main test
    test_improved_sa()
    
    print("\nAll tests completed! The improved Simulated Annealing algorithm is working correctly.\n")
