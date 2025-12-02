"""
Debug script to identify fitness calculation issues.
"""
import os
from problem.tw_tsp import TWTSPProblem
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.genetic_algorithm import GeneticAlgorithm

# Load a small test instance
test_instance = 'data/SolomonPotvinBengio/rc_201.1.txt'

if os.path.exists(test_instance):
    problem = TWTSPProblem(test_instance, 'spb')
    
    print("=" * 70)
    print("FITNESS CALCULATION DEBUG")
    print("=" * 70)
    print(f"\nProblem: {problem.instance_name}")
    print(f"Customers: {problem.num_customers}")
    
    if problem.has_best_known_solution():
        best_known = problem.get_best_known_solution()
        print(f"\nBest-Known Solution Available:")
        print(f"  - Best-Known Fitness (calculated): {best_known['fitness']:.2f}")
        print(f"  - Best-Known Distance: {best_known['distance']:.2f}")
        print(f"  - Best-Known Violations: {best_known['violations']}")
        print(f"  - Best-Known Reported Cost: {best_known['reported_cost']:.2f}")
    else:
        print("\nNo best-known solution available")
    
    # Run SA with minimal iterations to test
    print("\n" + "-" * 70)
    print("Testing Simulated Annealing - DETAILED DEBUG")
    print("-" * 70)
    
    sa_config = {
        'initial_temperature': 100.0,
        'final_temperature': 1.0,
        'cooling_rate': 0.95,
        'iterations_per_temp': 50,
        'show_progress': False
    }
    
    sa = SimulatedAnnealing(problem, sa_config)
    best_solution, best_fitness_sa, history_sa = sa.solve()
    
    # Calculate solution details
    details = problem.calculate_solution_details(best_solution)
    
    print(f"\n[ALGORITHM RESULTS]")
    print(f"  - Best solution: {best_solution}")
    print(f"  - Fitness from algorithm (best_fitness): {best_fitness_sa:.4f}")
    print(f"  - Final fitness_history entry: {history_sa[-1]:.4f}")
    
    print(f"\n[RECALCULATED RESULTS]")
    print(f"  - Fitness from calculate_solution_details: {details['fitness']:.4f}")
    print(f"  - Distance: {details['total_distance']:.4f}")
    print(f"  - Violations: {details['num_violations']}")
    print(f"  - Penalty: {details['total_penalty']:.4f}")
    
    print(f"\n[COMPARISON WITH BEST-KNOWN]")
    if problem.has_best_known_solution():
        best_known = problem.get_best_known_solution()
        print(f"  - Best-known fitness: {best_known['fitness']:.4f}")
        print(f"  - Best-known distance: {best_known['distance']:.4f}")
        print(f"  - Best-known violations: {best_known['violations']}")
        
        gap = ((details['fitness'] - best_known['fitness']) / best_known['fitness']) * 100
        print(f"\n  - Gap from best-known: {gap:.2f}%")
        
        if details['fitness'] < best_known['fitness']:
            print(f"  ✓ Algorithm fitness ({details['fitness']:.2f}) < best-known ({best_known['fitness']:.2f})")
            print(f"    This should NOT happen! Best-known should be optimal or near-optimal.")
        else:
            print(f"  ✓ Algorithm fitness ({details['fitness']:.2f}) >= best-known ({best_known['fitness']:.2f})")
    
    print(f"\n[FITNESS CONSISTENCY CHECK]")
    print(f"  - Difference between algo and recalculated: {abs(best_fitness_sa - details['fitness']):.6f}")
    
    if abs(best_fitness_sa - details['fitness']) > 0.01:
        print("  ⚠️  WARNING: Fitness calculation mismatch detected!")
    else:
        print("  ✓ Fitness calculations are consistent")
    
else:
    print(f"Test instance not found: {test_instance}")
