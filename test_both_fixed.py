"""
Comprehensive test of both SA and GA after fixes.
"""
import os
from problem.tw_tsp import TWTSPProblem
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.genetic_algorithm import GeneticAlgorithm

test_instance = 'data/SolomonPotvinBengio/rc_202.3.txt'

if os.path.exists(test_instance):
    problem = TWTSPProblem(test_instance, 'spb')
    
    print("=" * 70)
    print("ALGORITHM COMPARISON AFTER FIXES")
    print("=" * 70)
    print(f"\nProblem: {problem.instance_name}")
    print(f"Customers: {problem.num_customers}")
    
    if problem.has_best_known_solution():
        best_known = problem.get_best_known_solution()
        print(f"\nBest-Known Solution:")
        print(f"  Fitness: {best_known['fitness']:.4f}")
        print(f"  Distance: {best_known['distance']:.4f}")
        print(f"  Violations: {best_known['violations']}")
    
    # Test SA
    print("\n" + "-" * 70)
    print("SIMULATED ANNEALING")
    print("-" * 70)
    
    sa_config = {
        'initial_temperature': 1000.0,
        'final_temperature': 0.1,
        'cooling_rate': 0.96,
        'iterations_per_temp': 50,
        'show_progress': False
    }
    
    sa = SimulatedAnnealing(problem, sa_config)
    best_solution_sa, best_fitness_sa, history_sa = sa.solve()
    details_sa = problem.calculate_solution_details(best_solution_sa)
    
    print(f"\nResults:")
    print(f"  Fitness: {best_fitness_sa:.4f}")
    print(f"  Distance: {details_sa['total_distance']:.4f}")
    print(f"  Violations: {details_sa['num_violations']}")
    print(f"  Penalty: {details_sa['total_penalty']:.4f}")
    
    if best_known['fitness'] is not None:
        gap_sa = ((best_fitness_sa - best_known['fitness']) / best_known['fitness']) * 100
        print(f"  Gap from best-known: {gap_sa:.2f}%")
    
    # Test GA
    print("\n" + "-" * 70)
    print("GENETIC ALGORITHM")
    print("-" * 70)
    
    ga_config = {
        'population_size': 150,
        'num_generations': 2000,
        'mutation_rate': 0.08,
        'crossover_rate': 0.92,
        'tournament_size': 20,
        'elitism_count': 3
    }
    
    ga = GeneticAlgorithm(problem, ga_config)
    best_solution_ga, best_fitness_ga, history_ga = ga.solve()
    details_ga = problem.calculate_solution_details(best_solution_ga)
    
    print(f"\nResults:")
    print(f"  Fitness: {best_fitness_ga:.4f}")
    print(f"  Distance: {details_ga['total_distance']:.4f}")
    print(f"  Violations: {details_ga['num_violations']}")
    print(f"  Penalty: {details_ga['total_penalty']:.4f}")
    
    if best_known['fitness'] is not None:
        gap_ga = ((best_fitness_ga - best_known['fitness']) / best_known['fitness']) * 100
        print(f"  Gap from best-known: {gap_ga:.2f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nBest-known: {best_known['fitness']:.4f}")
    print(f"SA Result:  {best_fitness_sa:.4f} (Gap: {gap_sa:.2f}%)")
    print(f"GA Result:  {best_fitness_ga:.4f} (Gap: {gap_ga:.2f}%)")
    
    winner = "SA" if best_fitness_sa < best_fitness_ga else "GA"
    difference = abs(best_fitness_sa - best_fitness_ga)
    print(f"\nWinner: {winner} by {difference:.4f}")
    
    # Check feasibility
    print(f"\nFeasibility:")
    print(f"  SA: {'✓ Feasible' if details_sa['num_violations'] == 0 else '✗ Infeasible'}")
    print(f"  GA: {'✓ Feasible' if details_ga['num_violations'] == 0 else '✗ Infeasible'}")

else:
    print(f"Test instance not found: {test_instance}")
