"""
Debug SA algorithm step by step to find the bug.
"""
import os
from problem.tw_tsp import TWTSPProblem
from algorithms.simulated_annealing import SimulatedAnnealing

test_instance = 'data/SolomonPotvinBengio/rc_201.1.txt'

if os.path.exists(test_instance):
    problem = TWTSPProblem(test_instance, 'spb')
    
    print("=" * 70)
    print("SA ALGORITHM DEBUG - STEP BY STEP")
    print("=" * 70)
    print(f"\nProblem: {problem.instance_name}")
    print(f"Customers: {problem.num_customers}")
    
    if problem.has_best_known_solution():
        best_known = problem.get_best_known_solution()
        print(f"\nBest-Known Solution:")
        print(f"  Fitness: {best_known['fitness']:.4f}")
        print(f"  Distance: {best_known['distance']:.4f}")
        print(f"  Violations: {best_known['violations']}")
    
    # Test 1: Check initial solution quality
    print("\n" + "-" * 70)
    print("TEST 1: Initial Solution Quality")
    print("-" * 70)
    
    sa = SimulatedAnnealing(problem, {
        'initial_temperature': 100.0,
        'final_temperature': 1.0,
        'cooling_rate': 0.95,
        'iterations_per_temp': 1,
        'show_progress': False
    })
    
    initial_solution = sa._nearest_neighbor_init()
    initial_fitness = problem.calculate_fitness(initial_solution)
    initial_details = problem.calculate_solution_details(initial_solution)
    
    print(f"Initial Nearest Neighbor Solution:")
    print(f"  Solution: {initial_solution}")
    print(f"  Fitness: {initial_fitness:.4f}")
    print(f"  Distance: {initial_details['total_distance']:.4f}")
    print(f"  Violations: {initial_details['num_violations']}")
    print(f"  Penalty: {initial_details['total_penalty']:.4f}")
    
    if best_known['fitness'] is not None:
        gap = ((initial_fitness - best_known['fitness']) / best_known['fitness']) * 100
        print(f"  Gap from best-known: {gap:.2f}%")
    
    # Test 2: Check acceptance probability function
    print("\n" + "-" * 70)
    print("TEST 2: Acceptance Probability")
    print("-" * 70)
    
    test_temperatures = [100, 50, 10, 1, 0.1]
    test_deltas = [0, 10, 100, 1000]
    
    print("\nAcceptance probabilities at different temperatures and deltas:")
    print(f"{'Temp':<8} {' | '.join(f'Δ={d:>6}' for d in test_deltas)}")
    print("-" * 50)
    
    for temp in test_temperatures:
        probs = [sa._acceptance_probability(delta, temp) for delta in test_deltas]
        print(f"{temp:<8} {' | '.join(f'{p:>7.4f}' for p in probs)}")
    
    # Test 3: Run SA with very few iterations to see what happens
    print("\n" + "-" * 70)
    print("TEST 3: SA Convergence Trace (First 5 Iterations)")
    print("-" * 70)
    
    # Manually run SA with logging
    current_solution = sa._nearest_neighbor_init()
    current_fitness = problem.calculate_fitness(current_solution)
    sa.best_solution = current_solution.copy()
    sa.best_fitness = current_fitness
    
    temperature = sa.initial_temperature
    iteration = 0
    
    print(f"\nInitial: fitness={current_fitness:.4f}, temp={temperature:.4f}")
    
    while temperature > sa.final_temperature and iteration < 5:
        for _ in range(sa.iterations_per_temp):
            # Generate neighbor
            neighbor_solution = sa._get_neighbor(current_solution)
            neighbor_fitness = problem.calculate_fitness(neighbor_solution)
            
            delta = neighbor_fitness - current_fitness
            
            # Calculate acceptance
            import random
            if delta < 0 or random.random() < sa._acceptance_probability(delta, temperature):
                accept = True
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
            else:
                accept = False
            
            # Update best
            if current_fitness < sa.best_fitness:
                sa.best_fitness = current_fitness
                print(f"Iteration {iteration}: NEW BEST fitness={current_fitness:.4f}, accept={accept}, delta={delta:.2f}")
            
            iteration += 1
        
        temperature *= sa.cooling_rate
    
    print(f"\nAfter first cooling round:")
    print(f"  Current fitness: {current_fitness:.4f}")
    print(f"  Best fitness: {sa.best_fitness:.4f}")
    details = problem.calculate_solution_details(sa.best_solution)
    print(f"  Best solution violations: {details['num_violations']}")
    print(f"  Best solution penalty: {details['total_penalty']:.4f}")

else:
    print(f"Test instance not found: {test_instance}")
