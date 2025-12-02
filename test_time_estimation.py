#!/usr/bin/env python3
"""
Test script demonstrating time estimation features for algorithms.

Shows:
1. TimeEstimator: Real-time progress tracking during execution
2. AlgorithmTimeTracker: Execution time breakdown by phase
3. Integration with Simulated Annealing algorithm
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problem.tw_tsp import TWTSPProblem
from algorithms.simulated_annealing import SimulatedAnnealing
from utils.time_estimator import TimeEstimator, AlgorithmTimeTracker


def test_time_estimation_features():
    """Test time estimation features."""
    
    print("\n" + "="*70)
    print("TIME ESTIMATION FEATURES DEMONSTRATION")
    print("="*70 + "\n")
    
    # Test 1: TimeEstimator basics
    print("Test 1: TimeEstimator Class")
    print("-" * 70)
    
    estimator = TimeEstimator(total_iterations=1000, problem_size=100)
    estimator.start()
    
    print(f"Started timing for 1000 iterations on problem size 100\n")
    
    # Simulate progress
    import time
    for i in range(0, 1001, 100):
        time.sleep(0.1)  # Simulate work
        estimator.update(i)
        
        if i > 0:
            progress = estimator.get_progress_percentage()
            remaining = estimator.get_estimated_time_remaining()
            elapsed = estimator.get_elapsed_time()
            
            print(f"Iteration {i}/1000:")
            print(f"  Progress: {progress:.1f}%")
            print(f"  Elapsed: {TimeEstimator._format_time(elapsed)}")
            print(f"  Estimated remaining: {TimeEstimator._format_time(remaining) if remaining else 'N/A'}")
            print(f"  {estimator.get_progress_bar()}\n")
    
    # Test 2: AlgorithmTimeTracker
    print("\nTest 2: AlgorithmTimeTracker Class")
    print("-" * 70)
    
    tracker = AlgorithmTimeTracker()
    tracker.start_total()
    
    # Simulate phases
    tracker.start_phase("Data Loading")
    time.sleep(0.2)
    tracker.end_phase("Data Loading")
    
    tracker.start_phase("Initialization")
    time.sleep(0.15)
    tracker.end_phase("Initialization")
    
    tracker.start_phase("Main Computation")
    time.sleep(0.4)
    tracker.end_phase("Main Computation")
    
    tracker.end_total()
    
    print("\nPhase execution times:")
    for phase_name, duration in tracker.get_all_phases().items():
        print(f"  {phase_name}: {duration:.3f}s")
    
    print(f"\nTotal execution time: {AlgorithmTimeTracker._format_time(tracker.total_duration)}")
    
    # Test 3: Integration with Simulated Annealing
    print("\n\nTest 3: Time Estimation with Simulated Annealing")
    print("-" * 70)
    
    # Create problem
    problem = TWTSPProblem()  # Uses mock data
    
    # Configure SA with time estimation enabled
    config = {
        'initial_temperature': 500.0,
        'final_temperature': 1.0,
        'cooling_rate': 0.95,
        'iterations_per_temp': 50,
        'use_adaptive_cooling': True,
        'use_vnd': True,
        'show_progress': True  # Enable progress output
    }
    
    print(f"\nRunning Simulated Annealing with time estimation...")
    print(f"Problem: Mock instance with {problem.num_customers} customers\n")
    
    # Run algorithm
    sa = SimulatedAnnealing(problem, config)
    best_solution, best_fitness, history = sa.solve()
    
    # Display results
    print(f"\nResults:")
    print(f"  Best fitness: {best_fitness:.2f}")
    print(f"  Iterations: {len(history)}")
    
    stats = sa.get_statistics()
    print(f"\nAlgorithm Statistics:")
    print(f"  Execution time: {sa._format_time(stats['execution_time'])}")
    print(f"  Phase breakdown:")
    for phase_name, duration in stats['phase_times'].items():
        percentage = (duration / stats['execution_time'] * 100) if stats['execution_time'] > 0 else 0
        print(f"    - {phase_name}: {duration:.2f}s ({percentage:.1f}%)")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_time_estimation_features()
