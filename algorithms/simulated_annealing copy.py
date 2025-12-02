"""
Simulated Annealing implementation for TW-TSP.

Uses 2-opt neighborhood and exponential cooling schedule.
"""
import numpy as np
from typing import List, Tuple
import random
import math
from .base_algorithm import BaseAlgorithm
from problem.tw_tsp import TWTSPProblem


class SimulatedAnnealing(BaseAlgorithm):
    """
    Simulated Annealing algorithm for solving TW-TSP.
    
    Configuration parameters:
        - initial_temperature: Starting temperature
        - final_temperature: Ending temperature (stopping criterion)
        - cooling_rate: Rate of temperature reduction (0.0 to 1.0)
        - iterations_per_temp: Number of iterations at each temperature
    """
    
    def __init__(self, problem: TWTSPProblem, config: dict):
        """
        Initialize Simulated Annealing.
        
        Args:
            problem: TW-TSP problem instance
            config: Configuration dictionary
        """
        super().__init__(problem, config)
        
        # Extract configuration parameters with defaults
        self.initial_temperature = config.get('initial_temperature', 1000.0)
        self.final_temperature = config.get('final_temperature', 0.1)
        self.cooling_rate = config.get('cooling_rate', 0.95)
        self.iterations_per_temp = config.get('iterations_per_temp', 100)
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Execute Simulated Annealing to solve TW-TSP.
        
        Returns:
            Tuple of (best_solution, best_fitness, fitness_history)
        """
        # Initialize with random solution
        current_solution = list(range(1, self.problem.num_customers + 1))
        random.shuffle(current_solution)
        current_fitness = self.problem.calculate_fitness(current_solution)
        
        # Initialize best solution
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        
        # Temperature
        temperature = self.initial_temperature
        iteration = 0
        # Main annealing loop
        while temperature > self.final_temperature:
            for _ in range(self.iterations_per_temp):
                # Generate neighbor solution
                neighbor_solution = self._get_neighbor(current_solution)
                neighbor_fitness = self.problem.calculate_fitness(neighbor_solution)
                
                # Calculate fitness difference
                delta = neighbor_fitness - current_fitness
                
                # Accept or reject neighbor
                if delta < 0 or random.random() < self._acceptance_probability(delta, temperature):
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
                    
                    # Update best solution if improved
                    if current_fitness < self.best_fitness:
                        self.best_solution = current_solution.copy()
                        self.best_fitness = current_fitness
                
                # Record fitness history
                self.fitness_history.append(self.best_fitness)
                iteration += 1
            
            # Cool down
            temperature *= self.cooling_rate
        
        return self.best_solution, self.best_fitness, self.fitness_history
    
    def _get_neighbor(self, solution: List[int]) -> List[int]:
        """
        Generate a neighbor solution using 2-opt swap.
        
        2-opt: Reverse a segment of the tour to create a neighbor.
        
        Args:
            solution: Current solution
        
        Returns:
            Neighbor solution
        """
        neighbor = solution.copy()
        size = len(neighbor)
        
        # Select two random positions
        i, j = sorted(random.sample(range(size), 2))
        
        # Reverse the segment between i and j
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        
        return neighbor
    
    def _acceptance_probability(self, delta: float, temperature: float) -> float:
        """
        Calculate probability of accepting a worse solution.
        
        Uses the Metropolis criterion: P = exp(-delta / temperature)
        
        Args:
            delta: Fitness difference (positive for worse solutions)
            temperature: Current temperature
        
        Returns:
            Acceptance probability between 0 and 1
        """
        if temperature <= 0:
            return 0.0
        
        try:
            probability = math.exp(-delta / temperature)
        except OverflowError:
            probability = 0.0
        
        return probability
