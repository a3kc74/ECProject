"""
Genetic Algorithm implementation for TW-TSP.

Uses tournament selection, ordered crossover (OX), and swap mutation.
"""
import numpy as np
from typing import List, Tuple
import random
from .base_algorithm import BaseAlgorithm
from problem.tw_tsp import TWTSPProblem


class GeneticAlgorithm(BaseAlgorithm):
    """
    Genetic Algorithm for solving TW-TSP.
    
    Configuration parameters:
        - population_size: Number of individuals in population
        - num_generations: Number of generations to evolve
        - mutation_rate: Probability of mutation (0.0 to 1.0)
        - crossover_rate: Probability of crossover (0.0 to 1.0)
        - tournament_size: Number of individuals in tournament selection
        - elitism_count: Number of best individuals to preserve each generation
    """
    
    def __init__(self, problem: TWTSPProblem, config: dict):
        """
        Initialize Genetic Algorithm.
        
        Args:
            problem: TW-TSP problem instance
            config: Configuration dictionary
        """
        super().__init__(problem, config)
        
        # Extract configuration parameters with defaults
        self.population_size = config.get('population_size', 100)
        self.num_generations = config.get('num_generations', 500)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.tournament_size = config.get('tournament_size', 5)
        self.elitism_count = config.get('elitism_count', 2)
        
        self.population: List[List[int]] = []
        self.fitness_values: List[float] = []
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Execute the Genetic Algorithm to solve TW-TSP.
        
        Returns:
            Tuple of (best_solution, best_fitness, fitness_history)
        """
        # Initialize population
        self._initialize_population()
        
        # Evolution loop
        for generation in range(self.num_generations):
            # Evaluate fitness for all individuals
            self._evaluate_population()
            
            # Track best solution
            best_idx = np.argmin(self.fitness_values)
            generation_best_fitness = self.fitness_values[best_idx]
            
            if generation_best_fitness < self.best_fitness:
                self.best_fitness = generation_best_fitness
                self.best_solution = self.population[best_idx].copy()
            
            # Record fitness history
            self.fitness_history.append(self.best_fitness)
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            sorted_indices = np.argsort(self.fitness_values)
            for i in range(self.elitism_count):
                new_population.append(self.population[sorted_indices[i]].copy())
            
            # Generate offspring to fill remaining population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._selection()
                parent2 = self._selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = self._mutation(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self._mutation(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            self.population = new_population
        
        # Final evaluation
        self._evaluate_population()
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness_values[best_idx]
            self.best_solution = self.population[best_idx].copy()
        
        return self.best_solution, self.best_fitness, self.fitness_history
    
    def _initialize_population(self) -> None:
        """
        Initialize population with random permutations of customer IDs.
        """
        customer_ids = list(range(1, self.problem.num_customers + 1))
        
        for _ in range(self.population_size):
            individual = customer_ids.copy()
            random.shuffle(individual)
            self.population.append(individual)
    
    def _evaluate_population(self) -> None:
        """
        Evaluate fitness for all individuals in the population.
        """
        self.fitness_values = []
        for individual in self.population:
            fitness = self.problem.calculate_fitness(individual)
            self.fitness_values.append(fitness)
    
    def _selection(self) -> List[int]:
        """
        Tournament selection: select best individual from random subset.
        
        Returns:
            Selected individual
        """
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitness = [self.fitness_values[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Ordered Crossover (OX) operator.
        
        Preserves the relative order of cities from parents while creating offspring.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
        
        Returns:
            Tuple of two offspring
        """
        size = len(parent1)
        
        # Select two random crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Create offspring 1
        offspring1 = [None] * size
        offspring1[point1:point2] = parent1[point1:point2]
        
        # Fill remaining positions from parent2
        parent2_filtered = [gene for gene in parent2 if gene not in offspring1]
        idx = 0
        for i in range(size):
            if offspring1[i] is None:
                offspring1[i] = parent2_filtered[idx]
                idx += 1
        
        # Create offspring 2
        offspring2 = [None] * size
        offspring2[point1:point2] = parent2[point1:point2]
        
        # Fill remaining positions from parent1
        parent1_filtered = [gene for gene in parent1 if gene not in offspring2]
        idx = 0
        for i in range(size):
            if offspring2[i] is None:
                offspring2[i] = parent1_filtered[idx]
                idx += 1
        
        return offspring1, offspring2
    
    def _mutation(self, individual: List[int]) -> List[int]:
        """
        Swap mutation: randomly swap two genes.
        
        Args:
            individual: Chromosome to mutate
        
        Returns:
            Mutated chromosome
        """
        mutated = individual.copy()
        
        # Select two random positions and swap
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
