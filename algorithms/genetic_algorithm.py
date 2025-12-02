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
        Initialize population with diverse strategies for better coverage.
        
        Uses multiple initialization methods:
        - Nearest Neighbor variants (30%)
        - Earliest Deadline First (20%)
        - Savings Algorithm (20%)
        - Regret Insertion (15%)
        - Pure Random (15%)
        """
        self.population = []
        
        # 30% Nearest Neighbor variants
        nn_count = int(0.3 * self.population_size)
        for _ in range(nn_count):
            self.population.append(self._nearest_neighbor_init())
        
        # 20% Earliest Deadline First
        edf_count = int(0.2 * self.population_size)
        for _ in range(edf_count):
            self.population.append(self._earliest_deadline_first())
        
        # 20% Savings Algorithm
        savings_count = int(0.2 * self.population_size)
        for _ in range(savings_count):
            self.population.append(self._savings_init())
        
        # 15% Regret Insertion
        regret_count = int(0.15 * self.population_size)
        for _ in range(regret_count):
            self.population.append(self._regret_insertion())
        
        # Fill remaining with pure random (for diversity)
        customer_ids = list(range(1, self.problem.num_customers + 1))
        remaining = self.population_size - len(self.population)
        for _ in range(remaining):
            individual = customer_ids.copy()
            random.shuffle(individual)
            self.population.append(individual)
    
    def _nearest_neighbor_init(self) -> List[int]:
        """
        Time-window-aware nearest neighbor heuristic initialization.
        
        Greedy construction that respects time windows:
        1. Start from depot at time 0
        2. At each step, find nearest unvisited customer that can be reached
           within their time window, minimizing lateness
        3. Fallback to any feasible customer if no perfect fit exists
        4. Last resort: pick nearest customer (may create violations)
        
        Returns:
            Initial solution
        """
        unvisited = set(range(1, self.problem.num_customers + 1))
        solution = []
        current_location = 0  # Start at depot
        current_time = 0.0
        
        while unvisited:
            best_candidate = None
            best_score = float('inf')
            
            for customer_id in unvisited:
                # Calculate arrival time at this customer
                travel_dist = self.problem.distance_matrix[current_location][customer_id]
                arrival_time = current_time + travel_dist
                
                customer = self.problem.customers[customer_id - 1]
                
                # Calculate a score that prefers feasible visits (no violations)
                # Score = arrival_time if feasible, else heavily penalize violations
                if arrival_time <= customer.due_time:
                    # Feasible: prioritize by arrival time (closer is better)
                    # Prefer to arrive closer to ready_time to minimize waiting
                    if arrival_time < customer.ready_time:
                        score = customer.ready_time  # Will wait anyway
                    else:
                        score = arrival_time
                else:
                    # Infeasible: penalize by violation amount
                    violation = arrival_time - customer.due_time
                    score = customer.due_time + 10000 * violation  # Heavy penalty for violations
                
                if score < best_score:
                    best_score = score
                    best_candidate = customer_id
            
            # Add best candidate to solution
            solution.append(best_candidate)
            unvisited.remove(best_candidate)
            
            # Update current location and time
            travel_dist = self.problem.distance_matrix[current_location][best_candidate]
            current_time += travel_dist
            
            customer = self.problem.customers[best_candidate - 1]
            # Simulate arrival and service
            if current_time < customer.ready_time:
                current_time = customer.ready_time
            current_time += customer.service_time
            
            current_location = best_candidate
        
        return solution
    
    def _earliest_deadline_first(self) -> List[int]:
        """
        Sort customers by due_time (earliest deadline first).
        
        Idea: Prioritize customers with tight deadlines to minimize
        late arrivals. Works well when time windows are tight.
        """
        customers = [(i, self.problem.customers[i-1].due_time) 
                     for i in range(1, self.problem.num_customers + 1)]
        
        # Sort by due_time (ascending)
        customers.sort(key=lambda x: x[1])
        
        return [c[0] for c in customers]
    
    def _savings_init(self) -> List[int]:
        """
        Clarke-Wright Savings Algorithm initialization.
        
        Idea: Calculate savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
        Build tour by merging pairs with highest savings.
        Classic TSP heuristic for distance minimization.
        """
        n = self.problem.num_customers
        customer_ids = list(range(1, n + 1))
        
        # Calculate savings for all pairs
        savings = []
        for i in customer_ids:
            for j in customer_ids:
                if i < j:
                    save = (self.problem.distance_matrix[0][i] + 
                           self.problem.distance_matrix[0][j] - 
                           self.problem.distance_matrix[i][j])
                    savings.append((save, i, j))
        
        # Sort by savings (descending)
        savings.sort(reverse=True)
        
        # Build tour using savings
        solution = []
        used = set()
        
        for save_val, i, j in savings:
            if i not in used and j not in used:
                solution.extend([i, j])
                used.add(i)
                used.add(j)
            elif i not in used:
                solution.append(i)
                used.add(i)
            elif j not in used:
                solution.append(j)
                used.add(j)
            
            if len(used) == n:
                break
        
        # Add any remaining customers
        for c in customer_ids:
            if c not in used:
                solution.append(c)
        
        return solution
    
    def _regret_insertion(self) -> List[int]:
        """
        Regret-based insertion heuristic.
        
        Idea: At each step, insert the customer with largest 'regret'
        (difference between best and second-best insertion cost).
        Prioritizes hard-to-place customers first.
        """
        unvisited = set(range(1, self.problem.num_customers + 1))
        
        # Start with a random customer
        if unvisited:
            solution = [unvisited.pop()]
        else:
            return []
        
        while unvisited:
            max_regret = -float('inf')
            best_customer = None
            best_position = 0
            
            for customer in unvisited:
                insertion_costs = []
                
                # Calculate insertion cost at each position
                for pos in range(len(solution) + 1):
                    if pos == 0:
                        cost = self.problem.distance_matrix[0][customer]
                        if len(solution) > 0:
                            cost += self.problem.distance_matrix[customer][solution[0]]
                            cost -= self.problem.distance_matrix[0][solution[0]]
                    elif pos == len(solution):
                        cost = self.problem.distance_matrix[solution[-1]][customer]
                    else:
                        cost = (self.problem.distance_matrix[solution[pos-1]][customer] +
                               self.problem.distance_matrix[customer][solution[pos]] -
                               self.problem.distance_matrix[solution[pos-1]][solution[pos]])
                    
                    insertion_costs.append((cost, pos))
                
                # Sort by cost
                insertion_costs.sort()
                
                # Calculate regret (difference between best and second-best)
                if len(insertion_costs) >= 2:
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                else:
                    regret = insertion_costs[0][0]
                
                if regret > max_regret:
                    max_regret = regret
                    best_customer = customer
                    best_position = insertion_costs[0][1]
            
            # Insert customer with highest regret
            solution.insert(best_position, best_customer)
            unvisited.remove(best_customer)
        
        return solution

    
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
