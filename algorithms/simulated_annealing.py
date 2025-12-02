"""
Simulated Annealing implementation for TW-TSP.

Uses multiple neighborhood operators and exponential cooling schedule.
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
        - initial_solution: Method for initial solution ('random', 'nearest_neighbor', 'earliest_deadline')
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
        self.initial_solution_method = config.get('initial_solution', 'nearest_neighbor')
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Execute Simulated Annealing to solve TW-TSP.
        
        Returns:
            Tuple of (best_solution, best_fitness, fitness_history)
        """
        # Initialize with heuristic solution
        current_solution = self._generate_initial_solution()
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
                # Generate neighbor solution using multiple operators
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
    
    def _generate_initial_solution(self) -> List[int]:
        """
        Generate initial solution using specified method.
        
        Returns:
            Initial solution
        """
        if self.initial_solution_method == 'nearest_neighbor':
            return self._nearest_neighbor_init()
        elif self.initial_solution_method == 'earliest_deadline':
            return self._earliest_deadline_first()
        elif self.initial_solution_method == 'savings':
            return self._savings_init()
        else:  # 'random' or any other value
            solution = list(range(1, self.problem.num_customers + 1))
            random.shuffle(solution)
            return solution
    
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
                if arrival_time <= customer.due_time:
                    # Feasible: prioritize by arrival time (closer is better)
                    if arrival_time < customer.ready_time:
                        score = customer.ready_time  # Will wait anyway
                    else:
                        score = arrival_time
                else:
                    # Infeasible: penalize by violation amount
                    violation = arrival_time - customer.due_time
                    score = customer.due_time + 10000 * violation
                
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
        
        Idea: Prioritize customers with tight deadlines to minimize late arrivals.
        """
        customers = [(i, self.problem.customers[i-1].due_time) 
                     for i in range(1, self.problem.num_customers + 1)]
        
        # Sort by due_time (ascending)
        customers.sort(key=lambda x: x[1])
        
        return [c[0] for c in customers]
    
    def _savings_init(self) -> List[int]:
        """
        Clarke-Wright Savings Algorithm initialization.
        
        Calculate savings s(i,j) = d(0,i) + d(0,j) - d(i,j)
        Build tour by merging pairs with highest savings.
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
    
    def _get_neighbor(self, solution: List[int]) -> List[int]:
        """
        Generate a neighbor solution using multiple operators.
        
        Randomly selects from:
        - 2-opt: Reverse a segment
        - Swap: Exchange two customers
        - Insert: Remove and reinsert a customer
        - Or-opt: Move a sequence of customers
        
        Args:
            solution: Current solution
        
        Returns:
            Neighbor solution
        """
        operator = random.choice(['2-opt', 'swap', 'insert', 'or-opt'])
        
        if operator == '2-opt':
            return self._two_opt(solution)
        elif operator == 'swap':
            return self._swap(solution)
        elif operator == 'insert':
            return self._insert(solution)
        else:  # 'or-opt'
            return self._or_opt(solution)
    
    def _two_opt(self, solution: List[int]) -> List[int]:
        """
        2-opt operator: Reverse a segment of the tour.
        
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
    
    def _swap(self, solution: List[int]) -> List[int]:
        """
        Swap operator: Exchange two customers.
        
        Args:
            solution: Current solution
        
        Returns:
            Neighbor solution
        """
        neighbor = solution.copy()
        size = len(neighbor)
        
        # Select two random positions
        i, j = random.sample(range(size), 2)
        
        # Swap
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor
    
    def _insert(self, solution: List[int]) -> List[int]:
        """
        Insert operator: Remove a customer and reinsert at different position.
        
        Args:
            solution: Current solution
        
        Returns:
            Neighbor solution
        """
        neighbor = solution.copy()
        size = len(neighbor)
        
        # Select customer to move and new position
        remove_idx = random.randint(0, size - 1)
        insert_idx = random.randint(0, size - 1)
        
        if remove_idx != insert_idx:
            customer = neighbor.pop(remove_idx)
            neighbor.insert(insert_idx, customer)
        
        return neighbor
    
    def _or_opt(self, solution: List[int]) -> List[int]:
        """
        Or-opt operator: Move a sequence of 1-3 customers to another position.
        
        Args:
            solution: Current solution
        
        Returns:
            Neighbor solution
        """
        neighbor = solution.copy()
        size = len(neighbor)
        
        if size < 4:
            return self._swap(solution)
        
        # Random sequence length (1-3)
        seq_len = random.randint(1, min(3, size - 1))
        
        # Select start position of sequence
        start_idx = random.randint(0, size - seq_len)
        
        # Extract sequence
        sequence = neighbor[start_idx:start_idx + seq_len]
        
        # Remove sequence
        del neighbor[start_idx:start_idx + seq_len]
        
        # Select insertion position
        insert_idx = random.randint(0, len(neighbor))
        
        # Insert sequence at new position
        for i, customer in enumerate(sequence):
            neighbor.insert(insert_idx + i, customer)
        
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
