"""
IGA-ACO Hybrid Algorithm for TW-TSP.

Combines Improved Genetic Algorithm with Ant Colony Optimization using
a dual-population structure for solution exchange.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
import random
from .base_algorithm import BaseAlgorithm
from problem.tw_tsp import TWTSPProblem


class ImprovedGeneticAlgorithm:
    """
    Improved Genetic Algorithm with Variable Neighborhood Descent.
    
    Features:
    - Hybrid initialization (Solomon I1 + Random)
    - Elite retention + Roulette wheel selection
    - VND with 6 neighborhood operators
    """
    
    def __init__(self, problem: TWTSPProblem, config: Dict[str, Any]):
        """
        Initialize IGA.
        
        Args:
            problem: TW-TSP problem instance
            config: Configuration dictionary
        """
        self.problem = problem
        self.population_size = config.get('iga_population_size', 100)
        self.elite_ratio = config.get('iga_elite_ratio', 0.1)
        self.vnd_probability = config.get('iga_vnd_probability', 0.3)
        
        self.population: List[List[int]] = []
        self.fitness_values: List[float] = []
        self.best_solution: List[int] = None
        self.best_fitness: float = float('inf')
    
    def initialize_population(self) -> None:
        """
        Initialize population using hybrid approach:
        - 50% Solomon I1 heuristic variants
        - 50% Random generation
        """
        self.population = []
        
        # 50% Solomon I1 heuristic
        solomon_count = self.population_size // 2
        for _ in range(solomon_count):
            solution = self._solomon_i1_init()
            self.population.append(solution)
        
        # 50% Random generation
        customer_ids = list(range(1, self.problem.num_customers + 1))
        for _ in range(self.population_size - solomon_count):
            individual = customer_ids.copy()
            random.shuffle(individual)
            self.population.append(individual)
        
        self._evaluate_population()
    
    def _solomon_i1_init(self) -> List[int]:
        """
        Solomon's I1 insertion heuristic for TW-TSP.
        
        Greedy construction considering both distance and time windows:
        1. Start with the customer farthest from depot
        2. Insert remaining customers at position that minimizes cost
        3. Cost considers distance increase and time window urgency
        
        Returns:
            Solution constructed using I1 heuristic
        """
        unvisited = set(range(1, self.problem.num_customers + 1))
        
        # Find customer farthest from depot to start
        max_dist = -1
        seed_customer = None
        for cid in unvisited:
            dist = self.problem.distance_matrix[0][cid]
            if dist > max_dist:
                max_dist = dist
                seed_customer = cid
        
        solution = [seed_customer]
        unvisited.remove(seed_customer)
        
        # Insert remaining customers
        while unvisited:
            best_customer = None
            best_position = 0
            best_cost = float('inf')
            
            for customer in unvisited:
                for pos in range(len(solution) + 1):
                    # Calculate insertion cost
                    cost = self._calculate_insertion_cost(solution, customer, pos)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_customer = customer
                        best_position = pos
            
            # Insert best customer at best position
            solution.insert(best_position, best_customer)
            unvisited.remove(best_customer)
        
        return solution
    
    def _calculate_insertion_cost(self, solution: List[int], customer: int, position: int) -> float:
        """
        Calculate TIME-AWARE cost of inserting a customer at a specific position.
        
        This simulates the actual route execution to evaluate:
        - Distance increase
        - Time window feasibility (arrival vs ready_time and due_time)
        - Impact on subsequent customers' time windows
        
        Args:
            solution: Current partial solution
            customer: Customer to insert
            position: Position to insert at
        
        Returns:
            Insertion cost (lower is better)
        """
        c1 = 1.0     # Distance weight
        c2 = 2.0     # Time window violation weight
        c3 = 0.5     # Time window slack weight
        
        # Create temporary solution with customer inserted
        temp_solution = solution[:position] + [customer] + solution[position:]
        
        # Simulate route execution to calculate time-aware cost
        current_time = 0.0
        current_location = 0  # Depot
        total_distance = 0.0
        total_violations = 0
        total_slack = 0.0  # How much slack time we have
        
        for cust_id in temp_solution:
            # Travel to customer
            travel_dist = self.problem.distance_matrix[current_location][cust_id]
            total_distance += travel_dist
            arrival_time = current_time + travel_dist
            
            # Get customer info
            cust = self.problem.customers[cust_id - 1]
            
            # Check time window
            if arrival_time < cust.ready_time:
                # Early arrival: wait until ready_time
                wait_time = cust.ready_time - arrival_time
                service_start = cust.ready_time
                total_slack += wait_time  # Waiting is wasted time
            elif arrival_time <= cust.due_time:
                # On time: no violation
                service_start = arrival_time
                slack = cust.due_time - arrival_time
                total_slack += slack
            else:
                # Late arrival: VIOLATION
                lateness = arrival_time - cust.due_time
                total_violations += lateness
                service_start = arrival_time
            
            # Update time and location
            current_time = service_start + cust.service_time
            current_location = cust_id
        
        # Return to depot
        total_distance += self.problem.distance_matrix[current_location][0]
        
        # Calculate insertion cost considering distance, violations, and slack
        # Penalize violations heavily, reward tight but feasible insertions
        cost = c1 * total_distance + c2 * total_violations - c3 * total_slack
        
        return cost
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals and update best solution."""
        self.fitness_values = []
        for individual in self.population:
            fitness = self.problem.calculate_fitness(individual)
            self.fitness_values.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = individual.copy()
    
    def evolve_generation(self) -> None:
        """
        Evolve one generation using elite retention + roulette wheel selection.
        """
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_values)
        
        # Elite retention
        elite_count = max(1, int(self.elite_ratio * self.population_size))
        new_population = []
        for i in range(elite_count):
            new_population.append(self.population[sorted_indices[i]].copy())
        
        # Roulette wheel selection for remaining
        while len(new_population) < self.population_size:
            parent1 = self._roulette_wheel_selection()
            parent2 = self._roulette_wheel_selection()
            
            # Crossover
            offspring1, offspring2 = self._order_crossover(parent1, parent2)
            
            # VND local search with probability
            if random.random() < self.vnd_probability:
                offspring1 = self._vnd(offspring1)
            if random.random() < self.vnd_probability and len(new_population) < self.population_size - 1:
                offspring2 = self._vnd(offspring2)
            
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        self.population = new_population
        self._evaluate_population()
    
    def _roulette_wheel_selection(self) -> List[int]:
        """
        Roulette wheel selection based on fitness.
        
        Selection probability: P(chr) = (max_fitness - fitness) / sum
        
        Returns:
            Selected individual
        """
        # Convert to maximization (invert fitness)
        max_fitness = max(self.fitness_values)
        adjusted_fitness = [max_fitness - f + 1 for f in self.fitness_values]
        total_fitness = sum(adjusted_fitness)
        
        # Roulette wheel spin
        spin = random.uniform(0, total_fitness)
        cumulative = 0
        
        for i, fitness in enumerate(adjusted_fitness):
            cumulative += fitness
            if cumulative >= spin:
                return self.population[i].copy()
        
        # Fallback
        return self.population[-1].copy()
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Ordered Crossover (OX) operator.
        
        Args:
            parent1: First parent
            parent2: Second parent
        
        Returns:
            Two offspring
        """
        size = len(parent1)
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Offspring 1
        offspring1 = [None] * size
        offspring1[point1:point2] = parent1[point1:point2]
        parent2_filtered = [gene for gene in parent2 if gene not in offspring1]
        idx = 0
        for i in range(size):
            if offspring1[i] is None:
                offspring1[i] = parent2_filtered[idx]
                idx += 1
        
        # Offspring 2
        offspring2 = [None] * size
        offspring2[point1:point2] = parent2[point1:point2]
        parent1_filtered = [gene for gene in parent1 if gene not in offspring2]
        idx = 0
        for i in range(size):
            if offspring2[i] is None:
                offspring2[i] = parent1_filtered[idx]
                idx += 1
        
        return offspring1, offspring2
    
    def _vnd(self, solution: List[int]) -> List[int]:
        """
        Variable Neighborhood Descent with 6 operators.
        
        Systematically explores neighborhoods until no improvement.
        
        Args:
            solution: Solution to optimize
        
        Returns:
            Improved solution
        """
        current_solution = solution.copy()
        current_fitness = self.problem.calculate_fitness(current_solution)
        
        # Neighborhood operators in order
        operators = [
            self._relocation_operator,
            self._two_opt_operator,
            self._three_opt_operator,
            self._or_opt_operator,
            self._swap_operator,
        ]
        
        k = 0  # Current neighborhood index
        max_k = len(operators)
        
        while k < max_k:
            # Apply k-th neighborhood operator
            new_solution = operators[k](current_solution)
            new_fitness = self.problem.calculate_fitness(new_solution)
            
            if new_fitness < current_fitness:
                # Improvement found: restart from first neighborhood
                current_solution = new_solution
                current_fitness = new_fitness
                k = 0
            else:
                # No improvement: try next neighborhood
                k += 1
        
        return current_solution
    
    def _relocation_operator(self, solution: List[int]) -> List[int]:
        """
        Relocation: Move a single node to a different position in the route.
        
        Args:
            solution: Current solution
        
        Returns:
            Modified solution
        """
        new_solution = solution.copy()
        size = len(new_solution)
        
        if size < 2:
            return new_solution
        
        # Select random node and new position
        node_idx = random.randint(0, size - 1)
        new_pos = random.randint(0, size - 1)
        
        if node_idx != new_pos:
            node = new_solution.pop(node_idx)
            new_solution.insert(new_pos, node)
        
        return new_solution
    
    def _two_opt_operator(self, solution: List[int]) -> List[int]:
        """
        2-opt: Reverse a segment between two randomly selected nodes.
        
        Args:
            solution: Current solution
        
        Returns:
            Modified solution
        """
        new_solution = solution.copy()
        size = len(new_solution)
        
        if size < 2:
            return new_solution
        
        # Select two random points
        i, j = sorted(random.sample(range(size), 2))
        
        # Reverse segment between i and j
        new_solution[i:j+1] = reversed(new_solution[i:j+1])
        
        return new_solution
    
    def _three_opt_operator(self, solution: List[int]) -> List[int]:
        """
        3-opt: Select three nodes and perform segment reversals/swaps.
        
        Args:
            solution: Current solution
        
        Returns:
            Modified solution
        """
        new_solution = solution.copy()
        size = len(new_solution)
        
        if size < 3:
            return new_solution
        
        # Select three random points
        indices = sorted(random.sample(range(size), 3))
        i, j, k = indices
        
        # Perform one of several 3-opt reconnections randomly
        reconnection = random.randint(0, 3)
        
        if reconnection == 0:
            # Reverse middle segment
            new_solution[i:j+1] = reversed(new_solution[i:j+1])
        elif reconnection == 1:
            # Reverse last segment
            new_solution[j:k+1] = reversed(new_solution[j:k+1])
        elif reconnection == 2:
            # Swap middle and last segments
            middle = new_solution[i:j+1]
            last = new_solution[j+1:k+1]
            new_solution = new_solution[:i] + last + middle + new_solution[k+1:]
        else:
            # Reverse both middle and last
            new_solution[i:j+1] = reversed(new_solution[i:j+1])
            new_solution[j:k+1] = reversed(new_solution[j:k+1])
        
        return new_solution
    
    def _or_opt_operator(self, solution: List[int]) -> List[int]:
        """
        Or-opt: Extract a segment of length 1-3 and reinsert elsewhere.
        
        Args:
            solution: Current solution
        
        Returns:
            Modified solution
        """
        new_solution = solution.copy()
        size = len(new_solution)
        
        if size < 2:
            return new_solution
        
        # Random segment length (1-3)
        segment_len = random.randint(1, min(3, size))
        
        # Random segment start position
        start_pos = random.randint(0, size - segment_len)
        
        # Extract segment
        segment = new_solution[start_pos:start_pos + segment_len]
        remaining = new_solution[:start_pos] + new_solution[start_pos + segment_len:]
        
        # Random insertion position
        if len(remaining) > 0:
            insert_pos = random.randint(0, len(remaining))
            new_solution = remaining[:insert_pos] + segment + remaining[insert_pos:]
        else:
            new_solution = segment
        
        return new_solution
    
    def _swap_operator(self, solution: List[int]) -> List[int]:
        """
        Swap: Exchange positions of two nodes.
        
        Args:
            solution: Current solution
        
        Returns:
            Modified solution
        """
        new_solution = solution.copy()
        size = len(new_solution)
        
        if size < 2:
            return new_solution
        
        # Select two random positions
        i, j = random.sample(range(size), 2)
        
        # Swap
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        
        return new_solution
    
    def inject_solution(self, solution: List[int]) -> None:
        """
        Inject external solution into population (from ACO).
        
        Replaces worst individual if external solution is better.
        
        Args:
            solution: External solution to inject
        """
        fitness = self.problem.calculate_fitness(solution)
        
        # Find worst individual
        worst_idx = np.argmax(self.fitness_values)
        
        # Replace if external solution is better
        if fitness < self.fitness_values[worst_idx]:
            self.population[worst_idx] = solution.copy()
            self.fitness_values[worst_idx] = fitness
            
            # Update best if necessary
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()


class AntColonyOptimization:
    """
    Ant Colony Optimization for TW-TSP.
    
    Features:
    - Nearest-neighbor initialization
    - Probabilistic solution construction
    - Local and global pheromone updates
    """
    
    def __init__(self, problem: TWTSPProblem, config: Dict[str, Any]):
        """
        Initialize ACO.
        
        Args:
            problem: TW-TSP problem instance
            config: Configuration dictionary
        """
        self.problem = problem
        self.num_ants = config.get('aco_num_ants', 50)
        self.alpha = config.get('aco_alpha', 1.0)  # Pheromone importance
        self.beta = config.get('aco_beta', 2.0)   # Heuristic importance
        self.rho = config.get('aco_rho', 0.1)     # Evaporation rate
        self.q0 = config.get('aco_q0', 0.9)       # Exploitation vs exploration
        
        # Pheromone matrix
        n = problem.num_customers + 1
        self.pheromone = np.ones((n, n))
        self.tau0 = 1.0  # Initial pheromone
        
        # Initialize with nearest neighbor
        self._initialize_pheromone()
        
        self.best_solution: List[int] = None
        self.best_fitness: float = float('inf')
    
    def _initialize_pheromone(self) -> None:
        """
        Initialize pheromone using nearest-neighbor heuristic.
        """
        # Build nearest-neighbor solution
        nn_solution = self._nearest_neighbor_heuristic()
        nn_fitness = self.problem.calculate_fitness(nn_solution)
        
        # Set initial pheromone based on NN solution quality
        self.tau0 = 1.0 / (self.problem.num_customers * nn_fitness)
        self.pheromone.fill(self.tau0)
        
        # Store as initial best
        self.best_solution = nn_solution
        self.best_fitness = nn_fitness
    
    def _nearest_neighbor_heuristic(self) -> List[int]:
        """
        Construct solution using nearest-neighbor heuristic.
        
        Returns:
            Nearest-neighbor solution
        """
        unvisited = set(range(1, self.problem.num_customers + 1))
        solution = []
        current = 0  # Start at depot
        
        while unvisited:
            nearest = None
            min_dist = float('inf')
            
            for customer in unvisited:
                dist = self.problem.distance_matrix[current][customer]
                if dist < min_dist:
                    min_dist = dist
                    nearest = customer
            
            solution.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return solution
    
    def construct_solutions(self) -> List[List[int]]:
        """
        Construct solutions for all ants.
        
        Returns:
            List of solutions constructed by ants
        """
        solutions = []
        
        for _ in range(self.num_ants):
            solution = self._construct_ant_solution()
            solutions.append(solution)
        
        return solutions
    
    def _construct_ant_solution(self) -> List[int]:
        """
        Construct TIME-AWARE solution for a single ant using ACO probability rules.
        
        Now tracks current time to make time-aware decisions.
        
        Returns:
            Solution constructed by ant
        """
        unvisited = set(range(1, self.problem.num_customers + 1))
        solution = []
        current = 0  # Start at depot
        current_time = 0.0  # Track time for time-aware decisions
        
        while unvisited:
            # Select next customer based on current time (TIME-AWARE)
            next_customer = self._select_next_customer(current, unvisited, current_time)
            
            # Local pheromone update
            self._local_pheromone_update(current, next_customer)
            
            # Update current time based on travel and service
            travel_dist = self.problem.distance_matrix[current][next_customer]
            arrival_time = current_time + travel_dist
            
            customer = self.problem.customers[next_customer - 1]
            
            # Update service start time (wait if early)
            if arrival_time < customer.ready_time:
                service_start = customer.ready_time
            else:
                service_start = arrival_time
            
            current_time = service_start + customer.service_time
            
            solution.append(next_customer)
            unvisited.remove(next_customer)
            current = next_customer
        
        return solution
    
    def _select_next_customer(self, current: int, unvisited: set, current_time: float = 0.0) -> int:
        """
        Select next customer using TIME-AWARE ACO probability rules.
        
        P_ij = (tau_ij^alpha * eta_ij^beta) / sum(...)
        
        Enhanced with time window awareness in heuristic calculation.
        
        Args:
            current: Current location
            unvisited: Set of unvisited customers
            current_time: Current time in route (for time window checking)
        
        Returns:
            Selected customer
        """
        # Exploitation vs exploration
        if random.random() < self.q0:
            # Exploitation: choose best customer
            best_customer = None
            best_value = -float('inf')
            
            for customer in unvisited:
                tau = self.pheromone[current][customer]
                eta = self._heuristic_info(current, customer, current_time)
                value = (tau ** self.alpha) * (eta ** self.beta)
                
                if value > best_value:
                    best_value = value
                    best_customer = customer
            
            return best_customer
        else:
            # Exploration: probabilistic selection
            probabilities = []
            customers = list(unvisited)
            
            for customer in customers:
                tau = self.pheromone[current][customer]
                eta = self._heuristic_info(current, customer, current_time)
                prob = (tau ** self.alpha) * (eta ** self.beta)
                probabilities.append(prob)
            
            # Normalize probabilities
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
            else:
                # Uniform distribution if all probabilities are 0
                probabilities = [1.0 / len(customers)] * len(customers)
            
            # Roulette wheel selection
            return np.random.choice(customers, p=probabilities)
    
    def _heuristic_info(self, i: int, j: int, current_time: float = 0.0) -> float:
        """
        Calculate TIME-AWARE heuristic information (visibility).
        
        Enhanced heuristic that considers:
        - Distance (shorter is better)
        - Time window feasibility (prefer feasible arrivals)
        - Time window urgency (prefer customers with earlier deadlines)
        
        eta_ij = base_visibility * time_window_factor
        
        Args:
            i: From location
            j: To location
            current_time: Current time in route
        
        Returns:
            Heuristic value (higher is better)
        """
        dist = self.problem.distance_matrix[i][j]
        base_visibility = 1.0 / (dist + 1e-10)  # Avoid division by zero
        
        # Time window awareness
        if j > 0:  # Not depot
            customer = self.problem.customers[j - 1]
            arrival_time = current_time + dist
            
            # Calculate time window factor
            if arrival_time < customer.ready_time:
                # Early: will have to wait (less desirable)
                wait_time = customer.ready_time - arrival_time
                time_factor = 1.0 / (1.0 + 0.1 * wait_time)
            elif arrival_time <= customer.due_time:
                # On time: GOOD! Prefer customers we can reach without violations
                slack = customer.due_time - arrival_time
                time_factor = 1.5 + 0.5 * (1.0 / (1.0 + slack))  # Boost feasible customers
            else:
                # Late: VIOLATION! Heavily penalize
                lateness = arrival_time - customer.due_time
                time_factor = 0.1 / (1.0 + lateness)  # Strong penalty
            
            # Also consider due_time urgency (prefer customers with tighter deadlines)
            urgency_factor = 1.0 / (customer.due_time + 1.0)
            
            return base_visibility * time_factor * (1.0 + urgency_factor)
        
        return base_visibility
    
    def _local_pheromone_update(self, i: int, j: int) -> None:
        """
        Local pheromone update (during solution construction).
        
        tau_ij = (1 - rho) * tau_ij + rho * tau0
        
        Args:
            i: From location
            j: To location
        """
        self.pheromone[i][j] = (1 - self.rho) * self.pheromone[i][j] + self.rho * self.tau0
        self.pheromone[j][i] = self.pheromone[i][j]  # Symmetric
    
    def global_pheromone_update(self, best_solution: List[int], best_fitness: float) -> None:
        """
        Global pheromone update (after all ants finish).
        
        Only updates pheromone on the best path:
        tau_ij = (1 - rho) * tau_ij + delta_tau_ij
        
        where delta_tau_ij = 1 / fitness if edge in best path, else 0
        
        Args:
            best_solution: Best solution found
            best_fitness: Fitness of best solution
        """
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Deposit pheromone on best path
        delta_tau = 1.0 / best_fitness
        
        # Depot to first customer
        self.pheromone[0][best_solution[0]] += delta_tau
        self.pheromone[best_solution[0]][0] += delta_tau
        
        # Between customers
        for i in range(len(best_solution) - 1):
            from_node = best_solution[i]
            to_node = best_solution[i + 1]
            self.pheromone[from_node][to_node] += delta_tau
            self.pheromone[to_node][from_node] += delta_tau
        
        # Last customer to depot
        self.pheromone[best_solution[-1]][0] += delta_tau
        self.pheromone[0][best_solution[-1]] += delta_tau
    
    def update_from_iga(self, iga_best_solution: List[int], iga_best_fitness: float) -> None:
        """
        Update pheromone matrix using best solution from IGA.
        
        This is the IGA -> ACO exchange mechanism.
        
        Args:
            iga_best_solution: Best solution from IGA
            iga_best_fitness: Fitness of IGA's best solution
        """
        # Update pheromone based on IGA solution
        self.global_pheromone_update(iga_best_solution, iga_best_fitness)
        
        # Update ACO's best if IGA found better
        if iga_best_fitness < self.best_fitness:
            self.best_fitness = iga_best_fitness
            self.best_solution = iga_best_solution.copy()


class IGAACOHybrid(BaseAlgorithm):
    """
    Hybrid IGA-ACO algorithm for TW-TSP.
    
    Combines Improved Genetic Algorithm with Ant Colony Optimization
    using dual-population structure and solution exchange.
    
    Configuration parameters:
        - num_iterations: Total number of iterations
        - iga_population_size: IGA population size
        - iga_elite_ratio: IGA elite retention ratio
        - iga_vnd_probability: Probability of applying VND
        - aco_num_ants: Number of ants in ACO
        - aco_alpha: Pheromone importance
        - aco_beta: Heuristic importance
        - aco_rho: Evaporation rate
        - aco_q0: Exploitation parameter
        - exchange_interval: Interval for solution exchange
    """
    
    def __init__(self, problem: TWTSPProblem, config: Dict[str, Any]):
        """
        Initialize IGA-ACO hybrid.
        
        Args:
            problem: TW-TSP problem instance
            config: Configuration dictionary
        """
        super().__init__(problem, config)
        
        self.num_iterations = config.get('num_iterations', 500)
        self.exchange_interval = config.get('exchange_interval', 10)
        
        # Initialize IGA and ACO
        self.iga = ImprovedGeneticAlgorithm(problem, config)
        self.aco = AntColonyOptimization(problem, config)
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Execute the hybrid IGA-ACO algorithm.
        
        Returns:
            Tuple of (best_solution, best_fitness, fitness_history)
        """
        # Initialize IGA population
        self.iga.initialize_population()
        
        # Main iteration loop
        for iteration in range(self.num_iterations):
            # IGA evolution
            self.iga.evolve_generation()
            
            # ACO solution construction
            aco_solutions = self.aco.construct_solutions()
            
            # Find best ACO solution in this iteration
            best_aco_solution = None
            best_aco_fitness = float('inf')
            
            for solution in aco_solutions:
                fitness = self.problem.calculate_fitness(solution)
                if fitness < best_aco_fitness:
                    best_aco_fitness = fitness
                    best_aco_solution = solution
            
            # Update ACO best
            if best_aco_fitness < self.aco.best_fitness:
                self.aco.best_fitness = best_aco_fitness
                self.aco.best_solution = best_aco_solution.copy()
            
            # Global pheromone update with iteration best
            self.aco.global_pheromone_update(best_aco_solution, best_aco_fitness)
            
            # Solution exchange at specified intervals
            if (iteration + 1) % self.exchange_interval == 0:
                # IGA -> ACO: Update pheromone with IGA's best
                self.aco.update_from_iga(self.iga.best_solution, self.iga.best_fitness)
                
                # ACO -> IGA: Inject ACO's best into IGA population
                self.iga.inject_solution(self.aco.best_solution)
            
            # Track overall best
            current_best_fitness = min(self.iga.best_fitness, self.aco.best_fitness)
            
            if self.iga.best_fitness < self.aco.best_fitness:
                current_best_solution = self.iga.best_solution
            else:
                current_best_solution = self.aco.best_solution
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution.copy()
            
            # Record fitness history
            self.fitness_history.append(self.best_fitness)
            
            # Optional: Print progress
            if (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iterations}: "
                      f"Best Fitness = {self.best_fitness:.2f} "
                      f"(IGA: {self.iga.best_fitness:.2f}, ACO: {self.aco.best_fitness:.2f})")
        
        return self.best_solution, self.best_fitness, self.fitness_history
