"""
Time-Windowed Traveling Salesman Problem (TW-TSP) implementation.

This module defines the problem structure including customers with time windows
and the fitness evaluation function.
"""
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import os


@dataclass
class Customer:
    """
    Represents a customer location with time window constraints.
    
    Attributes:
        id: Unique customer identifier
        x: X-coordinate position
        y: Y-coordinate position
        demand: Customer demand (not used in basic TW-TSP but useful for extensions)
        ready_time: Earliest service start time (beginning of time window)
        due_time: Latest service start time (end of time window)
        service_time: Time required to service this customer
    """
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float


class TWTSPProblem:
    """
    Time-Windowed Traveling Salesman Problem class.
    
    Handles problem instance loading, distance calculation, and fitness evaluation
    with time window constraint penalties.
    """
    
    def __init__(self, filepath: str = None, format: str = 'solomon'):
        """
        Initialize TW-TSP problem instance.
        
        Args:
            filepath: Path to problem instance file.
            format: Format of the instance file. Options:
                   - 'solomon': Original Solomon format (default)
                   - 'spb': Solomon-Potvin-Bengio format
                   If None or file doesn't exist, creates a mock instance.
        """
        self.customers: List[Customer] = []
        self.depot: Customer = None
        self.distance_matrix: np.ndarray = None
        self.num_customers: int = 0
        self.instance_name: str = ""
        self.best_known_fitness: float = None
        self.best_known_distance: float = None
        self.best_known_violations: int = None
        self.best_known_reported_cost: float = None  # Original cost from file
        
        if filepath and os.path.exists(filepath):
            if format == 'spb':
                self._load_solomon_potvin_bengio_instance(filepath)
            else:
                self._load_solomon_instance(filepath)
            
            # Extract instance name from filepath
            self.instance_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # Try to load best-known solution if available
            self._load_best_known_solution()
        else:
            # Create mock data for testing
            self.instance_name = "mock_instance"
            self._create_mock_instance()
        
        self._calculate_distance_matrix()
    
    def _load_solomon_instance(self, filepath: str) -> None:
        """
        Load problem instance from Solomon format file.
        
        Solomon format structure:
        - Header lines with problem name and vehicle info
        - Customer data: CUST_NO X Y DEMAND READY_TIME DUE_TIME SERVICE_TIME
        - First customer (0) is the depot
        
        Args:
            filepath: Path to the problem instance file
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines (typically 4 lines in Solomon format)
        # Adjust this based on actual file format
        customer_data_start = 0
        for i, line in enumerate(lines):
            if 'CUST' in line.upper() or 'CUSTOMER' in line.upper():
                customer_data_start = i + 1
                break
        
        # If header not found, assume data starts after 9 lines (typical Solomon format)
        if customer_data_start == 0:
            customer_data_start = 9
        
        customers_list = []
        for line in lines[customer_data_start:]:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            
            try:
                cust_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_time = float(parts[5])
                service_time = 0.0  # Set service time to 0
                
                customer = Customer(
                    id=cust_id,
                    x=x,
                    y=y,
                    demand=demand,
                    ready_time=ready_time,
                    due_time=due_time,
                    service_time=service_time
                )
                
                if cust_id == 0:
                    self.depot = customer
                else:
                    customers_list.append(customer)
            except (ValueError, IndexError):
                continue
        
        self.customers = customers_list
        self.num_customers = len(self.customers)
    
    def _create_mock_instance(self) -> None:
        """
        Create a mock problem instance for testing purposes.
        
        Creates 10 customers in a 100x100 grid with varying time windows.
        """
        # Depot at center
        self.depot = Customer(
            id=0,
            x=50.0,
            y=50.0,
            demand=0.0,
            ready_time=0.0,
            due_time=230.0,
            service_time=0.0
        )
        
        # Create 10 random customers
        np.random.seed(42)  # For reproducibility
        num_customers = 10
        
        for i in range(1, num_customers + 1):
            x = np.random.uniform(10, 90)
            y = np.random.uniform(10, 90)
            ready = np.random.uniform(0, 100)
            due = ready + np.random.uniform(30, 80)
            service = 0.0  # Set service time to 0
            
            customer = Customer(
                id=i,
                x=x,
                y=y,
                demand=np.random.uniform(5, 25),
                ready_time=ready,
                due_time=due,
                service_time=service
            )
            self.customers.append(customer)
        
        self.num_customers = len(self.customers)
    
    def _load_solomon_potvin_bengio_instance(self, filepath: str) -> None:
        """
        Load problem instance from Solomon-Potvin-Bengio format.
        
        Format:
        - Line 1: Number of nodes (including depot)
        - Lines 2 to n+1: Distance matrix (n x n)
        - Lines n+2 to 2n+1: Time windows (earliest, latest) for each node
        - Optional: Comments prefixed by #
        
        Args:
            filepath: Path to the problem instance file
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
        
        # Line 1: Number of nodes
        num_nodes = int(lines[0])
        self.num_customers = num_nodes - 1  # Exclude depot
        
        # Lines 2 to n+1: Distance matrix
        self.distance_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            row_data = lines[1 + i].split()
            for j in range(num_nodes):
                self.distance_matrix[i][j] = float(row_data[j])
        
        # Lines n+2 to 2n+1: Time windows
        time_windows = []
        for i in range(num_nodes):
            tw_data = lines[1 + num_nodes + i].split()
            earliest = float(tw_data[0])
            latest = float(tw_data[1])
            time_windows.append((earliest, latest))
            
        # Create depot (node 0)
        self.depot = Customer(
            id=0,
            x=0.0,  # Coordinates not provided in SPB format
            y=0.0,
            demand=0.0,
            ready_time=time_windows[0][0],
            due_time=time_windows[0][1],
            service_time=0.0
        )
        assert(self.depot is not None)
        
        # Create customers (nodes 1 to n-1)
        self.customers = []
        for i in range(1, num_nodes):
            customer = Customer(
                id=i,
                x=0.0,  # Coordinates not provided in SPB format
                y=0.0,
                demand=0.0,  # Demand not provided in SPB format
                ready_time=time_windows[i][0],
                due_time=time_windows[i][1],
                service_time=0.0  # Service time already included in distance matrix
            )
            self.customers.append(customer)
    
    def _load_best_known_solution(self) -> None:
        """
        Load best-known solution from tsptw-2010-best-known directory.
        
        The file format is expected to be:
        # Instance                  Cost CV Permutation
        instance_name.txt          cost cv node1 node2 node3 ...
        """
        # Look for best-known solutions file
        best_known_dir = 'data/tsptw-2010-best-known'
        if not os.path.exists(best_known_dir):
            return
        
        # Try to find the SolomonPotvinBengio.best file
        possible_files = [
            os.path.join(best_known_dir, 'SolomonPotvinBengio.best'),
        ]
        
        # Also check for any TXT or CSV file in the directory
        # if os.path.isdir(best_known_dir):
        #     for file in os.listdir(best_known_dir):
        #         if file.endswith('.best') or file.endswith('.txt') or file.endswith('.csv'):
        #             filepath = os.path.join(best_known_dir, file)
        #             if filepath not in possible_files:
        #                 possible_files.append(filepath)
        
        for filepath in possible_files:
            if os.path.exists(filepath):
                try:
                    print(f"Loading best-known solution from: {filepath}")
                    self._parse_best_known_file(filepath)
                    if self.best_known_fitness is not None:
                        break
                except Exception as e:
                    print(f"Warning: Could not load from {filepath}: {e}")
                    continue
    
    def _parse_best_known_file(self, filepath: str) -> None:
        """
        Parse best-known solution file and extract values for this instance.
        
        Expected format (SolomonPotvinBengio.best):
        # Instance                  Cost CV Permutation
        instance_name.txt          cost cv node1 node2 node3 ...
        
        Where:
        - Instance: filename (e.g., rc_201.1.txt)
        - Cost: reported cost from the file (kept as-is)
        - CV: constraint violations count from file (NOT used)
        - Permutation: tour sequence that will be evaluated using calculate_solution_details()
        
        This function will:
        1. Keep the Cost value from file as best_known_cost
        2. Calculate fitness and violations by evaluating the permutation
        
        Args:
            filepath: Path to the best-known solutions file
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header and comments
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Split by whitespace
            parts = line.split()
            
            if len(parts) < 4:  # Need at least: instance cost cv node1
                continue
            
            # Check if this line is for our instance
            instance_name_in_file = parts[0].lower().replace('.txt', '')
            our_instance = self.instance_name.lower().replace('.txt', '')
            
            # print('###', instance_name_in_file, our_instance)

            if instance_name_in_file == our_instance or instance_name_in_file in our_instance:
                try:
                    # Parse: instance_name cost cv [permutation...]
                    reported_cost = float(parts[1])
                    reported_cv = int(parts[2])
                    permutation = [int(x) for x in parts[3:]]
                    
                    if not permutation:
                        print(f"Warning: No permutation found in line: {line}")
                        continue
                    
                    # Calculate fitness and violations from permutation using our fitness function
                    details = self.calculate_solution_details(permutation)
                    
                    # Store the calculated values
                    self.best_known_fitness = details['fitness']
                    self.best_known_distance = details['total_distance']
                    self.best_known_violations = details['num_violations']
                    
                    # Also store the reported cost for comparison
                    self.best_known_reported_cost = reported_cost
                    
                    print(f"  ✓ Best-known solution parsed:")
                    print(f"    - Reported Cost: {reported_cost:.2f}")
                    print(f"    - Calculated Fitness: {self.best_known_fitness:.2f}")
                    print(f"    - Calculated Distance: {self.best_known_distance:.2f}")
                    print(f"    - Calculated Violations: {self.best_known_violations}")
                    
                    break
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line} - {e}")
                    continue
    
    def _calculate_distance_matrix(self) -> None:
        """
        Calculate Euclidean distance matrix between all locations.
        
        Matrix includes depot (index 0) and all customers (indices 1 to n).
        Note: This is only called when coordinates are available (not for SPB format).
        """
        # Skip if distance matrix already loaded (e.g., from SPB format)
        if self.distance_matrix is not None:
            return
        
        n = self.num_customers + 1  # +1 for depot
        self.distance_matrix = np.zeros((n, n))
        
        all_locations = [self.depot] + self.customers
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = all_locations[i].x - all_locations[j].x
                    dy = all_locations[i].y - all_locations[j].y
                    self.distance_matrix[i][j] = np.sqrt(dx * dx + dy * dy)
    
    def calculate_fitness(self, solution: List[int]) -> float:
        """
        Calculate fitness value for a given solution.
        
        Fitness = Total Distance + Penalty for Time Window Violations
        
        The function simulates the route execution:
        1. Start at depot at time 0
        2. Travel to each customer in sequence
        3. If arrive early, wait until ready_time
        4. If arrive late (after due_time), apply heavy penalty
        5. Service the customer and continue
        
        Args:
            solution: List of customer IDs in visit order (e.g., [1, 5, 2, 4, 3])
                     Should NOT include depot (0)
        
        Returns:
            Total fitness value (lower is better)
        """
        if not solution or len(solution) == 0:
            return float('inf')
        
        total_distance = 0.0
        total_penalty = 0.0
        current_time = 0.0
        current_location = 0  # Start at depot (index 0)
        
        # Heavy penalty coefficient for time window violations
        PENALTY_COEFFICIENT = 1000.0
        
        for customer_id in solution:
            # Customer ID in solution corresponds to index in distance matrix
            next_location = customer_id
            
            # Validate customer ID
            if next_location < 1 or next_location > self.num_customers:
                return float('inf')
            
            # Travel to next customer
            travel_distance = self.distance_matrix[current_location][next_location]
            total_distance += travel_distance
            
            # Assume travel time equals distance (unit speed = 1)
            travel_time = travel_distance
            arrival_time = current_time + travel_time
            
            # Get customer details
            customer = self.customers[customer_id - 1]  # -1 because customer list doesn't include depot
            
            # Check time window constraints
            if arrival_time < customer.ready_time:
                # Arrive early: wait until ready_time
                wait_time = customer.ready_time - arrival_time
                service_start = customer.ready_time
            elif arrival_time <= customer.due_time:
                # Arrive within time window
                wait_time = 0.0
                service_start = arrival_time
            else:
                # Arrive late: VIOLATION - apply penalty
                lateness = arrival_time - customer.due_time
                total_penalty += PENALTY_COEFFICIENT * lateness
                service_start = arrival_time
                wait_time = 0.0
            
            # Complete service and update current time
            current_time = service_start
            current_location = next_location
        
        # Return to depot
        return_distance = self.distance_matrix[current_location][0]
        total_distance += return_distance
        
        # Total fitness
        fitness = total_distance + total_penalty
        
        return fitness
    
    def get_distance_matrix(self) -> np.ndarray:
        """
        Get the distance matrix.
        
        Returns:
            Distance matrix as numpy array
        """
        return self.distance_matrix.copy()
    
    def get_customer_details(self) -> Dict[int, Dict]:
        """
        Get detailed information about all customers.
        
        Returns:
            Dictionary mapping customer ID to customer details
        """
        details = {}
        assert(self.depot is not None)
        # Include depot
        details[0] = {
            'id': self.depot.id,
            'x': self.depot.x,
            'y': self.depot.y,
            'demand': self.depot.demand,
            'ready_time': self.depot.ready_time,
            'due_time': self.depot.due_time,
            'service_time': self.depot.service_time
        }
        
        # Include all customers
        for customer in self.customers:
            details[customer.id] = {
                'id': customer.id,
                'x': customer.x,
                'y': customer.y,
                'demand': customer.demand,
                'ready_time': customer.ready_time,
                'due_time': customer.due_time,
                'service_time': customer.service_time
            }
        
        return details
    
    def is_valid_solution(self, solution: List[int]) -> bool:
        """
        Check if a solution is valid (all customers visited exactly once).
        
        Args:
            solution: List of customer IDs
        
        Returns:
            True if valid, False otherwise
        """
        if len(solution) != self.num_customers:
            return False
        
        expected_ids = set(range(1, self.num_customers + 1))
        solution_ids = set(solution)
        
        return expected_ids == solution_ids
    
    def get_best_known_solution(self) -> Dict[str, float]:
        """
        Get best-known solution metrics if available.
        
        Returns:
            Dictionary containing best-known metrics or None values if not available.
            Includes both calculated values and reported cost from file.
        """
        return {
            'fitness': self.best_known_fitness,
            'distance': self.best_known_distance,
            'violations': self.best_known_violations,
            'reported_cost': self.best_known_reported_cost
        }
    
    def has_best_known_solution(self) -> bool:
        """
        Check if best-known solution is available.
        
        Returns:
            True if best-known solution is loaded, False otherwise
        """
        return self.best_known_fitness is not None
    
    def calculate_solution_details(self, solution: List[int]) -> Dict[str, float]:
        """
        Calculate detailed metrics for a solution including cost and constraint violations.
        
        This function provides a breakdown of:
        - Total distance traveled (without penalties)
        - Number of time window violations
        - Total penalty amount
        - Final fitness (distance + penalties)
        
        Args:
            solution: List of customer IDs in visit order
        
        Returns:
            Dictionary containing:
                - 'total_distance': Pure travel distance
                - 'num_violations': Count of customers visited late
                - 'total_penalty': Sum of all time window violation penalties
                - 'fitness': Total fitness (distance + penalties)
        """
        if not solution or len(solution) == 0:
            return {
                'total_distance': float('inf'),
                'num_violations': self.num_customers,
                'total_penalty': float('inf'),
                'fitness': float('inf')
            }
        
        total_distance = 0.0
        total_penalty = 0.0
        num_violations = 0
        current_time = 0.0
        current_location = 0  # Start at depot
        
        # Heavy penalty coefficient for time window violations
        PENALTY_COEFFICIENT = 1000.0
        
        for customer_id in solution:
            # Validate customer ID
            if customer_id < 1 or customer_id > self.num_customers:
                return {
                    'total_distance': float('inf'),
                    'num_violations': self.num_customers,
                    'total_penalty': float('inf'),
                    'fitness': float('inf')
                }
            
            next_location = customer_id
            
            # Travel to next customer
            travel_distance = self.distance_matrix[current_location][next_location]
            total_distance += travel_distance
            
            # Calculate arrival time
            travel_time = travel_distance
            arrival_time = current_time + travel_time
            
            # Get customer details
            customer = self.customers[customer_id - 1]
            
            # Check time window constraints
            if arrival_time < customer.ready_time:
                # Arrive early: wait until ready_time
                wait_time = customer.ready_time - arrival_time
                service_start = customer.ready_time
            elif arrival_time <= customer.due_time:
                # Arrive within time window: OK
                wait_time = 0.0
                service_start = arrival_time
            else:
                # Arrive late: VIOLATION
                lateness = arrival_time - customer.due_time
                total_penalty += PENALTY_COEFFICIENT * lateness
                num_violations += 1
                service_start = arrival_time
                wait_time = 0.0
            
            # Complete service and update current time
            current_time = service_start + customer.service_time
            current_location = next_location
        
        # Return to depot
        return_distance = self.distance_matrix[current_location][0]
        total_distance += return_distance
        
        # Calculate final fitness
        fitness = total_distance + total_penalty
        
        return {
            'total_distance': total_distance,
            'num_violations': num_violations,
            'total_penalty': total_penalty,
            'fitness': fitness
        }
