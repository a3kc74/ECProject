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
    
    def __init__(self, filepath: str = None):
        """
        Initialize TW-TSP problem instance.
        
        Args:
            filepath: Path to problem instance file (Solomon format expected).
                     If None or file doesn't exist, creates a mock instance.
        """
        self.customers: List[Customer] = []
        self.depot: Customer = None
        self.distance_matrix: np.ndarray = None
        self.num_customers: int = 0
        
        if filepath and os.path.exists(filepath):
            self._load_solomon_instance(filepath)
        else:
            # Create mock data for testing
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
                service_time = float(parts[6])
                
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
            service = np.random.uniform(5, 15)
            
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
    
    def _calculate_distance_matrix(self) -> None:
        """
        Calculate Euclidean distance matrix between all locations.
        
        Matrix includes depot (index 0) and all customers (indices 1 to n).
        """
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
            current_time = service_start + customer.service_time
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
