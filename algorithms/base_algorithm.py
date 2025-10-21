"""
Base algorithm class for all optimization algorithms.

Provides abstract interface that all algorithms must implement.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem.tw_tsp import TWTSPProblem


class BaseAlgorithm(ABC):
    """
    Abstract base class for all optimization algorithms.
    
    All algorithm implementations must inherit from this class and implement
    the solve() method.
    """
    
    def __init__(self, problem: TWTSPProblem, config: Dict[str, Any]):
        """
        Initialize the algorithm.
        
        Args:
            problem: TW-TSP problem instance to solve
            config: Dictionary containing algorithm-specific configuration parameters
        """
        self.problem = problem
        self.config = config
        self.best_solution: List[int] = None
        self.best_fitness: float = float('inf')
        self.fitness_history: List[float] = []
    
    @abstractmethod
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Solve the problem using the specific algorithm.
        
        This method must be implemented by all concrete algorithm classes.
        
        Returns:
            Tuple containing:
                - best_solution: List of customer IDs representing the best route found
                - best_fitness: Fitness value of the best solution
                - fitness_history: List of best fitness values over iterations/generations
        """
        pass
    
    def get_best_solution(self) -> List[int]:
        """
        Get the best solution found.
        
        Returns:
            Best solution as list of customer IDs
        """
        return self.best_solution
    
    def get_best_fitness(self) -> float:
        """
        Get the best fitness value found.
        
        Returns:
            Best fitness value
        """
        return self.best_fitness
    
    def get_fitness_history(self) -> List[float]:
        """
        Get the history of best fitness values.
        
        Returns:
            List of fitness values over time
        """
        return self.fitness_history
