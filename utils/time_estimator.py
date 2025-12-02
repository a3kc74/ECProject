"""
Time estimation utilities for algorithm execution.

Provides time prediction, progress tracking, and real-time updates.
"""
import time
import math
from typing import Optional, Tuple, Dict


class TimeEstimator:
    """Estimates algorithm execution time and provides progress tracking."""
    
    def __init__(self, total_iterations: int, problem_size: int):
        """
        Initialize time estimator.
        
        Args:
            total_iterations: Expected total iterations for the algorithm
            problem_size: Number of customers in the problem
        """
        self.total_iterations = total_iterations
        self.problem_size = problem_size
        self.start_time = None
        self.iterations_completed = 0
        self.iteration_times = []  # Track individual iteration times
        self.sampling_interval = max(1, total_iterations // 100)  # Sample ~100 times
        
    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()
        self.iterations_completed = 0
        self.iteration_times = []
    
    def update(self, iterations_completed: int) -> None:
        """
        Update progress.
        
        Args:
            iterations_completed: Number of iterations completed so far
        """
        self.iterations_completed = iterations_completed
    
    def record_iteration_time(self, iteration_num: int, elapsed_time: float) -> None:
        """
        Record time for an iteration (for accurate estimation).
        
        Args:
            iteration_num: Current iteration number
            elapsed_time: Time spent on this iteration
        """
        if iteration_num % self.sampling_interval == 0:
            self.iteration_times.append(elapsed_time)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """
        Estimate remaining execution time in seconds.
        
        Returns:
            Estimated seconds remaining, or None if not enough data
        """
        if self.iterations_completed == 0:
            return None
        
        elapsed = self.get_elapsed_time()
        
        if elapsed == 0:
            return None
        
        # Average time per iteration
        avg_time_per_iteration = elapsed / self.iterations_completed
        
        # Remaining iterations
        remaining_iterations = self.total_iterations - self.iterations_completed
        
        # Estimate remaining time
        estimated_remaining = avg_time_per_iteration * remaining_iterations
        
        return estimated_remaining
    
    def get_estimated_total_time(self) -> Optional[float]:
        """
        Estimate total execution time.
        
        Returns:
            Estimated total seconds, or None if not enough data
        """
        if self.iterations_completed == 0:
            return None
        
        elapsed = self.get_elapsed_time()
        remaining = self.get_estimated_time_remaining()
        
        if remaining is None:
            return None
        
        return elapsed + remaining
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_iterations == 0:
            return 0.0
        return min(100.0, (self.iterations_completed / self.total_iterations) * 100)
    
    def get_progress_bar(self, width: int = 30) -> str:
        """
        Get visual progress bar.
        
        Args:
            width: Width of the progress bar in characters
        
        Returns:
            String representation of progress bar
        """
        percentage = self.get_progress_percentage()
        filled = int((percentage / 100.0) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {percentage:.1f}%"
    
    def get_time_estimate_string(self) -> str:
        """
        Get formatted string with time estimates.
        
        Returns:
            Human-readable time estimate string
        """
        elapsed = self.get_elapsed_time()
        remaining = self.get_estimated_time_remaining()
        total = self.get_estimated_total_time()
        progress = self.get_progress_percentage()
        
        elapsed_str = self._format_time(elapsed)
        
        if remaining is None:
            return f"Elapsed: {elapsed_str} | Progress: {progress:.1f}%"
        
        remaining_str = self._format_time(remaining)
        total_str = self._format_time(total) if total else "??:??"
        
        return (f"Progress: {progress:.1f}% | "
                f"Elapsed: {elapsed_str} | "
                f"Remaining: {remaining_str} | "
                f"Total: {total_str}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format seconds into human-readable time string.
        
        Args:
            seconds: Time in seconds
        
        Returns:
            Formatted time string (HH:MM:SS or MM:SS)
        """
        if seconds is None or math.isnan(seconds) or math.isinf(seconds):
            return "??:??"
        
        seconds = max(0, seconds)  # Ensure non-negative
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get complete time summary.
        
        Returns:
            Dictionary with timing information
        """
        return {
            'elapsed_time': self.get_elapsed_time(),
            'estimated_remaining': self.get_estimated_time_remaining(),
            'estimated_total': self.get_estimated_total_time(),
            'progress_percentage': self.get_progress_percentage(),
            'iterations_completed': self.iterations_completed,
            'total_iterations': self.total_iterations
        }


class AlgorithmTimeTracker:
    """Tracks timing for different phases of algorithm execution."""
    
    def __init__(self):
        """Initialize tracker."""
        self.phase_times: Dict[str, Tuple[float, float]] = {}  # phase -> (start_time, duration)
        self.total_start = None
        self.total_duration = 0.0
    
    def start_total(self) -> None:
        """Start tracking total time."""
        self.total_start = time.time()
    
    def end_total(self) -> None:
        """End tracking total time."""
        if self.total_start:
            self.total_duration = time.time() - self.total_start
    
    def start_phase(self, phase_name: str) -> None:
        """
        Start tracking a phase.
        
        Args:
            phase_name: Name of the phase
        """
        self.phase_times[phase_name] = (time.time(), 0.0)
    
    def end_phase(self, phase_name: str) -> float:
        """
        End tracking a phase.
        
        Args:
            phase_name: Name of the phase
        
        Returns:
            Duration of the phase
        """
        if phase_name not in self.phase_times:
            return 0.0
        
        start_time, _ = self.phase_times[phase_name]
        duration = time.time() - start_time
        self.phase_times[phase_name] = (start_time, duration)
        
        return duration
    
    def get_phase_time(self, phase_name: str) -> float:
        """Get duration of a phase."""
        if phase_name not in self.phase_times:
            return 0.0
        _, duration = self.phase_times[phase_name]
        return duration
    
    def get_all_phases(self) -> Dict[str, float]:
        """Get all phase times."""
        return {name: duration for name, (_, duration) in self.phase_times.items()}
    
    def print_summary(self) -> None:
        """Print summary of all phases."""
        print("\n" + "=" * 60)
        print("EXECUTION TIME BREAKDOWN")
        print("=" * 60)
        
        for phase_name, duration in self.get_all_phases().items():
            percentage = (duration / self.total_duration * 100) if self.total_duration > 0 else 0
            print(f"{phase_name:.<40} {duration:>8.2f}s ({percentage:>5.1f}%)")
        
        print("-" * 60)
        print(f"{'TOTAL':.<40} {self.total_duration:>8.2f}s (100.0%)")
        print("=" * 60 + "\n")
