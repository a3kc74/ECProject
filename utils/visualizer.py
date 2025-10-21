"""
Visualization utilities for plotting results and routes.

Provides functions for convergence plots, benchmark comparisons, and route visualization.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem.tw_tsp import TWTSPProblem


def plot_convergence(history_data: Dict[str, List[float]], title: str, save_path: str) -> None:
    """
    Plot convergence curves for multiple algorithms.
    
    Args:
        history_data: Dictionary mapping algorithm names to fitness history lists
                     Example: {'GA': [100, 90, 85], 'SA': [110, 95, 88]}
        title: Title for the plot
        save_path: Path where the plot image will be saved
    """
    plt.figure(figsize=(10, 6))
    
    for algo_name, history in history_data.items():
        iterations = range(1, len(history) + 1)
        plt.plot(iterations, history, label=algo_name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Iteration / Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence plot saved to: {save_path}")


def plot_benchmark_results(results_df: pd.DataFrame, title: str, save_path: str) -> None:
    """
    Plot box-plot comparison of algorithm performances.
    
    Args:
        results_df: DataFrame with columns including 'algorithm' and 'best_fitness'
        title: Title for the plot
        save_path: Path where the plot image will be saved
    """
    plt.figure(figsize=(12, 6))
    
    # Get unique algorithms and problems
    algorithms = results_df['algorithm'].unique()
    
    # Check if multiple problems exist
    if 'problem' in results_df.columns:
        problems = results_df['problem'].unique()
        
        if len(problems) > 1:
            # Create grouped box plot
            data_to_plot = []
            labels = []
            
            for problem in problems:
                for algo in algorithms:
                    subset = results_df[(results_df['algorithm'] == algo) & 
                                       (results_df['problem'] == problem)]
                    if len(subset) > 0:
                        data_to_plot.append(subset['best_fitness'].values)
                        labels.append(f"{algo}\n{problem}")
            
            plt.boxplot(data_to_plot, labels=labels)
            plt.xticks(rotation=45, ha='right')
        else:
            # Single problem: simple box plot by algorithm
            data_to_plot = [results_df[results_df['algorithm'] == algo]['best_fitness'].values 
                           for algo in algorithms]
            plt.boxplot(data_to_plot, labels=algorithms)
    else:
        # No problem column: plot by algorithm only
        data_to_plot = [results_df[results_df['algorithm'] == algo]['best_fitness'].values 
                       for algo in algorithms]
        plt.boxplot(data_to_plot, labels=algorithms)
    
    plt.ylabel('Best Fitness', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Benchmark results plot saved to: {save_path}")


def plot_best_route(problem: TWTSPProblem, solution: List[int], title: str, save_path: str) -> None:
    """
    Visualize the best route found on a 2D map.
    
    Args:
        problem: TW-TSP problem instance containing customer locations
        solution: List of customer IDs representing the route
        title: Title for the plot
        save_path: Path where the plot image will be saved
    """
    plt.figure(figsize=(12, 10))
    
    # Get customer details
    customer_details = problem.get_customer_details()
    
    # Extract coordinates
    depot = customer_details[0]
    
    # Plot depot
    plt.scatter(depot['x'], depot['y'], c='red', s=300, marker='s', 
               label='Depot', zorder=5, edgecolors='black', linewidths=2)
    plt.text(depot['x'], depot['y'], '0', fontsize=10, ha='center', va='center', 
            fontweight='bold', color='white')
    
    # Plot customers
    customer_x = [customer_details[cid]['x'] for cid in range(1, problem.num_customers + 1)]
    customer_y = [customer_details[cid]['y'] for cid in range(1, problem.num_customers + 1)]
    
    plt.scatter(customer_x, customer_y, c='blue', s=150, marker='o', 
               label='Customers', zorder=4, edgecolors='black', linewidths=1.5, alpha=0.7)
    
    # Add customer ID labels
    for cid in range(1, problem.num_customers + 1):
        plt.text(customer_details[cid]['x'], customer_details[cid]['y'], 
                str(cid), fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Plot route
    if solution:
        # Start from depot
        route_x = [depot['x']]
        route_y = [depot['y']]
        
        # Add all customers in solution order
        for customer_id in solution:
            route_x.append(customer_details[customer_id]['x'])
            route_y.append(customer_details[customer_id]['y'])
        
        # Return to depot
        route_x.append(depot['x'])
        route_y.append(depot['y'])
        
        # Plot route lines
        plt.plot(route_x, route_y, 'g-', linewidth=2, alpha=0.6, label='Route', zorder=3)
        
        # Add arrows to show direction
        for i in range(len(route_x) - 1):
            dx = route_x[i+1] - route_x[i]
            dy = route_y[i+1] - route_y[i]
            plt.arrow(route_x[i], route_y[i], dx*0.7, dy*0.7, 
                     head_width=2, head_length=2, fc='green', ec='green', 
                     alpha=0.5, zorder=2)
    
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Route visualization saved to: {save_path}")
