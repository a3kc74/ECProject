"""
Advanced visualization utilities using Seaborn.

Provides enhanced statistical visualization for benchmark results including
pairplots, correlation heatmaps, boxplots, and violin plots.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import os


# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_palette("husl")


def plot_correlation_heatmap(results_df: pd.DataFrame, 
                             title: str = "Correlation Heatmap", 
                             save_path: str = "plots/correlation_heatmap.png") -> None:
    """
    Plot correlation heatmap for numerical features in results.
    
    Shows correlations between fitness, runtime, and other numerical metrics.
    
    Args:
        results_df: DataFrame containing benchmark results
        title: Title for the plot
        save_path: Path where the plot will be saved
    """
    # Select only numerical columns
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Not enough numerical columns for correlation heatmap.")
        return
    
    # Calculate correlation matrix
    corr_matrix = results_df[numeric_cols].corr()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={'shrink': 0.8})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to: {save_path}")


def plot_pairplot(results_df: pd.DataFrame,
                  hue_column: str = 'algorithm',
                  save_path: str = "plots/pairplot.png") -> None:
    """
    Create pairplot showing relationships between all numerical variables.
    
    Useful for exploring relationships between fitness, runtime, and other metrics
    across different algorithms.
    
    Args:
        results_df: DataFrame containing benchmark results
        hue_column: Column to use for color coding (typically 'algorithm')
        save_path: Path where the plot will be saved
    """
    # Select numerical columns
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Not enough numerical columns for pairplot.")
        return
    
    # Add hue column if it exists
    plot_cols = numeric_cols.copy()
    if hue_column in results_df.columns and hue_column not in plot_cols:
        plot_cols.append(hue_column)
    
    # Create pairplot
    pairplot_fig = sns.pairplot(results_df[plot_cols], 
                                hue=hue_column if hue_column in results_df.columns else None,
                                diag_kind='kde',
                                plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                                height=2.5)
    
    pairplot_fig.fig.suptitle('Pairplot of Benchmark Results', 
                              fontsize=16, 
                              fontweight='bold', 
                              y=1.01)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pairplot saved to: {save_path}")


def plot_boxplot(results_df: pd.DataFrame,
                 x_column: str = 'algorithm',
                 y_column: str = 'best_fitness',
                 title: str = "Algorithm Performance Comparison (Boxplot)",
                 save_path: str = "plots/seaborn_boxplot.png") -> None:
    """
    Create enhanced boxplot using seaborn.
    
    Shows distribution of performance metrics across algorithms with
    better styling and statistical information.
    
    Args:
        results_df: DataFrame containing benchmark results
        x_column: Column for x-axis (categorical, e.g., 'algorithm')
        y_column: Column for y-axis (numerical, e.g., 'best_fitness')
        title: Title for the plot
        save_path: Path where the plot will be saved
    """
    plt.figure(figsize=(12, 7))
    
    # Create boxplot with swarm plot overlay
    ax = sns.boxplot(data=results_df, 
                    x=x_column, 
                    y=y_column,
                    palette='Set2',
                    width=0.6,
                    linewidth=2)
    
    # Add individual points
    sns.swarmplot(data=results_df, 
                 x=x_column, 
                 y=y_column,
                 color='black',
                 alpha=0.5,
                 size=6,
                 ax=ax)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(x_column.capitalize(), fontsize=13, fontweight='bold')
    plt.ylabel(y_column.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot saved to: {save_path}")


def plot_violinplot(results_df: pd.DataFrame,
                   x_column: str = 'algorithm',
                   y_column: str = 'best_fitness',
                   title: str = "Algorithm Performance Distribution (Violin Plot)",
                   save_path: str = "plots/violinplot.png") -> None:
    """
    Create violin plot showing distribution density.
    
    Violin plots show the full distribution of the data, making it easier
    to see multimodal distributions and density.
    
    Args:
        results_df: DataFrame containing benchmark results
        x_column: Column for x-axis (categorical, e.g., 'algorithm')
        y_column: Column for y-axis (numerical, e.g., 'best_fitness')
        title: Title for the plot
        save_path: Path where the plot will be saved
    """
    plt.figure(figsize=(12, 7))
    
    # Create violin plot
    ax = sns.violinplot(data=results_df, 
                       x=x_column, 
                       y=y_column,
                       palette='muted',
                       inner='box',
                       linewidth=2)
    
    # Add individual points
    sns.stripplot(data=results_df, 
                 x=x_column, 
                 y=y_column,
                 color='black',
                 alpha=0.4,
                 size=5,
                 ax=ax)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(x_column.capitalize(), fontsize=13, fontweight='bold')
    plt.ylabel(y_column.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Violin plot saved to: {save_path}")


def plot_comparative_performance(results_df: pd.DataFrame,
                                save_path: str = "plots/comparative_performance.png") -> None:
    """
    Create comprehensive comparative plot with multiple subplots.
    
    Combines boxplot, violin plot, and strip plot in one figure for
    complete performance comparison.
    
    Args:
        results_df: DataFrame containing benchmark results
        save_path: Path where the plot will be saved
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Algorithm Performance Analysis', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Boxplot for best_fitness
    sns.boxplot(data=results_df, 
               x='algorithm', 
               y='best_fitness',
               palette='Set2',
               ax=axes[0, 0])
    axes[0, 0].set_title('Best Fitness Distribution (Boxplot)', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Violin plot for best_fitness
    sns.violinplot(data=results_df, 
                  x='algorithm', 
                  y='best_fitness',
                  palette='muted',
                  inner='quartile',
                  ax=axes[0, 1])
    axes[0, 1].set_title('Best Fitness Distribution (Violin Plot)', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Runtime comparison
    if 'run_time' in results_df.columns:
        sns.boxplot(data=results_df, 
                   x='algorithm', 
                   y='run_time',
                   palette='coolwarm',
                   ax=axes[1, 0])
        sns.swarmplot(data=results_df, 
                     x='algorithm', 
                     y='run_time',
                     color='black',
                     alpha=0.5,
                     ax=axes[1, 0])
        axes[1, 0].set_title('Runtime Comparison', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Run-by-run comparison
    if 'run' in results_df.columns:
        sns.lineplot(data=results_df, 
                    x='run', 
                    y='best_fitness',
                    hue='algorithm',
                    marker='o',
                    markersize=8,
                    linewidth=2,
                    ax=axes[1, 1])
        axes[1, 1].set_title('Performance Across Runs', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Run Number', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(title='Algorithm', fontsize=10, title_fontsize=11)
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparative performance plot saved to: {save_path}")


def plot_statistical_summary(stats_df: pd.DataFrame,
                            save_path: str = "plots/statistical_summary.png") -> None:
    """
    Create statistical summary visualization from statistics DataFrame.
    
    Displays mean, std, best, worst in grouped bar charts.
    
    Args:
        stats_df: DataFrame containing statistical summary (from benchmark.get_statistics())
        save_path: Path where the plot will be saved
    """
    if 'algorithm' not in stats_df.columns:
        print("Statistics DataFrame must contain 'algorithm' column.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Statistical Summary of Algorithm Performance', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Best fitness
    if 'best' in stats_df.columns:
        sns.barplot(data=stats_df, 
                   x='algorithm', 
                   y='best',
                   palette='viridis',
                   ax=axes[0, 0])
        axes[0, 0].set_title('Best Fitness Achieved', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in axes[0, 0].containers:
            axes[0, 0].bar_label(container, fmt='%.2e', fontsize=9)
    
    # 2. Mean fitness
    if 'mean' in stats_df.columns:
        sns.barplot(data=stats_df, 
                   x='algorithm', 
                   y='mean',
                   palette='rocket',
                   ax=axes[0, 1])
        axes[0, 1].set_title('Mean Fitness', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Mean Fitness', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for container in axes[0, 1].containers:
            axes[0, 1].bar_label(container, fmt='%.2e', fontsize=9)
    
    # 3. Standard deviation
    if 'std' in stats_df.columns:
        sns.barplot(data=stats_df, 
                   x='algorithm', 
                   y='std',
                   palette='mako',
                   ax=axes[1, 0])
        axes[1, 0].set_title('Standard Deviation', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Std. Deviation', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for container in axes[1, 0].containers:
            axes[1, 0].bar_label(container, fmt='%.2e', fontsize=9)
    
    # 4. Average runtime
    if 'avg_run_time' in stats_df.columns:
        sns.barplot(data=stats_df, 
                   x='algorithm', 
                   y='avg_run_time',
                   palette='flare',
                   ax=axes[1, 1])
        axes[1, 1].set_title('Average Runtime', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Avg Runtime (seconds)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for container in axes[1, 1].containers:
            axes[1, 1].bar_label(container, fmt='%.3f', fontsize=10)
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical summary plot saved to: {save_path}")
