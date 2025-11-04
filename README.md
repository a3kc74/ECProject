# TW-TSP Optimization Framework

A comprehensive framework for solving the Traveling Salesman Problem with Time Windows (TW-TSP) using metaheuristic algorithms, with built-in benchmarking capabilities against known optimal solutions.

## Overview

The Time Window Traveling Salesman Problem (TW-TSP) is an extension of the classic TSP where each customer must be visited within a specified time window. This framework implements:

- **Genetic Algorithm (GA)** with 5 diverse initialization strategies
- **Simulated Annealing (SA)** with 2-opt neighborhood search
- **Benchmark System** for systematic algorithm comparison
- **Best-Known Solution Loading** from literature (Solomon-Potvin-Bengio dataset)
- **Gap Analysis** to measure solution quality vs. optimal solutions

## Features

- **Multiple Algorithms**: Genetic Algorithm and Simulated Annealing implementations
- **Diverse Initialization**: 5 different GA initialization strategies for better solution diversity
- **Automatic Benchmarking**: Run experiments across multiple problem instances automatically
- **Best-Known Solutions**: Load and compare against best-known solutions from literature
- **Gap Analysis**: Calculate percentage gap from optimal solutions
- **CSV Export**: Detailed results and statistics exported with configuration tracking
- **SPB Format Support**: Parse Solomon-Potvin-Bengio format problem instances

## Installation

### Requirements

- Python 3.10+
- numpy>=1.24.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- scipy>=1.10.0
- seaborn>=0.12.0

### Setup

```bash
pip install numpy pandas matplotlib scipy seaborn
```

## Project Structure

```
ECProject/
├── problem/
│   └── tw_tsp.py              # TW-TSP problem definition and SPB format loader
├── algorithms/
│   ├── genetic_algorithm.py   # GA with 5 initialization strategies
│   └── simulated_annealing.py # SA with 2-opt neighborhood
├── benchmark.py               # Benchmark system for algorithm comparison
├── main.py                    # Main execution script
├── data/
│   ├── SolomonPotvinBengio/   # Problem instances (.txt files)
│   └── tsptw-2010-best-known/ # Best-known solutions
│       └── SolomonPotvinBengio.best
└── results/                   # Output directory for CSV files
```

## Quick Start

1. **Place problem instances** in `data/SolomonPotvinBengio/` directory (`.txt` files)
2. **Place best-known solutions** in `data/tsptw-2010-best-known/SolomonPotvinBengio.best`
3. **Run the benchmark**:

```bash
python main.py
```

The framework will automatically:
- Detect all `.txt` files in the data directory
- Run each algorithm on each instance for 10 independent runs
- Calculate gaps vs. best-known solutions
- Export results to timestamped CSV files

## Benchmark System

### How It Works

The `Benchmark` class in `benchmark.py` provides a systematic framework for comparing algorithms:

1. **Problem Loading**: All problem instances are loaded once and cached
2. **Algorithm Execution**: Each algorithm runs on each instance multiple times
3. **Best-Known Comparison**: Loads best-known solutions and calculates gaps
4. **Statistical Analysis**: Aggregates results across runs (best, mean, std, median, worst)
5. **CSV Export**: Saves raw results and statistics with configuration parameters

### Using the Benchmark

```python
from benchmark import Benchmark
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing

# Define problem files
problem_files = ['data/SolomonPotvinBengio/n20w20.001.txt']

# Configure algorithms
ga_config = {
    'population_size': 200,
    'num_generations': 500,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'tournament_size': 5,
    'elitism_count': 2
}

sa_config = {
    'initial_temperature': 1000,
    'cooling_rate': 0.95,
    'max_iterations': 10000,
    'stopping_temperature': 0.01
}

algorithms = [
    ('GA', GeneticAlgorithm, ga_config),
    ('SA', SimulatedAnnealing, sa_config)
]

# Run benchmark
benchmark = Benchmark(algorithms, problem_files, num_runs=10)
benchmark.run()

# Export results
results_df = benchmark.get_results()
stats_df = benchmark.get_statistics()
```

## Configuration

### Genetic Algorithm Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `population_size` | Number of solutions in population | 200 |
| `num_generations` | Number of generations to evolve | 500 |
| `mutation_rate` | Probability of mutation | 0.1 |
| `crossover_rate` | Probability of crossover | 0.8 |
| `tournament_size` | Tournament selection size | 5 |
| `elitism_count` | Number of elite solutions preserved | 2 |

### Simulated Annealing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `initial_temperature` | Starting temperature | 1000 |
| `cooling_rate` | Temperature reduction rate | 0.95 |
| `max_iterations` | Maximum iterations | 10000 |
| `stopping_temperature` | Temperature threshold to stop | 0.01 |

## Output Files

The benchmark system generates two CSV files with timestamps:

### 1. Raw Results CSV

Filename format: `raw_results_{algorithms}_{config}_{timestamp}.csv`

| Column | Description |
|--------|-------------|
| `algorithm` | Algorithm name (GA or SA) |
| `problem` | Problem instance name |
| `run` | Run number (0-indexed) |
| `best_fitness` | Best fitness found (distance + penalties) |
| `run_time` | Execution time in seconds |
| `total_distance` | Total travel distance |
| `num_violations` | Number of time window violations |
| `total_penalty` | Total penalty for violations |
| `best_known` | Best-known fitness from literature |
| `gap` | Percentage gap from best-known: `(fitness - best_known) / best_known * 100` |

### 2. Statistics CSV

Filename format: `statistics_{algorithms}_{config}_{timestamp}.csv`

Aggregates results across runs for each (algorithm, problem) pair:

| Column | Description |
|--------|-------------|
| `algorithm` | Algorithm name |
| `problem` | Problem instance name |
| `best` | Best fitness across all runs |
| `mean` | Mean fitness |
| `std` | Standard deviation of fitness |
| `median` | Median fitness |
| `worst` | Worst fitness |
| `avg_run_time` | Average execution time |
| `avg_distance` | Average travel distance |
| `avg_violations` | Average number of violations |
| `total_violations` | Total violations across all runs |
| `best_known` | Best-known fitness from literature |
| `gap` | Gap of the best solution |
| `population_size` | GA population size (if GA) |
| `num_generations` | GA generations (if GA) |
| `mutation_rate` | GA mutation rate (if GA) |
| `crossover_rate` | GA crossover rate (if GA) |
| `tournament_size` | GA tournament size (if GA) |
| `elitism_count` | GA elitism count (if GA) |
| `initial_temperature` | SA initial temperature (if SA) |
| `cooling_rate` | SA cooling rate (if SA) |
| `max_iterations` | SA max iterations (if SA) |
| `stopping_temperature` | SA stopping temperature (if SA) |

## Algorithm Details

### Genetic Algorithm

The GA implementation includes **5 diverse initialization strategies** to improve solution diversity:

1. **Nearest Neighbor (30%)**: Greedy construction starting from depot, always visiting the nearest unvisited customer
2. **Earliest Deadline First (20%)**: Prioritizes customers with earlier due times to reduce violations
3. **Savings Algorithm (20%)**: Clarke-Wright savings heuristic based on `savings(i,j) = d(0,i) + d(0,j) - d(i,j)`
4. **Regret Insertion (15%)**: Inserts customers at positions where not inserting them would cause maximum regret
5. **Random (15%)**: Pure random permutations for exploration

**Operators:**
- **Selection**: Tournament selection
- **Crossover**: Ordered Crossover (OX)
- **Mutation**: Swap mutation
- **Elitism**: Preserves best solutions across generations

### Simulated Annealing

- **Neighborhood**: 2-opt moves (reverses sub-tours)
- **Cooling Schedule**: Exponential cooling with configurable rate
- **Acceptance**: Metropolis criterion allows uphill moves with probability `exp(-ΔE/T)`

## Problem Format

### Solomon-Potvin-Bengio (SPB) Format

Problem files in `data/SolomonPotvinBengio/` follow this structure:

```
n                          # Number of nodes (including depot)
d(0,0) d(0,1) ... d(0,n-1)    # Distance matrix row 0
d(1,0) d(1,1) ... d(1,n-1)    # Distance matrix row 1
...
d(n-1,0) ... d(n-1,n-1)       # Distance matrix row n-1
e(0) l(0)                  # Depot time window [e, l]
e(1) l(1)                  # Customer 1 time window
...
e(n-1) l(n-1)              # Customer n-1 time window
```

Where:
- `n`: Total number of nodes (depot + customers)
- `d(i,j)`: Distance from node i to node j
- `e(i)`, `l(i)`: Earliest and latest service time for node i

**Note**: All customer service times are set to 0.

### Best-Known Solutions Format

The `SolomonPotvinBengio.best` file contains:

```
instance_name cost cv permutation
n20w20.001 6061.0 0 0 7 14 12 8 16 6 3 9 4 15 11 5 13 1 10 2 17 18 19 0
...
```

Where:
- `instance_name`: Problem instance filename
- `cost`: Best-known total cost (distance + penalties)
- `cv`: Number of constraint violations
- `permutation`: Best-known tour (node sequence)

The framework calculates fitness from the permutation using the same method as the evaluation function.

## Usage Examples

### Running on All Instances

The `main.py` script automatically detects all problem instances:

```python
# Automatically finds all .txt files in data directory
data_dir = 'data/SolomonPotvinBengio'
all_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
problem_files = [os.path.join(data_dir, f) for f in sorted(all_files)]
```

### Custom Configuration

Modify `main.py` to experiment with different configurations:

```python
# Example: More generations, larger population
ga_config = {
    'population_size': 300,
    'num_generations': 1000,
    'mutation_rate': 0.05,
    'crossover_rate': 0.9,
    'tournament_size': 7,
    'elitism_count': 5
}

# Example: Slower cooling, more iterations
sa_config = {
    'initial_temperature': 2000,
    'cooling_rate': 0.98,
    'max_iterations': 20000,
    'stopping_temperature': 0.001
}
```

### Running Specific Algorithms

To run only GA or only SA, modify the algorithms list in `main.py`:

```python
# Only GA
algorithms = [('GA', GeneticAlgorithm, ga_config)]

# Only SA
algorithms = [('SA', SimulatedAnnealing, sa_config)]
```

## Interpreting Results

### Gap Analysis

The **gap** measures how far a solution is from the best-known solution:

```
gap = (fitness - best_known) / best_known * 100
```

- **gap = 0%**: Optimal solution found
- **gap > 0%**: Suboptimal solution (higher is worse)
- **gap < 0%**: Better than best-known (verify feasibility!)

### Feasibility

A solution is **feasible** if `num_violations = 0`. Solutions with violations have penalties:

```
fitness = total_distance + (num_violations * 1000.0)
```

### Comparing Algorithms

Use the statistics CSV to compare:
- **best**: Which algorithm finds better solutions?
- **mean/std**: Which algorithm is more consistent?
- **avg_run_time**: Which algorithm is faster?
- **gap**: Which algorithm is closer to optimal?

## References

- Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research*, 35(2), 254-265.
- Potvin, J. Y., & Bengio, S. (1996). The vehicle routing problem with time windows part II: Genetic search. *INFORMS Journal on Computing*, 8(2), 165-172.
- Dataset: [TSPTW Benchmark Instances](https://lopez-ibanez.eu/tsptw-instances)

---

## License

This project is for educational and research purposes.
