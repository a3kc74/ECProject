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
# Run both algorithms on all instances (default)
python main.py

# Run only Genetic Algorithm
python main.py --algorithm GA

# Run only Simulated Annealing
python main.py --algorithm SA

# Run on a specific instance
python main.py --instance n20w20.001.txt

# Run with custom parameters
python main.py --num-runs 10 --ga-population 300 --ga-generations 3000
```

The framework will automatically:
- Detect all `.txt` files in the data directory (or use specified instance)
- Run selected algorithm(s) on each instance for the specified number of runs
- Calculate gaps vs. best-known solutions
- Export results to timestamped CSV files

## Benchmark System

### Command-Line Interface

The framework supports comprehensive command-line arguments for full control over experiments:

```bash
python main.py [OPTIONS]
```

**Basic Options:**
- `--algorithm {GA,SA,both}` or `-a` - Select algorithm(s) to run (default: both)
- `--instance FILE` or `-i` - Run on specific instance file (default: all instances in data-dir)
- `--data-dir DIR` or `-d` - Directory containing problem instances (default: data/SolomonPotvinBengio)
- `--num-runs N` or `-n` - Number of independent runs per configuration (default: 3)
- `--output-dir DIR` or `-o` - Directory to save CSV results (default: results)
- `--verbose` or `-v` - Enable detailed output

**Genetic Algorithm Parameters:**
- `--ga-population` - Population size (default: 150)
- `--ga-generations` - Number of generations (default: 2000)
- `--ga-mutation-rate` - Mutation probability (default: 0.08)
- `--ga-crossover-rate` - Crossover probability (default: 0.92)
- `--ga-tournament-size` - Tournament selection size (default: 20)
- `--ga-elitism-count` - Elite solutions preserved (default: 3)

**Simulated Annealing Parameters:**
- `--sa-initial-temp` - Initial temperature (default: 1000.0)
- `--sa-final-temp` - Final temperature (default: 0.1)
- `--sa-cooling-rate` - Cooling rate (default: 0.96)
- `--sa-iterations-per-temp` - Iterations per temperature (default: 200)

### Usage Examples

```bash
# View all available options
python main.py --help

# Run both algorithms on all instances with defaults
python main.py

# Run only GA on all instances
python main.py --algorithm GA

# Run only SA on a specific instance
python main.py -a SA -i n20w20.001.txt

# Run with more runs and verbose output
python main.py -n 10 -v

# Custom GA parameters
python main.py -a GA --ga-population 300 --ga-generations 3000 --ga-mutation-rate 0.1

# Custom SA parameters
python main.py -a SA --sa-initial-temp 2000 --sa-cooling-rate 0.98

# Full customization
python main.py -a GA -i n20w20.001.txt -n 5 \
  --ga-population 200 --ga-generations 1500 \
  --ga-mutation-rate 0.1 --ga-crossover-rate 0.85 \
  --output-dir my_results -v
```

### How It Works

The `Benchmark` class in `benchmark.py` provides a systematic framework for comparing algorithms:

1. **Problem Loading**: All problem instances are loaded once and cached
2. **Algorithm Execution**: Each algorithm runs on each instance multiple times
3. **Best-Known Comparison**: Loads best-known solutions and calculates gaps
4. **Statistical Analysis**: Aggregates results across runs (best, mean, std, median, worst)
5. **CSV Export**: Saves raw results and statistics with configuration parameters

All parameters are configurable via command-line arguments, eliminating the need to modify source code for experiments.

### Using the Benchmark

The benchmark system is fully controlled via command-line arguments:

```bash
# Run both GA and SA on all instances
python main.py

# Run only GA with custom parameters
python main.py -a GA --ga-population 300 --ga-generations 1500 -n 5

# Run on specific instance
python main.py -i n20w20.001.txt --num-runs 10

# Run SA with custom temperature schedule
python main.py -a SA --sa-initial-temp 2000 --sa-cooling-rate 0.98
```

For programmatic usage, you can still import and use the classes directly:

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

algorithms = {
    'GA': GeneticAlgorithm,
    'SA': SimulatedAnnealing
}

algorithm_configs = {
    'GA': ga_config,
    'SA': sa_config
}

# Run benchmark
benchmark = Benchmark(
    algorithms=algorithms,
    problem_paths=problem_files,
    num_runs=10,
    algorithm_configs=algorithm_configs
)
raw_results = benchmark.run()
stats_results = benchmark.get_statistics(raw_results)
```

## Configuration

All configuration parameters can be set via command-line arguments. Use `python main.py --help` to see all options.

### Genetic Algorithm Parameters

| Parameter | CLI Argument | Default | Description |
|-----------|-------------|---------|-------------|
| `population_size` | `--ga-population` | 150 | Number of solutions in population |
| `num_generations` | `--ga-generations` | 2000 | Number of generations to evolve |
| `mutation_rate` | `--ga-mutation-rate` | 0.08 | Probability of mutation |
| `crossover_rate` | `--ga-crossover-rate` | 0.92 | Probability of crossover |
| `tournament_size` | `--ga-tournament-size` | 20 | Tournament selection size |
| `elitism_count` | `--ga-elitism-count` | 3 | Number of elite solutions preserved |

### Simulated Annealing Parameters

| Parameter | CLI Argument | Default | Description |
|-----------|-------------|---------|-------------|
| `initial_temperature` | `--sa-initial-temp` | 1000.0 | Starting temperature |
| `final_temperature` | `--sa-final-temp` | 0.1 | Final temperature threshold |
| `cooling_rate` | `--sa-cooling-rate` | 0.96 | Temperature reduction rate |
| `iterations_per_temp` | `--sa-iterations-per-temp` | 200 | Iterations at each temperature |

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

### Basic Usage

```bash
# Run both algorithms on all instances with default settings
python main.py

# Run only Genetic Algorithm
python main.py --algorithm GA

# Run only Simulated Annealing
python main.py --algorithm SA
```

### Instance Selection

```bash
# Run on all instances in default directory
python main.py

# Run on specific instance
python main.py --instance n20w20.001.txt

# Run on all instances in custom directory
python main.py --data-dir path/to/instances
```

### Custom Parameters

```bash
# Increase population and generations for GA
python main.py -a GA --ga-population 300 --ga-generations 3000

# Adjust SA cooling schedule
python main.py -a SA --sa-initial-temp 2000 --sa-cooling-rate 0.98

# Run more independent trials
python main.py --num-runs 20

# Combine multiple options
python main.py -a GA -n 10 --ga-population 200 --ga-mutation-rate 0.1 -v
```

### Output Control

```bash
# Save results to custom directory
python main.py --output-dir my_experiments

# Enable verbose output for detailed information
python main.py --verbose

# Full experiment with custom settings
python main.py -a both -i n20w20.001.txt -n 5 \
  --ga-population 200 --ga-generations 1500 \
  --sa-initial-temp 1500 \
  --output-dir results/experiment_001 -v
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
