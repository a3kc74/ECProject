#!/bin/bash

# Bash script to run Genetic Algorithm with different configurations
# Tests various population sizes and generation counts

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================="
echo "Genetic Algorithm Configuration Tests"
echo "=================================="
echo ""

# Create output directory
mkdir -p results
mkdir -p logs

# Define GA configurations to test
# Format: population_size:num_generations:description
GA_CONFIGS=(
    "50:500:Small"
    "100:1000:Medium"
    "150:2000:Large"
    "200:3000:XLarge"
)

MUTATION_RATE=0.08
CROSSOVER_RATE=0.92
TOURNAMENT_SIZE=20
ELITISM_COUNT=3
NUM_RUNS=3
DATA_DIR="data/SolomonPotvinBengio"

echo "Running Genetic Algorithm with different configurations..."
echo "Base settings:"
echo "  - Mutation Rate: $MUTATION_RATE"
echo "  - Crossover Rate: $CROSSOVER_RATE"
echo "  - Tournament Size: $TOURNAMENT_SIZE"
echo "  - Elitism Count: $ELITISM_COUNT"
echo "  - Runs per config: $NUM_RUNS"
echo ""

# Run each configuration
for config in "${GA_CONFIGS[@]}"; do
    IFS=':' read -r pop_size num_gen desc <<< "$config"
    
    echo "========================================="
    echo "Configuration: $desc"
    echo "  Population Size: $pop_size"
    echo "  Number of Generations: $num_gen"
    echo "========================================="
    
    # Create log filename with timestamp
    LOG_FILE="logs/ga_${desc}_pop${pop_size}_gen${num_gen}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run the GA
    python main.py \
        --algorithm GA \
        --ga-population "$pop_size" \
        --ga-generations "$num_gen" \
        --ga-mutation-rate "$MUTATION_RATE" \
        --ga-crossover-rate "$CROSSOVER_RATE" \
        --ga-tournament-size "$TOURNAMENT_SIZE" \
        --ga-elitism-count "$ELITISM_COUNT" \
        --num-runs "$NUM_RUNS" \
        --data-dir "$DATA_DIR" \
        --verbose \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Configuration $desc completed"
    echo "  Log saved to: $LOG_FILE"
    echo ""
done

echo "=================================="
echo "All GA configurations completed!"
echo "=================================="
echo ""
echo "Results saved to: results/"
echo "Logs saved to: logs/"
