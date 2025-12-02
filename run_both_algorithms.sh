#!/bin/bash

# Bash script to run both GA and SA algorithms with different settings
# Allows comparison of both algorithms on the same problem instances

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================="
echo "GA and SA Configuration Tests"
echo "=================================="
echo ""

# Create output directory
mkdir -p results
mkdir -p logs

# Test scenarios combining GA and SA with different settings
# Each scenario has GA config (pop:gen) and SA config (temp:cool:iter)

declare -a SCENARIOS=(
    "50:500|500.0:0.95:100:Scenario1_Conservative"
    "100:1000|1000.0:0.96:200:Scenario2_Standard"
    "150:2000|1500.0:0.97:300:Scenario3_Aggressive"
    "200:3000|2000.0:0.98:400:Scenario4_VeryAggressive"
)

FINAL_TEMP=0.1
NUM_RUNS=3
DATA_DIR="data/SolomonPotvinBengio"

echo "Running GA and SA with different settings..."
echo "Base settings:"
echo "  - Runs per config: $NUM_RUNS"
echo "  - SA Final Temperature: $FINAL_TEMP"
echo ""

# Run each scenario
for scenario in "${SCENARIOS[@]}"; do
    # Parse the scenario string: GA_config|SA_config:description
    IFS='|' read -r ga_config sa_full <<< "$scenario"
    IFS=':' read -r init_temp cool_rate iter_per_temp sa_desc <<< "$sa_full"
    IFS=':' read -r pop_size num_gen <<< "$ga_config"
    
    echo "========================================="
    echo "Scenario: $sa_desc"
    echo "  GA - Population: $pop_size, Generations: $num_gen"
    echo "  SA - InitTemp: $init_temp, CoolRate: $cool_rate, IterPerTemp: $iter_per_temp"
    echo "========================================="
    
    # Create log filename with timestamp
    LOG_FILE="logs/both_${sa_desc}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run both algorithms
    python main.py \
        --algorithm both \
        --ga-population "$pop_size" \
        --ga-generations "$num_gen" \
        --sa-initial-temp "$init_temp" \
        --sa-final-temp "$FINAL_TEMP" \
        --sa-cooling-rate "$cool_rate" \
        --sa-iterations-per-temp "$iter_per_temp" \
        --num-runs "$NUM_RUNS" \
        --data-dir "$DATA_DIR" \
        --verbose \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Scenario $sa_desc completed"
    echo "  Log saved to: $LOG_FILE"
    echo ""
done

echo "=================================="
echo "All scenarios completed!"
echo "=================================="
echo ""
echo "Results saved to: results/"
echo "Logs saved to: logs/"
