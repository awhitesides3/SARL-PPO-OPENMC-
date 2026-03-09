#!/bin/bash
#SBATCH --job-name=test_optimal_design_run
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=test_optimal_design_run.out

source /home/awhitesides3/miniconda3/etc/profile.d/conda.sh
conda activate openneomc

# --- Simulation Parameters ---

# --- Geometry Parameters ---
lower_bound=0.01
upper_bound=10.0
number_layers=2
# --- Simulation Parameters ---


ARGS="
--lower_bound $lower_bound
--upper_bound $upper_bound
--number_layers $number_layers
"

python create_optimal_design.py $ARGS > run.out 2>&1 &