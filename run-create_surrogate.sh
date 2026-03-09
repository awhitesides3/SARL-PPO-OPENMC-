#!/bin/bash
#SBATCH --job-name=test_surrogate_run
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=test_surrogate_run.out
# --- Necessary Dependencies ---
source /home/awhitesides3/miniconda3/etc/profile.d/conda.sh
conda activate openneomc
# --- Simulation Parameters ---
dose_constraint=0.0936
hard_constraint=0
normalization=1
scalingFactor=1e3
nps=1e5
# --- Geometry Parameters ---
lower_bound=0.01
upper_bound=10.0
number_layers=2
# --- Simulation Parameters ---
number_random_points=2
# --- Arguments ---
ARGS="
--dose_constraint $dose_constraint
--hard_constraint $hard_constraint
--normalization $normalization
--scalingFactor $scalingFactor
--nps $nps
--lower_bound $lower_bound
--upper_bound $upper_bound
--number_layers $number_layers
--number_random_points $number_random_points
"
# --- Command Line ---
python create_surrogate.py $ARGS > run.out 2>&1 &