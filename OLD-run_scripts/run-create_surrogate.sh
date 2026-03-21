#!/bin/bash
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
number_layers=3
# --- Simulation Parameters ---
number_random_points=1
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
# --- Feedback ---
echo "-------------------------------------" >> run_surrogate.out
echo "Starting surrogate run: $(date)" >> run_surrogate.out
echo "Layers: $number_layers" >> run_surrogate.out
echo "Particles (nps): $nps" >> run_surrogate.out
echo "Dose constraint: $dose_constraint" >> run_surrogate.out
echo "-------------------------------------" >> run_surrogate.out
# --- Command Line ---
python -u create_surrogate.py $ARGS >> run_surrogate.out 2>&1 &

PID=$!
echo "Process ID: $PID" >> run_surrogate.out