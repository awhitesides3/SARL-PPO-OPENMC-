#!/bin/bash
#SBATCH --job-name=test_optimal_design_run
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=test_optimal_design_run.out
# --- Necessary Dependencies ---
source /home/awhitesides3/miniconda3/etc/profile.d/conda.sh
conda activate openneomc
# --- Data Parameters ---
npz_file_path="./test-results-create_surrogate/2L/data/0.0936-0-1-1e+03-1e+05-2.npz"
npz_file_name=0.0936-0-1-1e+03-1e+05-1.npz
# --- Geometry Parameters ---
number_layers=2
lower_bound=0.01
upper_bound=10.0
# --- Simulation Parameters ---
total_timesteps=1e2
iterations=1e2
dose_c=0.0936
nps=1e5
episode_length=1
mode='max'
policy='MlpPolicy'
check_freq=1
n_steps=32
nminibatches=4
seed=1
validation_threshold=0.05
# --- Save Parameters ---
results_path=./test-results-create_surrogate/2L/
# --- Arguments ---
ARGS="
--npz_file_path $npz_file_path
--npz_file_name $npz_file_name
--number_layers $number_layers
--lower_bound $lower_bound
--upper_bound $upper_bound
--total_timesteps $total_timesteps
--iterations $iterations
--dose_c $dose_c
--nps $nps
--episode_length $episode_length
--mode $mode
--policy $policy
--check_freq $check_freq
--n_steps $n_steps
--nminibatches $nminibatches
--seed $seed
--validation_threshold $validation_threshold
--results_path $results_path
"
# --- Feedback ---
echo "-------------------------------------" >> run_optimal_design.out
echo "Starting surrogate run: $(date)" >> run_optimal_design.out
echo "Layers: $number_layers" >> run_optimal_design.out
echo "Finding optimal design from this data set: $npz_file_path" >> run_optimal_design.out
echo "Dose constraint: $dose_constraint" >> run_optimal_design.out
echo "-------------------------------------" >> run_optimal_design.out
# --- Command Line ---
python -u create_optimal_design.py $ARGS >> run_optimal_design.out 2>&1 &

PID=$!
echo "Process ID: $PID" >> run_optimal_design.out