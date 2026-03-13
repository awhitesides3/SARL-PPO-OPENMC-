#!/bin/bash
# --- Necessary Dependencies ---
source /home/awhitesides3/miniconda3/etc/profile.d/conda.sh
conda activate openneomc
# --- Parameters ---
# likely to change
run_name="Run02"
surrogate_path="/home/awhitesides3/openneomc/pporuns/RESULTS/Run01/Run01-surrogate_data.npz"
number_layers=2
number_random_points=1
dose_constraint=0.0936
nps=1e5
# unlikely to change
hard_constraint=0
normalization=1
scalingFactor=1e3
total_timesteps=1e2
iterations=1e2
lower_bound=0.01
upper_bound=10.0
validation_threshold=0.5
# very ulikely to change
episode_length=1
mode='max'
policy='MlpPolicy'
check_freq=1
n_steps=32
nminibatches=4
seed=1
# Save Parameters
run_dir="./RESULTS/${run_name}/"
save_path="${run_dir}/${run_name}"
out_file="${save_path}-output.out"
params_file="${save_path}-parameters.txt"
# --- Arguments ---
ARGS_ppo="
--number_layers $number_layers
--lower_bound $lower_bound
--upper_bound $upper_bound
--total_timesteps $total_timesteps
--iterations $iterations
--dose_constraint $dose_constraint
--nps $nps
--episode_length $episode_length
--mode $mode
--policy $policy
--check_freq $check_freq
--n_steps $n_steps
--nminibatches $nminibatches
--seed $seed
--validation_threshold $validation_threshold
--save_path $save_path
--surrogate_path $surrogate_path
"
# --- Save Command ---
mkdir -p "${run_dir}"   # create directory if it doesn't exist
# --- Feedback ---
# Log Parameters
echo "PPO Arguments" > "$params_file"
echo "$ARGS_ppo" >> "$params_file" 
# Log ppo
echo "-------------------------------------" >> "$out_file"
echo "Starting ppo run: $(date)" >> "$out_file"
echo "Finding optimal design from this data set: $surrogate_path" >> "$out_file"
echo "-------------------------------------" >> "$out_file"
# Run PPO
python -u create_optimal_design.py $ARGS_ppo >> "$out_file" 2>&1 