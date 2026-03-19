#!/bin/bash
# --- Necessary Dependencies ---
source /home/awhitesides3/miniconda3/etc/profile.d/conda.sh
conda activate openneomc
# --- Parameters ---
# likely to change
run_name="Run03"
# data_path="/home/awhitesides3/openneomc/pporuns/RESULTS/Run03/Run03-ppo_data.csv"
# gpr_dose_path=
# gpr_cost_path="/home/awhitesides3/openneomc/pporuns/RESULTS/Run03/Run03-gpr_cost_model.pkl"
# Save Parameters
run_dir="./RESULTS/${run_name}/"
save_path="${run_dir}/${run_name}"
data_path="${save_path}-ppo_data.npz"
gpr_dose_path="${save_path}-gpr_dose_model.pkl"
gpr_cost_path="${save_path}-gpr_cost_model.pkl"
out_file="${save_path}-output.out"
params_file="${save_path}-parameters.txt"
# --- Arguments ---
ARGS_data_analysis="
--save_path $save_path
--data_path $data_path
--gpr_dose_path $gpr_dose_path
--gpr_cost_path $gpr_cost_path
"
# --- Feedback ---
# Log Parameters
echo "Data Analysis Arguments" >> "$params_file"
echo "$ARGS_data_analysis" >> "$params_file" 
# Log ppo
echo "-------------------------------------" >> "$out_file"
echo "Starting data analysis: $(date)" >> "$out_file"
echo "-------------------------------------" >> "$out_file"
# Run PPO
python -u create_plots.py $ARGS_data_analysis >> "$out_file" 2>&1 