#!/bin/bash
# Run Instructions: insert into command line -> |bash solve.sh &|
# Note: solve.sh should be ran on an agent that has already been initialized using the initialize.sh. It assumes that the
# --- Necessary Dependencies ---
source /home/awhitesides3/miniconda3/etc/profile.d/conda.sh
conda activate openneomc
# --- Arguments ---
ARGS=($(grep -v 'Arguments' parameters.txt))
# --- OVERRIDE Arguments ---
# ARGS+=(
# --[ARGUMENT] [NEW VALUE]
# )
# --- Feedback ---
# README
echo "$(date) | 'solve' run: [INSERT TEXT]" >> "${saveDir}README.txt"
# Log Parameters
echo "$(date) | Arguments in 'solve' run" > "${saveDir}solve_parameters.txt"
echo "$ARGS" >> "${saveDir}parameters.txt"
# Log Surrogate
echo "-------------------------------------" >> "${saveDir}out.out"
echo "$(date) | Starting a 'solve' run" >> "${saveDir}out.out"
echo "-------------------------------------" >> "${saveDir}out.out"
# Run Agent Functions - set python path since solve.py imports a .py in the directory one level above.
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"
python -u "solve.py" "${ARGS[@]}" >> "${saveDir}out.out" 2>&1