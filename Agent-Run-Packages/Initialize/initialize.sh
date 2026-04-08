
#!/bin/bash
# to run insert into command line -> |bash initialize.sh &|
# --- Necessary Dependencies ---
source /home/awhitesides3/miniconda3/etc/profile.d/conda.sh
conda activate openneomc
# --- Parameters ---
# ALWAYS CHANGE/CHECK
Agent=[INSERT VALUE]
nL=[INSERT VALUE]
rps=[INSERT VALUE]
dose_limit=[INSERT VALUE]
# MAY/UNLIKELY TO CHANGE
nps=1e5
lB=0.01
uB=10.0
scalingFactor=1000
policy='MlpPolicy'
threshold=0.05
n_steps=32
nminibatches=4
seed=1
chuncks=100
steps=100
# DON'T CHANGE
saveDir="/home/awhitesides3/openneomc/pporuns/Agent-Run-Packages/Agents/${Agent}/"
# --- Arguments ---
ARGS="
--saveDir $saveDir
--lB $lB
--uB $uB
--nL $nL
--dose_limit $dose_limit
--nps $nps
--threshold $threshold
--scalingFactor $scalingFactor
--rps $rps
--policy $policy
--n_steps $n_steps
--nminibatches $nminibatches
--seed $seed
--chuncks $chuncks
--steps $steps
"
# --- Save Command ---
mkdir -p "${saveDir}"   # create directory if it doesn't exist
# --- Feedback ---
# README
echo "$(date) | 'initialize' run: [INSERT TEXT]" > "${saveDir}README.txt"
# Log Parameters
echo "$(date) | Arguments in 'initialize' run" > "${saveDir}parameters.txt"
echo "$ARGS" >> "${saveDir}parameters.txt"
# Log Surrogate
echo "-------------------------------------" > "${saveDir}out.out"
echo "$(date) | Starting a 'initialize' run" >> "${saveDir}out.out"
echo "-------------------------------------" >> "${saveDir}out.out"
# Run Agent Functions - set python path since initialize.py imports a .py in the directory one level above.
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"
python -u "initialize.py" $ARGS >> "${saveDir}out.out" 2>&1