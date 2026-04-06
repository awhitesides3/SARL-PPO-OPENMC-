
#!/bin/bash
# --- Necessary Dependencies ---
source /home/awhitesides3/miniconda3/etc/profile.d/conda.sh
conda activate openneomc
# --- Parameters ---
# CHANGE
Agent=Agent1
bounds=(0.0, 10.0)
nL=2
dose_limit=0.0936
nps=1e5
theshold=0.05
scalingFactor=1000
rps=1
policy='MlpPolicy'
n_steps=32
nminibatches=4
seed=1
chuncks=100
steps=100
# don't change
saveDir="/home/awhitesides3/openneomc/pporuns/Agents/${Agent}/"
# --- Arguments ---
ARGS="
--saveDir $saveDir
--bounds $bounds
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
echo "First Agent" > "${saveDir}README.txt"
# Log Parameters
echo "Arguments" > "${saveDir}parameters.txt"
echo "$ARGS" >> "${saveDir}parameters.txt"
# Log Surrogate
echo "-------------------------------------" > "${saveDir}out.out"
echo "Starting Run: $(date)" >> "${saveDir}out.out"
echo "-------------------------------------" >> "${saveDir}out.out"
# Run Agent Functions
python -u "initializeAgent.py" $ARGS >> "${saveDir}out.out" 2>&1