"""
NOTES: 
This script initializes the necessary components for an agent sensitivity study. This script should not be changed.

This script should only be ran through executing the initialize.sh job script. Edit the necessary components
in the job script (agent name, number of layers, dose limit, number of random points, etc...) and run the script to
create a new directory for the newly initialized agent.

Then all agent development should take place in the specific agent's directory, accessing the SARL.py to call functions. 
"""
###############################   Imports   ########################################
import SARL as sarl
import numpy as np
###############################   Application   ########################################
i = sarl.override_default(sarl.default) #grabs the arguments that were passed through the 'python' command in the initialize.sh job script and overrides the SARL.py default parameters
# Initialize beginning architecture
surgPath = sarl.initialize_Surrogate(saveDir=i.saveDir, nL=i.nL, rps=i.rps, bounds=np.array([i.lB, i.uB]), dose_limit=i.dose_limit)
gprDosePath, gprCostPath = sarl.initialize_GPRs(path=surgPath, saveDir=i.saveDir, nL=i.nL)
env = sarl.create_Env(gpr_dose=sarl.load_GPR(gprDosePath), gpr_cost=sarl.load_GPR(gprCostPath), dose_limit=i.dose_limit, nL=i.nL, bounds=np.array([i.lB, i.uB]))
agentPath = sarl.initialize_Agent(env=env, saveDir=i.saveDir, n_steps=i.n_steps, nminibatches=i.nminibatches, policy=i.policy, seed=i.seed)
# Update & save the configuration class with the paths to each component of the architecture
sarl.update_Config(i, "surgPath", surgPath)
sarl.update_Config(i, "gprDosePath", gprDosePath)
sarl.update_Config(i, "gprCostPath", gprCostPath)
sarl.update_Config(i, "agentPath", agentPath)
sarl.save_Config(i)
# COMPLETE
print("Successfully completed the Agent Initialization Process!")
