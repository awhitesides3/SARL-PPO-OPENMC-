"""
NOTES: 
"""
###############################   Imports   ########################################
import SARL as sarl
import numpy as np
###############################   Configuration   ########################################
i = sarl.override_default(sarl.default) #grabs the arguments that were passed through the 'python' command in the solve.sh job script and overrides the SARL.py default parameters
params = sarl.load_Config(f"{i.saveDir}/config.json") #grabs original parameters that were passed through initialize.sh 
# Retreive current architecture
surrogate = sarl.load_Surrogate(params["surgPath"])
gprDose = sarl.load_GPR(params["gprDoseModel"])
gprCost = sarl.load_GPR(params["gprCostModel"])
env = sarl.create_Env(gprDose, gprCost, dose_limit=i.dose_limit, nL=i.nL, bounds=np.array([i.lB, i.uB]))
agent = sarl.update_Agent_Environment(params["agentPath"], env)
###############################   Application   ########################################
