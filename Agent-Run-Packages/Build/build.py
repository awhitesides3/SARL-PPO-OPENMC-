"""
NOTES: 
This script is used to develop the agent's architecture, building upon the basis provided by the intialize script.

To develop the agent's architecture you may update the surrogate data (i.e. add additional discrete data points
from which the agent can learn from) and/or update the gprs (i.e. retraining them with an updated surrogate or
with new gpr-specific parameters). Once the surrogate and gprs have been rebuilt, you must create a new env and 
then update the agent with its new evironment. - Training and solving of the agent is reserved for the 'train' and
'solve' agent packages, repsectively.  

This script should only be ran by executing the build.sh job script. Edit the necessary components
in the job script (agent name, number of layers, dose limit, number of random points, etc...) and run the script to
create a new directory for the newly initialized agent.

The agent development should take place in the specific agent's directory, accessing the SARL.py to call functions. 
"""
###############################   Imports   ########################################
import SARL as sarl
import numpy as np
###############################   Application   ########################################
i = sarl.override_default(sarl.default) #grabs the arguments that were passed through the 'python' command in the build.sh job script and overrides the SARL.py default parameters
params = sarl.load_Config(f"{i.saveDir}/config.json") #grabs original parameters that were passed through initialize.sh 
surrogate = sarl.load_Surrogate(params["surgPath"])
gprDose = sarl.load_GPR(params["gprDoseModel"])
gprCost = sarl.load_GPR(params["gprCostModel"])
env = sarl.create_Env(gprDose, gprCost, dose_limit=i.dose_limit, nL=i.nL, bounds=np.array([i.lB, i.uB]))
agent = sarl.update_Agent_Environment(params["agentPath"], env)
# build surrogate
[INSERT BUILD SURROGATE CODE]
# build gprs
[INSERT BUILD GPRS CODE]
# update the agent with the new architecture
[INSERT UPDATE AGENT CODE]



