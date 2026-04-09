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
import random
###############################   Configuration   ########################################
i = sarl.override_default(sarl.default) #grabs the arguments that were passed through the 'python' command in the build.sh job script and overrides the SARL.py default parameters
params = sarl.load_Config(f"{i.saveDir}/config.json") #grabs original parameters that were passed through initialize.sh 
###############################   Application   ########################################
# BUILD SURROGATE
    # Option 1: 
        # Adding to the surrogate by providing a desired thickness vector. 
        # This tells SARL "Evaluate the openMC model with the given thicknesses for each layer."
new_thicknesses = np.random.uniform(i.lB, i.uB, i.nL) # random thickness array
points, rewards, doses, costs = sarl.update_Surrogate(params["surgPath"], new_thicknesses, bounds=i.bounds, dose_limit=i.dose_limit)
    # Option 2:
        # The other method by which the surrogate is updated is through the active learning loop. 
        # Instead of telling SARL to evaluate at a desired OpenMC design case, 
        # this promps SARL to "Evaluate the openMC model at the proposed optimal design."
        # This proposed solution is provided by the agent which evaluates the continuous solution 
        # space provided by the GPR models. 
        # **THIS OPTION IS RESERVED FOR IMPLICIT USE DURING THE solve.sh JOB SUBMISSION**
# BUILD GPRS
gprDose = sarl.update_GPR(params["gprDoseModel"], points, doses)
gprCost = sarl.update_GPR(params["gprCostModel"], points, costs)
# UPDATE THE AGENT WITH THE NEW ARCHITECTURE
env = sarl.create_Env(gprDose, gprCost, dose_limit=i.dose_limit, nL=i.nL, bounds=np.array([i.lB, i.uB]))
agent = sarl.update_Agent_Environment(params["agentPath"], env)



