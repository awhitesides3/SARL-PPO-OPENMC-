###############################   Imports   ########################################
import AgentSARL as sarl
import numpy as np
###############################   Application   ########################################
i = sarl.override_default(sarl.default)
surgPath = sarl.initialize_Surrogate(saveDir=i.saveDir)
gprDosePath, gprCostPath = sarl.initialize_GPRs(path=surgPath, saveDir=i.saveDir)
env = sarl.create_Env(gpr_dose=sarl.load_GPR(gprDosePath), gpr_cost=sarl.load_GPR(gprCostPath))
agentPath = sarl.initialize_Agent(env=env, saveDir=i.saveDir)