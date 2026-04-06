###############################   Imports   ########################################
import AgentSARL as sarl
import numpy as np
###############################   Application   ########################################
i = sarl.override_default(sarl.default)
surgPath = sarl.initialize_Surrogate(saveDir=i.saveDir, nL=i.nL, rps=i.rps, bounds=np.array(i.bounds), dose_limit=i.dose_limit)
gprDosePath, gprCostPath = sarl.initialize_GPRs(path=surgPath, saveDir=i.saveDir, nL=i.nL)
env = sarl.create_Env(gpr_dose=sarl.load_GPR(gprDosePath), gpr_cost=sarl.load_GPR(gprCostPath), dose_limit=i.dose_limit, nL=i.nL, bounds=np.array(i.bounds))
agentPath = sarl.initialize_Agent(env=env, saveDir=i.saveDir, n_steps=i.n_steps, nminibatches=i.nminibatches, policy=i.policy, seed=i.seed)