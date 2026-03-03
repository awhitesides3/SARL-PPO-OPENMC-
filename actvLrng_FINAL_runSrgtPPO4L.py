import numpy as np
import gym
from neorl import PPO2, MlpPolicy, RLLogger, CreateEnvironment
import matplotlib.pyplot as plt
import shutil, datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import gym
from gym import spaces
from datetime import datetime
from pathlib import Path
from FINAL_srgtPPO4L import npzFile as npz
from testOpenMC import calc_dose
#############################'''SET INPUT PARAMS'''####################################
npzFile = npz #use if running this script directly after running 'FINAL_srgtPPO3L'
# npzFile = '' #use if pulling from old .npz
tts = '1e2'
itrs = '1e2'
dose_c = 0.0175
###############################'''BEGIN CODE'''########################################

# load data
data = np.load('surrogate_results/4_layer_sensitivity_runs/data/'+npzFile+'.npz')
x_train = data["x_train"] #thicknesses
reward_train = data["reward_train"] #reward
dose_train = data["dose_train"] #dose
cost_train = data["cost_train"] #cost

# train surrogate
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0, 1.0, 1.0], (1e-3, 1e3))
GP_dose = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    normalize_y=True
)
GP_dose.fit(x_train, dose_train)
GP_cost = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    normalize_y=True
)
GP_cost.fit(x_train, cost_train)

class SurrogateEnv(gym.Env):
    def __init__(self, GP_dose, GP_cost, dose_c, bounds):
        super().__init__()
        self.GP_dose = GP_dose
        self.GP_cost = GP_cost
        self.dose_c = dose_c

        self.low = np.array([0.01, 0.01, 0.01, 0.01])
        self.high = np.array([10.0, 10.0, 10.0, 10.0])

        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
    def reset(self):
        self.state = np.random.uniform(self.low, self.high)
        return self.state
    def step(self, action):
        action = np.clip(action, self.low, self.high)
        dose = self.GP_dose.predict(action.reshape(1,-1))[0]
        cost = self.GP_cost.predict(action.reshape(1,-1))[0]
        penalty = max(0.0, (dose - self.dose_c) * 1000 / self.dose_c)
        reward = -cost - penalty

        done=True
        next_state = self.reset()

        info = {"dose": dose, "cost": cost}
        return next_state, reward, done, info

nx = 4
bounds = {f'x{i+1}': ['float', 0.01, 10.0] for i in range(nx)}
episode_length = 1
mode = 'max'
policy = 'MlpPolicy'
check_freq = 1
n_steps = 32
nminibatches = 4
seed = 1

total_timesteps = int(float(tts))
iterations = int(float(itrs))

accuracy_check = False
while not accuracy_check:
    env = SurrogateEnv(GP_dose=GP_dose, GP_cost=GP_cost, dose_c=dose_c, bounds=bounds)
    agent = PPO2(
        env=env,
        policy=policy,
        n_steps=n_steps,
        nminibatches=nminibatches,
        seed=seed
    )

    reward_log = []
    thickness_log = []
    dose_log = []
    cost_log = []
    for i in range(iterations):
        agent.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        obs = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        reward_log.append(reward)
        dose_log.append(info["dose"])
        cost_log.append(info["cost"])
        thickness_log.append(action.copy())

    thickness_log = np.array(thickness_log)
    best_idx = np.argmax(reward_log)
    pred_opt_dose = dose_log[best_idx]
    best_thicknesses = thickness_log[best_idx]
    actual_dose = calc_dose(best_thicknesses)
    accuracy = abs(pred_opt_dose - actual_dose) / max(actual_dose, 1e-12)
    
    print("Predicted Reward:", reward_log[best_idx])
    print("Predicted Dose:", pred_opt_dose)
    print("Predicted Optimal Thicknesses [t1, t2, t3, t4]:", best_thicknesses)
    print("Accuracy:", accuracy)

    if accuracy <= 0.1:
        accuracy_check = True
        print("Predicted result is accurate!")
        break
    else:
        x_train = np.vstack([x_train, best_thicknesses])
        dose_train = np.append(dose_train, actual_dose)
        cost_train = np.append(cost_train, cost_log[best_idx])

        GP_dose.fit(x_train, dose_train)
        GP_cost.fit(x_train, cost_train)

resultsFile = tts+'-'+itrs+'-'+npzFile
new_directory = '/home/awhitesides3/openneomc/pporuns/surrogate_results/4_layer_sensitivity_runs/'+resultsFile
Path(new_directory).mkdir()
with open(new_directory+'/'+resultsFile+'.txt', "a") as f:
    f.write("=== New Run ===\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    f.write(f"Best reward: {reward_log[best_idx]:.6f}\n")
    f.write("Optimal thicknesses [t1, t2, t3, t4]: " )
    f.write(np.array2string(best_thicknesses, precision=4))
    f.write("\n\n")

plt.figure(1)
plt.plot(reward_log)
plt.xlabel("Surrogate training step")
plt.ylabel("Reward")
plt.title("Reward")
plt.grid(True)
plt.savefig(new_directory+'/'+'reward.png')
plt.close()

plt.figure(2)
plt.plot(thickness_log[:, 0], label = 't1')
plt.plot(thickness_log[:, 1], label = 't2')
plt.plot(thickness_log[:, 2], label = 't3')
plt.plot(thickness_log[:, 3], label = 't4')
plt.xlabel("Surrogate training step")
plt.ylabel("Thickness (cm)")
plt.title("Evolution of Shield Thicknesses during PPO Training")
plt.grid(True)
plt.legend()
plt.savefig(new_directory+'/'+"thickness.png")
plt.close()

plt.figure(3)
plt.plot(cost_log)
plt.xlabel("Surrogate training step")
plt.ylabel("Cost")
plt.title("Evolution of Cost during PPO Training")
plt.grid(True)
plt.legend()
plt.savefig(new_directory+'/'+"cost.png")
plt.close()

plt.figure(4)
plt.plot(dose_log, marker='o', label = 'Dose')
plt.axhline(dose_c, color='r', linestyle='--', label = 'Dose Limit')
plt.xlabel("Surrogate training step")
plt.ylabel("Dose")
plt.title("Evolution of Dose during PPO Training")
plt.grid(True)
plt.legend()
plt.savefig(new_directory+'/'+"dose.png")
plt.close()

# visualize GP process
t1_fixed = best_thicknesses[0]
t2_fixed = best_thicknesses[1]
t3 = np.linspace(0.01, 10.0, 50)
t4 = np.linspace(0.01, 10.0, 50)

T3, T4 = np.meshgrid(t3, t4)
x_arr = np.column_stack([np.full(T3.size, t1_fixed), np.full(T3.size, t2_fixed), T3.ravel(), T4.ravel()])
mean, std = GP_dose.predict(x_arr, return_std=True)
mean = mean.reshape(T3.shape)
std = std.reshape(T3.shape)
mean_c, std_c = GP_cost.predict(x_arr, return_std=True)
mean_c = mean_c.reshape(T3.shape)
std_c = std_c.reshape(T3.shape)
# plot GP mean dose
plt.figure(10)
plt.contour(T3, T4, mean, levels=50, cmap='viridis')
plt.colorbar(label="Predicted Dose")
plt.contour(T3, T4, mean, levels=[dose_c], colors='red', linewidths=2, linestyles='--', label='Dose Limit')
plt.xlabel("Layer 3 thickness")
plt.ylabel("Layer 4 thickness")
plt.title("GP Mean Dose Prediction: Fixed at Optimal L1 & L2 Thickness")
# overlay OpenMC runs
plt.scatter(x_train[:,2], x_train[:,3], c="red", s=30, label="OpenMC Samples")
plt.scatter(best_thicknesses[2], best_thicknesses[3], c="green", s=30, label="Gp Optimal Solution")
plt.legend()
plt.savefig(new_directory+'/'+"GP_dose_mean_fL1L2.png")
plt.close()
# plot GP dose uncertainty
plt.figure(11)
plt.contour(T3, T4, std, levels=50)
plt.colorbar(label="GP Std. Dev.")
plt.xlabel("Layer 3 thickness")
plt.ylabel("Layer 4 thickness")
plt.title("GP Dose Prediction Uncertainty: Fixed at Optimal L1 & L2 Thickness")
plt.scatter(x_train[:,2], x_train[:,3], c="red", s=30, label="OpenMC Samples")
plt.scatter(best_thicknesses[2], best_thicknesses[3], c="green", s=30, label="Gp Optimal Solution")
plt.legend()
plt.savefig(new_directory+'/'+"GP_dose_uncertainty_fL1L2.png")
plt.close()
# plot GP mean cost
plt.figure(12)
plt.contour(T3, T4, mean_c, levels=50)
plt.colorbar(label="Predicted Cost")
plt.xlabel("Layer 3 thickness")
plt.ylabel("Layer 4 thickness")
plt.title("GP Mean Cost Prediction: Fixed at Optimal L1 & L2 Thickness")
# overlay OpenMC runs
plt.scatter(x_train[:,2], x_train[:,3], c="red", s=30, label="OpenMC Samples")
plt.scatter(best_thicknesses[2], best_thicknesses[3], c="green", s=30, label="Gp Optimal Solution")
plt.legend()
plt.savefig(new_directory+'/'+"GP_cost_mean_fL1L2.png")
plt.close()
# plot GP cost uncertainty
plt.figure(13)
plt.contour(T3, T4, std_c, levels=50)
plt.colorbar(label="GP Std. Dev.")
plt.xlabel("Layer 3 thickness")
plt.ylabel("Layer 4 thickness")
plt.title("GP Cost Prediction Uncertainty: Fixed at Optimal L1 & L2 Thickness")
plt.scatter(x_train[:,2], x_train[:,3], c="red", s=30, label="OpenMC Samples")
plt.scatter(best_thicknesses[2], best_thicknesses[3], c="green", s=30, label="Gp Optimal Solution")
plt.legend()
plt.savefig(new_directory+'/'+"GP_cost_uncertainty_fL1L2.png")
plt.close()

t2_fixed = best_thicknesses[1]
t3_fixed = best_thicknesses[2]
t1 = np.linspace(0.01, 10.0, 50)
t4 = np.linspace(0.01, 10.0, 50)

T1, T4 = np.meshgrid(t1, t4)
x_arr = np.column_stack([T1.ravel(), np.full(T1.size, t2_fixed), np.full(T1.size, t3_fixed), T4.ravel()])
mean, std = GP_dose.predict(x_arr, return_std=True)
mean = mean.reshape(T4.shape)
std = std.reshape(T4.shape)
mean_c, std_c = GP_cost.predict(x_arr, return_std=True)
mean_c = mean_c.reshape(T4.shape)
std_c = std_c.reshape(T4.shape)
# plot GP mean dose
plt.figure(14)
plt.contour(T1, T4, mean, levels=50, cmap='viridis')
plt.colorbar(label="Predicted Dose")
plt.contour(T1, T4, mean, levels=[dose_c], colors='red', linewidths=2, linestyles='--', label='Dose Limit')
plt.xlabel("Layer 1 thickness")
plt.ylabel("Layer 4 thickness")
plt.title("GP Mean Dose Prediction: Fixed at Optimal L2 & L3 Thickness")
# overlay OpenMC runs
plt.scatter(x_train[:,0], x_train[:,3], c="red", s=30, label="OpenMC Samples")
plt.scatter(best_thicknesses[0], best_thicknesses[3], c="green", s=30, label="Gp Optimal Solution")
plt.legend()
plt.savefig(new_directory+'/'+"GP_dose_mean_fL2L3.png")
plt.close()
# plot GP dose uncertainty
plt.figure(15)
plt.contour(T1, T4, std, levels=50)
plt.colorbar(label="GP Std. Dev.")
plt.xlabel("Layer 1 thickness")
plt.ylabel("Layer 4 thickness")
plt.title("GP Dose Prediction Uncertainty: Fixed at Optimal L2 & L3 Thickness")
plt.scatter(x_train[:,0], x_train[:,3], c="red", s=30, label="OpenMC Samples")
plt.scatter(best_thicknesses[0], best_thicknesses[3], c="green", s=30, label="Gp Optimal Solution")
plt.legend()
plt.savefig(new_directory+'/'+"GP_dose_uncertainty_fL2L3.png")
plt.close()
# plot GP mean cost
plt.figure(16)
plt.contour(T1, T4, mean_c, levels=50)
plt.colorbar(label="Predicted Cost")
plt.xlabel("Layer 1 thickness")
plt.ylabel("Layer 4 thickness")
plt.title("GP Mean Cost Prediction: Fixed at Optimal L2 & L3 Thickness")
# overlay OpenMC runs
plt.scatter(x_train[:,0], x_train[:,3], c="red", s=30, label="OpenMC Samples")
plt.scatter(best_thicknesses[0], best_thicknesses[3], c="green", s=30, label="Gp Optimal Solution")
plt.legend()
plt.savefig(new_directory+'/'+"GP_cost_mean_fL2L3.png")
plt.close()
# plot GP cost uncertainty
plt.figure(17)
plt.contour(T1, T4, std_c, levels=50)
plt.colorbar(label="GP Std. Dev.")
plt.xlabel("Layer 1 thickness")
plt.ylabel("Layer 4 thickness")
plt.title("GP Cost Prediction Uncertainty: Fixed at Optimal L2 & L3 Thickness")
plt.scatter(x_train[:,0], x_train[:,3], c="red", s=30, label="OpenMC Samples")
plt.scatter(best_thicknesses[0], best_thicknesses[3], c="green", s=30, label="Gp Optimal Solution")
plt.legend()
plt.savefig(new_directory+'/'+"GP_cost_uncertainty_fL2L3.png")
plt.close()