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
#############################'''SET INPUT PARAMS'''####################################
npzFile = npz #use if running this script directly after running 'FINAL_srgtPPO3L'
# npzFile = '' #use if pulling from old .npz
tts = '1e3'
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

env = SurrogateEnv(GP_dose=GP_dose, GP_cost=GP_cost, dose_c=dose_c, bounds=bounds)
agent = PPO2(
    env=env,
    policy=policy,
    n_steps=n_steps,
    nminibatches=nminibatches,
    seed=seed
)

total_timesteps = int(float(tts))
iterations = int(float(itrs))

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
best_thicknesses = thickness_log[best_idx]
print("Best reward:", reward_log[best_idx])
print("Optimal thicknesses [t1, t2, t3, t4]:", best_thicknesses)

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