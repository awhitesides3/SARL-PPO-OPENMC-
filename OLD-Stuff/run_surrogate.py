import openmc
import numpy as np
import gym
from neorl import PPO2, MlpPolicy, RLLogger, CreateEnvironment
import os
import glob
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import shutil, datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import gym
from gym import spaces
from datetime import datetime
# load data
data = np.load("v2surrogateData_4L_1e3Penalty_20mcr.npz")
x_train = data["x_train"] #thicknesses
y_train = data["y_train"] #reward

dose_c = 0.1
prices = np.array([0.0093, 3.70, 0.0093, 3.70]) #prices of water and ss-316L taken from the dataset.jsons 
cost_arr = x_train @ prices
penalty_arr = -cost_arr - y_train
# penalty_arr = max(0, (dose - dose_c)*10/dose_c) #negative = 0
dose_arr = np.where(
    penalty_arr > 0,
    dose_c + (penalty_arr*dose_c) / 10,
    dose_c
)
# train surrogate
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0, 1.0, 1.0], (1e-3, 1e3))
GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
GP.fit(x_train, y_train)

class SurrogateEnv(gym.Env):
    def __init__(self, GP_model, bounds):
        super().__init__()
        self.GP = GP_model
        self.low = np.array([0.01, 0.01, 0.01, 0.01])
        self.high = np.array([10.0, 10.0, 10.0, 10.0])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
    def reset(self):
        self.state = np.random.uniform(self.low, self.high)
        return self.state
    def step(self, action):
        action = np.clip(action, self.low, self.high)
        reward = self.GP.predict(action.reshape(1, -1))[0]
        done=True
        next_state = np.random.uniform(self.low, self.high)
        return next_state, reward, done, {}

nx = 4
bounds = {f'x{i+1}': ['float', 0.01, 10.0] for i in range(nx)}
episode_length = 1
mode = 'max'
policy = 'MlpPolicy'
check_freq = 1
n_steps = 32
nminibatches = 4
seed = 1
total_timesteps = 100

env = SurrogateEnv(GP, bounds=bounds)
agent = PPO2(env=env, policy=policy, n_steps=n_steps, nminibatches=nminibatches, seed=seed)

reward_log = []
thickness_log = []
for i in range(100):
    agent.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    obs = env.reset()
    action, _ = agent.predict(obs)
    obs, reward, done, _ = env.step(action)
    reward_log.append(reward)
    thickness_log.append(action.copy())

thickness_log = np.array(thickness_log)
best_idx = np.argmax(reward_log)
best_thicknesses = thickness_log[best_idx]
print("Best reward:", reward_log[best_idx])
print("Optimal thicknesses [t1, t2, t3, t4]:", best_thicknesses)

with open("surrogate_results/2_layer_sensitivity_runs/.txt", "a") as f:
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

plt.figure(2)
plt.plot(thickness_log[:,0], label = 't1')
plt.plot(thickness_log[:,1], label = 't2')
plt.plot(thickness_log[:,2], label = 't3')
plt.plot(thickness_log[:,3], label = 't4')
plt.xlabel("Surrogate training step")
plt.ylabel("Thickness (cm)")
plt.title("Evolution of Shield Thicknesses during PPO Training")
plt.grid(True)
plt.legend()

plt.figure(3)
plt.plot(cost_arr)
plt.xlabel("Surrogate training step")
plt.ylabel("Cost")
plt.title("Evolution of Cost during PPO Training")
plt.grid(True)
plt.legend()

plt.figure(4)
plt.plot(dose_arr, marker='o', label = 'Dose')
plt.axhline(dose_c, color='r', linestyle='--', label = 'Dose Limit')
plt.xlabel("Surrogate training step")
plt.ylabel("Dose")
plt.title("Evolution of Dose during PPO Training")
plt.grid(True)
plt.legend()
plt.show()
