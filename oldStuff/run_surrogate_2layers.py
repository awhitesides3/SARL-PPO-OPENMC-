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
from surrogateppo_2layers import openmc_fitness
# load data
data = np.load("surrogate_results/2_layer_sensitivity_runs/bounded0.1-10-0-1-1e3-1e3.npz")
x_train = data["x_train"] #thicknesses
# y_train = data["y_train"] #reward
reward_train = data["reward_train"] #reward
dose_train = data["dose_train"] #dose
cost_train = data["cost_train"] #cost

dose_c = 0.1
# prices = np.array([0.0093, 3.70]) #prices of water and ss-316L taken from the dataset.jsons 
# cost_arr = x_train @ prices
# penalty_arr = -cost_arr - reward_train
# # penalty_arr = max(0, (dose - dose_c)*10/dose_c) #negative = 0
# dose_arr = np.where(
#     penalty_arr > 0,
#     dose_c + (penalty_arr*dose_c) / 10,
#     dose_c
# )

# train surrogate
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-3, 1e3))
# GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
# GP.fit(x_train, y_train)
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

# class SurrogateEnv(gym.Env):
#     def __init__(self, GP_model, bounds):
#         super().__init__()
#         self.GP = GP_model
#         self.low = np.array([0.01, 0.01])
#         self.high = np.array([10.0, 10.0])
#         self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
#         self.action_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
#     def reset(self):
#         self.state = np.random.uniform(self.low, self.high)
#         return self.state
#     def step(self, action):
#         action = np.clip(action, self.low, self.high)
#         # reward = self.GP.predict(action.reshape(1, -1))[0]
#         mean, std = self.GP.predict(action.reshape(1, -1), return_std=True)
#         reward = min(mean-2.0*std, 0.0)
#         done=True
#         next_state = np.random.uniform(self.low, self.high)
#         return next_state, reward, done, {}
class SurrogateEnv(gym.Env):
    def __init__(self, GP_dose, GP_cost, dose_c, bounds):
        super().__init__()
        self.GP_dose = GP_dose
        self.GP_cost = GP_cost
        self.dose_c = dose_c

        self.low = np.array([0.01, 0.01])
        self.high = np.array([10.0, 10.0])

        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
    def reset(self):
        self.state = np.random.uniform(self.low, self.high)
        return self.state
    def step(self, action):
        action = np.clip(action, self.low, self.high)
        # reward = self.GP.predict(action.reshape(1, -1))[0]
        # mean, std = self.GP.predict(action.reshape(1, -1), return_std=True)
        # reward = min(mean-2.0*std, 0.0)
        dose = self.GP_dose.predict(action.reshape(1,-1))[0]
        cost = self.GP_cost.predict(action.reshape(1,-1))[0]
        penalty = max(0.0, (dose - self.dose_c) * 1000 / self.dose_c)
        reward = -cost - penalty

        done=True
        # next_state = np.random.uniform(self.low, self.high)
        next_state = self.reset()

        info = {"dose": dose, "cost": cost}
        return next_state, reward, done, info

nx = 2
bounds = {f'x{i+1}': ['float', 0.01, 10.0] for i in range(nx)}
episode_length = 1
mode = 'max'
policy = 'MlpPolicy'
check_freq = 1
n_steps = 32
nminibatches = 4
seed = 1
total_timesteps = 100

# env = SurrogateEnv(GP, bounds=bounds)
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
for i in range(100):
    agent.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    obs = env.reset()
    action, _ = agent.predict(obs)
    obs, reward, done, info = env.step(action)
    reward_log.append(reward)
    dose_log.append(info["dose"])
    cost_log.append(info["cost"])
    thickness_log.append(action.copy())

# # active learning
# history = []
# best_reward = -np.inf
# best_thicknesses = None
# best_dose = None
# best_cost = None
# for iter in range(3):
#     env = SurrogateEnv(GP_dose, GP_cost, dose_c)
#     agent = PPO2(
#         env=env,
#         policy="MlpPolicy",
#         n_steps=32,
#         nminibatches=4,
#         seed=iter #might need to change seed to '1' so that it doesn't change each run.
#     ) 
#     agent.learn(total_timesteps=1000)

#     obs = env.reset()
#     action, _ = agent.predict(obs)
#     candidate = np.clip(action, env.low, env.high)

#     true_reward, true_dose, true_cost = openmc_fitness(candidate) 
#     history.append({
#         "iteration": iter,
#         "thicknesses": candidate.copy(),
#         "reward": true_reward,
#         "dose": true_dose,
#         "cost": true_cost
#     })

#     if true_reward > best_reward:
#         best_reward = true_reward
#         best_thicknesses = candidate.copy()
#         best_dose = true_dose
#         best_cost = true_cost

#     x_train = np.vstack([x_train, candidate])
#     cost_train = np.append(cost_train, true_cost)
#     dose_train = np.append(dose_train, true_dose)

#     GP_dose.fit(x_train, dose_train)
#     GP_cost.fit(x_train, cost_train)


thickness_log = np.array(thickness_log)
best_idx = np.argmax(reward_log)
best_thicknesses = thickness_log[best_idx]
print("Best reward:", reward_log[best_idx])
print("Optimal thicknesses [t1, t2]:", best_thicknesses)
# print("Best reward:", best_reward)
# print("Optimal thicknesses [t1, t2]:", best_thicknesses)

# rewards = [h["reward"] for h in history]
# thicknesses_arr = np.array([h["thicknesses"] for h in history])
# doses = [h["dose"] for h in history]
# costs = [h["cost"] for h in history]

with open("surrogate_results/2_layer_sensitivity_runs/100-bounded0.1-10-0-1-1e3-1e3.txt", "a") as f:
    f.write("=== New Run ===\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    f.write(f"Best reward: {reward_log[best_idx]:.6f}\n")
    # f.write(f"Best reward: {best_reward:.6f}\n")
    f.write("Optimal thicknesses [t1, t2]: " )
    f.write(np.array2string(best_thicknesses, precision=4))
    f.write("\n\n")

plt.figure(1)
plt.plot(reward_log)
# plt.plot(rewards)
plt.xlabel("Surrogate training step")
plt.ylabel("Reward")
plt.title("Reward")
plt.grid(True)

plt.figure(2)
plt.plot(thickness_log[:, 0], label = 't1')
plt.plot(thickness_log[:, 1], label = 't2')
# plt.plot(thicknesses_arr[:, 0], label = 't1')
# plt.plot(thicknesses_arr[:, 1], label = 't2')
plt.xlabel("Surrogate training step")
plt.ylabel("Thickness (cm)")
plt.title("Evolution of Shield Thicknesses during PPO Training")
plt.grid(True)
plt.legend()

plt.figure(3)
plt.plot(cost_log)
# plt.plot(costs)
plt.xlabel("Surrogate training step")
plt.ylabel("Cost")
plt.title("Evolution of Cost during PPO Training")
plt.grid(True)
plt.legend()

plt.figure(4)
plt.plot(dose_log, marker='o', label = 'Dose')
# plt.plot(doses, marker='o', label = 'Dose')
plt.axhline(dose_c, color='r', linestyle='--', label = 'Dose Limit')
plt.xlabel("Surrogate training step")
plt.ylabel("Dose")
plt.title("Evolution of Dose during PPO Training")
plt.grid(True)
plt.legend()
plt.show()

# validation plot
# high = np.array([10.0, 10.0])
# low = np.array([0.01, 0.01])
# best = max(history, key=lambda h: h["reward"])
# x_start = best["thicknesses"]
# noise = 0.1 * (high - low)
# test_points = x_start + np.random.uniform(-noise, noise, size=(5,2))
# test_points = np.clip(test_points, low, high)

# true_rewards = []
# surrogate_rewards = []
# for x in test_points:
#     r_true, _, _ = openmc_fitness(x)
#     r_surr = env.step(x)[1]
#     true_rewards.append(r_true)
#     surrogate_rewards.append(r_surr)

# plt.scatter(true_rewards, surrogate_rewards)
# plt.plot([min(true_rewards), 0], [min(true_rewards), 0], 'r--')
# plt.xlabel("OpenMC reward")
# plt.ylabel("Surrogate reward")
# plt.title("Surrogate fidelity")
# plt.grid(True)
# plt.show()