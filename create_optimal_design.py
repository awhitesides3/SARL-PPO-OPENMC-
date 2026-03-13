###############################   IMPORTS   ########################################
import numpy as np
import gym
from neorl import PPO2
import matplotlib.pyplot as plt
import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import gym
from gym import spaces
from datetime import datetime
from pathlib import Path
from create_surrogate import dose_calc
from argparse import ArgumentParser
import pandas as pd
import joblib
###############################   DEFINITIONS   ########################################
class SurrogateEnv(gym.Env):
    def __init__(self, GP_dose, GP_cost, dose_c, number_layers):
        super().__init__()
        self.GP_dose = GP_dose
        self.GP_cost = GP_cost
        self.dose_c = dose_c

        self.low = np.array([0.01]*number_layers)
        self.high = np.array([10.0]*number_layers)

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
def retrieve_surrogate_data(surrogate_data):
    data = np.load(surrogate_data)
    surrogate_points = data["surrogate_points"] #thicknesses
    surrogate_rewards = data["surrogate_rewards"] #reward
    surrogate_doses = data["surrogate_doses"] #dose
    surrogate_costs = data["surrogate_costs"] #cost
    return surrogate_points, surrogate_rewards, surrogate_doses, surrogate_costs
def train_GPRs(number_layers, surrogate_points, surrogate_doses, surrogate_costs):
    kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * number_layers, (1e-3, 1e3))
    gpr_dose = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True
    )
    gpr_cost = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True
    )
    gpr_dose.fit(surrogate_points, surrogate_doses)
    gpr_cost.fit(surrogate_points, surrogate_costs)
    return gpr_dose, gpr_cost
def parse_arguments():
    parser = ArgumentParser()
    for name, dtype in PARAMs.items():
        parser.add_argument(f"--{name}", type=dtype)
    return parser.parse_args()
def ppo_chuncking(iterations, total_timesteps, ppo_agent, ppo_environment):
    #this def is simply for recording data at chuncks of 'timesteps'. It is equivalent to running PPO at #timesteps = iterations*total_timesteps.
    reward_log = []
    dose_log = []
    cost_log = []
    thickness_log = []
    for i in range(iterations):
            ppo_agent.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
            observation = ppo_environment.reset()
            action, _ = ppo_agent.predict(observation, deterministic=True)
            observation, reward, done, info = ppo_environment.step(action)
            reward_log.append(reward)
            dose_log.append(info["dose"])
            cost_log.append(info["cost"])
            thickness_log.append(action.copy())
    return reward_log, dose_log, cost_log, np.array(thickness_log)
def active_learning_loop(gpr_dose, gpr_cost, dose_constraint, number_layers, iterations, total_timesteps, surrogate_points, surrogate_doses, surrogate_costs, validation_threshold, accuracy_check = False):
    while not accuracy_check:
        environment = SurrogateEnv(
            GP_dose=gpr_dose,
            GP_cost=gpr_cost,
            dose_c=dose_constraint,
            number_layers=number_layers
        )
        ppo_agent = PPO2(
            env=environment,
            policy=policy,
            n_steps=n_steps,
            nminibatches=nminibatches,
            seed=seed
        )
        reward_log, dose_log, cost_log, thickness_log = ppo_chuncking(iterations, total_timesteps, ppo_agent, environment)

        optimal_idx = np.argmax(reward_log)
        pred_optimal_dose = dose_log[optimal_idx]
        optimal_thicknesses = thickness_log[optimal_idx]
        actual_dose = dose_calc(optimal_thicknesses, layers=[], nps=nps, iteration=1)
        accuracy = abs(pred_optimal_dose - actual_dose) / max(actual_dose, 1e-12)
        
        print("Predicted Reward:", reward_log[optimal_idx])
        print("Predicted Dose:", pred_optimal_dose)
        print("Predicted Optimal Thicknesses [t1, t2, t3]:", optimal_thicknesses)
        print("Accuracy:", accuracy)

        if accuracy <= validation_threshold:
            accuracy_check = True
            print("Predicted result is accurate!")
            optimal_design_dict = {
                "optimal index": [optimal_idx],
                "optimal thicknesses": optimal_thicknesses,
                "predicted dose": [pred_optimal_dose],
                "actual dose": [actual_dose],
                "accuracy": [accuracy],
                "thickness values": thickness_log,
                "dose values": dose_log,
                "cost values": cost_log,
                "reward values": reward_log
            }
            break
        else:
            surrogate_points = np.vstack([surrogate_points, optimal_thicknesses])
            surrogate_doses = np.append(surrogate_doses, actual_dose)
            surrogate_costs = np.append(surrogate_costs, cost_log[optimal_idx])

            gpr_dose.fit(surrogate_points, surrogate_doses)
            gpr_cost.fit(surrogate_points, surrogate_costs)
    return optimal_design_dict, gpr_dose, gpr_cost, ppo_agent
###############################   Application   ########################################
if __name__ == "__main__":
    ################   SET INPUT PARAMS   ################
    PARAMs = {
        "number_layers": int,
        "lower_bound": float,
        "upper_bound": float,
        "total_timesteps": float,
        "iterations": float,
        "dose_constraint": float,
        "nps": float,
        "episode_length": int,
        "mode": str,
        "policy": str,
        "check_freq": int,
        "n_steps": int,
        "nminibatches": int,
        "seed": int,
        "validation_threshold": float,
        "save_path": str,
        "surrogate_path": str
    }
    params = vars(parse_arguments())
    lower_bound=params["lower_bound"]
    upper_bound=params["upper_bound"]
    number_layers=params["number_layers"]
    total_timesteps=int(params["total_timesteps"]) 
    iterations=int(params["iterations"])
    dose_constraint=params["dose_constraint"]
    nps=params["nps"]
    episode_length=params["episode_length"]
    mode=params["mode"]
    policy=params["policy"]
    check_freq=params["check_freq"]
    n_steps=params["n_steps"]
    nminibatches=params["nminibatches"]
    seed=params["seed"]
    validation_threshold=params["validation_threshold"]
    save_path=params["save_path"]
    surrogate_path=params["surrogate_path"]
    ################  CREATE OTHER VARIABLES   ################
    bounds = np.array([lower_bound, upper_bound])
    if surrogate_path == None:
        surrogate_data = f"{save_path}-surrogate_data.npz"
    else:
        surrogate_data = surrogate_path
    ################  FIND OPTIMAL DESIGN   ################
    surrogate_points, surrogate_rewards, surrogate_doses, surrogate_costs = retrieve_surrogate_data(surrogate_data)
    print("Retrieved surrogate data!")
    gpr_dose, gpr_cost = train_GPRs(number_layers, surrogate_points, surrogate_doses, surrogate_costs)
    print("Trained GPRs!")
    optimal_design_dict, optimal_gpr_dose, optimal_gpr_cost, ppo_agent = active_learning_loop(gpr_dose, gpr_cost, dose_constraint, number_layers, iterations, total_timesteps, surrogate_points, surrogate_doses, surrogate_costs, validation_threshold)
    optimal_index = optimal_design_dict['optimal index'][0]
    optimal_thicknesses = optimal_design_dict["optimal thicknesses"]
    print("Found the optimal design!")
    ################  SAVE RESULTS   ################
    # save the dictionary which contains all result data
    max_length = max(len(value) for value in optimal_design_dict.values())
    print(f"max length: {max_length}")
    for key,value in optimal_design_dict.items():
        optimal_design_dict[key] = list(value) + ([np.nan] * (max_length-len(value)))
    print(optimal_design_dict)
    results_data_frame = pd.DataFrame(optimal_design_dict)
    # save the data and agents which will be accessed for data analysis. additionally, these can be used to build upon with more surrogate data.
    results_data_frame.to_csv(f"{save_path}-ppo_data.csv", index=False)
    joblib.dump(gpr_cost, f"{save_path}-gpr_cost_model.pkl")
    joblib.dump(gpr_dose, f"{save_path}-gpr_dose_model.pkl")
    ppo_agent.save(f"{save_path}-ppo_agent")
    # save a .txt with info on the run
    with open(f"{save_path}-ppo_info.txt", "a") as f:
        f.write("=== New Run ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Pulled surrogate data from this location:{surrogate_data}\n")
        f.write(f"Results saved at this location: {save_path}-ppo_data.csv\n")
        f.write(f"Cost GPR saved at this location: {save_path}-gpr_cost_model.pkl\n")
        f.write(f"Dose GPR saved at this location: {save_path}-gpr_dose_model.pkl\n")
        f.write(f"PPO Agent saved at this location: {save_path}-ppo_agent\n")
        f.write(f"Best reward: {optimal_design_dict['reward values'][optimal_index]:.6f}\n")
        f.write("Optimal thicknesses:" )
        f.write(np.array2string(optimal_thicknesses, precision=4))
        f.write("\n\n")
    print("_____________________________________________________________________Successful Run!_____________________________________________________________________")