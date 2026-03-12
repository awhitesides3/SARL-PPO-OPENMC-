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
def retrieve_surrogate_data(npz_file_path):
    data = np.load(npz_file_path)
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
        agent = PPO2(
            env=environment,
            policy=policy,
            n_steps=n_steps,
            nminibatches=nminibatches,
            seed=seed
        )
        reward_log, dose_log, cost_log, thickness_log = ppo_chuncking(iterations, total_timesteps, agent, environment)

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
                "optimal index": optimal_idx,
                "optimal thicknesses": optimal_thicknesses,
                "predicted dose": pred_optimal_dose,
                "actual dose": actual_dose,
                "accuracy": accuracy,
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
    return optimal_design_dict, gpr_dose, gpr_cost
###############################   Application   ########################################
if __name__ == "__main__":
    ################   SET INPUT PARAMS   ################
    PARAMs = {
        "npz_file_path": str,
        "npz_file_name": str,
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
        "results_path": str
    }
    params = vars(parse_arguments())
    npz_file_path=params["npz_file_path"]
    npz_file_name=params["npz_file_name"]
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
    results_path=params["results_path"]
    ################  CREATE OTHER VARIABLES   ################
    bounds = np.array([lower_bound, upper_bound])
    print(f"results_path: {results_path}")
    results_directory = f"{results_path}{total_timesteps:.0e}-{iterations:.0e}-{npz_file_name}"
    print(f"results_directory: {results_directory}")
    results_file_name = f"{total_timesteps:.0e}-{iterations:.0e}-{npz_file_name}.txt"
    print(f"results_file_name: {results_file_name}")
    ################  FIND OPTIMAL DESIGN   ################
    surrogate_points, surrogate_rewards, surrogate_doses, surrogate_costs = retrieve_surrogate_data(npz_file_path)
    print("Retrieved surrogate data!")
    gpr_dose, gpr_cost = train_GPRs(number_layers, surrogate_points, surrogate_doses, surrogate_costs)
    print("Trained GPRs!")
    optimal_design_dict, optimal_gpr_dose, optimal_gpr_cost = active_learning_loop(gpr_dose, gpr_cost, dose_constraint, number_layers, iterations, total_timesteps, surrogate_points, surrogate_doses, surrogate_costs, validation_threshold)
    print("Found the optimal design!")
    ################  SAVE RESULTS   ################
    Path(results_directory).mkdir()
    with open(results_directory+'/'+results_file_name+'.txt', "a") as f:
        f.write("=== New Run ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Best reward: {optimal_design_dict['reward values'][optimal_design_dict['optimal index']]:.6f}\n")
        f.write("Optimal thicknesses:" )
        f.write(np.array2string(optimal_design_dict["optimal thicknesses"], precision=4))
        f.write("\n\n")
###############################   PLOTTING   ########################################
# plt.figure(1)
# plt.plot(reward_log)
# plt.xlabel("Surrogate training step")
# plt.ylabel("Reward")
# plt.title("Reward")
# plt.grid(True)
# plt.savefig(results_directory+'/'+'reward.png')
# plt.close()

# plt.figure(2)
# plt.plot(thickness_log[:, 0], label = 't1')
# plt.plot(thickness_log[:, 1], label = 't2')
# plt.plot(thickness_log[:, 2], label = 't3')
# plt.xlabel("Surrogate training step")
# plt.ylabel("Thickness (cm)")
# plt.title("Evolution of Shield Thicknesses during PPO Training")
# plt.grid(True)
# plt.legend()
# plt.savefig(results_directory+'/'+"thickness.png")
# plt.close()

# plt.figure(3)
# plt.plot(cost_log)
# plt.xlabel("Surrogate training step")
# plt.ylabel("Cost")
# plt.title("Evolution of Cost during PPO Training")
# plt.grid(True)
# plt.legend()
# plt.savefig(results_directory+'/'+"cost.png")
# plt.close()

# plt.figure(4)
# plt.plot(dose_log, marker='o', label = 'Dose')
# plt.axhline(dose_constraint, color='r', linestyle='--', label = 'Dose Limit')
# plt.xlabel("Surrogate training step")
# plt.ylabel("Dose")
# plt.title("Evolution of Dose during PPO Training")
# plt.grid(True)
# plt.legend()
# plt.savefig(results_directory+'/'+"dose.png")
# plt.close()

# # visualize GP process
# t1_fixed = best_thicknesses[0]
# t2 = np.linspace(0.01, 10.0, 50)
# t3 = np.linspace(0.01, 10.0, 50)

# T2, T3 = np.meshgrid(t2, t3)
# x_arrF1 = np.column_stack([np.full(T2.size, t1_fixed), T2.ravel(), T3.ravel()])
# mean, std = gpr_dose.predict(x_arrF1, return_std=True)
# mean = mean.reshape(T3.shape)
# std = std.reshape(T3.shape)
# mean_c, std_c = gpr_cost.predict(x_arrF1, return_std=True)
# mean_c = mean_c.reshape(T3.shape)
# std_c = std_c.reshape(T3.shape)
# # plot GP mean dose
# plt.figure(10)
# plt.contour(T2, T3, mean, levels=50, cmap='viridis')
# plt.colorbar(label="Predicted Dose")
# plt.contour(T2, T3, mean, levels=[dose_constraint], colors='red', linewidths=2, linestyles='--', label='Dose Limit')
# plt.xlabel("Layer 2 thickness")
# plt.ylabel("Layer 3 thickness")
# plt.title("GP Mean Dose Prediction: Fixed at Optimal L1 Thickness")
# # overlay OpenMC runs
# plt.scatter(x_train[:,1], x_train[:,2], c="red", s=30, label="OpenMC Samples")
# plt.scatter(best_thicknesses[1], best_thicknesses[2], c="green", s=30, label="Gp Optimal Solution")
# plt.legend()
# plt.savefig(results_directory+'/'+"GP_dose_mean_fL1.png")
# plt.close()
# # plot GP dose uncertainty
# plt.figure(11)
# plt.contour(T2, T3, std, levels=50)
# plt.colorbar(label="GP Std. Dev.")
# plt.xlabel("Layer 2 thickness")
# plt.ylabel("Layer 3 thickness")
# plt.title("GP Dose Prediction Uncertainty: Fixed at Optimal L1 Thickness")
# plt.scatter(x_train[:,1], x_train[:,2], c="red", s=30, label="OpenMC Samples")
# plt.scatter(best_thicknesses[1], best_thicknesses[2], c="green", s=30, label="Gp Optimal Solution")
# plt.legend()
# plt.savefig(results_directory+'/'+"GP_dose_uncertainty_fL1.png")
# plt.close()
# # plot GP mean cost
# plt.figure(12)
# plt.contour(T2, T3, mean_c, levels=50)
# plt.colorbar(label="Predicted Cost")
# plt.xlabel("Layer 2 thickness")
# plt.ylabel("Layer 3 thickness")
# plt.title("GP Mean Cost Prediction: Fixed at Optimal L1 Thickness")
# # overlay OpenMC runs
# plt.scatter(x_train[:,1], x_train[:,2], c="red", s=30, label="OpenMC Samples")
# plt.scatter(best_thicknesses[1], best_thicknesses[2], c="green", s=30, label="Gp Optimal Solution")
# plt.legend()
# plt.savefig(results_directory+'/'+"GP_cost_mean_fL1.png")
# plt.close()
# # plot GP cost uncertainty
# plt.figure(13)
# plt.contour(T2, T3, std_c, levels=50)
# plt.colorbar(label="GP Std. Dev.")
# plt.xlabel("Layer 2 thickness")
# plt.ylabel("Layer 3 thickness")
# plt.title("GP Cost Prediction Uncertainty: Fixed at Optimal L1 Thickness")
# plt.scatter(x_train[:,1], x_train[:,2], c="red", s=30, label="OpenMC Samples")
# plt.scatter(best_thicknesses[1], best_thicknesses[2], c="green", s=30, label="Gp Optimal Solution")
# plt.legend()
# plt.savefig(results_directory+'/'+"GP_cost_uncertainty_fL1.png")
# plt.close()

# t2_fixed = best_thicknesses[1]
# t1 = np.linspace(0.01, 10.0, 50)
# t3 = np.linspace(0.01, 10.0, 50)

# T1, T3 = np.meshgrid(t1, t3)
# x_arrF2 = np.column_stack([T1.ravel(), np.full(T1.size, t2_fixed), T3.ravel()])
# mean, std = gpr_dose.predict(x_arrF2, return_std=True)
# mean = mean.reshape(T3.shape)
# std = std.reshape(T3.shape)
# mean_c, std_c = gpr_cost.predict(x_arrF2, return_std=True)
# mean_c = mean_c.reshape(T3.shape)
# std_c = std_c.reshape(T3.shape)
# # plot GP mean dose
# plt.figure(14)
# plt.contour(T1, T3, mean, levels=50, cmap='viridis')
# plt.colorbar(label="Predicted Dose")
# plt.contour(T1, T3, mean, levels=[dose_constraint], colors='red', linewidths=2, linestyles='--', label='Dose Limit')
# plt.xlabel("Layer 1 thickness")
# plt.ylabel("Layer 3 thickness")
# plt.title("GP Mean Dose Prediction: Fixed at Optimal L2 Thickness")
# # overlay OpenMC runs
# plt.scatter(x_train[:,0], x_train[:,2], c="red", s=30, label="OpenMC Samples")
# plt.scatter(best_thicknesses[0], best_thicknesses[2], c="green", s=30, label="Gp Optimal Solution")
# plt.legend()
# plt.savefig(results_directory+'/'+"GP_dose_mean_fL2.png")
# plt.close()
# # plot GP dose uncertainty
# plt.figure(15)
# plt.contour(T1, T3, std, levels=50)
# plt.colorbar(label="GP Std. Dev.")
# plt.xlabel("Layer 1 thickness")
# plt.ylabel("Layer 3 thickness")
# plt.title("GP Dose Prediction Uncertainty: Fixed at Optimal L2 Thickness")
# plt.scatter(x_train[:,0], x_train[:,2], c="red", s=30, label="OpenMC Samples")
# plt.scatter(best_thicknesses[0], best_thicknesses[2], c="green", s=30, label="Gp Optimal Solution")
# plt.legend()
# plt.savefig(results_directory+'/'+"GP_dose_uncertainty_fL2.png")
# plt.close()
# # plot GP mean cost
# plt.figure(16)
# plt.contour(T1, T3, mean_c, levels=50)
# plt.colorbar(label="Predicted Cost")
# plt.xlabel("Layer 1 thickness")
# plt.ylabel("Layer 3 thickness")
# plt.title("GP Mean Cost Prediction: Fixed at Optimal L2 Thickness")
# # overlay OpenMC runs
# plt.scatter(x_train[:,0], x_train[:,2], c="red", s=30, label="OpenMC Samples")
# plt.scatter(best_thicknesses[0], best_thicknesses[2], c="green", s=30, label="Gp Optimal Solution")
# plt.legend()
# plt.savefig(results_directory+'/'+"GP_cost_mean_fL2.png")
# plt.close()
# # plot GP cost uncertainty
# plt.figure(17)
# plt.contour(T1, T3, std_c, levels=50)
# plt.colorbar(label="GP Std. Dev.")
# plt.xlabel("Layer 1 thickness")
# plt.ylabel("Layer 3 thickness")
# plt.title("GP Cost Prediction Uncertainty: Fixed at Optimal L2 Thickness")
# plt.scatter(x_train[:,0], x_train[:,2], c="red", s=30, label="OpenMC Samples")
# plt.scatter(best_thicknesses[0], best_thicknesses[2], c="green", s=30, label="Gp Optimal Solution")
# plt.legend()
# plt.savefig(results_directory+'/'+"GP_cost_uncertainty_fL2.png")
# plt.close()