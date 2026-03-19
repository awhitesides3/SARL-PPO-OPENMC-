###############################   IMPORTS   ########################################
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib
###############################   DEFINITIONS   ########################################
def plot_reward(save_path, reward_values, optimal_index):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.scatter(range(1, len(reward_values)+1), reward_values)
    ax.axvline(x=optimal_index[0], color='green', linestyle='--')
    ax.set_xlabel('PPO Training Step', fontweight="bold", fontsize=11)
    ax.set_ylabel('Reward [-]', fontweight="bold", fontsize=11)
    ax.grid(True)
    fig.suptitle('Reward vs. PPO Training Step', fontsize=14, y=0.92, fontweight="bold")
    fig.savefig(f"{save_path}-reward_plot.png", dpi=300, bbox_inches='tight')
    print("plotted the reward values!")
    return
def plot_dose(save_path, dose_values, optimal_index, dose_constraint):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.scatter(range(1, len(dose_values)+1), dose_values)
    ax.axvline(x=optimal_index[0], color='green', linestyle='--')
    ax.axhline(y=dose_constraint[0], color='red', linestyle='--')
    ax.set_xlabel('PPO Training Step', fontweight="bold", fontsize=11)
    ax.set_ylabel('Dose [-]', fontweight="bold", fontsize=11)
    ax.grid(True)
    fig.suptitle('Dose vs. PPO Training Step', fontsize=14, y=0.92, fontweight="bold")
    fig.savefig(f"{save_path}-dose_plot.png", dpi=300, bbox_inches='tight')
    print("plotted the dose values!")
    return
def plot_cost(save_path, cost_values, optimal_index):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.scatter(range(1, len(cost_values)+1), cost_values)
    ax.axvline(x=optimal_index[0], color='green', linestyle='--')
    ax.set_xlabel('PPO Training Step', fontweight="bold", fontsize=11)
    ax.set_ylabel('Cost [-]', fontweight="bold", fontsize=11)
    ax.grid(True)
    fig.suptitle('Cost vs. PPO Training Step', fontsize=14, y=0.92, fontweight="bold")
    fig.savefig(f"{save_path}-cost_plot.png", dpi=300, bbox_inches='tight')
    print("plotted the cost values!")
    return
def plot_thickness(save_path, thickness_values, optimal_index):
    fig, ax = plt.subplots(figsize=(14,6))
    for layer, thicknesses in enumerate(thickness_values.T):
        print(thicknesses)
        ax.scatter(range(1, len(thicknesses)+1), thicknesses, label=f"layer #{layer+1}")
    ax.legend()
    ax.axvline(x=optimal_index[0], color='green', linestyle='--', label="Optimal Design")
    ax.set_xlabel('PPO Training Step', fontweight="bold", fontsize=11)
    ax.set_ylabel('Thickness [cm]', fontweight="bold", fontsize=11)
    ax.grid(True)
    fig.suptitle('Thickness vs. PPO Training Step', fontsize=14, y=0.92, fontweight="bold")
    fig.savefig(f"{save_path}-thickness_plot.png", dpi=300, bbox_inches='tight')
    print("plotted the thickness values!")
    return
def plot_gpr_dose():
    return
def plot_gpr_cost():
    return
def retrieve_run_data(data_path):
    data = np.load(data_path)
    return data["optimal_index"], data["dose_constraint"], data["reward_log"], data["dose_log"], data["cost_log"], data["thickness_log"]
def parse_arguments():
    parser = ArgumentParser()
    for name, dtype in PARAMs.items():
        parser.add_argument(f"--{name}", type=dtype)
    return parser.parse_args()
###############################   Application   ########################################
if __name__ == "__main__":
    ################   SET INPUT PARAMS   ################
    PARAMs = {
        "save_path": str,
        "data_path": str,
        "gpr_dose_path": str,
        "gpr_cost_path": str,
    }
    params = vars(parse_arguments())
    save_path=params["save_path"]
    data_path=params["data_path"]
    gpr_dose_path=params["gpr_dose_path"]
    gpr_cost_path=params["gpr_cost_path"]
    ################  CREATE OTHER VARIABLES   ################
    # retrieve data from respective run
    optimal_index, dose_constraint, reward_values, dose_values, cost_values, thickness_values = retrieve_run_data(data_path)
    # retrieve gpr models
    with open(gpr_dose_path, "rb") as f:
        gpr_dose = joblib.load(f)
    with open(gpr_cost_path, "rb") as f:
        gpr_cost = joblib.load(f)
    ################  PERFORM DATA ANALYSIS   ################
    ################  SAVE RESULTS   ################
    print("_____________________________________________________________________Saving Data Analysis!_____________________________________________________________________")
    plot_reward(save_path, reward_values, optimal_index)
    plot_dose(save_path, dose_values, optimal_index, dose_constraint)
    plot_cost(save_path, cost_values, optimal_index)
    plot_thickness(save_path, thickness_values, optimal_index)
# plt.figure(1)
# plt.plot(reward_values)
# plt.xlabel("Surrogate training step")
# plt.ylabel("Reward")
# plt.title("Reward")
# plt.grid(True)
# plt.savefig(results_directory+'/'+'reward.png')
# plt.close()

# plt.figure(2)
# plt.plot(thickness_values[:, 0], label = 't1')
# plt.plot(thickness_values[:, 1], label = 't2')
# plt.plot(thickness_values[:, 2], label = 't3')
# plt.xlabel("Surrogate training step")
# plt.ylabel("Thickness (cm)")
# plt.title("Evolution of Shield Thicknesses during PPO Training")
# plt.grid(True)
# plt.legend()
# plt.savefig(results_directory+'/'+"thickness.png")
# plt.close()

# plt.figure(3)
# plt.plot(cost_values)
# plt.xlabel("Surrogate training step")
# plt.ylabel("Cost")
# plt.title("Evolution of Cost during PPO Training")
# plt.grid(True)
# plt.legend()
# plt.savefig(results_directory+'/'+"cost.png")
# plt.close()

# plt.figure(4)
# plt.plot(dose_values, marker='o', label = 'Dose')
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