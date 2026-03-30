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
from itertools import combinations
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
###############################   DEFINITIONS   ########################################
def plot_reward(save_path, reward_values, optimal_index):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.scatter(range(1, len(reward_values)+1), reward_values)
    ax.axvline(x=optimal_index[0], color='green', linestyle='--', label="Optimal Design")
    ax.set_xlabel('PPO Training Step', fontweight="bold", fontsize=11)
    ax.set_ylabel('Reward [-]', fontweight="bold", fontsize=11)
    ax.grid(True)
    ax.legend()
    fig.suptitle('Reward vs. PPO Training Step', fontsize=14, y=0.92, fontweight="bold")
    fig.savefig(f"{save_path}-reward_plot.png", dpi=300, bbox_inches='tight')
    print("plotted the reward values!")
    return
def plot_dose(save_path, dose_values, optimal_index, dose_constraint):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.scatter(range(1, len(dose_values)+1), dose_values)
    ax.axvline(x=optimal_index[0], color='green', linestyle='--', label="Optimal Design")
    ax.axhline(y=dose_constraint[0], color='red', linestyle='--', label=f"Dose Limit: {dose_constraint[0]}")
    ax.set_xlabel('PPO Training Step', fontweight="bold", fontsize=11)
    ax.set_ylabel('Dose [-]', fontweight="bold", fontsize=11)
    ax.grid(True)
    ax.legend()
    fig.suptitle('Dose vs. PPO Training Step', fontsize=14, y=0.92, fontweight="bold")
    fig.savefig(f"{save_path}-dose_plot.png", dpi=300, bbox_inches='tight')
    print("plotted the dose values!")
    return
def plot_cost(save_path, cost_values, optimal_index):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.scatter(range(1, len(cost_values)+1), cost_values)
    ax.axvline(x=optimal_index[0], color='green', linestyle='--', label="Optimal Design")
    ax.set_xlabel('PPO Training Step', fontweight="bold", fontsize=11)
    ax.set_ylabel('Cost [-]', fontweight="bold", fontsize=11)
    ax.grid(True)
    ax.legend()
    fig.suptitle('Cost vs. PPO Training Step', fontsize=14, y=0.92, fontweight="bold")
    fig.savefig(f"{save_path}-cost_plot.png", dpi=300, bbox_inches='tight')
    print("plotted the cost values!")
    return
def plot_thickness(save_path, thickness_values, optimal_index):
    fig, ax = plt.subplots(figsize=(14,6))
    for layer, thicknesses in enumerate(thickness_values.T):
        ax.scatter(range(1, len(thicknesses)+1), thicknesses, label=f"layer #{layer+1}")
    ax.axvline(x=optimal_index[0], color='green', linestyle='--', label="Optimal Design")
    ax.set_xlabel('PPO Training Step', fontweight="bold", fontsize=11)
    ax.set_ylabel('Thickness [cm]', fontweight="bold", fontsize=11)
    ax.grid(True)
    ax.legend()
    fig.suptitle('Thickness vs. PPO Training Step', fontsize=14, y=0.92, fontweight="bold")
    fig.savefig(f"{save_path}-thickness_plot.png", dpi=300, bbox_inches='tight')
    print("plotted the thickness values!")
    return
def plot_gpr_dose(gpr_dose, thickness_values, optimal_index, dose_constraint, bounds):
    layers = thickness_values.shape[1]
    levels = 50
    for fixed_layers in combinations(range(0,layers), layers-2):
        grid_dims = np.linspace(bounds[0], bounds[1], levels)
        x_mesh, y_mesh = np.meshgrid(grid_dims, grid_dims)
        print(fixed_layers)
        optimal_thicknesses = thickness_values[optimal_index[0]]
        # predict mean and std using the gpr (requires grid points - random points between geometry bounds)
        master_list = []
        layers_plotted = []
        for layer in range(0, layers):
            if layer in fixed_layers:
                master_list.append(np.full(x_mesh.size, optimal_thicknesses[layer]))
            else:
                if len(layers_plotted) == 0:
                    mesh = x_mesh
                else:
                    mesh = y_mesh
                master_list.append(mesh.ravel())
                layers_plotted.append(layer)
        grid_points = np.column_stack(master_list)
        mean, std = gpr_dose.predict(grid_points, return_std=True)
        mean = mean.reshape(x_mesh.shape)
        std = std.reshape(x_mesh.shape)
        print(mean.min(), mean.max())
        # mean plot
        fig, ax = plt.subplots(figsize=(14,6))
        contour = ax.contour(x_mesh, y_mesh, mean, levels=levels, cmap='viridis')
        ax.contour(x_mesh, y_mesh, mean, levels=[dose_constraint], colors='red', linewidths=2, linestyles='--')
        ax.scatter(surrogate_points[:,layers_plotted[0]], surrogate_points[:,layers_plotted[1]], c="black", s=30, label="OpenMC Samples")
        ax.scatter(optimal_thicknesses[layers_plotted[0]], optimal_thicknesses[layers_plotted[1]], c="green", s=30, label="Gp Optimal Solution")
        legend_handles = [
            Line2D([0],[0], color='red', linestyle='--', linewidth=2, label=f"Dose Limit: {dose_constraint[0]}"),
            Line2D([0],[0], marker='o', color="black", linewidth=2, label="Surrogate Point"),
            Line2D([0],[0], marker='o', color="green", linewidth=2, label="Optimal Point"),
        ]
        ax.legend(handles=legend_handles)
        ax.set_xlabel(f"Layer {layers_plotted[0]+1} thickness")
        ax.set_ylabel(f"Layer {layers_plotted[1]+1} thickness")
        ax.annotate(
        "GP Optimal Solution", xy=(optimal_thicknesses[layers_plotted[0]], optimal_thicknesses[layers_plotted[1]]),
        xytext=(15,15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-"),
        bbox=dict(boxstyle="round", fc="white", ec="green", linewidth=2),
        fontsize=8, zorder=99
        )   
        fig.colorbar(contour, ax=ax, label="Predicted Dose")
        fig.suptitle("GP Predicted Mean Dose")
        fig.savefig(f"{save_path}-gpr_dose_mean_plot-L{layers_plotted[0]+1}-L{layers_plotted[1]+1}-.png", dpi=300, bbox_inches='tight')
        # uncertainty plot
        fig2, ax2 = plt.subplots(figsize=(14,6))
        contour2 = ax2.contour(x_mesh, y_mesh, std, levels=levels, cmap='viridis')
        ax2.scatter(surrogate_points[:,layers_plotted[0]], surrogate_points[:,layers_plotted[1]], c="black", s=30, label="OpenMC Samples")
        ax2.scatter(optimal_thicknesses[layers_plotted[0]], optimal_thicknesses[layers_plotted[1]], c="green", s=30, label="GP Optimal Solution")
        legend_handles2 = [
            Line2D([0],[0], marker='o', color="black", linewidth=2, label="Surrogate Point"),
            Line2D([0],[0], marker='o', color="green", linewidth=2, label="Optimal Point"),
        ]
        ax2.legend(handles=legend_handles2)
        ax2.set_xlabel(f"Layer {layers_plotted[0]+1} thickness")
        ax2.set_ylabel(f"Layer {layers_plotted[1]+1} thickness")
        ax2.annotate(
        "GP Optimal Solution", xy=(optimal_thicknesses[layers_plotted[0]], optimal_thicknesses[layers_plotted[1]]),
        xytext=(15,15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-"),
        bbox=dict(boxstyle="round", fc="white", ec="green", linewidth=2),
        fontsize=8, zorder=99
        )    
        fig2.colorbar(contour2, ax=ax2, label="Predicted Dose Uncertainty")
        fig2.suptitle("GP Predicted Dose Uncertainty")
        fig2.savefig(f"{save_path}-gpr_dose_uncertainty_plot-L{layers_plotted[0]+1}-L{layers_plotted[1]+1}-.png", dpi=300, bbox_inches='tight')
    print("plotted the gpr dose estimates!")
    return
def plot_gpr_cost(gpr_cost, thickness_values, optimal_index, bounds):
    layers = thickness_values.shape[1]
    levels = 50
    for fixed_layers in combinations(range(0,layers), layers-2):
        grid_dims = np.linspace(bounds[0], bounds[1], levels)
        x_mesh, y_mesh = np.meshgrid(grid_dims, grid_dims)
        print(fixed_layers)
        optimal_thicknesses = thickness_values[optimal_index[0]]
        # predict mean and std using the gpr (requires grid points - random points between geometry bounds)
        master_list = []
        layers_plotted = []
        for layer in range(0, layers):
            if layer in fixed_layers:
                master_list.append(np.full(x_mesh.size, optimal_thicknesses[layer]))
            else:
                if len(layers_plotted) == 0:
                    mesh = x_mesh
                else:
                    mesh = y_mesh
                master_list.append(mesh.ravel())
                layers_plotted.append(layer)
        grid_points = np.column_stack(master_list)
        mean, std = gpr_cost.predict(grid_points, return_std=True)
        mean = mean.reshape(x_mesh.shape)
        std = std.reshape(x_mesh.shape)
        print(mean.min(), mean.max())
        # mean plot
        fig, ax = plt.subplots(figsize=(14,6))
        contour = ax.contour(x_mesh, y_mesh, mean, levels=levels, cmap='viridis')
        ax.scatter(surrogate_points[:,layers_plotted[0]], surrogate_points[:,layers_plotted[1]], c="black", s=30, label="OpenMC Samples")
        ax.scatter(optimal_thicknesses[layers_plotted[0]], optimal_thicknesses[layers_plotted[1]], c="green", s=30, label="Gp Optimal Solution")
        legend_handles = [
            Line2D([0],[0], marker='o', color="black", linewidth=2, label="Surrogate Point"),
            Line2D([0],[0], marker='o', color="green", linewidth=2, label="Optimal Point"),
        ]
        ax.legend(handles=legend_handles)
        ax.set_xlabel(f"Layer {layers_plotted[0]+1} thickness")
        ax.set_ylabel(f"Layer {layers_plotted[1]+1} thickness")
        ax.annotate(
        "GP Optimal Solution", xy=(optimal_thicknesses[layers_plotted[0]], optimal_thicknesses[layers_plotted[1]]),
        xytext=(15,15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-"),
        bbox=dict(boxstyle="round", fc="white", ec="green", linewidth=2),
        fontsize=8, zorder=99
        )  
        fig.colorbar(contour, ax=ax, label="Predicted Cost")
        fig.suptitle("GP Predicted Mean Cost")
        fig.savefig(f"{save_path}-gpr_cost_mean_plot-L{layers_plotted[0]+1}-L{layers_plotted[1]+1}-.png", dpi=300, bbox_inches='tight')
        # uncertainty plot
        fig2, ax2 = plt.subplots(figsize=(14,6))
        contour2 = ax2.contour(x_mesh, y_mesh, std, levels=levels, cmap='viridis')
        ax2.scatter(surrogate_points[:,layers_plotted[0]], surrogate_points[:,layers_plotted[1]], c="black", s=30, label="OpenMC Samples")
        ax2.scatter(optimal_thicknesses[layers_plotted[0]], optimal_thicknesses[layers_plotted[1]], c="green", s=30, label="Gp Optimal Solution")
        legend_handles2 = [
            Line2D([0],[0], marker='o', color="black", linewidth=2, label="Surrogate Point"),
            Line2D([0],[0], marker='o', color="green", linewidth=2, label="Optimal Point"),
        ]
        ax2.legend(handles=legend_handles2)
        ax2.set_xlabel(f"Layer {layers_plotted[0]+1} thickness")
        ax2.set_ylabel(f"Layer {layers_plotted[1]+1} thickness")  
        ax2.annotate(
        "GP Optimal Solution", xy=(optimal_thicknesses[layers_plotted[0]], optimal_thicknesses[layers_plotted[1]]),
        xytext=(15,15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-"),
        bbox=dict(boxstyle="round", fc="white", ec="green", linewidth=2),
        fontsize=8, zorder=99
        )    
        fig2.colorbar(contour2, ax=ax2, label="Predicted Cost Uncertainty")
        fig2.suptitle("GP Predicted Cost Uncertainty")
        fig2.savefig(f"{save_path}-gpr_cost_uncertainty_plot-L{layers_plotted[0]+1}-L{layers_plotted[1]+1}-.png", dpi=300, bbox_inches='tight')
    print("plotted the gpr cost estimates!")    
    return
def retrieve_surrogate_data(surrogate_data):
    data = np.load(surrogate_data)
    return data["surrogate_points"], data["surrogate_rewards"], data["surrogate_doses"], data["surrogate_costs"]
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
        "surrogate_path": str,
        "gpr_dose_path": str,
        "gpr_cost_path": str,
        "lower_bound": float,
        "upper_bound": float
    }
    params = vars(parse_arguments())
    save_path=params["save_path"]
    data_path=params["data_path"]
    surrogate_path=params["surrogate_path"]
    gpr_dose_path=params["gpr_dose_path"]
    gpr_cost_path=params["gpr_cost_path"]
    lower_bound=params["lower_bound"]
    upper_bound=params["upper_bound"]
    ################  CREATE OTHER VARIABLES   ################
    bounds = np.array([lower_bound, upper_bound])
    # retrieve data from respective run
    surrogate_points, surrogate_rewards, surrogate_doses, surrogate_costs = retrieve_surrogate_data(surrogate_path)
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
    plot_gpr_dose(gpr_dose, thickness_values, optimal_index, dose_constraint, bounds)
    plot_gpr_cost(gpr_cost, thickness_values, optimal_index, bounds)
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
# # # overlay OpenMC runs
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