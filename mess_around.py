import numpy as np
def retrieve_run_data(data_path):
    data = np.load(data_path)
    return data["optimal_index"], data["dose_constraint"], data["reward_log"], data["dose_log"], data["cost_log"], data["thickness_log"]
data_path="/home/awhitesides3/openneomc/pporuns/RESULTS/Run03/Run03-ppo_data.npz"
optimal_index, dose_constraint, reward_values, dose_values, cost_values, thickness_values = retrieve_run_data(data_path)
# print(np.shape(thickness_values))
# print(thickness_values)
# print(np.shape(thickness_values[0]))
# print(thickness_values[0])
# print(len(thickness_values[0]))
# print(thickness_values.shape[0])
# for i, x in enumerate(thickness_values):
#     print(i)
#     print(x)
bounds = np.array([0.01, 10.0])
grid_dims = np.linspace(bounds[0], bounds[1], 50)
x_mesh, y_mesh = np.meshgrid(grid_dims, grid_dims)
x_arrF1 = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
print(f"grid_dims: {grid_dims}")
print(f"x_mesh: {x_mesh}")
print(f"y_mesh: {y_mesh}")
print(x_mesh.size)
print(x_mesh.shape)
print(y_mesh.shape)
print(x_mesh.ravel().shape)
print(np.full(x_mesh.size, 7))
print(type(np.full(x_mesh.size, 7)))
# print(f"array: {x_arrF1}")
# combinations for gpr plottings
from itertools import combinations

# # layers = np.array([0.01,10.0,4.0,6.0])
# layers=4

# for fixed_layers in combinations(range(0,layers), layers-2):
#     print(fixed_layers)

# for layer in range(0, layers):
#     print(layer)
print(optimal_index)
optimal_thicknesses = thickness_values[optimal_index]
print(optimal_thicknesses)
print(optimal_thicknesses.shape)
