"""NOTES: the agent always have to be accompanied by an environment. it needs an env to be loaded"""
###############################   IMPORTS   ########################################
import numpy as np
import gym
from neorl import PPO2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import gym
from gym import spaces
from argparse import ArgumentParser
import joblib
import os
import glob
import openmc
import itertools
from dataclasses import dataclass, asdict
###############################   Configuration   ########################################
@dataclass
class Config:
    # admin
    saveDir: str
    # openmc parameters
    bounds: tuple
    nL: int
    dose_limit: float
    nps: float
    theshold: float
    scalingFactor: int
    # surrogate parameters
    rps: int
    # agent parameters
    policy: str
    n_steps: int
    nminibatches: int
    seed: int 
    chuncks: int
    steps: int
default = Config(
    saveDir="/home/awhitesides3/openneomc/pporuns",
    bounds=(0.0, 10.0),
    nL=2,
    dose_limit=0.0936,
    nps=1e5,
    theshold=0.05,
    scalingFactor=1000,
    rps=1,
    policy='MlpPolicy',
    n_steps=32,
    nminibatches=4,
    seed=1,
    chuncks=100,
    steps=100
)
###############################   Agent Functions   ########################################
def initialize_Surrogate(saveDir, nL, rps, bounds, dose_limit):
    points = create_initial_training_points(nL, rps, bounds)
    results = [evaluate_openmc_model(point, bounds, dose_limit) for point in points]
    rewards, doses, costs = map(np.array, zip(*results))
    np.savez(f"{saveDir}_surrogateData.npz",
        surrogate_points=points,
        surrogate_rewards=rewards,
        surrogate_doses=doses,
        surrogate_costs=costs
    )
    return f"{saveDir}_surrogateData.npz"
def load_Surrogate(surgPath):
    data = np.load(surgPath)
    return data["surrogate_points"], data["surrogate_rewards"], data["surrogate_doses"], data["surrogate_costs"]
def update_Surrogate(surgPath, bounds, dose_limit, newThickness, reward=None, dose=None, cost=None):
    # retrieve
    points, rewards, doses, costs = load_Surrogate(surgPath)
    # if necessary - calculate output from new input
    if reward is None and dose is None and cost is None:
        newReward, newDose, newCost = evaluate_openmc_model(newThickness, bounds, dose_limit)
    else:
        newReward = reward
        newDose = dose
        newCost = cost
    # add
    points = np.vstack([points, newThickness]) #needs vstack & brackets because of type array, not list
    rewards = np.append(rewards, newReward)
    doses = np.append(doses, newDose)
    costs = np.append(costs, newCost)
    # save
    np.savez(surgPath,
        surrogate_points=points,
        surrogate_rewards=rewards,
        surrogate_doses=doses,
        surrogate_costs=costs
    )
    return points, rewards, doses, costs
def initialize_GPRs(saveDir, surgPath, nL):
    points, _, doses, costs = load_Surrogate(surgPath)
    kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * nL, (1e-3, 1e3))
    gpr_dose = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True
    )
    gpr_cost = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True
    )
    gpr_dose.fit(points, doses)
    gpr_cost.fit(points, costs)
    # save
    joblib.dump(gpr_dose, f"{saveDir}_gprDose.pkl")
    joblib.dump(gpr_cost, f"{saveDir}_gprCost.pkl")
    return f"{saveDir}_gprDose.pkl", f"{saveDir}_gprCost.pkl"
def load_GPR(gprPath):
    with open(gprPath, "rb") as f:
        gpr = joblib.load(f)
    return gpr
def update_GPR(gprPath, points, metric):
    gpr = load_GPR(gprPath)
    gpr.fit(points, metric)
    joblib.dump(gpr, gprPath)
    return gpr
def create_Env(gpr_dose, gpr_cost, dose_limit, nL, bounds):
    env = SurrogateEnv(
        dose_model=gpr_dose, cost_model=gpr_cost, dose_limit=dose_limit, nL=nL, lB=bounds[0], uB=bounds[1]
    )
    return env
def initialize_Agent(saveDir, env, policy, n_steps, nminibatches, seed):
    agent = PPO2(env=env, policy=policy, n_steps=n_steps, nminibatches=nminibatches, seed=seed)
    agent.save(f"{saveDir}_Agent")
    return f"{saveDir}_Agent"
def update_Agent_Environment(agentPath, env):
    agent = PPO2.load(load_path=agentPath, env=env)
    agent.save(agentPath)
    return agent
def train_Agent(agentPath, env, chuncks, steps):
    agent = PPO2.load(load_path=agentPath, env=env)
    thicknesses, rewards, doses, costs = [],[],[],[]
    for _ in range(chuncks):
            agent.learn(total_timesteps=steps, reset_num_timesteps=False)
            obs = env.reset()
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, _, info = env.step(action)
            thicknesses.append(action.copy())
            rewards.append(reward)
            doses.append(info["dose"])
            costs.append(info["cost"])
    agent_results = {
        "Thickness": thicknesses,
        "Reward": rewards,
        "Dose": doses,
        "Cost": costs
    }
    agent.save(agentPath)
    return agent, agent_results
###############################   Helper Functions   ########################################
def evaluate_openmc_model(point, bounds, dose_limit, nps=1e5, scalingFactor=1000):
    thicknesses = np.clip(point, bounds[0], bounds[1]) # clip thicknesses within bounds
    # build openmc model geometry
    model, layer_names = build_openmc_model(thicknesses, nps)
    # Delete old openmc run files & run model
    clean_dir()
    model.run(output=False, geometry_debug=True)
    # retrieve output params
    dose = retrieve_dose()
    cost = calculate_cost(thicknesses, layer_names)
    penalty = max(0, (dose - dose_limit)*int(scalingFactor)/dose_limit) 
    reward = -cost - penalty
    return reward, dose, cost
def create_initial_training_points(nl, rps, bounds):
    points = np.array(list(itertools.product(bounds.tolist(), repeat=int(nl))), dtype=float) #creates the bounded points depending on the number of layers
    rng = np.random.default_rng(42)
    if int(rps) == 0:
        training_points = points
    else:
        rand_points = points[0] + (points[-1]-points[0]) * rng.random((int(rps), int(nl)))
        training_points = np.vstack([points, rand_points])
    return training_points
def active_learning(agentPath, surgPath, gprDPath, gprCPath, chuncks, steps, threshold, bounds, dose_limit, nL):
    check = False
    while not check:
        _, results = train_Agent(agentPath, env, chuncks, steps)
        optimal_results = find_optimal_design(results)
        actual_dose = calculate_dose(optimal_results["Thickness"])
        accuracy = abs(optimal_results["Dose"] - actual_dose) / max(actual_dose, 1e-12)
        if accuracy <= threshold:
            check = True
            solution = optimal_results
        else:
            points, _, doses, costs = update_Surrogate(surgPath, bounds, dose_limit, 
                                                       optimal_results["Thickness"],
                                                       optimal_results["Reward"],
                                                       optimal_results["Dose"],
                                                       optimal_results["Cost"])
            gprDose = update_GPR(gprDPath, points, doses)
            gprCost = update_GPR(gprCPath, points, costs)
            env = create_Env(gprDose, gprCost, dose_limit, nL, bounds)
            update_Agent_Environment(agentPath, env)
    return solution
def override_default(default: Config):
    inputs = parse_arguments(default)
    for key, value in vars(inputs).items():
        if value is not None:
            setattr(default, key, tuple(value) if isinstance(getattr(default, key), tuple) else value)
    return default
###############################   Helper Helper Functions   ########################################
def find_optimal_design(agent_results):
    index = np.argmax(agent_results["Reward"])
    thicknesses = agent_results["Thickness"][index]
    reward = agent_results["Reward"][index]
    dose = agent_results["Dose"][index]
    cost = agent_results["Cost"][index]
    results = {
        "Thickness": thicknesses,
        "Reward": reward,
        "Dose": dose,
        "Cost": cost,
        "Index": [index]
    }
    return results
def build_openmc_model(thicknesses, nps=1e5):
    ################    MODEL    ################
    openmc.reset_auto_ids()
    model = openmc.examples.slab_mg(num_regions=len(thicknesses)+1)  
    ################    MATERIAL    ################
    # m1:water
    m1 = openmc.Material(material_id=1, name='water')
    m1.add_nuclide('H1', 2.0, 'ao')
    m1.add_nuclide('O16', 1.0, 'ao')
    m1.temperature = 600.0
    m1.set_density('g/cm3', 1.0)
    # m2:ss-316L
    m2 = openmc.Material(material_id=2, name='ss-316L')
    m2.add_element('C', 0.001384, 'ao')
    m2.add_nuclide('Mn55', 0.020165, 'ao')
    m2.add_nuclide('P31', 0.000805, 'ao')
    m2.add_element('S', 0.000518, 'ao')
    m2.add_element('Si', 0.019722, 'ao')
    m2.add_element('Cr', 0.181098, 'ao')
    m2.add_element('Ni', 0.113247, 'ao')
    m2.add_element('Mo', 0.014432, 'ao')
    m2.add_element('Fe', 0.648629, 'ao')
    m2.temperature = 600.0
    m2.set_density('g/cm3', 7.92) #ref uses 7.92, mcnp conpendium uses 8.00
    # m3:borated poly
    m3 = openmc.Material(material_id=3, name='borated polyethylene')
    m3.add_nuclide('H1', 0.627683, 'ao')
    m3.add_nuclide('H2', 0.000072, 'ao')
    m3.add_nuclide('B10', 0.009289, 'ao')
    m3.add_nuclide('B11', 0.037391, 'ao')
    m3.add_element('C', 0.325564, 'ao')
    m3.temperature = 600.0
    m3.set_density('g/cm3', 0.93) #ref uses 0.93, mcnp conpendium uses 1.00
    # m4:lead
    m4 = openmc.Material(material_id=4, name='lead')
    m4.add_element('Pb', 1.0, 'ao')
    m4.temperature = 600.0
    m4.set_density('g/cm3', 11.35)
    # m5:lead
    m5 = openmc.Material(material_id=5, name='air')
    m5.add_element('C', 0.000150, 'ao')
    m5.add_nuclide('N14', 0.781574, 'ao')
    m5.add_nuclide('N15', 0.002855, 'ao')
    m5.add_element('O', 0.21075, 'ao')
    m5.add_element('Ar', 0.004671, 'ao')
    m5.temperature = 600.0
    m5.set_density('g/cm3', 0.001205)
    # m6:boron carbide
    m6 = openmc.Material(material_id=6, name='B4C')
    m6.add_nuclide('B10', 0.1592, 'ao')
    m6.add_nuclide('B11', 0.6408, 'ao')
    m6.add_element('C', 0.2000, 'ao')
    m6.temperature = 600.0
    m6.set_density('g/cm3', 2.52)
    # m7:fuel
    m7 = openmc.Material(material_id=7, name='UO2-4.4wt%') #conversion from wt to at [https://www.wise-uranium.org/nfcjol.html]
    m7.add_element('O', 2.0, 'ao')
    m7.add_element('U', 1.0, enrichment=4.4)
    m7.temperature = 600.0
    m7.set_density('g/cm3', 10.96) #for simplicity, using density from compendium even though its only 3% enriched
    # create the core material using weight fractions from literature
    core_material = openmc.Material.mix_materials([m1, m2, m6, m7], [0.567654, 0.144916, 0.040532, 0.246898], 'vo')
    model.materials = openmc.Materials([m1,m2,m3,m4,m5, core_material])
    model.materials.cross_sections = '/home/awhitesides3/openneomc/openmc/Cross_Section_Libraries/endfb-vii.1-hdf5/cross_sections.xml'  
    ################    GEOMETRY    ################
    all_cells = model.geometry.get_all_cells()
    all_surfaces = model.geometry.get_all_surfaces()
    surfaces = list(all_surfaces.values())
    # create a list of the materials that correspond to the cells in the model 
    cell_materials = [core_material, m1, m2, m1, m2, m1, m2, m1, m2]
    # create the thicknesses for the model
    surface_positions = [0.0, 78.8] #
    for thickness in thicknesses:
        surface_positions.append(thickness)
    surface_positions = np.cumsum(surface_positions)
    # apply materials to slab geometry
    layer_names = []
    for i, cell in enumerate(model.geometry.root_universe.cells.values()):  
        cell.fill = cell_materials[i]
        if (i > 0):
            layer_names.append(cell_materials[i].name) #this variable is used later to match each layer with its repsective cost
    # set core cell to void - when using the surface source, the core can be set to void.
    all_cells[list(all_cells.keys())[0]].fill = None
    # change surface position
    for i, surface in enumerate(model.geometry.get_all_surfaces()):  
        all_surfaces[i+1].x0 = surface_positions[i]
    # set outer core surface boundary type to vacuum in stead of reflective
    all_surfaces[2].boundary_type = 'vacuum'
    ################    TALLIES    ################
    surface_filter = openmc.SurfaceFilter([surfaces[-1]])
    energy_filter = openmc.EnergyFilter([0.5, 2.0e7])
    particle_filter = openmc.ParticleFilter(['neutron'])
    fn_current = openmc.Tally(name='fast neutron current')
    fn_current.filters = [surface_filter, energy_filter, particle_filter]
    fn_current.scores = ['current']
    model.tallies = openmc.Tallies([fn_current])  
    ################    SETTINGS    ################
    model.settings.run_mode = 'fixed source'
    model.settings.energy_mode = 'continuous-energy'
    model.settings.photon_transport = False
    print(f"The NPS is {nps:.0e}")
    model.settings.particles = int(nps)
    model.settings.batches = 10
    model.settings.inactive = 0
    model.settings.surf_source_read = {
        'path': '/home/awhitesides3/openneomc/pporuns/surface_source.h5'
    }
    model.settings.source = []
    ################    EXPORT    ################
    model.export_to_xml()
    return model, layer_names
def calculate_dose(thicknesses):
    model, _ = build_openmc_model(thicknesses)
    clean_dir()
    model.run(output=False, geometry_debug=True)
    return retrieve_dose()
def retrieve_dose():
    dose = None
    # retrieve the tally score from the tally output file
    with open("tallies.out", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Current"):
                parts = line.split()
                dose = float(parts[1])   
    return dose
def calculate_cost(thicknesses, layer_names):
    prices = np.array([])
    for i, material in enumerate(layer_names):
        if material == "water":
            prices = np.append(prices, 0.0093)
        if material == "ss-316L":
            prices = np.append(prices, 3.70)  
    return np.dot(thicknesses, prices)
def parse_arguments(default: Config):
    parser = ArgumentParser()
    for attribute, value in asdict(default).items():
        dtype = type(value)
        if isinstance(value, tuple):
            parser.add_argument(f"--{attribute}", nargs=2, type=dtype)
        else:
            parser.add_argument(f"--{attribute}", type=dtype)
    return parser.parse_args()
def clean_dir():
    for f in glob.glob('statepoint.*.h5'):
        if os.path.exists(f):
            os.remove(f)
    return
###############################   Classes   ########################################
class SurrogateEnv(gym.Env):
    def __init__(self, gpr_dose, gpr_cost, dose_limit, nL, lB, uB):
        super().__init__()
        self.dose_model = gpr_dose
        self.cost_model = gpr_cost
        self.dose_limit = dose_limit
        self.low = np.array([lB]*nL)
        self.high = np.array([uB]*nL)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
    def reset(self):
        self.state = np.random.uniform(self.low, self.high)
        return self.state
    def step(self, action):
        action = np.clip(action, self.low, self.high)
        dose = self.dose_model.predict(action.reshape(1,-1))[0]
        cost = self.cost_model.predict(action.reshape(1,-1))[0]
        penalty = max(0.0, (dose - self.dose_limit) * 1000 / self.dose_limit)
        reward = -cost - penalty
        done=True
        next_state = self.reset()
        info = {"dose": dose, "cost": cost}
        return next_state, reward, done, info
###############################   Application   ########################################
# if __name__ == "__main__":
#     ################   SET INPUT PARAMS   ################
#     PARAMs = {
#         "number_layers": int,
#         "lower_bound": float,
#         "upper_bound": float,
#         "total_timesteps": float,
#         "iterations": float,
#         "dose_limit": float,
#         "nps": float,
#         "episode_length": int,
#         "mode": str,
#         "policy": str,
#         "check_freq": int,
#         "n_steps": int,
#         "nminibatches": int,
#         "seed": int,
#         "validation_threshold": float,
#         "save_path": str,
#         "surrogate_path": str
#     }
#     # params = vars(parse_arguments())
#     # lower_bound=params["lower_bound"]
#     # upper_bound=params["upper_bound"]
#     # number_layers=params["number_layers"]
#     # total_timesteps=int(params["total_timesteps"]) 
#     # iterations=int(params["iterations"])
#     # dose_limit=params["dose_limit"]
#     # nps=params["nps"]
#     # episode_length=params["episode_length"]
#     # mode=params["mode"]
#     # policy=params["policy"]
#     # check_freq=params["check_freq"]
#     # n_steps=params["n_steps"]
#     # nminibatches=params["nminibatches"]
#     # seed=params["seed"]
#     # validation_threshold=params["validation_threshold"]
#     # save_path=params["save_path"]
#     # surrogate_path=params["surrogate_path"]
#     ################  CREATE OTHER VARIABLES   ################
#     # bounds = np.array([lower_bound, upper_bound])
#     if surrogate_path == None:
#         surrogate_data = f"{save_path}-surrogate_data.npz"
#     else:
#         surrogate_data = surrogate_path
#     ################  FIND OPTIMAL DESIGN   ################
#     surrogate_points, surrogate_rewards, surrogate_doses, surrogate_costs = retrieve_surrogate_data(surrogate_data)
#     print("Retrieved surrogate data!")
#     gpr_dose, gpr_cost = train_GPRs(number_layers, surrogate_points, surrogate_doses, surrogate_costs)
#     print("Trained GPRs!")
#     optimal_design_dict, data_log, optimal_gpr_dose, optimal_gpr_cost, ppo_agent = active_learning_loop(PPO_Agent, PPO_Environment, gpr_dose, gpr_cost, dose_limit, number_of_layers, iterations, total_timesteps, surrogate_points, surrogate_doses, surrogate_costs, validation_threshold)
#     optimal_index = optimal_design_dict['optimal index'][0]
#     optimal_thicknesses = optimal_design_dict["optimal thicknesses"]
#     print("Found the optimal design!")
#     ################  SAVE RESULTS   ################
#     # save the dictionary which contains all result data
#     max_length = max(len(value) for value in optimal_design_dict.values())
#     print(f"max length: {max_length}")
#     for key,value in optimal_design_dict.items():
#         optimal_design_dict[key] = list(value) + ([np.nan] * (max_length-len(value)))
#     print(optimal_design_dict)
#     results_data_frame = pd.DataFrame(optimal_design_dict)
#     # save the data log as a .npz
#     np.savez(f"{save_path}-ppo_data.npz",
#              optimal_index=data_log["optimal index"],
#              dose_limit=data_log["dose constraint"],
#              reward_log=data_log["reward log"],
#              dose_log=data_log["dose log"],
#              cost_log=data_log["cost log"],
#              thickness_log=data_log["thickness log"])
#     # save the data and agents which will be accessed for data analysis. additionally, these can be used to build upon with more surrogate data.
#     results_data_frame.to_csv(f"{save_path}-ppo_data.csv", index=False)
#     joblib.dump(optimal_gpr_cost, f"{save_path}-gpr_cost_model.pkl")
#     joblib.dump(optimal_gpr_dose, f"{save_path}-gpr_dose_model.pkl")
#     ppo_agent.save(f"{save_path}-ppo_agent")
#     # save a .txt with info on the run
#     with open(f"{save_path}-ppo_info.txt", "a") as f:
#         f.write("=== New Run ===\n")
#         f.write(f"Timestamp: {datetime.now().isoformat()}\n")
#         f.write(f"Pulled surrogate data from this location:{surrogate_data}\n")
#         f.write(f"Results saved at this location: {save_path}-ppo_data.csv\n")
#         f.write(f"Cost GPR saved at this location: {save_path}-gpr_cost_model.pkl\n")
#         f.write(f"Dose GPR saved at this location: {save_path}-gpr_dose_model.pkl\n")
#         f.write(f"PPO Agent saved at this location: {save_path}-ppo_agent\n")
#         f.write(f"Best reward: {optimal_design_dict['reward values'][optimal_index]:.6f}\n")
#         f.write("Optimal thicknesses:" )
#         f.write(np.array2string(optimal_thicknesses, precision=4))
#         f.write("\n\n")
#     print("_____________________________________________________________________Successful Run!_____________________________________________________________________")