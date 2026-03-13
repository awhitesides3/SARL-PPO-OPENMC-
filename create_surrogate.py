###############################   IMPORTS   ########################################
import openmc
import numpy as np
import os
import glob
import itertools
from pathlib import Path
from argparse import ArgumentParser
###############################   DEFINITIONS   ########################################
def openmc_fitness(point, layers, nps, iteration):
    print(f"Monte Carlo Run #{iteration}")
    # clip the layer thicknesses to ensure they are within the bounds
    thicknesses = np.clip(point, bounds[0], bounds[1])
    print(f"Layer Thicknesses: {thicknesses}")
    # build the geometry for the model
    model, layers = build_geometry(thicknesses, layers, nps, iteration)
    # remove prior MC run files
    for f in glob.glob('statepoint.*.h5'):
        if os.path.exists(f):
            os.remove(f)
    # run model
    model.run(output=False, geometry_debug=True)
    # form key outputs (dose, cost, reward)
    dose = read_dose()
    cost = cost_calc(thicknesses, layers)
    penalty = max(0, (dose - dose_constraint)*int(scalingFactor)/dose_constraint) 
    reward = -cost - penalty
    print(f"dose: {dose}")
    print(f"cost: {cost}")
    print(f"reward: {reward}")
    return reward, dose, cost
def build_geometry(thicknesses, layers, nps, iteration):
    ################    MODEL    ################
    # reset the ids for cells from the previous MC run
    openmc.reset_auto_ids()
    # create the 1D multi-group slab model using OpenMCs builtin function 'slab_mg()'
    # you must generate (#layers + 1) regions to form the model 
    model = openmc.examples.slab_mg(num_regions=len(thicknesses)+1)  
    print(f"The number of regions in the Monte Carlo Model are {len(thicknesses)+1}")
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
    # compile materials and assign them to the model
    model.materials = openmc.Materials([m1,m2,m3,m4,m5, core_material])
    # assign cross section library to the model
    model.materials.cross_sections = '/home/awhitesides3/openneomc/openmc/Cross_Section_Libraries/endfb-vii.1-hdf5/cross_sections.xml'  
    ################    GEOMETRY    ################
    # grab slab geometry components
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
    print(f"Vector of the surface positions: {surface_positions}")
    # apply materials to slab geometry
    for i, cell in enumerate(model.geometry.root_universe.cells.values()):  
        cell.fill = cell_materials[i]
        if (i > 0):
            print(f"iteration = {iteration}, i = {i}, layers are being added")
            layers.append(cell_materials[i].name) #this variable is used later to match each layer with its repsective cost
        else:
            print(f"iteration = {iteration}, i = {i} - no layers are added")
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
    return model, layers
def read_dose():
    dose = None
    # grad the tally score from the tally output file
    with open("tallies.out", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Current"):
                parts = line.split()
                dose = float(parts[1])   
    return dose
def dose_calc(thicknesses, layers, nps, iteration):
    model, _ = build_geometry(thicknesses, layers, nps, iteration)
    for f in glob.glob('statepoint.*.h5'):
            if os.path.exists(f):
                os.remove(f)
    model.run(output=False, geometry_debug=True)
    dose = read_dose()
    print(f"The actual dose for the proposed optimal design is: {dose}")
    return dose
def cost_calc(thicknesses, layers):
    print(f"layers:{layers}")
    prices = np.array([])
    #prices of water and ss-316L taken from the dataset.jsons
    for i, material in enumerate(layers):
        if material == "water":
            prices = np.append(prices, 0.0093)
        if material == "ss-316L":
            prices = np.append(prices, 3.70)  
    print(f"thicknesses:{thicknesses}")
    print(f"prices:{prices}")
    return np.dot(thicknesses, prices)
def setUp(nl, rps):
    points = np.array(list(itertools.product(bounds.tolist(), repeat=int(nl))), dtype=float)
    print(f"points:{points}")
    rng = np.random.default_rng(42)
    if int(rps) == 0:
        training_points = points
    else:
        rand_points = points[0] + (points[-1]-points[0]) * rng.random((int(rps), int(nl)))
        print(f"rand points:{rand_points}")
        training_points = np.vstack([points, rand_points])
    print(f"training points: {training_points}")
    return training_points
def parse_arguments():
    parser = ArgumentParser()
    for name, dtype in PARAMs.items():
        parser.add_argument(f"--{name}", type=dtype)
    return parser.parse_args()
###############################   Application   ########################################
if __name__ == "__main__":
    ################   SET INPUT PARAMS   ################
    PARAMs = {
        "dose_constraint": float,
        "hard_constraint": str,
        "normalization": str,
        "scalingFactor": float,
        "nps": float,
        "lower_bound": float,
        "upper_bound": float,
        "number_layers": int,
        "number_random_points": int,
        "save_path": str
    }
    params = vars(parse_arguments())
    dose_constraint=params["dose_constraint"]
    hard_constraint=params["hard_constraint"]
    normalization=params["normalization"]
    scalingFactor=params["scalingFactor"]
    nps=params["nps"]
    lower_bound=params["lower_bound"]
    upper_bound=params["upper_bound"]
    number_layers=params["number_layers"]
    number_random_points=params["number_random_points"]
    save_path=params["save_path"]
    # Create the bounds
    bounds = np.array([lower_bound, upper_bound])
    # create the surrogate data file
    surrogate_data = f"{save_path}-surrogate_data.npz"
    ################  CREATE OTHER VARIABLES   ################
    layers = []
    iteration = 1
    surrogate_rewards = []
    surrogate_doses = []
    surrogate_costs = []
    ################  CREATE SURROGATE   ################
    surrogate_points = setUp(number_layers, number_random_points)
    for point in surrogate_points:
        reward, dose, cost = openmc_fitness(point, layers, nps, iteration)
        surrogate_rewards.append(reward)
        surrogate_doses.append(dose)
        surrogate_costs.append(cost)
        iteration += 1 
        layers = []
    surrogate_rewards = np.array(surrogate_rewards)
    surrogate_doses = np.array(surrogate_doses)
    surrogate_costs = np.array(surrogate_costs)
    ################  SAVE RESULTS   ################
    print("_____________________________________________________________________Reached save step!_____________________________________________________________________")
    np.savez(
        surrogate_data,
        surrogate_points=surrogate_points,
        surrogate_rewards=surrogate_rewards,
        surrogate_doses=surrogate_doses,
        surrogate_costs=surrogate_costs
        )