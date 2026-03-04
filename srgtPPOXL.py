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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import gym
from gym import spaces
import itertools
#############################'''SET INPUT PARAMS'''####################################
dc = '0.1'
mcrs = '4'
hard_constraint = '0'
normalization = '1'
scalingFactor = '1e3'
nps = '1e5'
#############################'''Variables'''####################################
npzFile = dc+'-'+mcrs+'-'+hard_constraint+'-'+normalization+'-'+scalingFactor+'-'+nps
bounds = np.array([0.01, 10.0])
dose_c = float(dc)
dose_list = []
cost_list = []
reward_list = []
layers = []
batches = None
tally_name = None
iteration = 1
###############################'''Definitions'''########################################
def openmc_fitness(point):
    global reward_list, dose_list, cost_list, layers, nps, batches, tally_name, iteration
    print(iteration)
    print(f"point: {point}")
    thicknesses = np.clip(point, bounds[0] * len(point), bounds[1] * len(point))
    print(f"thicknesses: {thicknesses}")
    model = build_geometry(thicknesses)
    # remove old results files in working directory
    for f in glob.glob('statepoint.*.h5'):
        if os.path.exists(f):
            os.remove(f)
    # run model
    model.run(output=False, geometry_debug=True)
    dose = read_dose()
    dose_list.append(dose)
    cost = cost_calc(thicknesses)
    cost_list.append(cost)
    penalty = max(0, (dose - dose_c)*int(float(scalingFactor))/dose_c) 
    reward = -cost - penalty
    reward_list.append(reward)
    print(f"dose: {dose}")
    print(f"cost: {cost}")
    print(f"reward: {reward}")
    return reward, dose, cost
def build_geometry(thicknesses):
    global reward_list, dose_list, cost_list, layers, nps, batches, tally_name, iteration
    '''
    Model
    '''
    openmc.reset_auto_ids()
    model = openmc.examples.slab_mg(num_regions=len(thicknesses)+1)  
    print(f"num_regions: {len(thicknesses)+1}")
    '''
    Materials
    '''
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
    core_material = openmc.Material.mix_materials([m1, m2, m6, m7], [0.567654, 0.144916, 0.040532, 0.246898], 'vo')
    # compile materials
    model.materials = openmc.Materials([m1,m2,m3,m4,m5, core_material])
    # specify cross sections
    model.materials.cross_sections = '/home/awhitesides3/openneomc/openmc/Cross_Section_Libraries/endfb-vii.1-hdf5/cross_sections.xml'  
    '''
    Geometry
    '''
    # cells
    num_cells = len(model.geometry.root_universe.cells)
    # grab slab geometry components
    all_cells = model.geometry.get_all_cells()
    cell_ids = list(all_cells.keys())
    cells = list(all_cells.values())
    all_surfaces = model.geometry.get_all_surfaces()
    surface_ids = list(all_surfaces.keys())
    surfaces = list(all_surfaces.values())
    shield_layer_mats = [core_material, m1, m2, m1, m2, m1, m2, m1, m2]
    model_ts = [0.0, 78.8]
    for thickness in thicknesses:
        model_ts.append(thickness)
    # full-model thicknesses = [0.0, 78.8, 9.525, 2.54, 5.715, 5.08, 11.938, 2.54, 7.62, 15.24]
    model_ts = np.cumsum(model_ts)
    print(f"model_ts: {model_ts}")
    # apply materials to slab geometry
    for i, cell in enumerate(model.geometry.root_universe.cells.values()):  
        cell.fill = shield_layer_mats[i]
        if (i > 0) & (iteration == 1):
            layers.append(shield_layer_mats[i].name)
    # set core cell to void
    all_cells[list(all_cells.keys())[0]].fill = None
    # change surface position
    for i, surface in enumerate(model.geometry.get_all_surfaces()):  
        all_surfaces[i+1].x0 = model_ts[i]
    # set outer core surface boundary type to vacuum in stead of reflective
    all_surfaces[2].boundary_type = 'vacuum'
    '''
    Source
    '''
    '''
    Tallies
    '''
    surface_filter = openmc.SurfaceFilter([surfaces[-1]])
    energy_filter = openmc.EnergyFilter([0.5, 2.0e7])
    particle_filter = openmc.ParticleFilter(['neutron'])
    fn_current = openmc.Tally(name='fast neutron current')
    tally_name = fn_current.name
    fn_current.filters = [surface_filter, energy_filter, particle_filter]
    fn_current.scores = ['current']
    model.tallies = openmc.Tallies([fn_current])  
    '''
    Settings
    '''
    model.settings.run_mode = 'fixed source'
    model.settings.energy_mode = 'continuous-energy'
    model.settings.photon_transport = False
    print(nps)
    model.settings.particles = int(float(nps))
    model.settings.batches = 10
    batches = model.settings.batches
    model.settings.inactive = 0
    model.settings.surf_source_read = {
        'path': '/home/awhitesides3/openMC/openmc/nre8803_scripts/slabexample/surface_source.h5'
    }
    model.settings.source = []
    '''
    Export
    '''
    model.export_to_xml()
    iteration += 1
    return model
def read_dose():
    dose = None
    with open("tallies.out", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Current"):
                parts = line.split()
                dose = float(parts[1])   
    return dose
def cost_calc(thicknesses):
    prices = np.array([])
    #prices of water and ss-316L taken from the dataset.jsons
    for i, material in enumerate(layers):
        if material == 'water':
           prices.append(0.0093)
        if material == 'ss-316L':
            prices.append(3.70)  
    return np.dot(thicknesses, prices)
def setUp(nl):
    points = np.array(list(itertools.product(bounds.tolist(), nl)))
    num_points = int(mcrs)
    num_rand_points = num_points - 4
    rng = np.random.default_rng(42)
    rand_points = points[0] + (points[0]-points[1]) * rng.random((num_rand_points, 2))
    training_points = np.vstack([points, rand_points])
    return training_points
###############################'''Application'''########################################
'''
PPO2 Implementation
'''
if __name__ == "__main__":
    num_layers = input("Enter the number of layers:")
    x_train = setUp(num_layers)
    reward_train = []
    dose_train = []
    cost_train = []
    for point in x_train:
        reward, dose, cost = openmc_fitness(point)
        reward_train.append(reward)
        dose_train.append(dose)
        cost_train.append(cost)
    reward_train = np.array(reward_train)
    dose_train = np.array(dose_train)
    cost_train = np.array(cost_train)
    print("Reached save step")
    np.savez('surrogate_results/'+str(num_layers)+'_layer_sensitivity_runs/data/TEST'+npzFile+'.npz', x_train=x_train, reward_train=reward_train, dose_train=dose_train, cost_train=cost_train)
