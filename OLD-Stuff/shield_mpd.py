import openmc
import numpy as np
from neorl import PPO2
import os
import glob

class openmcenv:
    def __init__(self, c_dose):
        # extract dose constraint
        self.c_dose = c_dose
        # set action space; action = [t1, t2, ..]
        self.num_actions = 2
        self.lower_bounds = [0.01, 0.01]
        self.upper_bounds = [10.0, 10.0]
        self.num_inputs = 2 #2 layers
    def reset(self):
        return np.zeros(self.num_inputs)
    def step(self, action):
        # constrain random thicknesses to bounds
        t1 = np.clip(action[0], 0.01, 10.0)
        t2 = np.clip(action[1], 0.01, 10.0)
        thicknesses = np.array([t1, t2])
        # build openmc model
        model = self.build_geometry(thicknesses)
        # remove old results files in working directory
        for f in glob.glob('statepoint.*.h5'):
            if os.path.exists(f):
                os.remove(f)
        # run model
        model.run(output=True, geometry_debug=True)
        sp_filename = 'statepoint.10.h5'
        dose = self.read_dose(sp_filename)
        cost = self.cost_calc(thicknesses)
        # reward
        penalty = max(0, dose - self.c_dose) #negative = 0
        reward = -cost - 1000*penalty**2
        done = True #single-step MDP
        obs = np.array([dose, cost])
        info = {'dose': dose, 'cost': cost}
        return obs, reward, done, info
    def build_geometry(self, thicknesses):
        '''
        Model
        '''
        model = openmc.examples.slab_mg(num_regions=self.num_inputs+1)  
        '''
        Materials
        '''
        # m1:water
        m1 = openmc.Material(material_id=1, name='water')
        m1.add_nuclide('H1', 2.0, 'ao')
        m1.add_nuclide('O16', 1.0, 'ao')
        # m1.add_s_alpha_beta('c_H_in_H2O') #unsure if this will actually make a difference
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
        print(all_surfaces)
        surface_ids = list(all_surfaces.keys())
        surfaces = list(all_surfaces.values())
        shield_layer_mats = [core_material, m1, m2, m1, m2, m1, m2, m1, m2]
        model_ts = [0.0, 78.8]
        for thickness in thicknesses:
            model_ts.append(thickness)
        # thicknesses = [0.0, 78.8, 9.525, 2.54, 5.715, 5.08, 11.938, 2.54, 7.62, 15.24]
        model_ts = np.cumsum(model_ts)
        # apply materials to slab geometry
        for i, cell in enumerate(model.geometry.root_universe.cells.values()):  
            cell.fill = shield_layer_mats[i]
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
        fn_current.filters = [surface_filter, energy_filter, particle_filter]
        fn_current.scores = ['current']
        model.tallies = openmc.Tallies([fn_current])  
        '''
        Settings
        '''
        model.settings.run_mode = 'fixed source'
        model.settings.energy_mode = 'continuous-energy'
        model.settings.photon_transport = False
        model.settings.particles = 10000
        model.settings.batches = 10
        model.settings.inactive = 0
        model.settings.surf_source_read = {
            'path': '/home/awhitesides3/openMC/openmc/nre8803_scripts/slabexample/surface_source.h5'
        }
        model.settings.source = []
        '''
        Export
        '''
        model.export_to_xml()
        return model
    def read_dose(self, sp_filename):
        sp = openmc.StatePoint(sp_filename)
        tally = sp.tallies[1]
        dose = tally.mean    
        return dose
    def cost_calc(self, thicknesses):
        # cost = sum(density*thickness*$/kg) <--- need to rework this formula
        prices = np.array([0.0093, 3.70]) #prices of water and ss-316L taken from the dataset.jsons 
        return np.dot(thicknesses, prices)
'''
PPO2 Implementation
'''
env = openmcenv(c_dose=0.025) #contraint = 0.025 current in units particle/(src*cm^2)
env_dict = {
    'state_dim': env.num_inputs,
    'action_dim': env.num_actions,
    'action_low': env.lower_bounds,
    'action_high': env.upper_bounds
}
mdp = MDP(env_dict, env)
agent = PPO2(mdp=mdp, seed=1)
agent.learn(total_timesteps=10)