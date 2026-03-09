import openmc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
################    MODEL    ################
model = openmc.examples.slab_mg(num_regions=1)  
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
thicknesses = [0.0, 78.8, 9.525, 2.54, 5.715, 5.08, 11.938, 2.54, 7.62, 15.24]
shield_layer_thick = np.cumsum(thicknesses)
# apply materials to slab geometry
for i, cell in enumerate(model.geometry.root_universe.cells.values()):  
    cell.fill = shield_layer_mats[i]
# change surface position
for i, surface in enumerate(model.geometry.get_all_surfaces()):  
    all_surfaces[i+1].x0 = shield_layer_thick[i]
################    SOURCE    ################
energy_distribution = openmc.stats.Watt() #default, included for robustness
angle_distribution = openmc.stats.Isotropic() #default, included for robustness
time_distribution = openmc.stats.Uniform() #default, included for robustness
space_distribution = openmc.stats.Box((0.0, -1000, -1000), (78.8, 1000, 1000)) #929.45 
source = openmc.IndependentSource(  
    energy=energy_distribution, 
    angle=angle_distribution,
    time=time_distribution,
    space=space_distribution
) 
################    TALLIES    ################
surface_filter = openmc.SurfaceFilter([surfaces[-1]])
energy_filter = openmc.EnergyFilter([0, 0.5, 2.0e7])
particle_filter = openmc.ParticleFilter(['neutron'])
n_current = openmc.Tally(name='surface neutron flux')
n_current.filters = [surface_filter, energy_filter, particle_filter]
n_current.scores = ['current']
model.tallies = openmc.Tallies([n_current])  
################    SETTINGS    ################
model.settings.run_mode = 'fixed source'
model.settings.energy_mode = 'continuous-energy'
model.settings.photon_transport = False
model.settings.source = source 
model.settings.particles = 100000
model.settings.batches = 10
model.settings.inactive = 0
model.settings.surf_source_write = {
    'surface_ids': [2],
    'max_particles': 10000
}
################    PLOT    ################
plot = openmc.Plot()
plot.filename = 'Slab xz'       
plot.basis = 'xz'               
plot.width = (500.0, 500.0)     
plot.pixels = (800, 800)        
plot.origin = (0.0, 0.0, 0.0)   
plot.color_by = 'cell'      
model.plots = openmc.Plots([plot])
model.plot_geometry()
# img = Image.open('Slab xz.png')
# plt.imshow(np.array(img))
# plt.axis('off')
# plt.show()
################    EXPORT    ################
model.export_to_xml()
################    RUN    ################
model.run(output=True, geometry_debug=True)