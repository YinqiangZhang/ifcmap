import os
import pickle
import trimesh
from tqdm import tqdm
import ifcopenshell
import ifcopenshell.geom as ifc_geom
from ifcfunctions import meshfromshape, getunitfactor
from ifcopenshell.util.element import get_decomposition
from matplotlib import cm


# read ifc file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
data_folder = 'real_models'
ifc_filename = 'structure_small.ifc'
ifc_file = ifcopenshell.open(os.path.join(data_folder, ifc_filename))

# parse ifc settings 
settings = ifc_geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

# get unit 
unitfactor = getunitfactor(ifc_file)
print('The Unit of IFC file: {}'.format(unitfactor)) # millimeter
storeys = ifc_file.by_type('IfcBuildingStorey')
colors = cm.get_cmap('viridis', len(storeys))(range(len(storeys)))

error_list = []
mesh_list=[]

for idx, storey in enumerate(storeys):
    print('Storey Label: {}'.format(storey.Name))
    storey_list = []
    elements = get_decomposition(storey)
    for ifc_entity in tqdm(elements, leave=False):
        if ifc_entity.is_a('IfcOpeningElement'):
            continue
        # check entities with no representation
        if ifc_entity.Representation is None:
            continue
        try:
            shape = ifc_geom.create_shape(settings, ifc_entity)
        except RuntimeError:
            # print(ifc_entity.Name)
            error_list.append(ifc_entity.GlobalId)
            continue
        mesh = meshfromshape(shape, colors[idx])
        storey_list.append(mesh)
        mesh_list.append(mesh)
    layer_combined = trimesh.util.concatenate(storey_list)
    layer_combined.export(os.path.join('result_map', 'layers', 'Storey_{}.ply'.format('_'.join(storey.Name.split('/')))))

print('Error Global ID: {}'.format(error_list))

combined = trimesh.util.concatenate(mesh_list)
combined.export(os.path.join('result_map', 'structure_small.ply'))

with open(os.path.join('result_map','error_ids.pkl'), 'wb') as f:
    pickle.dump(error_list, f)