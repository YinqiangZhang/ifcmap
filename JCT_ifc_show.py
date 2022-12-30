import os
import trimesh
import multiprocessing
from tqdm import tqdm
import ifcopenshell
import ifcopenshell.geom as ifc_geom
from ifcfunctions import meshfromshape, getunitfactor
from ifcopenshell.util.element import get_decomposition


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

error_list = []
meshlist=[]
meshcolor = [0,0,0,100]

iterator = ifc_geom.iterator(settings, ifc_file, multiprocessing.cpu_count())

for storey in storeys:
    print('Storey Label: {}'.format(storey.Name))
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
            error_list.append(ifc_entity.GlobalId)
            continue
        mesh = meshfromshape(shape, meshcolor)
        meshlist.append(mesh)

print('Error Global ID: {}'.format(error_list))

combined = trimesh.util.concatenate(meshlist)
combined.export(os.path.join('result_map', 'structure_small.ply'))
combined.show()