import os 
import glob
import ifcopenshell
import ifcopenshell.geom as geom
import numpy as np
from ifcopenshell.util.placement import get_storey_elevation
from ifcopenshell.util.element import get_decomposition
import pandas as pd

color_dict = {
    'IfcColumn': [106, 90, 205],
    'IfcSite': 	[65, 105, 225],
    'IfcRamp': [60, 179, 113], 
    'IfcFlowTerminal': [255, 215, 0],
    'IfcGrid': [205, 92, 92],
    'IfcWallStandardCase': [205, 133, 63],
    'IfcWall': [176, 48, 96],
    'IfcDoor': [255, 218, 185],
    'IfcSlab': [205, 205, 193],
    'IfcStair': [187, 255, 255],
    'IfcCovering': [192, 255, 62],
    'IfcBuildingElementProxy': [255, 246, 143],
    'IfcFurnishingElement': [205, 190, 112],
    'IfcWindow': [139, 101, 8],
    'IfcBeam': [255, 130, 71],
    'IfcRailing': [255, 140, 105],
    'IfcStairFlight': [238, 106, 80], 
    'IfcMember': [255, 181, 197], 
    'IfcPlate': [255, 225, 255], 
    'IfcFastener': [144, 238, 144], 
    'IfcCurtainWall': [224, 238, 224],
    'IfcRampFlight': [205, 183, 181], 
}

settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

root_path = os.path.dirname(os.path.abspath(__file__))    
ifc_filepath = glob.glob(os.path.join(root_path, 'real_models', '*.ifc'))[0]
ifc_model = ifcopenshell.open(ifc_filepath)

storeys = ifc_model.by_type('IfcBuildingStorey')

ifc_type_dict = dict()
for storey_idx, storey in enumerate(storeys, 0):
    elev = get_storey_elevation(storey)
    elements = get_decomposition(storey)
    print('Storey {}, Elevation: {:.4}'.format(storey.Name, elev/1000))
    types = list(elem.get_info()['type'] for elem in elements)
    for elem in elements:
        elem_ifc_type = elem.get_info()['type']
        elem_obj_type = elem.ObjectType if elem.ObjectType is not None else " "
        if ifc_type_dict.get(elem_ifc_type, None) is None:
            ifc_type_dict[elem_ifc_type] = [elem_obj_type]
        else:
            ifc_type_dict[elem_ifc_type].append(elem_obj_type)
    

color_dict = dict()
ifc_type_list = list(ifc_type_dict.keys())
color_dict['IfcType'] = ifc_type_list
color_dict['R'] = np.zeros((len(ifc_type_list),))
color_dict['G'] = np.zeros((len(ifc_type_list),))
color_dict['B'] = np.zeros((len(ifc_type_list),))
