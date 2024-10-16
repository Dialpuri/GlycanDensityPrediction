import gemmi
from glycan_prediction_model.constants import DATASET_BASE_DIR
import logging
import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from confection import Config
import argparse
from scipy.spatial.transform import Rotation


def zscore(arr: np.ndarray):
    mean = np.mean(arr)
    stddev = np.std(arr)
    return (arr-mean)/stddev

def plot_volumes():
    ...

def create_validation_set(model, config):
    validation_sample_base = DATASET_BASE_DIR / "1ac5" 
    validation_mtz_path = validation_sample_base / "deglycosylated.mtz"
    validation_mask_path = validation_sample_base / "mask.map"


    deglycosylated_mtz = gemmi.read_mtz_file(str(validation_mtz_path))
    deglycosylated_grid = deglycosylated_mtz.transform_f_phi_to_map("FWT", "PHWT")

    mask_grid = gemmi.read_ccp4_map(str(validation_mask_path)).grid
    # mask_grid.normalize()

    

    # input_array, mask_array, minimum = _interpolate_input_grids(deglycosylated_grid, mask_grid, config)
    while True: 
        translation = gemmi.Fractional(*np.random.rand(3))
        rotation = gemmi.Mat33(Rotation.random().as_matrix()) 
        exp_array = _interpolate(deglycosylated_grid, translation, rotation, False, config)
        tar_array = _interpolate(mask_grid, translation, rotation, False, config)
        thresold = pow(config["dataset"]["box_size"], 3) * config["dataset"]["sample_coverage_threshold"]
        if np.sum(tar_array) > thresold:
            break

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'volume'}, {'type': 'volume'}]])
    X, Y, Z = np.mgrid[0:22.4:32j, 0:22.4:32j, 0:22.4:32j]
    
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=zscore(exp_array).squeeze().flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1, 
        surface_count=17, 
        ), row=1, col=1)
    
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=tar_array.squeeze().flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1, 
        surface_count=17, 
        ), row=1, col=2)

    fig.update_layout(height=600, width=1000, title_text="")
    fig.write_image("/users/jsd523/scratch/glycan_prediction_model/tests/1ac5/density.png")

    cell = gemmi.UnitCell(22.4, 22.4, 22.4, 90, 90, 90)
    spg = gemmi.SpaceGroup(1)

    m1 = gemmi.Ccp4Map()
    g1 = gemmi.FloatGrid(zscore(exp_array).squeeze(), cell, spg)
    m1.grid = g1
    m1.update_ccp4_header()
    m1.write_ccp4_map("tests/1ac5/input_array.map")

    # tar_array[tar_array > 0.5] = 1.0
    # tar_array[tar_array <= 0.5] = 0.0

    m1 = gemmi.Ccp4Map()
    g1 = gemmi.FloatGrid(tar_array.squeeze(), cell, spg)
    m1.grid = g1
    m1.update_ccp4_header()
    m1.write_ccp4_map("tests/1ac5/mask_array.map")

def _interpolate(
    grid: gemmi.FloatGrid,
    translation: gemmi.Fractional,
    rotation: gemmi.Mat33,
    squeeze: bool,
    config: dict,
) -> np.ndarray:
    spacing = config["dataset"]["grid_spacing"]
    size = config["dataset"]["box_size"]

    translation = grid.unit_cell.orthogonalize(translation)
    scale = gemmi.Mat33([[spacing, 0, 0], [0, spacing, 0], [0, 0, spacing]])
    transform = gemmi.Transform(scale.multiply(rotation), translation)
    values = np.zeros((size, size, size), dtype=np.float32)
    grid.interpolate_values(values, transform)
    if squeeze:
        return values
    return values[..., np.newaxis]


def _interpolate_input_grids(input_grid: gemmi.FloatGrid, mask_grid: gemmi.FloatGrid, config: dict):
    logging.info("Interpolating the input map with a cuboid of points around the ASU")
    
    _SIZE = config["dataset"]["box_size"]
    _HALF = _SIZE // 2
    _SPACING = config["dataset"]["grid_spacing"]

    extent = gemmi.find_asu_brick(input_grid.spacegroup).get_extent()
    box = input_grid.unit_cell.orthogonalize_box(extent)
    box.add_margin(_HALF * _SPACING)
    size = box.get_size()
    num_x = -(-int(size.x / _SPACING) // _HALF * _HALF)
    num_y = -(-int(size.y / _SPACING) // _HALF * _HALF)
    num_z = -(-int(size.z / _SPACING) // _HALF * _HALF)
    scale = gemmi.Mat33([[_SPACING, 0, 0], [0, _SPACING, 0], [0, 0, _SPACING]])
    transform = gemmi.Transform(scale, box.minimum)
    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    input_grid.interpolate_values(array, transform)

    mask_array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    mask_grid.interpolate_values(mask_array, transform)

    logging.info("Created a %dx%dx%d box", num_x, num_y, num_z)
    return array, mask_array, box.minimum


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Path to config")

    args = parser.parse_args()
      
    config = Config().from_disk(args.c)

    create_validation_set(None, config=config)