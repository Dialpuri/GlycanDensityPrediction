#!/usr/bin/env python3

"Predict a high-resolution calculated map from a low-resolution map"

import argparse
import json
import logging
from types import SimpleNamespace
import gemmi
import numpy as np
# from scipy.stats import zscore
import tensorflow as tf
from confection import Config
from pathlib import Path
from glycan_prediction_model.training.loss import sigmoid_focal_crossentropy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report
from tqdm import tqdm 
from time import perf_counter
from datetime import datetime


def _main(args):
    t1 = perf_counter()
    output = Path(args.o)
    output.mkdir(exist_ok=True, parents=True)
    
    input = Path(args.i)
    if '.map' in input.suffixes:
        input_grid = gemmi.read_ccp4_map(args.i).grid
        difference_grid = gemmi.read_ccp4_map(args.d).grid
        
        input_grid.normalize()
        difference_grid.normalize()
        
    if '.mtz' in input.suffixes:
        input_mtz = gemmi.read_mtz_file(args.i)
        input_grid = input_mtz.transform_f_phi_to_map("FWT", "PHWT")
        difference_grid = input_mtz.transform_f_phi_to_map("DELFWT", "PHDELWT")

    _write_output(input_grid, str(output / f"input_best.map"))
    _write_output(difference_grid, str(output / f"input_difference.map"))

    mask_grid = None

    if args.mask:
        input = Path(args.mask)
        if '.map' in input.suffixes:
            mask_grid = gemmi.read_ccp4_map(args.mask).grid

        if '.mtz' in input.suffixes:
            input_mtz = gemmi.read_mtz_file(args.mask)
            mask_grid = input_mtz.transform_f_phi_to_map("FWT", "PHWT")


    input_array, difference_array, mask_array, minimum = _interpolate_input_grids(input_grid, difference_grid, mask_grid)
    output_array = _predict_output_array(input_array, difference_array, args.m, mask_array)

    output_grid = _interpolate_output_array(output_array, input_grid, minimum)
    _write_output(output_grid, str(output / f"output.map"))
    
    output_array = np.round(output_array)
    mask_array = np.round(mask_array)

    t2 = perf_counter()

    report = classification_report(mask_array.flatten(), output_array.flatten(), output_dict=True)
    
    data = {}
    data["runtime"] = t2-t1
    data["classification"] = report
    data["date"] = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    with open(output / "report.json", "w") as of:
        json.dump(data, of, indent=4)
        
    


def _interpolate_input_grids(input_grid: gemmi.FloatGrid, diff_grid: gemmi.FloatGrid, mask_grid: gemmi.FloatGrid):
    logging.info("Interpolating the input map with a cuboid of points around the ASU")
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

    diff_array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    diff_grid.interpolate_values(diff_array, transform)

    mask_array = None
    if mask_grid:
        mask_array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
        mask_grid.interpolate_values(mask_array, transform)

    logging.info("Created a %dx%dx%d box", num_x, num_y, num_z)
    return array, diff_array, mask_array, box.minimum


def _predict_output_array(
    input_array: np.ndarray, difference_array: np.ndarray,  model: str, mask_array: np.ndarray) -> np.ndarray:
    logging.info("Using the U-Net to predict overlapping samples")
    custom_objects={"sigmoid_focal_crossentropy": sigmoid_focal_crossentropy}

    model = tf.keras.models.load_model(model, custom_objects=custom_objects)
    total_array = np.zeros(input_array.shape, dtype=np.float32)
    count_array = np.zeros(input_array.shape, dtype=np.float32)
    # return count_array

    for i in tqdm(range(0, input_array.shape[0] - _HALF, _HALF)):
        for j in range(0, input_array.shape[1] - _HALF, _HALF):
            for k in range(0, input_array.shape[2] - _HALF, _HALF):
                input_sub = input_array[i : i + _SIZE, j : j + _SIZE, k : k + _SIZE]
                input_sub = input_sub[np.newaxis, ..., np.newaxis]
                
                difference_sub = difference_array[i : i + _SIZE, j : j + _SIZE, k : k + _SIZE]
                difference_sub = difference_sub[np.newaxis, ..., np.newaxis]
                
                combined_sub = tf.concat([input_sub, difference_sub], axis=-1)
                output_sub = model(combined_sub).numpy().squeeze()
                output_sub = np.argmax(output_sub, axis=-1)

                total_array[i : i + _SIZE, j : j + _SIZE, k : k + _SIZE] += output_sub
                count_array[i : i + _SIZE, j : j + _SIZE, k : k + _SIZE] += 1
    return total_array / count_array


def _interpolate_output_array(
    output_array: np.ndarray, input_grid: gemmi.FloatGrid, minimum: gemmi.Position
):
    logging.info("Interpolating the predicted values in the cuboid with the output ASU")
    output_grid = gemmi.FloatGrid()
    output_grid.spacegroup = input_grid.spacegroup
    output_grid.set_unit_cell(input_grid.unit_cell)
    output_grid.set_size_from_spacing(_SPACING, gemmi.GridSizeRounding.Nearest)
    size_x = output_array.shape[0] * _SPACING
    size_y = output_array.shape[1] * _SPACING
    size_z = output_array.shape[2] * _SPACING
    array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)
    array_grid = gemmi.FloatGrid(output_array, array_cell)
    for point in output_grid.masked_asu():
        position = output_grid.point_to_position(point) - minimum
        point.value = array_grid.interpolate_value(position)
    output_grid.symmetrize_max()
    return output_grid


def _write_output(output_grid: gemmi.FloatGrid, path: str):
    logging.info("Writing the output map to %s", path)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = output_grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Path to input best map or MTZ")
    parser.add_argument("-d", help="Path to input difference")
    parser.add_argument("-o", help="Path to output map")
    parser.add_argument("-c", help="Path to config file", required=True)
    parser.add_argument("-m", help="Path to model", required=True)
    parser.add_argument("--mask", help="Path to mask")

    args = parser.parse_args()
      
    config = Config().from_disk(args.c)
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    _SIZE = config["dataset"]["box_size"]
    _HALF = _SIZE // 2
    _SPACING = config["dataset"]["grid_spacing"]

    _main(args)
