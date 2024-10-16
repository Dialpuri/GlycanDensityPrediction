import os
from typing import Tuple
import urllib.error
from tqdm import tqdm
import gemmi
import wget
import requests
import pandas as pd
import shutil
import numpy as np


def get_bounding_box(grid: gemmi.FloatGrid) -> gemmi.PositionBox:
    extent = gemmi.find_asu_brick(grid.spacegroup).get_extent()
    corners = [
        grid.unit_cell.orthogonalize(fractional)
        for fractional in (
            extent.minimum,
            gemmi.Fractional(extent.maximum[0], extent.minimum[1], extent.minimum[2]),
            gemmi.Fractional(extent.minimum[0], extent.maximum[1], extent.minimum[2]),
            gemmi.Fractional(extent.minimum[0], extent.minimum[1], extent.maximum[2]),
            gemmi.Fractional(extent.maximum[0], extent.maximum[1], extent.minimum[2]),
            gemmi.Fractional(extent.maximum[0], extent.minimum[1], extent.maximum[2]),
            gemmi.Fractional(extent.minimum[0], extent.maximum[1], extent.maximum[2]),
            extent.maximum,
        )
    ]
    min_x = min(corner[0] for corner in corners)
    min_y = min(corner[1] for corner in corners)
    min_z = min(corner[2] for corner in corners)
    max_x = max(corner[0] for corner in corners)
    max_y = max(corner[1] for corner in corners)
    max_z = max(corner[2] for corner in corners)
    box = gemmi.PositionBox()
    box.minimum = gemmi.Position(min_x, min_y, min_z)
    box.maximum = gemmi.Position(max_x, max_y, max_z)
    return box


def load_and_interpolate_map(
    file_path: str, grid_spacing=0.7
) -> Tuple[gemmi.FloatGrid, gemmi.Transform]:
    map: gemmi.Ccp4Map = gemmi.read_ccp4_map(file_path)
    grid = map.grid
    grid.normalize()

    box: gemmi.PositionBox = get_bounding_box(grid)
    size: gemmi.Position = box.get_size()

    num_x = int(size.x / grid_spacing)
    num_y = int(size.y / grid_spacing)
    num_z = int(size.z / grid_spacing)

    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )

    transform: gemmi.Transform = gemmi.Transform(scale, box.minimum)
    grid.interpolate_values(array, transform)
    cell: gemmi.UnitCell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
    interpolated_grid: gemmi.FloatGrid = gemmi.FloatGrid(array, cell)

    return interpolated_grid, transform


def convert_map_to_mtz(mtz_file_path: str, pdb_code: str, output_dir: str):
    try:
        mtz = gemmi.read_mtz_file(mtz_file_path)
    except (RuntimeError, ValueError) as e:
        print(f"{pdb_code} raised {e}")
        return

    float_grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    output_path = os.path.join(output_dir, f"{pdb_code}.map")
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = float_grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_path)


def pdb_list_to_map():
    df = pd.read_csv("./data/low_res_data/pdb_list.txt")
    pdb_list = df.columns
    pdb_list = [x.lower() for x in pdb_list]

    mtz_base_dir = "https://edmaps.rcsb.org/coefficients/PDBCODE.mtz"
    pdb_base_dir = "https://files.rcsb.org/download/PDBCODE.pdb"

    output_dir = "data/low_res_data/"

    for pdb_code in tqdm(pdb_list):
        mtz_url = mtz_base_dir.replace("PDBCODE", pdb_code)
        mtz_dir = os.path.join(output_dir, "mtz")
        map_dir = os.path.join(output_dir, "map")

        if os.path.isfile(os.path.join(map_dir, f"{pdb_code}.map")):
            continue

        try:
            # with open(mtz_dir, "wb") as out_file:
            #     content = requests.get(mtz_url, stream=True).content
            #     out_file.write(content)
            file_name = wget.download(mtz_url, out=mtz_dir)
        except urllib.error.HTTPError as e:
            print(f"{pdb_code} skipped")
            continue
        except ValueError as e:
            continue

        print(pdb_code)
        convert_map_to_mtz(file_name, pdb_code, map_dir)

        # pdb_url = pdb_base_dir.replace("PDBCODE", pdb_code)
        # try:
        #     file_name = wget.download(pdb_url, out=pdb_output_dir)
        # except urllib.error.HTTPError as e:
        #     print(f"{pdb_code}.pdb could not be downloaded.")
        #     continue

        split_path = file_name.split("/")
        lower_case_pdb = split_path[-1].lower()
        base_dir = split_path[:-1]
        recombined_path = "/".join(base_dir) + "/" + lower_case_pdb

        shutil.move(file_name, recombined_path)


if __name__ == "__main__":
    pdb_list_to_map()
