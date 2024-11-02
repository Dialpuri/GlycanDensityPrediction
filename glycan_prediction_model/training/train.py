import os
import numpy as np
import gemmi
from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.stats import zscore
import itertools

import enum
import argparse
import random
from datetime import datetime
from glycan_prediction_model.training.loss import sigmoid_focal_crossentropy
from glycan_prediction_model.training.data import create_dataset
import glycan_prediction_model.training.unet as unet

from confection import Config
import polars as pl
from collections import defaultdict
# from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from random import randrange


class DataSources(enum.Enum):
    __order__ = "deposited deglyco"

    deposited = 1
    deglyco = 2



def save_coordinates(coordinates: List[np.ndarray], cell: gemmi.UnitCell, path: str ): 
    print("Writing coordinates to ", path)
    s = gemmi.Structure()
    s.cell = cell
    m = gemmi.Model("1")
    
    c = gemmi.Chain("A")


    # for i, coordinate in enumerate(coordinates):    
    i = 0
    r = gemmi.Residue()
    r.seqid = gemmi.SeqId(f"{i+1}")
    r.name = "DUM"
    for point in coordinates:
        a = gemmi.Atom()
        a.pos.fromlist(list(point))
        a.name = "X"
        a.b_iso = 20
        a.element = gemmi.Element("X")
        r.add_atom(a)
    
    c.add_residue(r)
    m.add_chain(c)
    s.add_model(m)
    
    s.write_pdb(str(path))


def sample_generator(df: pl.DataFrame, config: dict):
    # restrict = config["dataset"]["restrict_samples"]
    
    for source in itertools.cycle(DataSources):
        for restrict in itertools.cycle([True, True, True, False]):

            sample = df.sample(n=1)
            pdb = sample["pdb"].item()

            source_map_path = sample["source_path"].item()
            exp = gemmi.read_ccp4_map(source_map_path, setup=True).grid
            if config["dataset"]["normalise"]:
                exp.normalize()
        
            difference_map_path = sample["difference_path"].item()
            
            if not os.path.exists(difference_map_path): 
                continue
            
            try:
                diff = gemmi.read_ccp4_map(difference_map_path, setup=True).grid
            except:
                continue
            if config["dataset"]["normalise_difference"]:
                diff.normalize()

            target_map_path = sample["target_path"].item()
            target_map = gemmi.read_ccp4_map(target_map_path, setup=True)
            # target_map.grid.normalize()
            tar = target_map.grid

            # rotation = gemmi.Mat33(Rotation.random().as_matrix())
            rotation=gemmi.Mat33([[1,0,0], [0,1,0], [0,0,1]])

            if restrict:     
                help_df = pl.read_csv(sample["help_path"].item(), schema={"x":pl.Float32, "y":pl.Float32, "z": pl.Float32})
                if len(help_df) == 0:
                    continue
                
                help_sample = help_df.sample(n=1).row(0)
                jitter_range = 6
                jitter = gemmi.Position(randrange(-jitter_range, jitter_range), randrange(-jitter_range, jitter_range), randrange(-jitter_range, jitter_range))
                point = gemmi.Position(*help_sample) - gemmi.Position(11.2,11.2,11.2) - jitter
                translation = tar.unit_cell.fractionalize(point)
            else:
                translation = gemmi.Fractional(*np.random.rand(3))
            
            exp_array = _interpolate(exp, translation, rotation, False, config)
            diff_array = _interpolate(diff, translation, rotation, False, config)
            tar_array = _interpolate(tar, translation, rotation, True, config)        

            if restrict:
                thresold = pow(config["dataset"]["box_size"], 3) * config["dataset"]["sample_coverage_threshold"]
                if np.sum(tar_array) > thresold:
                    yield tf.concat([exp_array, diff_array], axis=-1), tf.one_hot(np.round(tar_array), depth=2)
                else:
                    print(np.sum(tar_array))
            else:
                yield tf.concat([exp_array, diff_array], axis=-1), tf.one_hot(np.round(tar_array), depth=2)


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


def dice_coe(y_true, y_pred, loss_type="jaccard", smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == "jaccard":
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == "sorensen":
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred, loss_type="jaccard", smooth=1.0):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == "jaccard":
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == "sorensen":
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return 1 - (2.0 * intersection + smooth) / (union + smooth)


def train(config):
    num_threads: int = 128
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    train, test = create_dataset()
    _train_gen = sample_generator(train, config)
    _test_gen = sample_generator(test, config)

    size = config["dataset"]["box_size"]
    epochs = config["training"]["epochs"]

    input = tf.TensorSpec(shape=(size, size, size, 2), dtype=tf.float32)
    output = tf.TensorSpec(shape=(size, size, size, 2), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: _train_gen, output_signature=(input, output)
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: _test_gen, output_signature=(input, output)
    )
    
    model = unet.binary_model_64()

    loss = sigmoid_focal_crossentropy

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=["accuracy"],
    )

    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=2,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=1e-7,
    )
    # epochs: int = 1000
    batch_size: int = 8
    steps_per_epoch: int = 1000
    validation_steps: int = 100
    name: str = f"{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}"
    print(f"Starting with {epochs} epochs")

    weight_path: str = f"models/{name}.keras"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )

    train_dataset = train_dataset.repeat(epochs).batch(batch_size=batch_size)

    test_dataset = test_dataset.repeat(epochs).batch(batch_size=batch_size)

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"logs/{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}_{type}.log"
    )

    callbacks_list = [
        checkpoint,
        reduce_lr_on_plat,
        TqdmCallback(verbose=2),
        csv_logger,
    ]

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=0,
    )

    model.save(f"models/{name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=False, default="runs")
    parsed_args = parser.parse_args()
    parsed_config = Config().from_disk(parsed_args.config)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    train(parsed_config)
