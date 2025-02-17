import ast
import random
import json
import datetime
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict
from configparser import ConfigParser
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split


def get_config(path="config.ini", comment_char=";"):
    config_file = ConfigParser(inline_comment_prefixes=comment_char)
    config_file.read(path)

    config_default = config_file["DEFAULT"]
    config_colours = config_file["COLOURS"]
    config_eyetracker = config_file["EYETRACKER"]
    config_tf = config_file["TF"]

    settings = {key: ast.literal_eval(config_default[key]) for key in config_default}
    colours = {key: ast.literal_eval(config_colours[key]) for key in config_colours}
    eyetracker = {
        key: ast.literal_eval(config_eyetracker[key]) for key in config_eyetracker
    }
    tf = {key: ast.literal_eval(config_tf[key]) for key in config_tf}

    return settings, colours, eyetracker, tf


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((5, 2), dtype=dtype)
    for i in range(0, 5):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def bgr_to_rgb(img):
    """Convert from opencv BGR to RGB"""
    return img[..., ::-1].copy()


def clamp_value(x, max_value):
    """Restrict values to a range"""
    if x < 0:
        return 0
    if x > max_value:
        return max_value
    return x


def plot_region_map(path, region_map, map_scale, cmap="inferno"):
    """Create plot of number of data samples at each screen coordinate"""
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(
        np.transpose(region_map).repeat(map_scale, axis=0).repeat(map_scale, axis=1),
        interpolation="bicubic",
        cmap=cmap,
    )
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(
        im, ticks=np.arange(np.min(region_map), np.max(region_map) + 1), cax=cax
    )
    ax.set_title("Number of samples at each screen region")
    plt.savefig(path)


def get_calibration_zones(w, h, target_radius):
    """Get coordinates for 9 point calibration"""
    xs = (0 + target_radius, w // 2, w - target_radius)
    ys = (0 + target_radius, h // 2, h - target_radius)
    zones = list(itertools.product(xs, ys))
    random.shuffle(zones)
    return zones


def get_undersampled_region(region_map, map_scale):
    """Get screen coordinates with fewest data samples"""
    min_coords = np.where(region_map == np.min(region_map))
    idx = random.randint(0, len(min_coords[0]) - 1)
    return (min_coords[0][idx] * map_scale, min_coords[1][idx] * map_scale)



def dir_name_string(trial):
    name = str(trial.experiment_tag)

    if len(name) > 100:
        return name[:100]
    else:
        return name




def get_tune_results(analysis):
    """Get results from single experiment"""

    if analysis.best_checkpoint:
        print(f"Directory: {analysis.best_checkpoint}")
    else:
        print(f"Directory: {analysis.best_logdir}")

    print(f"Loss: {round(analysis.best_result['loss'],2)}")
    print(f"Pixel error: {round(np.sqrt(analysis.best_result['loss']),2)}")
    print("Hyperparameters...")
    for hparam in analysis.best_config:
        print(f"- {hparam}: {analysis.best_config[hparam]}")



def save_model(model, config, path_weights, path_config):
    """Save trained torch weights with config"""
    torch.save(model.state_dict(), path_weights)

    with open(path_config, "w") as fp:
        json.dump(config, fp, indent=4)



def plot_screen_errors(x, y, z, path_plot=None, path_errors=None):
    """Plot prediction errors over screen space"""
    # create grid
    xi = np.arange(0, 1920, 1)
    yi = np.arange(0, 1080, 1)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method="nearest")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(xi, yi, zi)
    cbar = plt.colorbar()

    cbar.ax.set_ylabel("Error (px)")
    plt.xlabel("Screen X")
    plt.ylabel("Screen Y")
    plt.gca().invert_yaxis()

    if path_plot is not None:
        plt.savefig(path_plot, dpi=100)

    if path_errors is not None:
        np.save(path_errors, zi.T)

    plt.show()

    # Error histogram
    plt.hist(z, edgecolor="black")
    plt.xlabel("Error (px)")
    plt.ylabel("Count")
    plt.show()

    return zi.T


# Tensorflow test things below


class OrderedDictWithDefaultList(OrderedDict):
    """
    Used for tensorflow in-memory datasets
    """

    def __missing__(self, key):
        value = list()
        self[key] = value
        return value


def create_data_splits(data, train_size=0.80, shuffle=True, random_state=87):
    """
    Used for tensorflow in-memory datasets

    This mutates the data dict.
    To make it less brittle (at the cost of more memory), create and return a data.copy()
    """
    # Create training set
    split1 = train_test_split(
        *[v for v in data.values()],
        test_size=1 - train_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    train_data = split1[0::2]
    remaining_data = split1[1::2]

    # Split remaining into validation and test sets (50/50)
    split2 = train_test_split(
        *remaining_data,
        test_size=0.5,
        shuffle=shuffle,
        random_state=random_state,
    )
    val_data = split2[0::2]
    test_data = split2[1::2]

    for i, k in enumerate(data.keys()):
        data[k] = {}
        data[k]["train"] = train_data[i]
        data[k]["val"] = val_data[i]
        data[k]["test"] = test_data[i]

    return data