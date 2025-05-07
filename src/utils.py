import ast
import datetime
import itertools
import json
import random
from configparser import ConfigParser
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from typing import List
import numpy as np
import pytorch_lightning as pl
import torch
from models import (
    SingleModel,
    EyesModel,
    FullModel,
)
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler
from scipy.interpolate import griddata
from tqdm.autonotebook import tqdm
import datetime
import json
import random
from pathlib import Path
from typing import Sequence, Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import (
    TuneReportCheckpointCallback,
)

from models import (
    GazeDataModule,
    SingleModel,
    EyesModel,
    FullModel,
)


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


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


# Crop the right eye region
def getRightEye(image, landmarks, eye_center):
    eye_top = int(landmarks[257].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])
    eye_width = eye_right - eye_left

    top = eye_center[1] - int(eye_width / 2)
    bottom = eye_center[1] + int(eye_width / 2)

    eye_img = image[
              top:bottom,
              eye_left:eye_right
              ]
    return eye_img


def getLeftEye(image, landmarks, eye_center):
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])
    eye_width = eye_right - eye_left

    top = eye_center[1] - int(eye_width / 2)
    bottom = eye_center[1] + int(eye_width / 2)

    eye_img = image[
              top:bottom,
              eye_left:eye_right
              ]
    return eye_img

# Draw the face mesh annotations on the image.
def drawFaceMesh(image, results):
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #         print('face landmarks', face_landmarks)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        cv2.imshow('MediaPipe FaceMesh', image)


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
    xs = (0 + target_radius, w / 2, w - target_radius)
    ys = (0 + target_radius, h / 2, h - target_radius)
    zones = list(itertools.product(xs, ys))
    random.shuffle(zones)
    return zones


def get_undersampled_region(region_map, map_scale):
    """Get screen coordinates with fewest data samples"""
    min_coords = np.where(region_map == np.min(region_map))
    idx = random.randint(0, len(min_coords[0]) - 1)
    return (min_coords[0][idx] * map_scale, min_coords[1][idx] * map_scale)

# -----------------------------------------------------------------------------
# 1  Utility builders
# -----------------------------------------------------------------------------

def _build_datamodule(data_dir: Path | str, img_types: Sequence[str], cfg: Dict[str, Any]):
    """Return a *GazeDataModule* given CLI/Ray‑Tune `cfg`."""
    return GazeDataModule(
        data_dir=Path(data_dir),
        batch_size=int(cfg.get("bs", 128)),
        img_types=img_types,
        seed=int(cfg.get("seed", 87)),
    )


def _build_model(cfg: Dict[str, Any], img_types: Sequence[str]):
    """Instantiate the proper model, automatically stripping unknown kwargs."""
    import inspect

    # Decide which class to use
    if len(img_types) == 1:
        cls = SingleModel
    elif set(img_types) == {"l_eye", "r_eye"}:
        cls = EyesModel
    else:
        cls = FullModel

    # Keep only kwargs that the target class accepts
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters) - {"self", "args", "kwargs"}

    kwargs = {k: v for k, v in cfg.items() if k in valid_keys}

    if len(img_types) == 1:
        return cls(img_type=img_types[0], **kwargs)
    else:
        return cls(**kwargs)
# -----------------------------------------------------------------------------
# 2. Core Lightning training routine (shared by standalone + Ray‑Tune)
# -----------------------------------------------------------------------------

def _train_model(
    cfg: Dict[str, Any],
    img_types: Sequence[str],
    data_dir: Path | str | None = None,
    num_epochs: int = 20,
    tune_report: bool = False,
):
    """Single‑GPU/CPU train; reports metrics to Ray‑Tune when *tune_report* is True."""

    pl.seed_everything(int(cfg.get("seed", 87)), workers=True)
    data_dir = Path(data_dir or "data")

    dm = _build_datamodule(data_dir, img_types, cfg)
    model = _build_model(cfg, img_types)

    callbacks: List[Any] = []
    if tune_report:
        callbacks.append(
            TuneReportCheckpointCallback(
                metrics={"loss": "val_loss", "mae": "val_mae"}, on="validation_end"
            )
        )

    tb_logger = TensorBoardLogger(
        save_dir="tb_logs",
        name="gaze",
        version=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=tb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)

    if not tune_report:
        metrics = trainer.callback_metrics
        print(
            f"\nFinished – val_loss: {metrics['val_loss']:.3f}, val_mae: {metrics['val_mae']:.3f}"
        )
        return metrics

# -----------------------------------------------------------------------------
# 3. Thin wrappers – keep public API stable
# -----------------------------------------------------------------------------

def train_single(cfg: Dict[str, Any], img_types: Sequence[str], num_epochs: int = 15, data_dir: Path | str = "data"):
    assert len(img_types) == 1, "SingleModel expects exactly one img_type"
    _train_model(cfg, img_types, data_dir, num_epochs, tune_report=False)


def train_eyes(cfg: Dict[str, Any], img_types: Sequence[str] = ("l_eye", "r_eye"), num_epochs: int = 15, data_dir: Path | str = "data"):
    assert set(img_types) == {"l_eye", "r_eye"}, "EyesModel needs both eyes"
    _train_model(cfg, img_types, data_dir, num_epochs, tune_report=False)


def train_full(cfg: Dict[str, Any], img_types: Sequence[str] = ("face_aligned", "l_eye", "r_eye", "head_pos", "head_angle"), num_epochs: int = 20, data_dir: Path | str = "data"):
    _train_model(cfg, img_types, data_dir, num_epochs, tune_report=False)

# -----------------------------------------------------------------------------
# 4. Ray‑Tune – ASHA search
# -----------------------------------------------------------------------------

def tune_asha(
    config: Dict[str, Any],
    train_func: str,  # "single", "eyes", or "full"
    name: str,
    img_types: Sequence[str],
    num_samples: int = 10,
    num_epochs: int = 20,
    data_dir: Path | str = "data",
    seed: int = 1,
):
    """Run hyperparameter optimisation with ASHA and return ExperimentAnalysis."""

    train_map = {
        "single": train_single,
        "eyes": train_eyes,
        "full": train_full,
    }
    assert train_func in train_map, "train_func must be 'single', 'eyes', or 'full'"

    def _tune_wrapper(cfg):
        _train_model(cfg, img_types, data_dir, num_epochs, tune_report=True)

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    def _short_name(trial):
        # just use the numeric trial_id to avoid 260‑char shell‑shock
        return str(trial.trial_id)

    analysis = tune.run(
        _tune_wrapper,
        resources_per_trial={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0},
        metric="loss",
        mode="min",
        num_samples=num_samples,
        config=config,
        scheduler=scheduler,
        name=f"{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        storage_path=Path.cwd() / "logs",
        log_to_file=True,
        fail_fast=True,
        trial_dirname_creator=_short_name,
        verbose=3,
        raise_on_failed_trial=False,
    )

    print("Best hyperparameters →", analysis.best_config)
    return analysis

# -----------------------------------------------------------------------------
# 5. Visualisation helpers – *the* plotters referenced in the notebook
# -----------------------------------------------------------------------------

def _ensure_fig_dir() -> Path:
    fig_dir = Path.cwd() / "figs"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir


def plot_asha_scatter(analysis, param: str, save_path: str | Path | None = None):
    """Scatter plot of a single hyper‑parameter value vs. final validation loss."""
    df = analysis.dataframe()
    if f"config/{param}" not in df.columns:
        raise KeyError(f"Parameter '{param}' not found in analysis dataframe.")

    plt.figure(figsize=(6, 4))
    plt.scatter(df[f"config/{param}"], df["loss"], alpha=0.6, edgecolor="k")
    plt.xlabel(param)
    plt.ylabel("val_loss")
    plt.title(f"ASHA search – {param} vs. val_loss")

    save_path = save_path or _ensure_fig_dir() / f"asha_scatter_{param}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()

    return Path(save_path)


def plot_topk_learning_curves(analysis, k: int = 5, save_path: str | Path | None = None):
    """Overlay val_loss curves of the top‑k trials sorted by final loss."""
    df = analysis.dataframe()
    top_trials = df.nsmallest(k, "loss")["trial_id"]

    plt.figure(figsize=(7, 4))
    for tid in top_trials:
        tdf = analysis.trial_dataframes[tid]
        if "val_loss" in tdf.columns:
            plt.plot(tdf["training_iteration"], tdf["val_loss"], label=f"trial {tid}")
    plt.xlabel("Epoch")
    plt.ylabel("val_loss")
    plt.title(f"Top‑{k} learning curves (ASHA)")
    plt.legend(fontsize="small")

    save_path = save_path or _ensure_fig_dir() / f"asha_top{k}_curves.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()

    return Path(save_path)


def plot_asha_overview(analysis, params: List[str], top_k: int = 8):
    """Convenience one‑liner: scatter for *each* param + learning curves."""
    for p in params:
        plot_asha_scatter(analysis, p)
    plot_topk_learning_curves(analysis, k=top_k)

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


def get_best_results(path):
    """Get best results in a directory
    analysis = Analysis(path, default_metric="loss", default_mode="min")
    df = analysis.dataframe()
    df.sort_values("loss", inplace=True)
    best = df.head(1)

    print(f"\n--- Best of '{path}' ---\n")
    print(f"Directory: {best['logdir'].values[0]}")
    print(f"Loss: {round(best['loss'].values[0],2)}")
    print(f"Pixel error: {round(np.sqrt(best['loss'].values[0]),2)}")

    hyperparams = best.filter(like="config", axis=1)
    print("Hyperparameters...")
    for column in hyperparams:
        name = column.split("/")[1]
        value = hyperparams[column].values[0]
        print(f"- {name}: {value}")

    return analysis.get_best_config()"""


def save_model(model, config, path_weights, path_config):
    """Save trained torch weights with config"""
    torch.save(model.state_dict(), path_weights)

    with open(path_config, "w") as fp:
        json.dump(config, fp, indent=4)



def predict_screen_errors( *img_types, path_model, path_config, path_plot=None, path_errors=None, data_partial=True, steps=10):
    """Get prediction error for each screen coordinate
    with open(path_config) as json_file:
        config = json.load(json_file)

    if len(img_types) == 1:
        model = SingleModel(config, img_types[0])
    else:
        model = FullModel(config)

    model.load_state_dict(torch.load(path_model))
    model.cuda()
    model.eval()

    data = FaceDataset(Path.cwd(), data_partial, *img_types)

    x = []
    y = []
    error = []

    for i, d in tqdm(enumerate(data), total=len(data)):
        if i % steps == 0:
            img_list = [d[img].unsqueeze(0).cuda() for img in img_types]

            with torch.no_grad():
                target = d["targets"].cuda()
                predict = model(*img_list)[0]
                dist = torch.sqrt(((predict - target) ** 2).sum(axis=0))

                x.append(target.cpu().numpy()[0])
                y.append(target.cpu().numpy()[1])
                error.append(float(dist.cpu().numpy()))

    print("Average error: {}px over {} predictions".format(round(np.mean(error), 2), len(error)))
    errors = plot_screen_errors( x, y, error, path_plot=path_plot, path_errors=path_errors)

    return errors"""


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