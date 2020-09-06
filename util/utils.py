import importlib
import time
from datetime import date
import os

import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi
from scipy.io.wavfile import write


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (
        ".pth",
        ".tar",
    ), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


# def initialize_config(module_cfg, pass_args=True):
#     """According to config items, load specific module dynamically with params.
#     e.g., Config items as follow：
#         module_cfg = {
#             "module": "model.model",
#             "main": "Model",
#             "args": {...}
#         }
#     1. Load the module corresponding to the "module" param.
#     2. Call function (or instantiate class) corresponding to the "main" param.
#     3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
#     """
#     module = importlib.import_module(module_cfg["module"])
#
#     if pass_args:
#         return getattr(module, module_cfg["main"])(**module_cfg["args"])
#     else:
#         return getattr(module, module_cfg["main"])


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(sr, clean_signal, noisy_signal, "wb")


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


class OmniLogger:
    def __init__(self, ex, trainer_conf):
        self.ex = ex
        self.dir = os.path.join(trainer_conf.base_dir, trainer_conf.exp_name)
        self.speech_path = os.path.join(
            trainer_conf.base_dir, trainer_conf.exp_name, trainer_conf.speech_dir
        )
        os.makedirs(self.dir, exist_ok=True)

    def add_scalars(self, key, value, order):
        self.ex.log_scalar(key, float(value), order)

    def add_audio(self, name, array, epoch, sr):
        name = f"{name}_ep_{epoch}.wav"
        full_path = os.path.join(self.speech_path, name)
        write(full_path, sr, array)
        self.ex.add_artifact(full_path, name)

    def add_image(self, name, array):
        full_path = os.path.join(self.speech_path, name)
        self.ex.add_artifact(full_path, name)

    def add_figure(self, name, fig, epoch):
        name = f"{name}_ep_{epoch}.png"
        full_path = os.path.join(self.speech_path, name)
        fig.savefig(full_path)
        self.ex.add_artifact(full_path, name)
