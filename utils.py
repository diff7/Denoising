import importlib
import time
from datetime import date
import os

import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi
from scipy.io.wavfile import write


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
            trainer_conf.base_dir, trainer_conf.exp_name, trainer_conf.spectro_dir
        )
        os.makedirs(self.speech_path, exist_ok=True)

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
