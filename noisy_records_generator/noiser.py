import os
import torch
import random
import librosa
from multiprocess import get_context
from multiprocessing.pool import ThreadPool
import torchaudio as ta
from scipy.io import wavfile
from tqdm import tqdm
from omegaconf import OmegaConf


# Did not manage to speed up with gpu !!! Use librosa + numpy next time
# Preprocessing  on GPU is faster
# But need to fix GPU out memory


def read_A_file(file):
    waveform, sr = ta.load(file, normalization=True)
    return waveform


def align(waveform, new_length):
    waveform = waveform
    if len(waveform[0]) >= new_length:
        new_waveform = waveform[:, :new_length]
    if len(waveform[0]) < new_length:
        m, l = new_length // len(waveform[0]), new_length % len(waveform[0])
        new_waveform = [waveform] * m
        new_waveform.append(waveform[:, :l])
        new_waveform = torch.cat(new_waveform, 1)
    return new_waveform


def read_n_align(file, new_length, resample):
    waveform = read_A_file(file)
    #waveform = resample(waveform)
    waveform = librosa.resample(waveform.numpy()[0], 44100, 16000)
    waveform = torch.tensor(waveform).unsqueeze(0)
    return align(waveform, new_length)


def read_random_batch(all_files, resample, batch_size, workers, new_length):

    files_batch = random.sample(all_files, batch_size)
    new_length = [new_length] * batch_size
    resample = [resample] * batch_size
    with get_context("spawn").Pool(workers) as p:
        files = p.starmap(read_n_align, zip(files_batch, new_length, resample))
    return torch.cat(files, 0)


rms = lambda x: torch.pow(torch.pow(x, 2).mean(1), 0.5)


def snr(signal, noise):
    r_s = rms(signal)
    r_n = rms(noise)
    return 20 * torch.log10(r_s / r_n)


def get_noise_scale(signal, snr):
    r_s = rms(signal)
    snr_ten = torch.pow(10, torch.true_divide(snr, 10))
    return r_s / torch.pow(snr_ten, 0.5)


def get_random_noise_snr(signal, snr):
    scale = get_noise_scale(signal, snr)
    size = signal.shape[1]
    return torch.randn(size=(1, size)) * scale.view(
        -1, 1
    )  # torch.randn(size=(1, size)).cuda() * scale.view(-1, 1)


def get_snr_scaled_noise(signal, noise, snr):
    scale = get_noise_scale(signal, snr)
    r_n = rms(noise)
    return noise * torch.true_divide(scale, r_n).view(-1, 1)

def make_noisy_file(clean_file_path, cfg, noise_files, resample):
    print(f"CURRENT F: {clean_file_path}")
    output_file = open(cfg.save_records, "a")
    waveform = read_A_file(clean_file_path)
    file_name = os.path.join(
        cfg.result_dir, clean_file_path.split("/")[-1].split(".")[0]
    )

    snr = random.randint(cfg.snr_min, cfg.snr_max)
    file_name = file_name + f"_snr{snr}"
    # get random sounds files and align them with source file len
    base_len = len(waveform[0])
    noises_batch = read_random_batch(
        noise_files,
        resample,
        batch_size=cfg.num_copies,
        workers=cfg.workers,
        new_length=base_len,
    )

    noised = get_snr_scaled_noise(waveform, noises_batch, snr) + waveform
    del noises_batch
    if random.random() > cfg.white_noise:
        snr = random.randint(cfg.snr_min, cfg.snr_max)
        noised = noised + get_random_noise_snr(noised, snr)
        file_name = file_name + f"_snr{snr}"
    # CUDA OUT
    #noised = noised.cpu()
    #torch.cuda.empty_cache()

    def save_file(file, idx):
        # r_m = rms(file.unsqueeze(0))
        new_name = file_name + f"_idx{idx}.wav"
        new_name = os.path.abspath(new_name)
        # ta.save(new_name, file, sample_rate=cfg.source_sr)
        # torch save requiers normalized values in range [-1:1]
        wavfile.write(new_name, cfg.source_sr, file.numpy())
        output_file.write(f"{clean_file_path} {new_name}\n")

    with get_context("spawn").Pool(cfg.workers) as p:
        p.starmap(save_file, [(noised[i], i) for i in range(cfg.num_copies)])
    output_file.close()


if __name__ == "__main__":
    cfg = OmegaConf.load("noise_conf.yaml")
    os.makedirs(cfg.result_dir, exist_ok=True)
    clean_files = cfg.clean_files_txt
    noise_files = cfg.noise_files_txt

    with open(noise_files, "r") as f:
        noise_files = f.read().split("\n")

    with open(clean_files, "r") as f:
        clean_files = f.read().split("\n")

    # resample to match noise_sr == clean_sr
    resample = ta.transforms.Resample(
        orig_freq=cfg.noise_sr,
        new_freq=cfg.source_sr,
        resampling_method="sinc_interpolation",
    )

    clean_files = [os.path.abspath(f) for f in clean_files]
    random.shuffle(clean_files)
    pool = ThreadPool(processes=cfg.workers)
    result = pool.starmap(
        make_noisy_file, [(f, cfg, noise_files, resample) for f in clean_files]
    )
    result.get()
    #make_noisy_file(clean_files[0], cfg, noise_files, resample)
