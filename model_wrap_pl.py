import torch
import numpy as np
import librosa
import librosa.display

import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import compute_STOI, compute_PESQ


class Plwrap(pl.LightningModule):
    def __init__(self, cfg, model, writer, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric = writer
        self.cfg = cfg
        self.sample_len = cfg.sample_len
        self.sample_rate = cfg.trainer.sample_rate
        self.writer = writer
        self.loss = 0

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(), lr=self.cfg.optim.lr, weight_decay=1e-06
        )

    def forward(self, inp):
        out = self.model(inp)
        return out

    def training_step(self, batch, batch_nb):
        noisy_mix, clean, _ = batch

        enhanced = self.forward(noisy_mix)
        loss = self.loss_fn(clean, enhanced)

        if batch_nb%500==0:
            self.writer.add_scalars(f"Train/Loss", loss/500, batch_nb)
            self.loss = 0 

        else:
             self.loss+=loss.item()
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        # NOTE: Validation is super slow and done on gpu
        if self.trainer.root_gpu is not None:
            device = self.trainer.root_gpu
        else:
            device = "cpu"

        noisy_mix, clean, name = batch
        assert len(name) == 1, "Only support batch size is 1 in val/enhancement stage."
        assert len(noisy_mix) == len(
            clean
        ), "Lenght of noisy and clean files does not match"

        name = name[0]
        padded_length = 0

        # The input of the model should be fixed length.
        if noisy_mix.size(-1) % self.sample_len != 0:
            padded_length = self.sample_len - (noisy_mix.size(-1) % self.sample_len)
            noisy_mix = torch.cat(
                [noisy_mix, torch.zeros(size=(1, 1, padded_length), device=device)],
                dim=-1,
            )

        assert noisy_mix.size(-1) % self.cfg.sample_len == 0 and noisy_mix.dim() == 3
        noisy_chunks = list(torch.split(noisy_mix, self.cfg.sample_len, dim=-1))

        # Get enhanced full audio
        enhanced_chunks = [self.model(chunk).detach().cpu() for chunk in noisy_chunks]
        enhanced = torch.cat(enhanced_chunks, dim=-1)

        if padded_length != 0:
            enhanced = enhanced[:, :, :-padded_length]
            noisy_mix = noisy_mix[:, :, :-padded_length]

        enhanced = enhanced.reshape(-1).detach().cpu().numpy()
        clean = clean.cpu().numpy().reshape(-1)
        noisy_mix = noisy_mix.cpu().numpy().reshape(-1)

        stoi_c_n = compute_STOI(clean, noisy_mix, sr=self.sample_rate)
        stoi_c_e = compute_STOI(clean, enhanced, sr=self.sample_rate)
        pesq_c_n = compute_PESQ(clean, noisy_mix, sr=self.sample_rate)
        pesq_c_e = compute_PESQ(clean, enhanced, sr=self.sample_rate)

        if batch_nb <= self.cfg.trainer.visualize_waveform_limit:
            self.visualize_waveform(
                noisy_mix, enhanced, clean, self.current_epoch, name
            )
            self.visualize_spectrogram(
                noisy_mix, enhanced, clean, self.current_epoch, name
            )
            self.write_audio_samples(
                noisy_mix, enhanced, clean, self.current_epoch, name
            )

        return {
            "stoi_c_n": stoi_c_n,
            "stoi_c_e": stoi_c_e,
            "pesq_c_n": pesq_c_n,
            "pesq_c_e": pesq_c_e,
        }

    @staticmethod
    def _transform_pesq_range(pesq_score):
        """transform PESQ range. From [-0.5 ~ 4.5] to [0 ~ 1]."""
        return (pesq_score + 0.5) / 5

    def validation_epoch_end(self, validation_step_outputs):
        get_metrics_ave = lambda key: np.array(
            [m[key] for m in validation_step_outputs]
        ).mean()
        epoch = self.current_epoch

        self.writer.add_scalars(
            f"STOI Clean and noisy", get_metrics_ave("stoi_c_n"), epoch
        )
        self.writer.add_scalars(
            f"STOI Clean and enhanced", get_metrics_ave("stoi_c_e"), epoch
        )

        self.writer.add_scalars(
            f"PESQ Clean and noisy", get_metrics_ave("pesq_c_n"), epoch
        )
        self.writer.add_scalars(
            f"PESQ Clean and enhanced", get_metrics_ave("pesq_c_e"), epoch
        )

        score = (
            get_metrics_ave("stoi_c_e")
            + self._transform_pesq_range(get_metrics_ave("pesq_c_e"))
        ) / 2
        return {"score": torch.tensor(score)}

        # LOGGING FUNCS:
        # NOTE: move funcs to writer later

    def write_audio_samples(self, noisy_mix, enhanced, clean, epoch, name):
        self.writer.add_audio(
            f"Audio_{name}_Noisy", noisy_mix, epoch, sr=self.sample_rate
        )
        self.writer.add_audio(
            f"Audio_{name}_Enhanced", enhanced, epoch, sr=self.sample_rate
        )
        self.writer.add_audio(f"Audio_{name}_Clean", clean, epoch, sr=self.sample_rate)

    def visualize_waveform(self, noisy_mix, enhanced, clean, epoch, name):
        fig, ax = plt.subplots(3, 1)
        for j, y in enumerate([noisy_mix, enhanced, clean]):
            ax[j].set_title(
                "mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                    np.mean(y), np.std(y), np.max(y), np.min(y)
                )
            )
            librosa.display.waveplot(y, sr=self.sample_rate, ax=ax[j])
        plt.tight_layout()
        self.writer.add_figure(f"Waveform_{name}", fig, epoch)

    def visualize_spectrogram(self, noisy_mix, enhanced, clean, epoch, name):
        noisy_mag, _ = librosa.magphase(
            librosa.stft(noisy_mix, n_fft=320, hop_length=160, win_length=320)
        )
        enhanced_mag, _ = librosa.magphase(
            librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320)
        )
        clean_mag, _ = librosa.magphase(
            librosa.stft(clean, n_fft=320, hop_length=160, win_length=320)
        )

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for k, mag in enumerate(
            [
                noisy_mag,
                enhanced_mag,
                clean_mag,
            ]
        ):
            axes[k].set_title(
                f"mean: {np.mean(mag):.3f}, "
                f"std: {np.std(mag):.3f}, "
                f"max: {np.max(mag):.3f}, "
                f"min: {np.min(mag):.3f}"
            )
            librosa.display.specshow(
                librosa.amplitude_to_db(mag),
                cmap="magma",
                y_axis="linear",
                ax=axes[k],
                sr=self.sample_rate,
            )
        plt.tight_layout()
        self.writer.add_figure(f"Spectrogram_{name}", fig, epoch)