import os
import torch
import numpy as np
import librosa
import librosa.display

import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import compute_STOI, compute_PESQ


class PlTrainer(pl.LightningModule):
    def __init__(self,
        self,
        cfg,
        model,
        writer,
        loss_fn):

    self.model = model
    self.loss_fn = loss_fn
    self.metric = writer
    self.cfg = cfg
    self.sample_len = cfg.sample_len
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=1e-06)

    def forward(self, inp):
        out = self.model(inp)
        return out

    def training_step(self, batch, batch_nb):
        noisy_mix, clean, _  = batch
        
        self.optimizer.zero_grad()
        enhanced = self.forward(noisy_mix)
        loss = self.loss_fun(clean, enhanced)
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalars(f"Train/Loss", loss.item(), batch_nb)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        # NOTE: Validation is super slow and done on gpu
        device = torch.cuda.device(f'cuda:{self.trainer.root_gpu}')

        noisy_mix, clean, name = batch
        assert len(name) == 1, "Only support batch size is 1 in val/enhancement stage."
        assert len(noisy_mix) == len(clean), "Lenght of noisy and clean files does not match"

        name = name[0]
        padded_length = 0

        # The input of the model should be fixed length.
        if noisy_mix.size(-1) % self.sample_len != 0:
            padded_length = self.sample_len - (mixture.size(-1) % self.sample_len)
            noisy_mix = torch.cat([noisy_mix, torch.zeros(1, 1, padded_length, device=device)],
                                dim=-1)

        assert noisy_mix.size(-1) % self.sample_length == 0 and noisy_mix.dim() == 3
        noisy_chunks = list(torch.split(noisy_mix, self.sample_length, dim=-1))

        # Get enhanced full audio 
        enhanced_chunks = [self.model(chunk).detach().cpu() for chunk in noisy_chunks]
        enhanced = torch.cat(enhanced_chunks, dim=-1)  

        if padded_length != 0:
            enhanced = enhanced[:, :, :-padded_length]
            mixture = mixture[:, :, :-padded_length]

        enhanced = enhanced.reshape(-1).numpy()
        clean = clean.numpy().reshape(-1)
        mixture = mixture.cpu().numpy().reshape(-1)


        stoi_c_n = compute_STOI(clean, mixture, sr=self.sample_rate)
        stoi_c_e = compute_STOI(clean, enhanced, sr=self.sample_rate)
        pesq_c_n = compute_PESQ(clean, mixture, sr=self.sample_rate)
        pesq_c_e = compute_PESQ(clean, enhanced, sr=self.sample_rate)
    

        # TODO: Continue with spectrogramms 
        # Move them to a separate function

        return {"stoi_c_n": stoi_c_n, 
                "stoi_c_e": stoi_c_e, 
                "pesq_c_n": pesq_c_n, 
                "pesq_c_e": pesq_c_e}



    def validation_epoch_end(self, validation_step_outputs):







        ############################3
        # TODO rewrite for regression
        if self.task == "binary":
            y_hat = (
                torch.stack([x["y_hat"][:, 1] for x in validation_step_outputs]))
            y_hat = F.softmax(y_hat, 1).reshape(-1).cpu().numpy()
        elif self.task == "regression":
            y_hat = (
                torch.stack([x["y_hat"] for x in validation_step_outputs])
                .reshape(-1)
                .cpu()
                .numpy()
            )
        else:
            raise NotImplementedError

        y = (
            torch.stack([x["y"] for x in validation_step_outputs])
            .reshape(-1)
            .cpu()
            .numpy()
        )

        score = self.metric(y, y_hat)
        loss = (
            torch.stack([x["loss"] for x in validation_step_outputs])
            .reshape(-1)
            .cpu()
            .numpy()
            .mean()
        )
        print(f"SCORE: {score} mean LOSS {loss}")
        self.save_model(score)
        self.custom_logger(key="score", value=score, order=self.current_epoch)
        return {"score": torch.tensor(score)}



         def save_model(self, score):
            tp = "_red" if self.cfg["use_cnn_reduction"] else "_full"
        path = os.path.join("./my_check_points", self.cfg["dataset_name"] + tp)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), path + f"/model_{round(score,3)}.chk")
        print(f"MODEL SAVED {path}")