import librosa
import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import compute_STOI, compute_PESQ


import os
import torch
from auto_int import AutoInt
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F


class PlTrainer(pl.LightningModule):
    def __init__(self,
        self,
        config,
        model,
        writer,
        loss_function,
        optimizer):



    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-06)

    def forward(self, inp):
        preds = self.model(inp)
        return preds

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        if self.task == "regression":
            y = y.float().reshape(-1)
            y_hat = y_hat.float().reshape(-1)

        loss = self.loss_fn(y_hat, y)
        self.custom_logger(key="loss", value=loss, order=batch_nb)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        if self.task == "regression":
            y = y.float().reshape(-1)
            y_hat = y_hat.float().reshape(-1)

        loss = self.loss_fn(y_hat, y)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def save_model(self, score):
        tp = "_red" if self.cfg["use_cnn_reduction"] else "_full"
        path = os.path.join("./my_check_points", self.cfg["dataset_name"] + tp)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), path + f"/model_{round(score,3)}.chk")
        print(f"MODEL SAVED {path}")

    def validation_epoch_end(self, validation_step_outputs):
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