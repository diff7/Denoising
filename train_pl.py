import os
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sacred import Experiment
from sacred.observers import MongoObserver
from torch.utils.data import DataLoader

from dataset.waveform_dataset import DatasetAudio
from model.loss import l1_loss, mse_loss
from model.unet_basic import Model as Unet

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model_wrapper_pl import Plwrap
from utils import OmniLogger

config = OmegaConf.load("config.yaml")
ex = Experiment(config.exp_name)
ex.add_config(dict(config))
ex.observers.append(
    MongoObserver.create(
        url="mongodb://mongo_user:pass@dockersacredomni_mongo_1/", db_name="sacred"
    )
)


@ex.main
def main(_config):
    cfg = DictConfig(_config)
    print(config.pretty(resolve=True))
    writer = OmniLogger(ex, config.trainer)

    torch.manual_seed(cfg.seed)  # for both CPU and GPU
    np.random.seed(cfg.seed)

    loss_function = globals()[config.loss]()

    train_set = DatasetAudio(
        **dict(config.data.dataset_train), sample_len=cfg.sample_len
    )
    train_dataloader = DataLoader(
        dataset=train_set,
        **config.data.loader_train,
    )

    val_set = DatasetAudio(**config.data.dataset_val, sample_len=cfg.sample_len)
    val_dataloader = DataLoader(
        dataset=val_set, num_workers=1, batch_size=1, shuffle=True
    )

    model = Unet(**config.model)
    model_pl = Plwrap(cfg, model, writer, loss_function)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join("./check_points"),
        verbose=True,
        monitor="score",
        mode="max",
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        gpus=0,
        auto_select_gpus=False,
        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(model_pl, train_dataloader, val_dataloader)


if __name__ == "__main__":
    ex.run_commandline()
