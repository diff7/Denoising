import os
import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import DictConfig, OmegaConf
from sacred import Experiment
from sacred.observers.mongo import QueuedMongoObserver

# from sacred.observers import MongoObserver
from torch.utils.data import DataLoader

from dataset.waveform_dataset import DatasetAudio
from model.sftt_loss import MultiResolutionSTFTLoss
from model.demucs import Demucs
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model_wrap_pl import Plwrap
from utils import OmniLogger

config = OmegaConf.load("config.yaml")
ex = Experiment(config.trainer.exp_name)
ex.add_config(dict(config))
ex.observers.append(
    QueuedMongoObserver.create(
        url="mongodb://mongo_user:pass@egorsv_dockersacredomni_mongo_1/",
        db_name="sacred",
    )
)


@ex.main
def main(_config):
    cfg = DictConfig(_config)
    print(config.pretty(resolve=True))
    writer = OmniLogger(ex, config.trainer)

    torch.manual_seed(cfg.seed)  # for both CPU and GPU
    np.random.seed(cfg.seed)

    mrstftloss = MultiResolutionSTFTLoss(
        factor_sc=cfg.loss.stft_sc_factor,
        factor_mag=cfg.loss.stft_mag_factor,
    )

    def loss_function(x, y):
        sc_loss, mag_loss = mrstftloss(x.squeeze(1), y.squeeze(1))
        return F.l1_lomse_loss(x, y) + sc_loss + mag_loss

    train_set = DatasetAudio(
        **dict(config.data.dataset_train),
        sample_len=cfg.sample_len,
        shift=cfg.data.shift,
    )

    train_dataloader = DataLoader(
        dataset=train_set,
        **config.data.loader_train,
    )

    val_set = DatasetAudio(**config.data.dataset_val, sample_len=cfg.sample_len)
    val_dataloader = DataLoader(
        dataset=val_set, num_workers=1, batch_size=1, shuffle=True
    )

    model = Demucs(**config.demucs)
    model_pl = Plwrap(cfg, model, writer, loss_function)

    check_point_path = os.path.join(
        cfg.trainer.base_dir, cfg.trainer.exp_name, "checkpoints"
    )
    os.makedirs(check_point_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=check_point_path,
        verbose=True,
        monitor="score",
        mode="max",
        save_weights_only=False,
    )

    # adhoc load if only save weights

    # model_pl.load_state_dict(
    #     torch.load(os.path.join(check_point_path, c))["state_dict"]
    # )
    resume_from = None
    if cfg.trainer.resume is not None:
        resume_from = os.path.join(check_point_path, cfg.trainer.resume)
    print(resume_from)
    trainer = pl.Trainer(
        resume_from_checkpoint=resume_from,
        max_epochs=cfg.trainer.epochs,
        gpus=[2],
        auto_select_gpus=True,
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=cfg.trainer.validation_interval,
    )

    trainer.fit(
        model_pl,
        train_dataloader,
        val_dataloader,
    )


if __name__ == "__main__":
    ex.run_commandline()
