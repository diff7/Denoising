import os
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.unet_basic import Model as Unet
from dataset.waveform_dataset import DatasetAudio
from trainer.trainer import Trainer
from model.loss import mse_loss, l1_loss



@hydra.main(config_path="config_h.yaml")
def main(config):
    torch.manual_seed(config["seed"])  # for both CPU and GPU
    np.random.seed(config["seed"])

    train_set = DatasetAudio(**config.data.dataset_train)
    train_dataloader = DataLoader(dataset=train_set, **config.data.loader_train)

    val_set = DatasetAudio(**config.data.dataset_val)
    val_dataloader = DataLoader(
        dataset=val_set, num_workers=1, batch_size=1, shuffle=False
    )

    model = Unet(**config.model)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, config.optim.beta2),
    )

    loss_function = globals()[config.loss]()

    trainer = Trainer(
        config=config.trainer,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
    )

    trainer.train()


if __name__ == "__main__":
    main()
