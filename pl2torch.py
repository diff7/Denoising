import torch
from model.demucs import Demucs
from omegaconf import OmegaConf
from collections import OrderedDict


pl_model_path = 'results/Demucs_pl_ru_ukr/checkpoints/epoch=214.ckpt'
new_torch_model_path = 'results/Demucs_pl_ru_ukr/checkpoints/torch_214.ckpt'
config_path = './main/config.yaml'

if __name__ == "__main__":
    checkpoint = torch.load(pl_model_path)
    config = OmegaConf.load(config_path)
    model = Demucs(**config.demucs)

    keys_orig = list(model.state_dict().keys())
    keys_pl =  list(checkpoint['state_dict'].keys())

    new_state_dict = OrderedDict()

    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.','')
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    torch.save(model.state_dict(), new_torch_model_path)
