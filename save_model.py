import torch
import random
import logging
import hydra
import json
import yaml
import numpy as np
import lightning.pytorch as pl

from pathlib import Path
from data.bug import BUGBalanced
from configuration.tuner import MitigationConfig
from mitigation.tuner import GPT2FineTuningModule

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import os

file = ""
file_corr = "GroNLP/gpt2-small-dutch_random_seed_9_new"

for filename in os.listdir("checkpoints/GroNLP"):
    # Check if the path is a file (not a directory)
    if filename == "gpt2-small-dutch_random_seed_9_epoch=037_val_loss=3.74.ckpt":
        file = os.path.join("checkpoints/GroNLP", filename)

print(file)


ft_model = GPT2FineTuningModule.load_from_checkpoint(
    file, map_location=torch.device("cuda")
)
torch.save(
    ft_model.model,
    Path("/content/drive/My Drive/Mitigation_data")
    / (file_corr + "_epochs" + str(40) + ".pt"),
)

print(
    "Model succesfully saved at ",
    Path("/content/drive/My Drive/Mitigation_data")
    / (file_corr + "_epochs" + str(40) + ".pt"),
)
