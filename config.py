import os
from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 360,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "es",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_pathname(config, epoch: str):
    return str(os.path.join(config["model_folder"], f'{config["model_basename"]}{epoch}.pt'))
