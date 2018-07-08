import torch
from  src.Tacotron_model.taco_model import MelSpectrogramNet
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from src.Tacotron_model.util import collate_fn
from src.Tacotron_model.util import chars
import numpy as np


def fetch_dataloader(dataset,hparams):
    num_train = len(dataset)
    dataset_train_parts = {}
    indices = list(range(num_train))
    for i in range(num_train):
        dataset_train_parts[i] = (dataset[i]['embedded_text'], dataset[i]['mel_spectograms'])
    #dataset_train_parts = (embedded, mels)
    ##split dataset
    split = int(np.floor(hparams.valid_size * num_train))
    train_data, valid_data = dataset[:split], dataset[split:]
    dl_train = DataLoader(train_data, batch_size=hparams.batch_size, collate_fn=collate_fn)
    dl_val = DataLoader(valid_data, batch_size=hparams.batch_size, collate_fn=collate_fn)

    return {
        'train': dl_train,
        'val': dl_val
    }


def fetch_model(hparams):
    num_chars = len(chars)
    teacher_forcing_ratio = hparams.teacher_forcing_ratio
    model = MelSpectrogramNet(num_chars, teacher_forcing_ratio)
    return model


def fetch_optimizer(model,hparams):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

    return optimizer


