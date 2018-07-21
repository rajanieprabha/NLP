import torch
from  Tacotron_model.taco_model import MelSpectrogramNet
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from Tacotron_model.util import collate_fn
from Tacotron_model.util import chars
import numpy as np


def fetch_dataloader(dataset,hparams):
    num_train = len(dataset)
    dataset_train_parts = {}
    indices = list(range(num_train))
    for i in range(num_train):
        dataset_train_parts[i] = (dataset[i]['embedded_text'], dataset[i]['mel_spectograms'])
    #dataset_train_parts = (embedded, mels)
    split = int(np.floor(hparams.valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    dl_train = DataLoader(dataset_train_parts, batch_size=hparams.batch_size,sampler = train_sampler, collate_fn=collate_fn)
    dl_val = DataLoader(dataset_train_parts, batch_size=hparams.batch_size, sampler = valid_sampler, collate_fn=collate_fn)

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


