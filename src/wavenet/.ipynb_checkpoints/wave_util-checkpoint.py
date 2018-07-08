import src.wavenet.wave_model
from src.dataset import WavenetLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch


def fetch_dataloader(dataset, model, hparams):
	num_train = len(dataset)
	indices = list(range(num_train))
	valid_size = hparams.valid_size
	split = int(np.floor(valid_size * num_train))
	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	sample_size = model.receptive_field + hparams.output_length
	print("Fetch dataloader: ")
	print("   sample size", sample_size)
	print("   recp: ", model.receptive_field)
	print("   output length: ", hparams.output_length)

	dl_train = WavenetLoader(dataset, receptive_length=model.receptive_field,
							 sample_size=sample_size, batch_size=hparams.batch_size,
							 sampler=train_sampler, num_workers=hparams.num_workers)

	dl_val = WavenetLoader(dataset, receptive_length=model.receptive_field,
						   sample_size=sample_size, batch_size=hparams.batch_size,
						   sampler=valid_sampler, num_workers=hparams.num_workers)

	return {
		'train': dl_train,
		'val': dl_val
	}


def fetch_model(hparams):
	model = src.wavenet.wave_model.WavenetModel(layers=hparams.layers, stacks=hparams.stacks,
												dilation_channels=hparams.dilation_channels,
												residual_channels=hparams.residual_channels,
												skip_channels=hparams.skip_channels,
												end_channels=hparams.end_channels, classes=hparams.classes,
												output_length=hparams.output_length,
												kernel_size=hparams.kernel_size, cin_channels=hparams.cin_channels,
												bias=hparams.bias)
	return model


def fetch_optimizer(model, hparams):
	optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

	return optimizer


def mulaw_encode(input, channels=256):
	n_q = channels - 1
	mu = torch.tensor(n_q, dtype=torch.float)
	audio = torch.tensor(input)
	# audio = torch.abs(torch.tensor(input))
	# audio_abs = torch.min(torch.abs(audio), 1.0)
	mag = torch.log1p(mu * torch.abs(audio)) / torch.log1p(mu)
	signal = torch.sign(audio) * mag
	out = ((signal + 1) / 2 * mu + 0.5).int()
	return out


def mulaw_decode(input, channels=256):
	n_q = channels - 1
	mu = torch.tensor(n_q, dtype=torch.float)
	audio = torch.tensor(input, dtype=torch.float)
	audio = (audio / mu) * 2 - 1
	out = torch.sign(audio) * (torch.exp(torch.abs(audio) * torch.log1p(mu)) - 1) / mu
	return out


def one_hot_encode(input, channels=256):
	size = input.size()[0]
	one_hot = torch.FloatTensor(size, channels)
	one_hot = one_hot.zero_()
	in1 = torch.tensor(input, dtype=torch.long)
	one_hot.scatter_(1, in1.unsqueeze(1), 1.0)
	return one_hot


def one_hot_decode(input):
	_, i = input.max(1)
	return torch.tensor(i)


def mel_resize(audio, S):
	factor = audio.size / S.shape[0]
	print(factor)
	mel = np.repeat(S, factor, axis=0)
	print(S.shape)
	print(mel.shape)
	mel_pad = audio.size - mel.shape[0]
	print(mel_pad)
	mel = np.pad(mel, [(0, mel_pad), (0, 0)], mode="constant", constant_values=0)
	print(mel.shape)
	return mel
