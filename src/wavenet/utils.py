import numpy as np
import torch


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