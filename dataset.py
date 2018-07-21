import librosa.display
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
import os
import numpy as np
from Tacotron_model.util import load_wav

#import src.wavenet.wave_util as util

import torch.nn.functional as F

# from src.Tacotron_model import taco_util
import wavenet.utils as util


class TextToSpeechDataset(Dataset):
    def __init__(self, path , text_embeddings=None, mel_transforms=None,inference_mode =False):
        self.inference_mode = inference_mode
        self.path = path
        print(("data at path ") + (path))
        csv = path + '/metadata.csv'
        self.metadata = pd.read_csv(csv, sep = '|' ,names=['wav', 'transcription', 'text'],
                            usecols=['wav', 'text'])
    
        self.metadata.dropna(inplace= True)
        self.mel_transforms = mel_transforms

        self.text_embeddings = text_embeddings

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        text = self.metadata.iloc[idx]['text']
        embedded_text = text
        wav_filename = self.metadata.iloc[idx]['wav']
        audio, _ = load_wav(self.path + '/wavs/' + wav_filename + '.wav')
        if self.inference_mode:
            audio_array = np.zeros([1000000])
        else:
            audio_array = np.asarray(audio)  # convert tuple to np array
        text_for_embeddings = text
        #text = text.values  # convert pandas dataframe to np array
        if self.text_embeddings:
            embedded_text = self.text_embeddings(embedded_text)
            #print(embedded_text)

        # audio_for_mel = np.array(audio)
        if self.mel_transforms:
                audio = self.mel_transforms(audio)

        sample = {'text': text,
                  'speech': audio_array,
                  'embedded_text': embedded_text,
                  'mel_spectograms': audio
                  }
        return sample



class WavenetLoader(DataLoader):

	def __init__(self, dataset, receptive_length, sample_size, q_channels=256, batch_size=1,
				 sampler=None, num_workers=1):
		"""
		Gets the data as TTSDataset format and loads the audio outputs for training
			sample_size = receptive_length+target_size - 1

		Args:
			dataset ():
			receptive_length ():
			sample_size ():
			q_channels ():
			batch_size ():
			sampler ():
			num_workers ():
		"""

		self.dataset = dataset

		super().__init__(dataset, batch_size, shuffle=False, sampler=sampler) #num_workers

		self.sample_size = sample_size
		self.receptive_fields = receptive_length
		self.q_channels = q_channels
		self.collate_fn = self._collate_fn2

	def _collate_fn1(self, batch):
		"""
		Gets b batches and stacks them with the same size of samples.
		TODO: read mel frequency as an input! -> ground truth & prediction

		TODO: return the merged data from consecutive datafiles -> LATER
		Args:
			batch ():

		Returns:
			the audio and targets with required lengths during iteration.
		"""
		print("collate call: ", len(batch))

		input_batch, target_batch = [], []
		for sample in batch:
			ins, target = self._prepare1(sample)
			input_batch.append(ins)
			target_batch.append(target)
		return input_batch, target_batch

	def _prepare1(self, sample):
		audio = sample['speech']
		# mel = batch[0]['mel_spectograms']
		# mel_resized = util.mel_resize(audio, mel)
		sample_size = self.sample_size
		encoded = util.mulaw_encode(audio, self.q_channels)
		input_batch, target_batch = [], []
		while sample_size < len(encoded):
			inputs = encoded[:sample_size]
			targets = inputs[self.receptive_fields:sample_size]
			ins = inputs.view(1, 1, -1)

			# yield melTor as well
			# melT = np.transpose(mel_resized[:sample_size])
			# melTor= torch.from_numpy(melT).float().unsqueeze(0)

			input_batch.append(ins)
			target_batch.append(targets)
			encoded = encoded[sample_size - self.receptive_fields:]

		return input_batch, target_batch

	def _collate_fn2(self, batch):
		"""
		Gets batches, pads the short audio data, embeds and returns.
		Args:
			batch ():

		Returns:

		"""
		print("collate call!")
		batch_size = len(batch)
		input_lengths = [len(x['speech']) for x in batch]
		max_input_length = max(input_lengths)
		input_batch = torch.zeros(0, 0).int()
		target_batch = torch.zeros(0, 0).int()
		for x in batch:
			encoded = self._prepare2(x['speech'], max_input_length)
			ins = encoded.view(1, 1, -1)
			targets = encoded[:-self.receptive_fields]
			#targets = targets.view(1, -1)
			input_batch = torch.cat((input_batch, ins))
			#input_batch.append(ins)
			target_batch = torch.cat((target_batch, targets))

		return input_batch, target_batch

	def _prepare2(self, audio, max_len):
		encoded = util.mulaw_encode(audio, self.q_channels)
		padded = F.pad(encoded, (0, max_len - len(encoded)), mode='constant')
		return padded
