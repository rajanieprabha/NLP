import numpy as np
import librosa
import torch
import src.hparam as hp
import os


hparameters = hp.load_params_from_yaml('/home/rajaniep/code/UntitledFolder/project/src/Tacotron_model/taco_hparams.yaml')

eos = '~'
pad = '_'
chars = pad + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ' + eos


char_to_id = {char: i for i, char in enumerate(chars)}
id_to_char = {i: char for i, char in enumerate(chars)}

def load_wav(filename):
    return librosa.load(filename, sr=hparameters.sample_rate)


def text_to_sequence(text, eos=eos):
    text += eos
    return [char_to_id[char] for char in text]


def sequence_to_text(sequence):
    return "".join(id_to_char[i] for i in sequence)


def ms_to_frames(ms, sample_rate):
    return int((ms / 1000) * sample_rate)


def wav_to_spectrogram(wav, sample_rate=hparameters.sample_rate,
                       fft_frame_size=hparameters.fft_frame_size,
                       fft_hop_size=hparameters.fft_hop_size,
                       num_mels=hparameters.num_mels,
                       min_freq=hparameters.min_freq,
                       max_freq=hparameters.max_freq,
                       floor_freq=hparameters.floor_freq):
    """
    Converts a wav file to a transposed db scale mel spectrogram.
    Args:
        wav:
        sample_rate:
        fft_frame_size:
        fft_hop_size:
        num_mels:
        min_freq:
        max_freq:
        floor_freq:

    Returns:

    """
    n_fft = ms_to_frames(fft_frame_size, sample_rate)
    hop_length = ms_to_frames(fft_hop_size, sample_rate)
    mel_spec = librosa.feature.melspectrogram(wav, sr=sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=num_mels,
                                              fmin=min_freq,
                                              fmax=max_freq)
    return librosa.power_to_db(mel_spec, ref=floor_freq).T


def collate_fn(batch):
    """
    Pads Variable length sequence to size of longest sequence.
    Args:
        batch:
    Returns: Padded sequences and original sizes.
    """
    text = [item[0] for item in batch]
    audio = [item[1] for item in batch]

    text_lengths = [len(x) for x in text]
    audio_lengths = [len(x) for x in audio]

    max_text = max(text_lengths)
    max_audio = max(audio_lengths)

    text_batch = np.stack(pad_text(x, max_text) for x in text)
    audio_batch = np.stack(pad_spectrogram(x, max_audio) for x in audio)

    return (torch.LongTensor(text_batch),
            torch.FloatTensor(audio_batch).permute(1, 0, 2),
            text_lengths, audio_lengths)


def pad_text(text, max_len):
    return np.pad(text, (0, max_len - len(text)), mode='constant', constant_values=hparameters.padding_idx)


def pad_spectrogram(S, max_len):
    padded = np.zeros((max_len, 80))
    padded[:len(S), :] = S
    return padded

def save_checkpoint(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        log.info("Log directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    #if is_best:
     #   shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
    Return:
        (dict): return saved state
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    state = torch.load(checkpoint)

    return state