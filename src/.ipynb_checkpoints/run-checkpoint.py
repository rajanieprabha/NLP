"""Main script. Run your experiments from here"""
from src.dataset import TextToSpeechDataset
from src.Tacotron_model.taco_util import fetch_model,fetch_optimizer
from torch.utils.data.dataset import Subset
import src.hparam as hp
import torch
import src.Tacotron_model.util as utils
from src.Tacotron_model.util import text_to_sequence, wav_to_spectrogram
from src.Tacotron_model.taco_train import train

import torch



def main():
    hparams = hp.load_params_from_yaml('/home/rajaniep/code/UntitledFolder/project/src/Tacotron_model/taco_hparams.yaml')
    #print(hparams)
    # set seed for reproducible experiments
    torch.manual_seed(hparams.seed)
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        torch.cuda.manual_seed(hparams.seed)
        torch.backends.cudnn.benchmark = True
        hparams.device = torch.device("cuda:0")
        hparams.cuda = True
    else:
        print("CUDA is not available. Using CPU.")
        hparams.device = torch.device("cpu")
        hparams.cuda = False


    

    PATH = '/home/rajaniep/code/UntitledFolder/project/en_US/by_book/female/judy_bieber/ozma_of_oz'
    dataset = TextToSpeechDataset(path = PATH,
                                  text_embeddings=text_to_sequence,
                                  mel_transforms=wav_to_spectrogram)
    logdir = '/home/rajaniep/code/UntitledFolder/project/logs'
    checkpoint = '/home/rajaniep/code/UntitledFolder/project/logs/last.pth.tar'
    #dataset = Subset(dataset, range(10))
    melnet = fetch_model(hparams)
    optimizer = fetch_optimizer(melnet,hparams)
    #melnet.cuda(device:0)
 #   melnet #=melnet.load_state_dict(torch.load('/home/rajaniep/code/UntitledFolder/runs/melnet.pt'))
    if checkpoint:
        state = utils.load_checkpoint(checkpoint)
        if hparams.resume:
            print('Resuming training from checkpoint: {}'.format(checkpoint))
            optimizer.load_state_dict(state['optim_dict'])
        print('Loading model from checkpoint: {}'.format(checkpoint))
        melnet.load_state_dict(state['state_dict'])

    
    train(melnet, optimizer, dataset, hparams,logdir)

if __name__ == "__main__":
    main()