"""Main script. Run your experiments from here"""
from dataset import TextToSpeechDataset
from Tacotron_model.taco_util import fetch_model,fetch_optimizer
from torch.utils.data.dataset import Subset
import hparam as hp
import torch
import Tacotron_model.util as utils
from Tacotron_model.util import text_to_sequence, wav_to_spectrogram
from Tacotron_model.taco_train import train_and_evaluate

import torch
import argparse
import os



import logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger()


def parse_args():
    """Parse command line arguments.
    Returns:
		(Namespace): arguments
	"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logdir', type=str,
                        default='log/tacotron',
						help='parent directory of experiment logs (checkpoints/tensorboard events)')
    parser.add_argument('--log_level', type=str, default='WARNING',
						help='log level to be used')
    parser.add_argument('-n', '--name', type=str, default='Training-naive',
						help='name of experiment')
    parser.add_argument('-c', '--checkpoint', type=str, default='/home/rajaniep/code/UntitledFolder/project/src/log/tacotron/Training-naive/german_last.pth.tar/last.pth.tar',
						required=False, help='path to checkpoint')
    parser.add_argument('--default_hparams', type=str,
						default='Tacotron_model/taco_hparams.yaml', help='path to .yaml with default hparams')
    parser.add_argument('--hparams', type=str,
						required=False, help='comma separated name=value pairs')
    parser.add_argument('--data', type=str, default='/home/rajaniep/code/UntitledFolder/project/de_DE/by_book/female/angela_merkel/merkel_alone',
						help='csv file of texts and audio names in LLJDS Format')
    return parser.parse_args()

def main():
    args = parse_args()
    logdir = args.logdir
    checkpoint = args.checkpoint
    # set log level
    assert (args.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    log.setLevel(logging.getLevelName(args.log_level))
    # make sure log directory exists
    logdir = os.path.join(args.logdir, args.name)
    if not os.path.isdir(logdir):
        log.info("Creating directory {}".format(logdir))
        os.makedirs(logdir)
        os.chmod(logdir, 0o775)
    hparams = hp.load_params_from_yaml(args.default_hparams)
    hparams.parse(args.hparams)
    hp.write_params_to_yaml(hparams, os.path.join(logdir, 'hparams.yaml'))
    #print(hparams)


    # set seed for reproducible experiments
    torch.manual_seed(hparams.seed)
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        log.info("CUDA is available. Using GPU.")
        torch.cuda.manual_seed(hparams.seed)
        torch.backends.cudnn.benchmark = True
        hparams.device = torch.device("cuda:0")
        hparams.cuda = True
    else:
        log.info("CUDA is not available. Using CPU.")
        print("CUDA is not available. Using CPU.")
        hparams.device = torch.device("cpu")
        hparams.cuda = False


    

    PATH = args.data
    dataset = TextToSpeechDataset(path = PATH,
                                  text_embeddings=text_to_sequence,
                                  mel_transforms=wav_to_spectrogram)
    
    #print(dataset[0])
   
    
  

    train_and_evaluate(dataset, hparams, checkpoint,logdir)
if __name__ == "__main__":
    main()
