import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
from Tacotron_model.visualize import show_spectrogram, show_attention
import Tacotron_model.taco_util as util
import Tacotron_model.util as utils
import numpy as np

from  Tacotron_model.util import sequence_to_text
import IPython.display

import logging
log = logging.getLogger(__name__)

def train(model, optimizer, dataset, hparams, saved_checkpoint, log_interval=5):
    log.info("TRAINING STARTS")
    print("TRAINING STARTS")
    print("Total epochs: ", hparams.num_epochs)

    model.train()
    writer = SummaryWriter()
    loader = util.fetch_dataloader(dataset,hparams)
    train_data = loader['train']
    train_loss = []
    step = 0
    total=hparams.num_epochs
    for epoch in range(total):
        log.info("epoch :" ,epoch)
        total_loss = 0
        pbar = tqdm(train_data, total=len(loader), unit=' batches')
        print("\n Epoch: ", epoch)

        for b, (text_batch, audio_batch, text_lengths, audio_lengths) in enumerate(pbar):
            print("\n Batch :", b)
            text = Variable(text_batch)
            targets = Variable(audio_batch, requires_grad=False)

             #  create stop targets
            stop_targets = torch.zeros(targets.size(1), targets.size(0))

            for i in range(len(stop_targets)):
                stop_targets[i, audio_lengths[i] - 1] = 1
            stop_targets = Variable(stop_targets, requires_grad=False)
            outputs, stop_tokens, attention = model(text, targets, hparams.teacher_forcing_ratio)
            spec_loss = F.mse_loss(outputs, targets)
            stop_loss = F.binary_cross_entropy_with_logits(stop_tokens, stop_targets)
            loss = spec_loss + stop_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
            train_loss.append(total_loss)
            pbar.set_description(f'Train loss: {loss.data[0]:.4f}')

            log.info('loss :', loss.data[0], ' and step :',step)
            writer.add_scalar('Train loss', loss.data[0], step)
            if step % log_interval ==0:
                
                  # plot the first sample in the batch
                temp = sequence_to_text(np.array(text.data[0]))
                attention_plot = show_attention(attention[0], return_array=True)

                output_plot = show_spectrogram(outputs.data.permute(1, 2, 0)[0],
                                              temp,
                                              return_array=True)
                target_plot = show_spectrogram(targets.data.permute(1, 2, 0)[0],
                                               temp,
                                               return_array=True)

                writer.add_image('Train attention', attention_plot, step)
                writer.add_image('Train target', target_plot, step)
                writer.add_image('Train output', output_plot, step)
                utils.save_checkpoint({'step': step,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict()},
                            checkpoint=saved_checkpoint)
                
                 
              
               
               
            step += 1


    return np.mean(train_loss), step


def val(model, dataset, hparams, log_interval=5):
    log.info("VALIDATION STARTS")
    print("VALIDATION STARTS")
    writer = SummaryWriter()
    loader = util.fetch_dataloader(dataset, hparams)
    val_data = loader['val']
    model.eval()
    step = 0
    val_loss = []
    for epoch in tqdm(range(hparams.num_epochs), total=hparams.num_epochs, unit=' epochs'):
        log.info("epoch: ",epoch)
        total_loss = 0
        pbar = tqdm(val_data, total=len(loader), unit=' batches')

        for b, (text_batch, audio_batch, text_lengths, audio_lengths) in enumerate(pbar):
            text = Variable(text_batch)
            targets = Variable(audio_batch, requires_grad=False)

            #  create stop targets
            stop_targets = torch.zeros(targets.size(1), targets.size(0))

            for i in range(len(stop_targets)):
                stop_targets[i, audio_lengths[i] - 1] = 1
            stop_targets = Variable(stop_targets, requires_grad=False)
            outputs, stop_tokens, attention = model(text)
            spec_loss = F.mse_loss(outputs, targets)
            stop_loss = F.binary_cross_entropy_with_logits(stop_tokens, stop_targets)
            loss = spec_loss + stop_loss
            total_loss += loss.data[0]
            val_loss.append(total_loss)
            pbar.set_description(f'Val loss: {loss.data[0]:.4f}')
            writer.add_scalar('Val loss', loss.data[0], step)
            if step % log_interval == 0:

                # plot the first sample in the batch
                attention_plot = show_attention(attention[0], return_array=True)
                
                output_plot = show_spectrogram(outputs.data.permute(1, 2, 0)[0],
                                               sequence_to_text(text.data[0]),
                                               return_array=True)
                target_plot = show_spectrogram(targets.data.permute(1, 2, 0)[0],
                                               sequence_to_text(text.data[0]),
                                               return_array=True)


                writer.add_image('Val attention', attention_plot, step)
                writer.add_image('Val target', output_plot, step)
                writer.add_image('Val output', target_plot, step)
              
            step += 1
            print("Step:", step ," Loss: ", total_loss)



    return np.mean(val_loss)


def train_and_evaluate(dataset, hparams, checkpoint, logdir):
    melnet = util.fetch_model(hparams)
    optimizer = util.fetch_optimizer(melnet, hparams)
    if checkpoint:
        state = utils.load_checkpoint(checkpoint)
        if hparams.resume:
            log.info('Resuming training from checkpoint: ',checkpoint)
            optimizer.load_state_dict(state['optim_dict'])
        log.info('Loading model from checkpoint: ',checkpoint)
        melnet.load_state_dict(state['state_dict'])

    best_metric = 0.0

    train_metric, step = train(melnet, optimizer, dataset, hparams,logdir, log_interval=5)
    val_metric = val(melnet, dataset, hparams, log_interval=5)
    is_best = False
    if val_metric >= best_metric:
        is_best = True
        best_metric = val_metric

    if is_best:
        print("gotcha", best_metric)
