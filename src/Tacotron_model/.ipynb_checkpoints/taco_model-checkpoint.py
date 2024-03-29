import random

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class LocationAttention(nn.Module):
    """
    Calculates context vector based on previous decoder hidden state (query vector),
    encoder output features, and convolutional features extracted from previous attention weights.
    Attention-Based Models for Speech Recognition
    https://arxiv.org/pdf/1506.07503.pdf
    Query vector is either the previous output or the last decoder hidden state.
    """

    def __init__(self, dim, num_location_features=32):
        super(LocationAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=num_location_features,
                              kernel_size=31, padding=15)
        self.W = nn.Linear(dim, dim, bias=False)
        self.L = nn.Linear(num_location_features, dim, bias=False)

    def score(self, query_vector, encoder_out, mask=None):
        # linear transform encoder out (seq, batch, dim)
        encoder_out = self.W(encoder_out)
        # (batch, seq, dim) | (2, 15, 50)
        encoder_out = encoder_out.permute(1, 0, 2)
        if isinstance(mask, Variable):
            conv_features = self.conv(mask.permute(0, 2, 1))  # (batch, dim , seq)
            encoder_out = encoder_out + self.L(conv_features.permute(0, 2, 1))  # (batch, seq , dim)
        # (2, 15, 50) @ (2, 50, 1)
        return encoder_out @ query_vector.permute(1, 2, 0)

    def forward(self, query_vector, encoder_out, mask=None):
        energies = self.score(query_vector, encoder_out)
        mask = F.softmax(energies, dim=1)
        context = encoder_out.permute(
            1, 2, 0) @ mask  # (batch, dim, seq) @ (batch, seq, dim)
        context = context.permute(2, 0, 1)  # (seq, batch, dim)
        return context, mask
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return self.dropout(x)


class ConvTanhBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvTanhBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch = nn.BatchNorm1d(out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        return self.tanh(x)


class PreNet(nn.Module):
    """
    Extracts 256d features from 80d input spectrogram frame
    """

    def __init__(self, in_features=80, out_features=256, dropout=0):
        super(PreNet, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, previous_y):
        x = self.relu(self.fc1(previous_y))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return x


class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()
        self.conv1 = ConvTanhBlock(in_channels=1, out_channels=512, kernel_size=5, padding=2)
        self.conv2 = ConvTanhBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.conv3 = ConvTanhBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.conv4 = ConvTanhBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)


class Encoder(nn.Module):

    def __init__(self, num_chars):
        super(Encoder, self).__init__()
        self.char_embedding = nn.Embedding(num_embeddings=num_chars,
                                           embedding_dim=512, padding_idx=0)

        self.conv1 = ConvBlock(512, 512, 5, 2)
        self.conv2 = ConvBlock(512, 512, 5, 2)
        self.conv3 = ConvBlock(512, 512, 5, 2)
        self.birnn = nn.GRU(input_size=512, hidden_size=256, bidirectional=True, dropout=0.1)

    def forward(self, text):
        # input - (batch, maxseqlen) | (4, 156)
        # print(text.shape)
        x = self.char_embedding(text)  # (batch, seqlen, embdim) | (4, 156, 512)
        x = x.permute(0, 2, 1)  # swap to batch, channel, seqlen (4, 512, 156)
        x = self.conv1(x)  # (4, 512, 156)
        # print(x.shape)
        x = self.conv2(x)  # (4, 512, 156)
        x = self.conv3(x)  # (4, 512, 156)
        x = x.permute(2, 0, 1)  # swap seq, batch, dim for rnn | (156, 4, 512)
        x, hidden = self.birnn(x)  # (156, 4, 512) | 256 dims in either direction
        # sum bidirectional outputs
        x = (x[:, :, :256] + x[:, :, 256:])
        print(x.shape)
        return x, hidden


class Decoder(nn.Module):
    """
    Decodes encoder output and previous predicted spectrogram frame into next spectrogram frame.
    """

    def __init__(self, hidden_size=1024, num_layers=2):
        super(Decoder, self).__init__()
        self.prenet = PreNet(in_features=80, out_features=256)
        self.attention = LocationAttention(dim=256)
        self.rnn = nn.GRU(input_size=512, hidden_size=hidden_size, num_layers=num_layers, dropout=0.1)
        self.spec_out = nn.Linear(in_features=1024 + 256, out_features=80)
        self.stop_out = nn.Linear(in_features=1024 + 256, out_features=1)
        self.postnet = PostNet()

    def _forward(self, previous_out, encoder_out, decoder_hidden=None, mask=None):
        """
        Decodes a single frame
        """
        previous_out = self.prenet(previous_out)  # (4, 1, 256)
        context, mask = self.attention(previous_out, encoder_out, mask)
        rnn_input = torch.cat([previous_out, context], dim=2)
        rnn_out, hidden = self.rnn(rnn_input, decoder_hidden)
        spec_frame = self.spec_out(torch.cat([rnn_out, context], dim=2))  # predict next audio frame
        stop_token = self.stop_out(torch.cat([rnn_out, context], dim=2))  # predict stop token
        spec_frame = spec_frame.permute(1, 0, 2)
        spec_frame = spec_frame + self.postnet(spec_frame)  # add residual
        return spec_frame.permute(1, 0, 2), stop_token, decoder_hidden, mask

    def forward(self, encoder_out, targets, teacher_forcing_ratio=0.5):
        outputs = []
        stop_tokens = []
        masks = []

        start_token = torch.zeros_like(targets[:1])
        output, stop_token, hidden, mask = self._forward(start_token,
                                                         encoder_out)
        for t in range(len(targets)):
            output, stop, hidden, mask = self._forward(output,
                                                       encoder_out,
                                                       hidden,
                                                       mask.detach())
            outputs.append(output)
            stop_tokens.append(stop)
            masks.append(mask.data.permute(2, 0, 1))
            teacher = random.random() < teacher_forcing_ratio
            if teacher:
                output = targets[t].unsqueeze(0)

        outputs = torch.cat(outputs)
        stop_tokens = torch.cat(stop_tokens)
        masks = torch.cat(masks)

        stop_tokens = stop_tokens.transpose(1, 0).squeeze()
        if len(stop_tokens.size()) == 1:
            stop_tokens = stop_tokens.unsqueeze(0)

        return outputs, stop_tokens, masks.permute(1, 2, 0)


class MelSpectrogramNet(nn.Module):

    def __init__(self, num_chars, teacher_forcing_ratio):
        super(MelSpectrogramNet, self).__init__()

        self.encoder = Encoder(num_chars=num_chars)
        self.decoder = Decoder()

        #self.num_chars = num_chars

    def forward(self, text, targets, teacher_forcing_ratio):
        encoder_output, _ = self.encoder(text)
        outputs, stop_tokens, masks = self.decoder(encoder_output,
                                                   targets,teacher_forcing_ratio =teacher_forcing_ratio)
        return outputs, stop_tokens, masks



