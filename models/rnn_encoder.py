'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, config):
        super(EncoderRNN, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.hidden_size = config.hidden_size
        self.embedding_dim = config.embedding_dim
        

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=config.pad_token_id)
        self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size)

    def forward(self, inputs, lengths, return_packed=False):
        """
        Inputs:
            inputs: (seq_length, batch_size), non-packed inputs
            lengths: (batch_size)
        """
        # [seq_length, batch_size, embedding_dim]
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths=lengths,enforce_sorted=False)
        outputs, hiddens = self.rnn(packed)
        if not return_packed:
            return pad_packed_sequence(outputs)[0], hiddens
        return outputs, hiddens
