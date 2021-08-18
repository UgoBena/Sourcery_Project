'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, config,encoder):
        """ General attention in `Effective Approaches to Attention-based Neural Machine Translation`
            Ref: https://arxiv.org/abs/1508.04025
            
            Share input and output embeddings:
            Ref:
                - "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                   https://arxiv.org/abs/1608.05859
                - "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                   https://arxiv.org/abs/1611.01462
        """
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = encoder.hidden_size
        self.tie_embeddings = config.tie_embeddings
        
        self.vocab_size = encoder.vocab_size
        self.embedding_dim = config.embedding_dim
        
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding.weight = encoder.embedding.weight
        self.rnn = nn.GRU(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size)
        
        if self.tie_embeddings:
            self.W_proj = nn.Linear(self.hidden_size, self.embedding_dim, bias=bias)
            self.W_s = nn.Linear(self.embedding_dim, self.vocab_size, bias=bias)
            self.W_s.weight = self.embedding.weight
        else:
            self.W_s = nn.Linear(self.hidden_size, self.vocab_size, bias=bias)
        
    def forward(self, input_seq, decoder_hidden, encoder_outputs):
        """ Args:
            - input_seq      : (batch_size,seq_len=1)
            - decoder_hidden : (t=0) last encoder hidden state (num_layers * num_directions, batch_size, hidden_size) 
                               (t>0) previous decoder hidden state (num_layers, batch_size, hidden_size)
            - encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)
        
            Returns:
            - output           : (batch_size, vocab_size)
            - decoder_hidden   : (num_layers, batch_size, hidden_size)
            - attention_weights: (batch_size, max_src_len)
        """        
        # (batch_size,seq_len=1) => (seq_len=1, batch_size)
        if len(input_seq.size())>1:
            input_seq = input_seq.transpose(0,1)
        # (batch_size) => (seq_len=1, batch_size)
        else:
            input_seq = input_seq.unsqueeze(0)
        
        # (seq_len=1, batch_size) => (seq_len=1, batch_size, word_vec_size) 
        emb = self.embedding(input_seq)
        
        # rnn returns:
        # - decoder_output: (seq_len=1, batch_size, hidden_size)
        # - decoder_hidden: (num_layers, batch_size, hidden_size)
        decoder_output, decoder_hidden = self.rnn(emb, decoder_hidden)

        # (seq_len=1, batch_size, hidden_size) => (batch_size, seq_len=1, hidden_size)
        decoder_output = decoder_output.transpose(0,1)
        
        # If input and output embeddings are tied,
        # project `decoder_hidden_size` to `word_vec_size`.
        if self.tie_embeddings:
            output = self.W_s(self.W_proj(decoder_output))
        else:
            # (batch_size, seq_len=1, decoder_hidden_size) => (batch_size, seq_len=1, vocab_size)
            output = self.W_s(decoder_output)    
        
        # Prepare returns:
        # (batch_size, seq_len=1, vocab_size) => (batch_size, vocab_size)
        output = output.squeeze(1)
                
        return output, decoder_hidden

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
 
        # The input dimension will the the concatenation of
        # encoder_hidden_dim (hidden) and  decoder_hidden_dim(encoder_outputs)
        self.attn_hidden_vector = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
 
        # We need source len number of values for n batch as the dimension
        # of the attention weights. The attn_hidden_vector will have the
        # dimension of [source len, batch size, decoder hidden dim]
        # If we set the output dim of this Linear layer to 1 then the
        # effective output dimension will be [source len, batch size]
        self.attn_scoring_fn = nn.Linear(decoder_hidden_dim, 1, bias=False)
 
    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, decoder hidden dim]
        src_len = encoder_outputs.shape[0]
 
        # We need to calculate the attn_hidden for each source words.
        # Instead of repeating this using a loop, we can duplicate
        # hidden src_len number of times and perform the operations.
        hidden = hidden.repeat(src_len, 1, 1)
 
        # Calculate Attention Hidden values
        attn_hidden = torch.tanh(self.attn_hidden_vector(torch.cat((hidden, encoder_outputs), dim=2)))
 
        # Calculate the Scoring function. Remove 3rd dimension.
        attn_scoring_vector = self.attn_scoring_fn(attn_hidden).squeeze(2)
 
        # The attn_scoring_vector has dimension of [source len, batch size]
        # Since we need to calculate the softmax per record in the batch
        # we will switch the dimension to [batch size,source len]
        attn_scoring_vector = attn_scoring_vector.permute(1, 0)
 
        # Softmax function for normalizing the weights to
        # probability distribution
        return F.softmax(attn_scoring_vector, dim=1)

class DecoderRNNAttention(nn.Module):
    def __init__(self, config,encoder):
        """ General attention in `Effective Approaches to Attention-based Neural Machine Translation`
            Ref: https://arxiv.org/abs/1508.04025
            
            Share input and output embeddings:
            Ref:
                - "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                   https://arxiv.org/abs/1608.05859
                - "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                   https://arxiv.org/abs/1611.01462
        """
        super(DecoderRNNAttention, self).__init__()
        
        self.hidden_size = encoder.hidden_size
        self.vocab_size = encoder.vocab_size
        self.embedding_dim = config.embedding_dim

        self.attention = Attention(encoder.hidden_size,self.hidden_size)
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.GRU(input_size=self.embedding_dim + encoder.hidden_size,hidden_size=self.hidden_size)
        self.fc = nn.Linear(encoder.hidden_size + self.hidden_size + self.embedding_dim, self.embedding_dim)
        self.output_layer = nn.Linear(self.embedding_dim, self.vocab_size)
        self.output_layer.weight = self.embedding.weight

        self.dropout = nn.Dropout(config.drop_rate)
        
    def forward(self, input_seq, decoder_hidden, encoder_outputs):
        """ Args:
            - input_seq      : (batch_size,seq_len=1)
            - decoder_hidden : (t=0) last encoder hidden state (num_layers * num_directions, batch_size, hidden_size) 
                               (t>0) previous decoder hidden state (num_layers, batch_size, hidden_size)
            - encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)
        
            Returns:
            - output           : (batch_size, vocab_size)
            - decoder_hidden   : (num_layers, batch_size, hidden_size)
            - attention_weights: (batch_size, max_src_len)
        """        
        # (batch_size,seq_len=1) => (seq_len=1, batch_size)
        if len(input_seq.size())>1:
            input_seq = input_seq.transpose(0,1)
        # (batch_size) => (seq_len=1, batch_size)
        else:
            input_seq = input_seq.unsqueeze(0)
        
        # (seq_len=1, batch_size) => (seq_len=1, batch_size, word_vec_size) 
        embedded = self.dropout(self.embedding(input_seq))

        # Calculate the attention weights
        a = self.attention(decoder_hidden, encoder_outputs).unsqueeze(1)
 
        # We need to perform the batch wise dot product.
        # Hence need to shift the batch dimension to the front.
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
 
        # Use PyTorch's bmm function to calculate the weight W.
        W = torch.bmm(a, encoder_outputs)
 
        # Revert the batch dimension.
        W = W.permute(1, 0, 2)
 
        # concatenate the previous output with W
        rnn_input = torch.cat((embedded, W), dim=2)
 
        rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
 
        # Remove the sentence length dimension and pass them to the Linear layer
        output = self.output_layer(self.fc(torch.cat((rnn_output.squeeze(0), W.squeeze(0), embedded.squeeze(0)), dim=1)))

        return output,decoder_hidden