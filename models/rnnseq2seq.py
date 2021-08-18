'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import torch.nn as nn
import torch

import time

from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .rnn_encoder import EncoderRNN
from .rnn_decoder import DecoderRNNAttention,DecoderRNN
from .baseclass import Seq2SeqModelInterface




def sequence_mask(sequence_length, device,max_len=None):
    """
    Caution: Input and Return are VARIABLE.
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long().to(device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand).to(device)
    
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand
    
    return mask

def masked_cross_entropy(logits, target, length,device):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
        
    The code is same as:
    
    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    """
    logits=logits.to(device)
    target=target.to(device)
    length=length.to(device)
    # logits_flat: (batch * max_len, num_classes)
    
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat,dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1),device=device)
    # Note: mask need to bed casted to float!
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()    
    return loss

def _inflate(tensor, times, dim):
    # repeat_dims = [1] * tensor.dim()
    # repeat_dims[dim] = times
    # return tensor.repeat(*repeat_dims)
    return torch.repeat_interleave(tensor, times, dim)


class RNNSeq2SeqModel(Seq2SeqModelInterface):
    def __init__(self,config,device,vocabulary):
        """
        PyTorch Implementation of an RNN encoder and RNN decoder, with the option to add attention (Luang attention)
        """
        super(RNNSeq2SeqModel, self).__init__(device,config)
        assert config.model_type == "RNNEncoder", 'Error: Wrong model type!'

        self.device = device
        self.config = config

        self.vocabulary = vocabulary
        self.idx2word = {idx:w for (w,idx) in self.vocabulary.items()}
        self.V = len(vocabulary)
        self.encoder = EncoderRNN(config=config,vocab_size=self.V).to(device)
        if config.use_attention:
            self.decoder = DecoderRNNAttention(config,self.encoder).to(device)
        else:
            self.decoder = DecoderRNN(config,self.encoder).to(device)

        self.optimizer = Adam([p for p in self.encoder.parameters() if p.requires_grad] +
                       [p for p in self.decoder.parameters() if p.requires_grad],
                       lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer)
        
        self.max_output_len=config.max_output_seq_len
        self.teacher_forcing_ratio = config.tf_ratio
        self.max_grad_norm = config.max_grad_norm

        self.hidden_size = config.hidden_size
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        
    def get_models_inputs_from_pair_batch(self,batch):
        batch_size = len(batch)
        unzipped = list(zip(*batch))
        inputs,targets,inputs_lengths,targets_lengths = unzipped[0],unzipped[1],unzipped[2],unzipped[3]

        #Pad input sequences
        inputs_lengths_tensor = torch.LongTensor(inputs_lengths)
        inputs_tensor = Variable(torch.zeros(inputs_lengths_tensor.max(),batch_size)).long()

        for idx, (seq, seqlen) in enumerate(zip(inputs, inputs_lengths_tensor)):
            inputs_tensor[:seqlen,idx] = torch.LongTensor(seq)

        #do the same for the targets
        targets_lengths_tensor = torch.LongTensor(targets_lengths)
        targets_tensor = Variable(torch.zeros(targets_lengths_tensor.max(),batch_size)).long()

        for idx, (seq, seqlen) in enumerate(zip(targets, targets_lengths_tensor)):
            targets_tensor[:seqlen,idx] = torch.LongTensor(seq)
    

        return (inputs_tensor, targets_tensor,inputs_lengths_tensor,targets_lengths_tensor)
    def step(self, batch):
        """
        Args:
            batch: data from the data loaders (see training.py)
        Output:
            loss (tensor): PyTorch loss
            outputs (batch_size,seq_len,vocab_size): model outputs (raw predictions without softmax)
        
        Examples::
            >>> batch = next(iter(train_loader))
            >>> loss, outputs = model.step(batch)
        """
         #Unpack batch data
        src_seqs,tgt_seqs,src_lens,tgt_lens = self.get_models_inputs_from_pair_batch(batch)

        # -------------------------------------
        # Training mode (enable dropout)
        # -------------------------------------
        self.encoder.train()
        self.decoder.train()
        # -------------------------------------
        # Zero gradients, since optimizers will accumulate gradients for every backward.
        # -------------------------------------
        self.optimizer.zero_grad()

        # -------------------------------------
        # Forward model
        # -------------------------------------
        decoder_outputs = self.forward(src_seqs,tgt_seqs,src_lens,tgt_lens)
        
        # -------------------------------------
        # Compute loss
        # -------------------------------------
        loss = masked_cross_entropy(
            decoder_outputs.transpose(0,1).contiguous(), 
            tgt_seqs.transpose(0,1).contiguous(),
            tgt_lens,
            device=self.device
        )

        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
        
        # Update parameters with optimizer
        self.optimizer.step()

        return loss,decoder_outputs


    def forward(self, src_seqs,tgt_seqs,src_lens,tgt_lens):
         # Last batch might not have the same size as we set to the `batch_size`
        batch_size = src_seqs.size(1)
        assert(batch_size == tgt_seqs.size(1))

        # Pack tensors to variables for neural network inputs (in order to autograd)
        src_seqs = Variable(src_seqs).to(self.device)
        tgt_seqs = Variable(tgt_seqs).to(self.device)
        src_lens = Variable(src_lens).to(self.device)
        tgt_lens = Variable(tgt_lens).to(self.device)

        # Decoder's input
        input_seq = Variable(torch.LongTensor([self.bos_token_id] * batch_size)).to(self.device)

        # Decoder's output sequence length = max target sequence length of current batch.
        max_tgt_len = tgt_lens.data.max()

        # Store all decoder's outputs for loss computation
        decoder_outputs = Variable(torch.zeros((max_tgt_len, batch_size, self.V))).to(self.device)

        #Store actual predicted sequences lengths for metrics computation
        pred_lens = torch.ones(batch_size).to(self.device) * max_tgt_len
        pred_seq = torch.zeros((max_tgt_len,batch_size))

        # -------------------------------------
        # Forward encoder
        # -------------------------------------
        encoder_outputs, encoder_hidden = self.encoder(src_seqs, src_lens.data.tolist())

        # -------------------------------------
        # Forward decoder
        # -------------------------------------
        # Initialize decoder's hidden state as encoder's last hidden state.
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if np.random.random() < self.teacher_forcing_ratio else False
        # Run through decoder one time step at a time.
        for t in range(max_tgt_len):
            
            # decoder returns:
            # - decoder_output   : (batch_size, vocab_size)
            # - decoder_hidden   : (num_layers, batch_size, hidden_size)
            decoder_output, decoder_hidden = self.decoder(input_seq, decoder_hidden,encoder_outputs)

            # Store decoder outputs.
            decoder_outputs[t] = decoder_output
            val,predictions = decoder_output.data.topk(1)
            # Next input is current target if teacher forcing
            if use_teacher_forcing:
                input_seq = tgt_seqs[t]
            # Otherwise it's the current output most probable prediction
            else:
                input_seq = predictions.squeeze()

            pred_seq[t] = predictions.squeeze()

            finished_sequences = (input_seq.data == self.eos_token_id).nonzero(as_tuple=True)[0]
            currently_finished_sequences = finished_sequences[pred_lens[finished_sequences] == max_tgt_len] 
            pred_lens[currently_finished_sequences] = t + 1
            
        return decoder_outputs
    
    def _backtrack(self, nw_output,predecessors, symbols, scores, b,max_tgt_len,num_return_sequences):
        """Backtracks over batch to generate optimal k-sequences.

        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state

        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]

            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]

            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.

            score [batch, k]: A list containing the final scores for all top-k sequences

            length [batch, k]: A list specifying the length of each sequence in the top-k candidates

            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """
        start_time= time.time()
        

        # initialize return variables given different types
        p = list()
        output = list()
        l = torch.ones((b,num_return_sequences),dtype=torch.long).to(self.device) * max_tgt_len  # Placeholder for lengths of top-k sequences

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, num_return_sequences).topk(num_return_sequences)
        s = sorted_score.clone()

        batch_eos_found = [0] * b  # the number of EOS found
        # in the backward loop below for each batch

        t = max_tgt_len - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * num_return_sequences)
        start_loop = time.time()
        while t >= 0:
            
            current_symbol = symbols[t].index_select(0, t_predecessors)
            current_output = nw_output[t].index_select(0,t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()
            tricky_block_start = time.time()
            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.eos_token_id).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] // num_return_sequences)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = num_return_sequences - (batch_eos_found[b_idx] % num_return_sequences) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * num_return_sequences + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]                   
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    l[b_idx][res_k_idx] = t + 1
            tricky_block_time = time.time() - tricky_block_start
            # record the back tracked results
            p.append(current_symbol)
            output.append(current_output)
            t -= 1

        loop_time = time.time() - start_loop
        start_reverse = time.time()
        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        _, re_sorted_idx = s.topk(num_return_sequences)
        l = torch.gather(l,1,re_sorted_idx)

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * num_return_sequences)
        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        if b == 1:
            p = [step.index_select(0, re_sorted_idx).view(b, num_return_sequences, -1).squeeze().unsqueeze(0) for step in reversed(p)]
            output = [step.index_select(0, re_sorted_idx).view(b, num_return_sequences, -1).squeeze().unsqueeze(0) for step in reversed(output)]
        else :
            p = [step.index_select(0, re_sorted_idx).view(b, num_return_sequences, -1).squeeze() for step in reversed(p)]
            output = [step.index_select(0, re_sorted_idx).view(b, num_return_sequences, -1).squeeze() for step in reversed(output)]
        reverse_time = time.time() - start_reverse

        backtrack_time = time.time() - start_time


        # print(f"Total backtrack time: {backtrack_time}")
        # print(f"Part of loop computation : {loop_time/backtrack_time * 100}%")
        # print(f"Part of tricky block in loop computation : {tricky_block_time/loop_time * 100}%")
        # print(f"Part of reversing computation : {reverse_time/backtrack_time * 100}%")


        return l, p, output

    def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
        score[idx] = masking_score

    def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)


    def evaluate(self,eval_batch,max_seq_len=None,num_return_sequences=None):
        """
        Args:
            batch: data from the data loaders (similar to training data)
            num_sequences: the number of sequences to output 
        Output:
            top_seqences(batch_size,num_sequences,max_output_seq_len): The top num_sequences predictions
            top_lengths(batch_size,num_sequences): The actual lengths of the top num_sequences predictions
            target_sequences(batch_size,batch_tgt_max_seq_len): The target sequences corresponding to the predicted ones for metrics computation
            target_lengths(batch_size): The actual lengths of the top target sequences
            decoded_sequences List[List[string] * num_sequences]*batch_size: The top num_sequences predictions decoded (as strings)
            outputs_probability (batch_size,num_sequences,max_output_seq_len - 1, vocab_size): model outputs passed through a softmax to turn into probabilities
        """
        if max_seq_len is None:
            max_seq_len = self.config.max_output_seq_len
        if num_return_sequences is None:
            num_return_sequences = self.config.num_return_sequences
        with torch.no_grad():
            # Last batch might not have the same size as we set to the `batch_size`
            src_seqs,tgt_seqs,src_lens,tgt_lens = self.get_models_inputs_from_pair_batch(eval_batch)
            batch_size = src_seqs.size(1)

            # Pack tensors to variables for neural network inputs (in order to autograd)
            src_seqs = Variable(src_seqs).to(self.device)
            src_lens = Variable(src_lens).to(self.device)
            
            # -------------------------------------
            # Forward encoder
            # -------------------------------------
            encoder_outputs, encoder_hidden = self.encoder(src_seqs, src_lens.data.tolist())

            # ---------------------------------------------
            # Declare variables for beam search decoding
            # ---------------------------------------------
            
            self.pos_index = (torch.LongTensor(range(batch_size)) * num_return_sequences).view(-1, 1).to(self.device)

            # Initialize decoder's hidden state as encoder's last hidden state.
            # Inflate the initial hidden states to be of size: b*k x h
            decoder_hidden = _inflate(encoder_hidden,num_return_sequences,1,)

            # ... same idea for encoder_outputs
            inflated_encoder_outputs = _inflate(encoder_outputs, num_return_sequences, 1)

            # Initialize the scores; for the first step,
            # ignore the inflated copies to avoid duplicate entries in the top k
            sequence_scores = torch.Tensor(batch_size * num_return_sequences, 1)
            sequence_scores.fill_(-float('Inf'))
            sequence_scores.index_fill_(0, torch.LongTensor([i * num_return_sequences for i in range(0, batch_size)]), 0.0)
            sequence_scores = sequence_scores.to(self.device)

            # Initialize the input vector
            input_var = torch.transpose(torch.LongTensor([[self.bos_token_id] * batch_size * num_return_sequences]), 0, 1).to(self.device)

            # Store decisions for backtracking
            stored_scores = list()
            stored_predecessors = list()
            stored_emitted_symbols = list()
            stored_outputs = list()

            # Run through decoder one time step at a time.
            for t in range(self.max_output_len):
                # decoder returns:
                # - decoder_output   : (batch_size * k, vocab_size)
                # - decoder_hidden   : (num_layers, batch_size * k, hidden_size)
                # - attention_weights: (batch_size * k, max_src_len)
                decoder_output, decoder_hidden = self.decoder(input_var, decoder_hidden,inflated_encoder_outputs)
                
                log_softmax_output = nn.functional.log_softmax(decoder_output,dim=1)

                # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
                sequence_scores = _inflate(sequence_scores, self.V, 1)
                sequence_scores += log_softmax_output.squeeze(1)
                scores, candidates = sequence_scores.view(batch_size, -1).topk(num_return_sequences, dim=1)

                # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
                input_var = (candidates % self.V).view(batch_size * num_return_sequences, 1)
                sequence_scores = scores.view(batch_size * num_return_sequences, 1)

                # Update fields for next timestep
                predecessors = (candidates // self.V + self.pos_index.expand_as(candidates)).view(batch_size * num_return_sequences, 1)

                decoder_hidden = decoder_hidden.index_select(1, predecessors.squeeze())

                # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
                stored_scores.append(sequence_scores.clone())
                eos_indices = input_var.data.eq(self.eos_token_id)
                if eos_indices.nonzero().dim() > 0:
                    sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

                # Cache results for backtracking
                stored_predecessors.append(predecessors)
                stored_emitted_symbols.append(input_var)
                stored_outputs.append(decoder_output)

            # Do backtracking to return the optimal values
            l, p, outputs = self._backtrack(stored_outputs,stored_predecessors,stored_emitted_symbols,stored_scores, batch_size,self.max_output_len,num_return_sequences)

            # Build return objects
            topk_length = l
            topk_sequence = torch.stack(p).transpose(0,1).transpose(1,2)
            length = l[:,0]
            sequence = topk_sequence[:,0,:]
            outputs = torch.stack(outputs).transpose(0,1).transpose(1,2)
            output_prob = torch.nn.functional.softmax(outputs,dim=3)

            #decode sequences
            top_sequence_list = topk_sequence.cpu().numpy().tolist()
            decoded_sequences = []
            for i in range(batch_size):
                for k in range(num_return_sequences):
                    decoded_sequences.append(self.decode_sequence(top_sequence_list[i][k]))
        #tgt_seqs (max_seq_len,batch_size) => (batch_size,max_seq_len)
        return topk_sequence,topk_length,tgt_seqs.transpose(0,1),tgt_lens,output_prob,decoded_sequences

    def single_inference(self, function_string, num_sequences=1):
        """
        Args:
            function_string: raw text data (i.e a function extracted using ast)
            num_sequences: the number of sequences to output 
        Output:
            decoded_sequence [num_sequences]: The top num_sequences predictions decoded (as strings)
            sequence_scores [num_sequences]: Probability for each sequence
        
        Examples:
            >>> decoded_sequences,sequence_scores = model.single_inference(function_string,num_sequences=num_output_sequences)
        """
        raise NotImplementedError
    
    def decode_sequence(self,encoded_sequence):
        decoded_sequence = []
        for token in encoded_sequence:
            if token == self.bos_token_id:
                continue
            if token == self.eos_token_id:
                break
            decoded_sequence.append(self.idx2word[token])
        return token