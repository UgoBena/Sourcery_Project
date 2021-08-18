'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''
import time
import json
import os
from transformers import EncoderDecoderModel, RobertaTokenizer
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .baseclass import Seq2SeqModelInterface
from utils import printc


def set_dropout(model, drop_rate=0.1):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)

class CodeBERTaEncoderDecoder(Seq2SeqModelInterface):
    def __init__(self,config,device):
        """
        RoBERTa to RoBERTa encoder-decoder using HuggingFace pretrained CodeBERTa models trained on the CodeNet challenge dataset on LM tasks.
        Elements in training batch for this model should be tuples (inputs,labels,inputs_lengths,labels_lengths). 
        Inputs and labels do not need any padding.
        """
        super(CodeBERTaEncoderDecoder, self).__init__(device, config)
        self.config = config
        assert config.model_type == "CodeBERTa", 'Error: Wrong model type!'

        self.model_name = config.model_name
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.model_name, self.model_name).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        self.max_output_seq_len = config.max_output_seq_len
        self.learning_rate = config.learning_rate
        self.max_grad_norm = config.max_grad_norm

        self.optimizer = Adam(self.model.parameters(), lr = self.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer)

        if config.resume:
            self.resume(config)
        printc("Successfully loaded\n", "SUCCESS")

        self.drop_rate = config.drop_rate
        if self.drop_rate:
            set_dropout(self.model, drop_rate=self.drop_rate)
            print(f"Dropout rate set to {self.drop_rate}")
    
    def get_models_inputs_from_pair_batch(self,batch):
        batch_size = len(batch)
        unzipped = list(zip(*batch))
        inputs,targets,inputs_lengths,targets_lengths = unzipped[0],unzipped[1],unzipped[2],unzipped[3]

        PAD_token = self.tokenizer.pad_token_id

        #Build input tensor and pad
        inputs_lengths_tensor = torch.LongTensor(inputs_lengths)
        inputs_tensor = torch.ones(batch_size,inputs_lengths_tensor.max()).long() * PAD_token
        for idx, (seq, seqlen) in enumerate(zip(inputs, inputs_lengths_tensor)):
            inputs_tensor[idx,:seqlen] = torch.LongTensor(seq)

        inputs_attention_mask = (inputs_tensor != PAD_token) * 1

        #Build target tensor and pad
        targets_lengths_tensor = torch.LongTensor(targets_lengths)
        targets_tensor = torch.ones(batch_size,targets_lengths_tensor.max()).long() * PAD_token

        for idx, (seq, seqlen) in enumerate(zip(targets, targets_lengths_tensor)):
            targets_tensor[idx,:seqlen] = torch.LongTensor(seq)

        targets_attention_mask = (targets_tensor != PAD_token) * 1

        return (inputs_tensor, targets_tensor,targets_lengths_tensor,inputs_attention_mask,targets_attention_mask)

    def step(self, batch):
        """
        Args:
            batch: a batch of training data in the form described above in init
        Output:
            loss (tensor): PyTorch loss
            outputs (batch_size,seq_len,vocab_size): model outputs (predictions or something else)
        """
        #Unpack batch data
        src_seqs,tgt_seqs,tgt_lens,src_mask,tgt_mask = self.get_models_inputs_from_pair_batch(batch)
        
        src_seqs = src_seqs.to(self.device)
        tgt_seqs = tgt_seqs.to(self.device)

        tgt_lens = tgt_lens.to(self.device)

        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        # -------------------------------------
        # Training mode (enable dropout)
        # -------------------------------------
        self.model.train()    

        loss,outputs = self.forward(src_seqs,tgt_seqs,src_mask,tgt_mask)

        # -------------------------------------
        # Backward and optimize
        # -------------------------------------
        # Backward to get gradients w.r.t parameters in model.
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.max_grad_norm)
        
        # Update parameters with optimizer
        self.optimizer.step()

        return loss,outputs
            

    def forward(self,src_seqs,tgt_seqs,src_mask,tgt_mask):
        output = self.model(input_ids=src_seqs,decoder_input_ids=tgt_seqs,labels=tgt_seqs,encoder_attention_mask=src_mask,decoder_attention_mask=tgt_mask)
        return output.loss,output.logits

    def evaluate(self, eval_batch, max_seq_len=None,num_return_sequences=None,num_beams=5):
        """
        Args:
            eval_batch: batch data in the same form as train data (described in init)
            num_sequences: the number of sequences to output 
            max_seq_len: Maximum output sequence length
            num_beams: Number of beams for beam search. 
        Output:
            top_seqences(batch_size,num_sequences,max_seq_len): The top num_sequences predictions
            top_lengths(batch_size,num_sequences): The actual lengths of the top num_sequences predictions
            target_sequences(batch_size,batch_tgt_max_seq_len): The target sequences corresponding to the predicted ones for metrics computation
            target_lengths(batch_size): The actual lengths of the top target sequences
            decoded_sequences List[List[string] * num_sequences]*batch_size: The top num_sequences predictions decoded (as strings)
            outputs_probability (batch_size,num_sequences,max_seq_len - 1, vocab_size): model outputs passed through a softmax to turn into probabilities
        """
        if max_seq_len is None:
            max_seq_len = self.config.max_output_seq_len
        if num_return_sequences is None:
            num_return_sequences = self.config.num_return_sequences
        with torch.no_grad():
            batch_size = len(eval_batch)

            #Unpack batch data
            src_seqs,tgt_seqs,tgt_lens,_,_ = self.get_models_inputs_from_pair_batch(eval_batch)

            src_seqs = src_seqs.to(self.device)
            tgt_seqs = tgt_seqs.to(self.device)
            tgt_lens = tgt_lens.to(self.device)


            # -------------------------------------
            # Eval mode mode (disable dropout)
            # -------------------------------------
            self.model.eval()

            # -------------------------------------
            # Forward model
            # -------------------------------------
            start_beam_search = time.time()
            beam_output = self.model.generate(
                                src_seqs, 
                                max_length=self.max_output_seq_len, 
                                num_beams=num_beams, 
                                num_return_sequences=num_return_sequences, 
                                early_stopping=True,
                                output_scores = True,
                                return_dict_in_generate=True,
                                no_repeat_ngram_size = 1
                            )
            beam_search_time = time.time() - start_beam_search
            #top_sequence = (batch_size,num_sequences,max_seq_len)
            #top_length = (batch_size,num_sequences)
            top_sequence = beam_output["sequences"].view(batch_size,num_return_sequences,self.max_output_seq_len)
            # non zero values mask
            eos_mask = top_sequence == self.tokenizer.eos_token_id

            # operations on the mask to find first EOS_token in the rows
            mask_max_values, eos_index = torch.max(eos_mask, dim=2)
            # Actual length is one more than the index
            top_length = eos_index + 1

            # if the max-mask is zero, there is no pad index in the row, the length is the length of the sequence
            top_length[mask_max_values == 0] = top_sequence.size(2)

            #get output probabilites
            outputs = torch.stack(beam_output['scores']).transpose(0,1).view(batch_size,num_return_sequences,self.max_output_seq_len - 1,self.tokenizer.vocab_size)
            output_prob = torch.nn.functional.softmax(outputs,dim=3)

            #decode sequences
            decoded_sequences = [self.tokenizer.batch_decode(top_sequence[i],skip_special_tokens=True) for i in range(batch_size)]

            del outputs,src_seqs,eos_index,eos_mask,mask_max_values,beam_output

        return top_sequence,top_length,tgt_seqs,tgt_lens,output_prob,decoded_sequences

    def single_inference(self, function_string,num_return_sequences=None):
        """
        Args:
            function_string: raw text data (i.e a function extracted using ast)
            num_sequences: the number of sequences to output 
        Output:
            decoded_sequence [num_sequences]: The top num_sequences predictions decoded (as strings)
            sequence_scores [num_sequences]: Probability for each sequence
        """
        raise NotImplementedError