'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import torch.nn as nn
import torch
from transformers import get_linear_schedule_with_warmup
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils import printc
class Seq2SeqModelInterface(torch.nn.Module):
    def __init__(self,device,config):
        """
        PyTorch Seq2SeqModel interface
        Every model has to inherit from Seq2SeqModelInterface so training and testing run correctly

        At least the methods defined below and which raise NotImplementedError must be implemented
        - self.optimizer
        - self.scheduler
        """
        super(Seq2SeqModelInterface, self).__init__()
        self.device = device
        self.config = config
        
    def initialize_scheduler(self, total_steps=0):
        """
        Creates a scheduler for a given otimizer
        """
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps=2, # Default value
                                                        num_training_steps=total_steps)
    def resume(self, config):
        """
        Resumes with a given checkpoint. Loads the saved parameters, optimizer and scheduler.
        """
        printc(f"Resuming with model at {config.resume}...", "INFO")
        path_checkpoint = os.path.join(config.resume, 'checkpoint.pth')
        assert os.path.isfile(path_checkpoint), 'Error: no checkpoint found!'
        checkpoint = torch.load(path_checkpoint, map_location=self.device)

        printc(f"Previous model had best val top5 f1 score of {checkpoint['best_topK_f1_score']} at epoch {checkpoint['epoch']}", "INFO")
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        printc("Model loaded !")

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
        raise NotImplementedError

    def forward(self,*args, **kwargs):
        """
        PyTorch nn.Module forward
        It is specific to the model, and the args have no specific format
        """
        raise NotImplementedError

    def evaluate(self, batch, num_sequences=1):
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
        
        Examples::
            >>> batch = next(iter(eval_loader))
            >>> (top_seqences,top_lengths,target_sequences, target_lengths, 
                decoded_sequences,outputs_probability) = model.evaluate(batch,num_sequences=num_output_sequences)
        """
        raise NotImplementedError

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