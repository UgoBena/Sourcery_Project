'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import torch
import os
import warnings
import re
warnings.filterwarnings("ignore", category=UserWarning)

from .baseclass import PreprocessorInterface

def subTokenize(function_name,add_special_tokens=True):
    
    subtokens = function_name.split("_")
    if subtokens[0] == '': subtokens.pop(0)
    #Check if name contained underscore (snake_case)
    #Otherwise assume it's camelCase or PascalCase
    if len(subtokens) == 1:
        subtokens = re.findall('^[a-z]+|[A-Z][^A-Z]*',function_name)
    for i in range(len(subtokens)):
        subtokens[i] = subtokens[i].lower()
    #Add beginning and end of sequence special tokens
    if add_special_tokens:
        subtokens.append("</s>")
        subtokens.insert(0,"<s>")
    return subtokens


class ClassicPreprocessor(PreprocessorInterface):
    def __init__(self,path_dataset,config):
        """
        Preprocessor class for non transformers (huggingface) models
        """
        super(ClassicPreprocessor,self).__init__(path_dataset,config)
        self.path_dataset = path_dataset
        self.dataset_size = config.dataset_size
        self.config = config

        with open(os.path.join(self.path_dataset,"train.txt"),"r") as f:
            train_lines = f.readlines()
        with open(os.path.join(self.path_dataset,"valid.txt"),"r") as f:
            val_lines = f.readlines()
        with open(os.path.join(self.path_dataset,"test.txt"),"r") as f:
            test_lines = f.readlines()

        #TRAIN
        train_functions = []
        train_names = []
        for line in train_lines:
            file_path,function_name,has_docstring = tuple(line.split(","))
            #reformat file_path according to current os
            file_path = os.sep.join(file_path.split("/"))
            try:
                with open(os.path.join(self.path_dataset,file_path + ".py"),"r") as f:
                    train_functions.append(re.sub("FUNCTION_NAME_REPLACEMENT_TOKEN","<mask>",f.read(),count=1))
                train_names.append(function_name)
            except Exception as e:
                continue

        
        #VAL
        val_functions = []
        val_names = []
        for line in val_lines:
            file_path,function_name,has_docstring = tuple(line.split(","))
            file_path = os.sep.join(file_path.split("/"))
            try:
                with open(os.path.join(self.path_dataset,file_path + ".py"),"r") as f:
                    val_functions.append(re.sub("FUNCTION_NAME_REPLACEMENT_TOKEN","<mask>",f.read(),count=1))
                val_names.append(function_name)
            except Exception as e:
                continue

        
        test_functions = []
        test_names = []
        for line in test_lines:
            file_path,function_name,has_docstring = tuple(line.split(","))
            file_path = os.sep.join(file_path.split("/"))

            try:
                with open(os.path.join(self.path_dataset,file_path + ".py"),"r") as f:
                    test_functions.append(f.read())
                test_names.append(function_name)
            except Exception as e:
                continue
        
        self.train_cleaned_inputs = list(map(self.preprocess_input,train_functions))
        self.val_cleaned_inputs = list(map(self.preprocess_input,val_functions))
        self.test_cleaned_inputs = list(map(self.preprocess_input,test_functions))

        self.train_names_subtokens = list(map(subTokenize,train_names))
        self.val_names_subtokens = list(map(subTokenize,val_names))
        self.test_names_subtokens = list(map(subTokenize,test_names))


    def build_vocabulary(self):
        """
        Returns a vocabulary dictionnary whose keys are the tokens
        """
        vocab = {
            "<s>":self.config.bos_token_id,
            "<pad>":self.config.pad_token_id,
            "</s>":self.config.eos_token_id,
            "<mask>":self.config.mask_token_id
        }
        idx = 4
        for function in self.train_cleaned_inputs:
            tokens = function.split()
            for token in tokens:
                if token == "<s>" or token == "<pad>" or token == "</s>" or token =="<mask>":
                    continue
                subtokens = subTokenize(token,add_special_tokens=False)
                for name in subtokens:
                    if name not in vocab.keys():
                        vocab[name] = idx
                        idx += 1

        for function in self.val_cleaned_inputs:
            tokens = function.split()
            for token in tokens:
                if token == "<s>" or token == "<pad>" or token == "</s>" or token =="<mask>":
                    continue
                subtokens = subTokenize(token,add_special_tokens=False)
                for name in subtokens:
                    if name not in vocab.keys():
                        vocab[name] = idx
                        idx += 1
        
        for name in self.train_names_subtokens:
            for subtoken in name:
                if subtoken not in vocab.keys():
                    vocab[subtoken] = idx
                    idx += 1

        for name in self.val_names_subtokens:
            for subtoken in name:
                if subtoken not in vocab.keys():
                    vocab[subtoken] = idx
                    idx += 1

        self.vocabulary = vocab
        return vocab
    
    def preprocess_input(self,input_method):
        def replace(matchobj):
            return " " + matchobj.group(0) + " "

        return re.sub(r'\W',replace,input_method)
    
    def tokenize(self,text):
        tokenized_text = []
        split_text = text.split()
        for word in split_text:
            split_words = subTokenize(word,add_special_tokens=False)
            for split_word in split_words:
                if split_word in self.vocabulary.keys():
                    tokenized_text.append(self.vocabulary[split_word])
        return tokenized_text

    
    def tokenize_subtokens_list(self,subtokens_list):
        tokenized_text = []
        for subtoken in subtokens_list:
            if subtoken in self.vocabulary.keys():
                tokenized_text.append(self.vocabulary[subtoken])
        return tokenized_text
    


    def cut(self,tokenized_text):
        if len(tokenized_text) < self.config.max_input_len:
            return tokenized_text
        else:
            return tokenized_text[:self.config.max_input_len]


    def get_pairs(self,dataset_type="train"):
        """
        Returns the encoded pairs corresponding to the requested dataset (either train, val or test)
        """
        if dataset_type == "train":
            tokenized_inputs = [self.tokenize(function) for function in self.train_cleaned_inputs]
            tokenized_outputs = [self.tokenize_subtokens_list(function_name) for function_name in self.train_names_subtokens]
        elif dataset_type == "valid":
            tokenized_inputs = [self.tokenize(function) for function in self.val_cleaned_inputs]
            tokenized_outputs = [self.tokenize_subtokens_list(function_name) for function_name in self.val_names_subtokens]
        elif dataset_type == "test":
            tokenized_inputs = [self.tokenize(function) for function in self.test_cleaned_inputs]
            tokenized_outputs = [self.tokenize_subtokens_list(function_name) for function_name in self.test_names_subtokens]
        else:
            raise "Incorrect dataset type required"
        
        #Cut inputs and outputs max seq len
        tokenized_inputs = [self.cut(tokenized_text) for tokenized_text in tokenized_inputs]
        pairs = list(zip(tokenized_inputs,tokenized_outputs))

        pairs = [pair for pair in pairs if len(pair[1])<self.config.max_output_seq_len]

        return pairs



    
