'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

from transformers import RobertaTokenizer
import json
import os
import re
import ast

from .baseclass import PreprocessorInterface

class CodeBERTaPreprocessor(PreprocessorInterface):
    def __init__(self,path_dataset,config):
        """
        Preprocessor class for transformers (huggingface) models
        """
        super(CodeBERTaPreprocessor,self).__init__(path_dataset,config)
        self.path_dataset = path_dataset
        self.dataset_size = config.dataset_size
        self.tokenizer = RobertaTokenizer.from_pretrained(config.model_name)

    def build_vocabulary(self):
        """
        Returns a vocabulary dictionnary whose keys are the tokens
        """
        self.vocabulary = self.tokenizer.get_vocab()

        return self.vocabulary
    
    def get_func_and_name(self,data):
        try:
            node = ast.parse(data).body[0]
            function_name = node.name
            function = data
            docstring = ast.get_docstring(node)
            #remove docstring
            if docstring is not None:
                function = re.sub(r'\"\"\"(.*)\"\"\"',"",function,count=1,flags=re.DOTALL)
            #remove function name
            function = re.sub(function_name,"<mask>",function,count=1)
            return function,function_name
        except:
            return None
    
    def get_pairs(self,dataset_type="train"):
        """
        Returns the encoded pairs corresponding to the requested dataset (either train, val or test)
        """
        if dataset_type == "train":
            if self.dataset_size == "small":
                nb_files = 1
            else:
                nb_files = 14
        else:
            nb_files = 1
        jsons = []
        for i in range(nb_files):
            with open(os.path.join(self.path_dataset,dataset_type,f"python_{dataset_type}_{i}.jsonl")) as f:
                jsonl_content = f.readlines()
                jsons += [json.loads(json_line) for json_line in jsonl_content]
        
        pairs_raw = [self.get_func_and_name(line["code"]) for line in jsons if self.get_func_and_name(line["code"]) is not None]

        inputs_raw = [x for (x,y) in pairs_raw]
        labels_raw = [y for (x,y) in pairs_raw]

        inputs = self.tokenizer.batch_encode_plus(inputs_raw)["input_ids"]
        labels = self.tokenizer.batch_encode_plus(labels_raw)["input_ids"]

        #Remove underscore tokens from inputs and labels, and truncate up to max model length
        underscore_token = self.vocabulary["_"]

        inputs = [[token for token in single_input if token != underscore_token][:self.tokenizer.model_max_length] for single_input in inputs]
        labels = [[token for token in single_label if token != underscore_token] for single_label in labels]

        pairs = list(zip(inputs,labels))

        return pairs
    
