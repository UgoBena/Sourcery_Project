'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''


from torch.utils.data import Dataset
import os

from preprocessing import CodeBERTaPreprocessor,ClassicPreprocessor

class MethodNamingDataset(Dataset):
    """PyTorch Dataset class for Method Naming"""

    def __init__(self, path_dataset,config,train_valid_test_set="train"):
        super(MethodNamingDataset, self).__init__()

        if config.model_type == "CodeBERTa":
            path_dataset = os.path.join(path_dataset,"codenet_data")
            preprocessor = CodeBERTaPreprocessor(path_dataset,config)
        else:
            path_dataset = os.path.join(path_dataset,"github_scrap_data")
            preprocessor = ClassicPreprocessor(path_dataset,config)
        
        #build vocabulary based on data given
        vocab = preprocessor.build_vocabulary()
        
        #build pairs based on require set (train, test or val)
        inputs_outputs_pairs = preprocessor.get_pairs(train_valid_test_set)

        self.vocabulary = vocab
        self.pairs = inputs_outputs_pairs
        self.n_examples = len(self.pairs)
    
    def __len__(self):
        r"""When used `len` return the number of examples.
        """

        return self.n_examples


    def __getitem__(self, item):
        r"""Given an index return a pair of input output
        """
        input,output = self.pairs[item]
        return (input,output,len(input),len(output))
    
    @classmethod
    def get_train_validation_sets(cls,path_dataset, config, get_only_val=False, **kwargs):
        """
        Returns train and validation set
        """
        if get_only_val:
            return cls(path_dataset,config, train_valid_test_set="valid")
        return cls(path_dataset,config, train_valid_test_set="train"), cls(path_dataset,config, train_valid_test_set="valid")
