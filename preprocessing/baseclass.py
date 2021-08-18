'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''
class PreprocessorInterface():
    def __init__(self,path_dataset,config):
        """
        Preprocessor class to build vocabulary and tokenize
        """
        self.path_dataset = path_dataset
    
    def build_vocabulary():
        """
        Returns a vocabulary dictionnary whose keys are the tokens
        """
        raise NotImplementedError
    
    def get_pairs(dataset_type="train"):
        """
        Returns the encoded pairs corresponding to the requested dataset (either train, val or test)
        """
        raise NotImplementedError
    
