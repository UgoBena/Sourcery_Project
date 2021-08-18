import json

class Config:
    """
    Stores useful information
    Many scripts uses a config instance. See utils.create_session for its initialization
    """
    
    model_type = "RNNEncoder" #CodeBERTa, RNNEncoder or CNNEncoder
    model_name = "huggingface/CodeBERTa-small-v1"
    use_attention = False # use attention for RNNEncoder model
    tie_embeddings = False # do tie embeddings between encoders and decoders (for RNNEncoder and CNNEncoder models)

    dataset_size = "small"
    data_folder = "data"

    print_every_k_batch = 8
    batch_size = 1
    learning_rate = 1e-5
    embedding_dim = 128
    hidden_size = 128
    epochs = 10
    weight_decay = 0
    drop_rate = .1
    tf_ratio = .5  #teacher forcing ration
    max_grad_norm = 2. #max norm for gradient clipping

    path_result = None
    resume = None
    run_number = None #For hyperparameter search

    num_return_sequences = 5 #number of sequences to return when making prediction with the models
    max_output_seq_len = 8
    max_input_len = 1000
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    mask_token_id = 3

    def __init__(self, args={}):
        for attr in dir(self):
            if not attr.startswith('__') and hasattr(args, attr):
                setattr(self, attr, getattr(args, attr))
            elif not attr.startswith('__'):
                setattr(self, attr, getattr(self, attr))

    def __repr__(self):
        return json.dumps(vars(self), sort_keys=True, indent=4)