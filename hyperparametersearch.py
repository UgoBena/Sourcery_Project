'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

import argparse

import optuna
from torch.utils.data import DataLoader
from torchtext.data import BucketIterator

from dataset import MethodNamingDataset
from utils import create_session
from models import CodeBERTaEncoderDecoder,AllamanisCNNModel,RNNSeq2SeqModel
from evaluation import evaluate_full_dataset
from training import train_and_validate


def main(args):
    """
    Runs optuna for hyperparameter optimization 
    """
    path_dataset, device, config = create_session(args)

    train_dataset, validation_dataset = MethodNamingDataset.get_train_validation_sets(path_dataset, config)
    vocabulary = train_dataset.vocabulary
    config.run_number = 0
    def objective(trial):
        config.run_number += 1

        config.batch_size = trial.suggest_categorical('batch_size', [8,16,32])
        if config.model_type == "CodeBERTa":
            config.learning_rate = trial.suggest_loguniform('learning_rate', 5e-7, 1e-4)
            model = CodeBERTaEncoderDecoder(device, config)
        else:
            config.learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            config.embedding_dim = trial.suggest_categorical('embedding_dim', [32,64,128,256])
            config.hidden_size = trial.suggest_categorical('hidden_size', [32,64,128,256])
            
            if config.model_type == "RNNEncoder":
                model = RNNSeq2SeqModel(device=device, config=config,vocabulary=vocabulary)
            elif config.model_type == "CNNEncoder":
                model = AllamanisCNNModel(device=device, config=config,vocabulary=vocabulary)

        train_dataloader,val_dataloader = BucketIterator.splits(
                        # Datasets for iterator to draw data from
                        (train_dataset,validation_dataset),
                        # Tuple of train and validation batch sizes.
                        batch_sizes=(config.batch_size,config.batch_size),
                        # Device to load batches on.
                        device=device, 
                        # Function to use for sorting examples.
                        sort_key=lambda x: x[2],
                        # Repeat the iterator for multiple epochs.
                        repeat=True, 
                        # Sort all examples in data using `sort_key`.
                        sort=False, 
                        # Shuffle data on each epoch run.
                        shuffle=True,
                        # Use `sort_key` to sort examples in each batch.
                        sort_within_batch=True,
                        )

        print(f"\nTrial config: {config}")
        
        return - train_and_validate(model, train_dataloader, val_dataloader, device, config)

    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n--- Finished trials ---\nBest params:\n{study.best_params}---\nBest accuracy:\n{-study.best_value}")
    save_json(config.path_result,"study",vars(study))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="data",
        help="The folder where all the data is stored (see dataset.py for further details on the formatting of this folder)")
    parser.add_argument("-s", "--dataset_size", type=str, default="small",choices=['small', 'large'],
        help="The size of the dataset to use (currently has no effect for models other than CodeBERTa)")
    parser.add_argument("-m", "--model_type", type=str, default="RNNEncoder", choices=['RNNEncoder', 'CNNEncoder', 'CodeBERTa'],
        help="Model type to use for training")
    parser.add_argument("-N", "--model_name", type=str, default="huggingface/CodeBERTa-small-v1",
        help="Model name to import pretrained weight from (only for CodeBERTa models)")
    parser.add_argument("-a", "--use_attention", type=bool, default="True",
        help="Whether to use attention layer or not (only for RNNEncoder model type)")
    parser.add_argument("-te", "--tie_embeddings", type=bool, default="True",
        help="Whether to use attention layer or not (only for RNNEncoder model type)")
    parser.add_argument("-n", "--n_trials", type=int, default=10, 
        help="number of trials")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=500, 
        help="prints training loss every k batch")
    parser.add_argument("-e", "--epochs", type=int, default=10, 
        help="number of epochs")
    parser.add_argument("-r", "--resume", type=str, default=None, 
        help="result folder in with the saved checkpoint will be reused")
    main(parser.parse_args())