'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

import numpy as np
import os
import json
import argparse
from time import time
from collections import defaultdict

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchtext.data import BucketIterator

from dataset import MethodNamingDataset
from utils import pretty_time, printc, create_session, save_json, now
from models import CodeBERTaEncoderDecoder,AllamanisCNNModel,RNNSeq2SeqModel
from evaluation import evaluate_full_dataset

import warnings
warnings.filterwarnings("ignore")

def train_and_validate(model, train_dataloader, val_dataloader, device, config):
    """
    train a model on the given data as loaders.
    Inputs: please refer bellow, to the argparse arguments.
    """
    printc("\n----- STARTING TRAINING -----")
    print(f"> config:\n{json.dumps(vars(config), sort_keys=True, indent=4)}\n")
    save_json(config.path_result, "confing", vars(config))


    losses = []
    val_top1_accuracies = []
    val_top1_f1_scores = []
    val_top5_accuracies = []
    val_top5_f1_scores = []
    best_topK_f1_score = 0

    if config.model_type == "CodeBERTa":
        model.initialize_scheduler(len(train_dataloader.dataset))
    for epoch in range(config.epochs):
        print("> EPOCH", epoch)
        model.train()
        epoch_loss, k_batch_loss = 0, 0
        epoch_start_time, k_batch_start_time = time(), time()
        train_dataloader.create_batches()
        #Training loop
        for i, batch in enumerate(train_dataloader.batches):

            loss, outputs = model.step(batch)

            epoch_loss += loss.item()
            k_batch_loss += loss.item()

            if (i+1) % config.print_every_k_batch == 0:
                average_loss = k_batch_loss / config.print_every_k_batch
                print(f'    [{i+1-config.print_every_k_batch}-{i+1}]  -  Average loss: {average_loss:.3f}  -  Time elapsed: {pretty_time(time()-k_batch_start_time)}')
                k_batch_loss = 0
                k_batch_start_time = time()

            
        #End of epoch
        printc("-----  Ended Train Epoch ---- Start of validation metrics computation  -----\n")
        val_top1_acc,val_top1_f1,val_topK_acc,val_topK_f1= evaluate_full_dataset(val_dataloader,model)
        print('\n' + '='*100)
        print('Training log:')
        print('- Epoch: {}/{}'.format(epoch, config.epochs))
        print('- Train loss: {}'.format(epoch_loss/len(train_dataloader)))
        print('- Val Top-1 Accuracy: {}'.format(val_top1_acc))
        print('- Val Top-1 F1 Score: {}'.format(val_top1_f1))
        print('- Val Top-K Accuracy: {}'.format(val_topK_acc))
        print('- Val Top-K F1 Score: {}'.format(val_topK_f1))
        print('='*100 + '\n')
        if best_topK_f1_score < val_topK_f1:
            best_topK_f1_score = val_topK_f1
            #run_number is set when doing hyper parameter optimization
            if config.run_number is not None:
                checkpoint_path = os.path.join(config.path_result,f'checkpoint_run_{config.run_number}.pth')
            else:
                checkpoint_path = os.path.join(config.path_result,'checkpoint.pth')
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_topK_f1_score': best_topK_f1_score,
                'optimizer': model.optimizer.state_dict(),
                'scheduler': model.scheduler.state_dict()
            }
            torch.save(checkpoint,checkpoint_path)
            
            print('\n' + '='*100)
            print('Saved checkpoint to "{}".'.format(checkpoint_path))
            print('Best top 5 F1-score value: ', best_topK_f1_score)
            print('='*100 + '\n')

        losses.append(epoch_loss/len(train_dataloader))
        val_top1_accuracies.append(val_top1_acc)
        val_top1_f1_scores.append(val_top1_f1)
        val_top5_accuracies.append(val_topK_acc)
        val_top5_f1_scores.append(val_topK_f1)

        if config.model_type == "CodeBERTa":
            model.scheduler.step()

    
    printc("-----  Ended Training  -----\n")

    print("Saving losses...")
    save_json(config.path_result, f"losses{config.run_number}", { "train": losses })
    print("Saving validation metrics")
    save_json(config.path_result, f"eval_metrics{config.run_number}", { "acc_1": val_top1_accuracies, "f1_score_1": val_top1_f1_scores,
                                            "acc_5":  val_top5_accuracies,"f1_score_5":  val_top5_f1_scores})
    epochs_realized = len(losses)
    #plot loss
    plt.plot(range(1, epochs_realized+1), losses)
    plt.legend(["Train loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss evolution")
    plt.savefig(os.path.join(config.path_result, f"loss{config.run_number}.png"))
    plt.close()

    #plot eval data
    plt.plot(range(1, epochs_realized+1), val_top5_accuracies)
    plt.legend(["Evaluation Top5 F1-score"])
    plt.xlabel("Epoch")
    plt.ylabel("Top5 F1-score")
    plt.title("Top5 F1-score Evolution")
    plt.savefig(os.path.join(config.path_result, f"eval_f1_score{config.run_number}.png"))
    plt.close()
    print("[DONE]")

    return best_topK_f1_score

from torch.utils.data import Dataset, DataLoader

# class FunctionNamingDataset(Dataset):
#     def __init__(self,data_pairs):
#         self.pairs = data_pairs
#         self.n_examples = len(self.pairs)
    
#     def __len__(self):
#         r"""When used `len` return the number of examples.
#         """

#         return self.n_examples


#     def __getitem__(self, item):
#         r"""Given an index return a pair of input output
#         """
#         input,output = self.pairs[item]
#         return (input,output,len(input),len(output))

def main(args):
    path_dataset, device, config = create_session(args)

    train_dataset, validation_dataset = MethodNamingDataset.get_train_validation_sets(path_dataset,config)
    vocabulary = train_dataset.vocabulary
    # validation_dataset = MethodNamingDataset.get_train_validation_sets(path_dataset,config,get_only_val=True)
    # dummy_pairs = [([1,2,3,4],[0,4,2])]*16
    # train_dataset = FunctionNamingDataset(dummy_pairs)
    # #validation_dataset = FunctionNamingDataset(dummy_pairs)
    # vocabulary = {
    #         "<s>":config.bos_token_id,
    #         "<pad>":config.pad_token_id,
    #         "</s>":config.eos_token_id,
    #         "<mask>":config.mask_token_id,
    #         "blabla":4
    #     }
    # vocabulary = validation_dataset.vocabulary
    
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
    if config.model_type == "CodeBERTa":
        model = CodeBERTaEncoderDecoder(config=config,device=device)
    else:
        model = RNNSeq2SeqModel(config=config,device=device,vocabulary=vocabulary)

    train_and_validate(model, train_dataloader, val_dataloader, device, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="RNNEncoder", choices=['RNNEncoder', 'CNNEncoder', 'CodeBERTa'],
        help="Model type to use for training")
    parser.add_argument("-N", "--model_name", type=str, default="huggingface/CodeBERTa-small-v1",
        help="Model name to import pretrained weight from (only for CodeBERTa models)")
    parser.add_argument("-a", "--use_attention", type=bool, default="True",
        help="Whether to use attention layer or not (only for RNNEncoder model type)")
    parser.add_argument("-d", "--data_folder", type=str, default="data",
        help="The folder where all the data is stored (see dataset.py for further details on the formatting of this folder)")
    parser.add_argument("-s", "--dataset_size", type=str, default="small",choices=['small', 'large'],
        help="The size of the dataset to use (currently has no effect for models other than CodeBERTa)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, 
        help="dataset batch size")
    parser.add_argument("-e", "--epochs", type=int, default=20, 
        help="number of epochs")
    parser.add_argument("-drop", "--drop_rate", type=float, default=.1, 
        help="dropout ratio. By default, uses p=0.1")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=100, 
        help="prints training loss every k batch")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, 
        help="model learning rate")
    parser.add_argument("-wg", "--weight_decay", type=float, default=0, 
        help="the weight decay for L2 regularization")
    parser.add_argument("-r", "--resume", type=str, default=None, 
        help="result folder in which the saved checkpoint will be reused")

    main(parser.parse_args())