'''
    Author: Ugo Benassayag
    Year: 2021
    Python Version: >= 3.7
'''

from tqdm import tqdm
import numpy as np
import time

def get_topK_metrics(tgt_seqs,topk_sequence,tgt_lens,topk_length):
    k = topk_length.size(1)
    batch_size = tgt_seqs.size(0)
    #get numpy arrays
    tgt_seqs = tgt_seqs.cpu().data.numpy()    
    topk_sequence = topk_sequence.cpu().data.numpy()
    topk_length = topk_length.cpu().data.numpy()
    tgt_lens = tgt_lens.cpu().numpy()
    
    #metrics to compute
    top1_f1 = 0
    top1_acc = 0
    topK_acc = 0
    topK_f1 = 0
    #loop: for each prediction, different pred_len and tgt_len make vectorized computation impossible
    for i in range(batch_size):
        tgt = tgt_seqs[i,1:tgt_lens[i].item()-1]
        best_acc = 0
        best_f1 = 0
        for j in range(k):
            pred = topk_sequence[i,j,1:topk_length[i,j]-1]

            tp = float((np.isin(pred,tgt)*1).sum())
            fp = float((np.isin(pred,tgt,invert=True)*1).sum())
            fn = float((np.isin(tgt,pred,invert=True)*1).sum())

            #Precision
            if (tp + fp != 0.): precision = tp/(tp + fp)
            else: precision = 0
            #Recall
            if (tp + fn != 0.): recall = tp/(tp + fn)
            else: recall = 0
            #Acc
            acc = (fp==0. and fn==0.) * 1.
            #F1
            if precision + recall != 0.:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.
            
            #record top1 value
            if j==0:
                top1_acc += acc
                top1_f1 += f1

            #keep best of K values
            if f1>best_f1:
                best_f1 = f1
            if acc>best_acc:
                best_acc = acc

        #add best values to topK metrics
        topK_acc += best_acc
        topK_f1 += best_f1
            

    #average values
    top1_acc /= batch_size
    top1_f1 /= batch_size
    topK_acc /= batch_size
    topK_f1 /= batch_size
    
    return top1_acc,top1_f1,topK_acc,topK_f1

def evaluate_full_dataset(val_dataloader,model):
    val_dataloader.create_batches()
    total_top1_acc = 0
    total_top1_f1 = 0
    total_topK_acc = 0
    total_topK_f1 = 0
    nb_eval = len(val_dataloader)
    for batch in tqdm(val_dataloader.batches):
        topk_sequence,topk_length,tgt_seqs,tgt_lens,output_prob,decoded_sequences = model.evaluate(batch)
        top1_acc,top1_f1,topK_acc,topK_f1 = get_topK_metrics(tgt_seqs,topk_sequence,tgt_lens,topk_length)

        total_top1_acc += top1_acc
        total_top1_f1 += top1_f1
        total_topK_acc += topK_acc
        total_topK_f1 += topK_f1

    #avg values
    total_top1_acc /= nb_eval
    total_top1_f1 /= nb_eval
    total_topK_acc /= nb_eval
    total_topK_f1 /= nb_eval

    
    return total_top1_acc,total_top1_f1,total_topK_acc,total_topK_f1