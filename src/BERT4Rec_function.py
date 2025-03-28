import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import time
from pynvml import *
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


from model import BERT4Rec

from box import Box

torch.set_printoptions(sci_mode=True)

TRACE_OP = 0
MODEL_FILE_NAME = dict({0: 'BERT4REC_movieLens_64dim_0923', 1: "BERT4REC_Twitch_64dim_0923"})
N_ITEMS = dict({0 : 3681, 1: 46875}) 

config = {
    'data_path' : "../data/" , 

    'max_len' : 10,
    'hidden_units' : 64, # Embedding size
    'num_heads' : 4, # Multi-head layer 의 수 (병렬 처리) movieLens: 4 (2)
    'num_layers': 4, # block의 개수 (encoder layer의 개수) movieLens: 4 (2)
    'dropout_rate' : 0.2, # dropout 비율 movieLens: 0.2 (0.5)
    'lr' : 0.001, # movieLens: 0.001 (0.004)
    'batch_size' : 200, # movieLens: 200
    'num_epochs' : 100, # movieLens: 100
    'num_workers' : 2, # movieLens: 2
    'mask_prob' : 0.15, # for cloze task,  movieLens: 0.15 (0.2)
    'device' : torch.device(f"cuda" if torch.cuda.is_available() else "cpu"),
}

config = Box(config)

def load_model(trace_op):
    global TRACE_OP

    model_name = f'../saved/{MODEL_FILE_NAME[trace_op]}.pt'
    TRACE_OP=trace_op
    print(model_name)
    
    inference_model = torch.load(model_name, map_location=config.device)
    embedding_matrix = inference_model.item_emb.weight  
    print("Embedding matrix shape:", embedding_matrix.shape)

    return inference_model


def read_processed_file(filename, tag):
    columns_name=['user_id_idx', 'item_id_idx', 'timestamp']

    ratings = pd.read_csv(f'{config.data_path}{filename}_{tag}.txt',sep="\t",names=columns_name)
    n_ratings = len(ratings)
    n_ratings_users = ratings['user_id_idx'].nunique()
    n_ratings_items = ratings['item_id_idx'].nunique()
    print(f"(Ratings, n_users, n_items)  :  {n_ratings, n_ratings_users, n_ratings_items}")

    return ratings, n_ratings_users, n_ratings_items

def generate_sequence_data(ratings):
    users = defaultdict(list)
    for user, item, time in zip(ratings['user_id_idx'], ratings['item_id_idx'], ratings['timestamp']):
            users[user].append(item)
    
    seqs = []

    for uid, seq in users.items():
        if(len(seq) <= (config.max_len + 1)):
            seqs.append(seq)
        else:
            for idx in range(0, len(seq), (config.max_len + 1)):
                sub_seq = seq[idx:idx + (config.max_len + 1)]
                seqs.append(sub_seq)


    print(f"(n_users, n_sequences)  :  {len(users.keys()), len(seqs)}")
    return seqs, len(seqs)

def split_ratings(seq_list):
    seq_train = {}
    seq_valid = {}

    for idx, seq in enumerate(seq_list):
        seq_train[idx] = seq[:-1]
        seq_valid[idx] = seq[-1:] 

    return seq_train, seq_valid

def random_neg_sampling(rated_item : list, num_item_sample : int, num_item: int):
    _all_items = set([i for i in range(1, num_item + 1)])
    try:
      nge_samples = random.sample(list(_all_items - set(rated_item)), num_item_sample)
    except:
        print(rated_item)
    return nge_samples

class BERTRecDataSet(Dataset):
    def __init__(self, train, max_len, num_seq, num_item, mask_prob):
        self.train = train
        self.max_len = max_len
        self.num_seq = num_seq
        self.num_item = num_item
        self.mask_prob = mask_prob
        self._all_items = set([i for i in range(1, self.num_item + 1)])

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_seq

    def __getitem__(self, idx): 
        
        seq = self.train[idx]
        tokens = []
        labels = []
        for s in seq[-self.max_len:]:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    # noise
                    tokens.extend(random_neg_sampling(rated_item = seq, num_item_sample = 1, num_item = self.num_item))  # item random sampling
                else:
                    tokens.append(s)
                labels.append(s) 
            else:
                tokens.append(s)
                labels.append(0) 

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

class EvaluationDataset(Dataset):
    def __init__(self, seq_train, seq_valid, max_len, num_item, num_item_sample=99):
        self.seq_train = seq_train
        self.seq_valid = seq_valid
        self.max_len = max_len
        self.num_item = num_item
        self.num_item_sample = num_item_sample
        self.num_seq = len(seq_train)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        seq = (self.seq_train[idx] + [self.num_item + 1])[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        
        rated = self.seq_train[idx] + self.seq_valid[idx]
        neg_candidates = self.seq_valid[idx] + random_neg_sampling(rated_item=rated, num_item_sample=self.num_item_sample, num_item=self.num_item)
        
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        neg_candidates_tensor = torch.tensor(neg_candidates, dtype=torch.long)
        
        return seq_tensor, neg_candidates_tensor



def train(model, criterion, optimizer, data_loader):
    model.train()
    loss_val = 0
    for seq, labels in tqdm(data_loader):
        logits = model(seq)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).to(config.device)

        optimizer.zero_grad()
        loss = criterion(logits, labels)

        loss_val += loss.item()

        loss.backward()
        optimizer.step()
    
    loss_val /= len(data_loader)

    return loss_val



def evaluate(model, seq_train, seq_valid, max_len, num_seq, num_item, subset_ratio=0.2):
    model.eval()

    eval_dataset = EvaluationDataset(seq_train, seq_valid, max_len, num_item)
    
    indices = list(range(num_seq))
    np.random.shuffle(indices)
    
    split = int(np.floor(subset_ratio * num_seq))
    subset_indices = indices[:split]

    sampler = SubsetRandomSampler(subset_indices)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers, pin_memory=True)

    NDCG = 0.0  
    HIT = 0.0   
    K = 10

    for batch_seqs, batch_neg_candidates in tqdm(eval_loader):
        with torch.no_grad():
            scores = -model(batch_seqs)  

        for i in range(len(batch_seqs)):
            sample_predictions = scores[i][-1][batch_neg_candidates[i]]  
            rank = sample_predictions.argsort().argsort()[0].item()

            if rank < K:  
                NDCG += 1 / np.log2(rank + 2)
                HIT += 1

    NDCG /= len(subset_indices)
    HIT /= len(subset_indices)

    return NDCG, HIT




def training(filename, tag):
    ratings, n_ratings_users, n_ratings_items = read_processed_file(filename, tag)
    seqs, num_seqs = generate_sequence_data(ratings)
    seq_train, seq_valid = split_ratings(seqs)

    print("================== training BERT4Rec ====================")

    bert4rec_dataset = BERTRecDataSet(train = seq_train, max_len = config.max_len, num_seq = num_seqs, num_item = n_ratings_items, mask_prob = config.mask_prob,)
    data_loader = DataLoader(bert4rec_dataset, batch_size = config.batch_size, shuffle = True, pin_memory = True, num_workers = config.num_workers, )
    
    model = BERT4Rec(num_item = n_ratings_items, hidden_units = config.hidden_units, num_heads = config.num_heads, num_layers = config.num_layers, max_len = config.max_len, dropout_rate = config.dropout_rate, device = config.device,).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_list = []
    ndcg_list = []
    hit_list = []
    best_hit = 0

    for epoch in range(1, config.num_epochs + 1):

        train_loss = train(
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            data_loader = data_loader)
        
        ndcg, hit = evaluate(
            model = model, 
            seq_train = seq_train, 
            seq_valid = seq_valid, 
            max_len = config.max_len,
            num_seq=num_seqs, 
            num_item=n_ratings_items,
            )

        loss_list.append(train_loss)
        ndcg_list.append(ndcg)
        hit_list.append(hit)

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')

        if(hit > best_hit):
            best_hit = hit
            torch.save(model, f'../saved/BERT4REC_{filename}_64dim_0923.pt')


def get_one_hot_emb(log):
    seq = np.zeros(shape=(1,N_ITEMS[TRACE_OP])).astype('float32')
    seq[:, log] = 1.0

    return seq


def get_emb(log, model):
    seq = (log + [N_ITEMS[TRACE_OP] + 1])[-config.max_len:]
    padding_len = config.max_len - len(seq)
    seq = [0] * padding_len + seq

    item_emb, pos_emb, norm_emb = model.forward_emb(np.array([seq]))

    return item_emb, pos_emb, norm_emb


def inference(input, model, K):
    log, emb = input
    seq = (log + [N_ITEMS[TRACE_OP] + 1])[-config.max_len:]
    padding_len = config.max_len - len(seq)
    seq = [0] * padding_len + seq

    with torch.no_grad():
        rec_vector = model.forward_bert(np.array([seq]), emb)
        rec_vector = rec_vector[:, -1, :]

        min_score = torch.min(rec_vector)
        max_score = torch.max(rec_vector)

        relative_score = (rec_vector - min_score) / (max_score - min_score)
        score = F.softmax(rec_vector, dim=-1)

        score = score.cpu().tolist()[0]
        relative_score = relative_score.cpu().tolist()[0]

        np_score = np.array(score)
        np_relative_score = np.array(relative_score)

        orig_score = deepcopy(np_score)
        orig_relative_score = deepcopy(np_relative_score)
    
        return orig_score, orig_relative_score