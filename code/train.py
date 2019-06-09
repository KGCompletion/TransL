from model import *
from data import *

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim

from _collections import defaultdict
import numpy as np
import random


class TrainDataset(Dataset):

    def __init__(self, file_name, neg_type):
        self.neg_type = neg_type

        self.entity2id, self.id2entity = load_dict('../' + file_name + '/entity2id.txt')
        self.rel2id, self.id2rel = load_dict('../' + file_name + '/relation2id.txt')

        self.entity2data_list, self.rel2entity_list = load_data('../' + file_name + '/train.txt', self.entity2id, self.rel2id)

        self.id2e1_train, self.id2e2_train, self.id2rel_train = load_train_data('../' + file_name + '/train.txt', self.entity2id, self.rel2id)
        self.len = len(self.id2e1_train)

        self.entity_len = len(self.entity2id)
        self.rel_len = len(self.rel2id)

        self.ok = defaultdict(int)
        for index in range(self.len):
            h = self.id2e1_train[index]
            r = self.id2rel_train[index]
            t = self.id2e2_train[index]
            self.ok[(h, r, t)] = 1

        self.prob_h = defaultdict(int)
        if self.neg_type == 'unif':
            for r in self.id2rel:
                self.prob_h[r] = 0.5
        else:
            self.get_prob()

    def get_prob(self):
        tph_dict = defaultdict(int)
        hpt_dict = defaultdict(int)
        for index in range(self.len):
            h = self.id2e1_train[index]
            r = self.id2rel_train[index]
            t = self.id2e2_train[index]
            tph_dict[(r, h)] += 1
            hpt_dict[(r, t)] += 1

        t_sum_dict = defaultdict(int)
        h_count_dict = defaultdict(int)
        for (r, h) in tph_dict.keys():
            t_sum_dict[r] += tph_dict[(r, h)]
            h_count_dict[r] += 1
        
        tph = defaultdict(int)
        for r in t_sum_dict.keys():
            tph[r] = t_sum_dict[r] * 1.0 / h_count_dict[r]

        h_sum_dict = defaultdict(int)
        t_count_dict = defaultdict(int)
        for (r, t) in hpt_dict.keys():
            h_sum_dict[r] += hpt_dict[(r, t)]
            t_count_dict[r] += 1

        hpt = defaultdict(int)
        for r in h_sum_dict.keys():
            hpt[r] = h_sum_dict[r] * 1.0 / t_count_dict[r]

        for r in self.id2rel:
            self.prob_h[r] = tph[r] / (tph[r] + hpt[r])


    def get_data(self, h, r, t):
        r_list = []
        e_list = []

        data = self.entity2data_list[h]
        if (r, t) in data:
            data.remove((r, t))
        data.append((self.rel_len, h))
        
        for (rel, entity) in data:
            r_list.append(rel * self.rel_len + r)
            e_list.append(entity)

        DATA_len = 50
        data_len = len(r_list)
        if data_len > DATA_len:
            ids = random.sample(range(0, data_len), DATA_len)
            r_data = [r_list[idx] for idx in ids]
            e_data = [e_list[idx] for idx in ids]
        else:
            r_data = r_list
            e_data = e_list

        data_len = len(r_data)
        data_r = torch.from_numpy(np.array(r_data)).int()
        data_r_temp = torch.ones((DATA_len - data_len), dtype=torch.int) * (self.rel_len * (self.rel_len + 1))
        data_r = torch.cat((data_r, data_r_temp), 0)
        
        data_e = torch.from_numpy(np.array(e_data)).int()
        data_e_temp = torch.ones((DATA_len - data_len), dtype=torch.int) * self.entity_len
        data_e = torch.cat((data_e, data_e_temp), 0)

        return data_r, data_e

    def get_neg(self, h, r, t):
        neg_entity = random.randint(0, self.entity_len - 1)

        if random.random() < self.prob_h[r]:
            while(self.ok[(neg_entity, r, t)] == 1):
                neg_entity = random.randint(0, self.entity_len - 1)
            
            h_neg = neg_entity
            t_neg = t
        else:
            while(self.ok[(h, r, neg_entity)] == 1):
                neg_entity = random.randint(0, self.entity_len - 1)
            
            h_neg = h
            t_neg = neg_entity

        return h_neg, r, t_neg

    def __getitem__(self, index):
        h = self.id2e1_train[index]
        r = self.id2rel_train[index]
        t = self.id2e2_train[index]

        data_r, data_e = self.get_data(h, r, t)

        h_neg, r_neg, t_neg = self.get_neg(h, r, t)
        data_r_neg, data_e_neg = self.get_data(h_neg, r_neg, t_neg)

        return data_r, data_e, r, t, data_r_neg, data_e_neg, r_neg, t_neg

    def __len__(self):
        return self.len


def train(dimension, margin1, margin2, nn_lr, batch_size, file_name, neg_type):
    train_epochs = 800
    file_dir = file_name + '/' + str(dimension) + '-' + str(margin1) + '-' + str(margin2) + '(' + str(nn_lr) + '-' + str(batch_size) + ')-' + neg_type
    
    train_dataset = TrainDataset(file_name, neg_type)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=5, batch_size=batch_size)

    net = Network(dimension, train_dataset.entity_len, train_dataset.rel_len)
    # net = torch.load('out/' + file_dir + '/net-' + str(500) + '.pt')
    
    loss_func = ContrastiveLoss(margin1, margin2)
    optimizer = optim.Adam(net.parameters(), lr=nn_lr)

    if torch.cuda.is_available() == True:
        net = net.cuda()
        loss_func = loss_func.cuda()
    
    start = time.time()

    for epoch in range(train_epochs):
        epoch_loss = 0
        current_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            data_r, data_e, r, t, data_r_neg, data_e_neg, r_neg, t_neg = data

            if torch.cuda.is_available() == True:
                data_r = data_r.cuda()
                data_e = data_e.cuda()
                r = r.cuda()
                t = t.cuda()
                data_r_neg = data_r_neg.cuda()
                data_e_neg = data_e_neg.cuda()
                r_neg = r_neg.cuda()
                t_neg = t_neg.cuda()

            vh = net.get_vh(data_r, data_e)
            vr = net.get_vr(r)
            vt = net.get_vt(t)

            vh_neg = net.get_vh(data_r_neg, data_e_neg)
            vr_neg = net.get_vr(r_neg)
            vt_neg = net.get_vt(t_neg)

            optimizer.zero_grad()
            loss = loss_func(vh, vr, vt, vh_neg, vr_neg, vt_neg)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available() == True:
                loss = loss.cpu()

            current_loss += loss.data.item()
            epoch_loss += loss.data.item()

            j = i * batch_size
            if j % 20000 == 0:
                print('%d %d %d%% (%s) %.4f' % (epoch, j, j * 100 / train_dataset.len, timeSince(start), current_loss))
                current_loss = 0

        if not os.path.exists('out/' + file_dir):
            os.makedirs('out/' + file_dir)
        
        j = epoch + 1
        if j % 10 == 0:
            model_name = 'out/' + file_dir + '/net-' + str(j) + '.pt'
            torch.save(net, model_name)

        loss_str = '%.4f' % epoch_loss
        write('out/' + file_dir + '/loss.txt', loss_str)

    print('train done!')

    

if __name__ == '__main__':
    dimension = 20

    margin1 = 1
    margin2 = 10

    nn_lr = 0.01
    batch_size = 100

    file_name = 'WN11'
    # file_name = 'FB13'
    
    # neg_type = 'unif'
    neg_type = 'bern'
    
    train(dimension, margin1, margin2, nn_lr, batch_size, file_name, neg_type)
