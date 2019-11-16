from model import *
from data import *

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from _collections import defaultdict
import numpy as np
import random
import argparse


class TestDataset(Dataset):

    def __init__(self, file_name, test_name):
        self.entity2id, self.id2entity = load_dict('../' + file_name + '/entity2id.txt')
        self.rel2id, self.id2rel = load_dict('../' + file_name + '/relation2id.txt')

        self.entity2data_list, self.rel2entity_list = load_data('../' + file_name + '/train.txt', self.entity2id, self.rel2id)
        self.hr2t_list, self.tr2h_list = load_allData(file_name, self.entity2id, self.rel2id)

        self.entity_len = len(self.entity2id)
        self.rel_len = len(self.rel2id)

        self.id2e1_test, self.id2e2_test, self.id2rel_test, _ = load_test_data('../' + file_name + '/' + test_name, self.entity2id, self.rel2id)
        self.len = len(self.id2e1_test)

    def get_data(self, h, r, t):
        r_list = []
        e_list = []

        data = self.entity2data_list[h]
        # if (r, t) in data:
        #     data.remove((r, t))
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

    def __getitem__(self, index):
        e1 = self.id2e1_test[index]
        e2 = self.id2e2_test[index]
        rel = self.id2rel_test[index]

        data_r, data_e = self.get_data(e1, rel, e2)

        return data_r, data_e, e1, rel, e2

    def __len__(self):
        return self.len


class Hit_entity():
    def __init__(self, embedding, entity_len):
        meeting_ids = range(entity_len)
        meeting_matrix = torch.from_numpy(np.array(meeting_ids)).long()
        self.embedding = embedding(meeting_matrix)


    def predict(self, out, d, t_list):
        pdist = nn.PairwiseDistance(p=2)
        dist = pdist(out, self.embedding)

        rank = 0
        fRank = 1
        for i in range(len(dist)):
            if(dist[i] <= d):
                rank += 1
            if(dist[i] <= d and i not in t_list):
                fRank += 1

        return rank, fRank


def test(file_name, net_path, test_name, out_path, epoch):
    test_dataset = TestDataset(file_name, test_name)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=1, batch_size=1000)

    # net = torch.load(net_path)
    net = Network(args.dim, test_dataset.entity_len, test_dataset.rel_len)
    net.load_state_dict(torch.load(net_path))
    net.eval()

    hit_entity = Hit_entity(net.eneity_embedding, test_dataset.entity_len)

    sum_rank = 0
    sum_frank = 0
    sum_hit1 = 0
    sum_hit3 = 0
    sum_hit10 = 0
    count = 0

    pdist = nn.PairwiseDistance(p=2)

    for i, data in enumerate(test_dataloader, 0):
        data_r, data_e, h, rel, t = data

        vh = net.get_vh(data_r, data_e)
        vr = net.get_vr(rel)
        vt = net.get_vt(t)

        ot = vh + vr
        dist = pdist(ot, vt)

        h = h.numpy()
        rel = rel.numpy()

        data_len = t.size(0)
        for idx in range(data_len): 
            rank, fRank = hit_entity.predict(ot[idx], dist[idx], test_dataset.hr2t_list[(h[idx], rel[idx])])
            count += 1
            sum_rank += 1.0 / rank
            sum_frank += 1.0 / fRank
            if fRank <= 1:
                sum_hit1 += 1
            if fRank <= 3:
                sum_hit3 += 1
            if fRank <= 10:
                sum_hit10 += 1
            
            line = ""
            if count % 10 == 0:
                line = '%d : %d : %.4f %.4f %.4f %.4f %.4f' % (epoch, count, sum_rank / count, sum_frank / count, sum_hit1 / count, sum_hit3 / count, sum_hit10 / count)
                print(line)

            if count % 100 == 0:
                write(out_path + str(epoch) + '.txt', line)

    line = '%d : %d : %.4f %.4f %.4f %.4f %.4f' % (epoch, count, sum_rank / count, sum_frank / count, sum_hit1 / count, sum_hit3 / count, sum_hit10 / count)
    print(line)
    write(out_path + str(epoch) + '.txt', line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dim', type=int, default=100, help='entity and relation sharing embedding dimension')
    parser.add_argument('-margin_pos', type=int, default=10, help='margin of positive triplets')
    parser.add_argument('-margin_neg', type=int, default=15, help='margin of negative triplets')
    parser.add_argument('-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-batch', type=int, default=100, help='batch size')
    parser.add_argument('-epoch', type=int, default=300, help='number of training epoch')
    parser.add_argument('-method', type=str, default='bern', help='stratege of constructing negative triplets')
    parser.add_argument('-data', type=str, default='FB15k-237', help='dataset of the model')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    file_name = args.data
    net_name = str(args.dim) + '-' + str(args.margin_pos) + '-' + str(args.margin_neg) + '(' + str(args.rate) + '-' + str(args.batch) + ')-' + args.method
    epoch = args.epoch

    test_name = 'test.txt'
    # test_name = 'valid.txt'

    net_path = 'out/' + file_name + '/' + net_name + '/net-' + str(epoch) + '.pt'
    out_path = 'out/' + file_name + '/' + net_name + '/test/' 
    # out_path = 'out/' + file_name + '/' + net_name + '/valid/' 

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    
    test(file_name, net_path, test_name, out_path, epoch)
    
