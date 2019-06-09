from model import *
from data import *

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from _collections import defaultdict
import numpy as np
import random


class TestDataset(Dataset):

    def __init__(self, file_name, test_name):
        self.entity2id, self.id2entity = load_dict('../' + file_name + '/entity2id.txt')
        self.rel2id, self.id2rel = load_dict('../' + file_name + '/relation2id.txt')

        self.entity2data_list, self.rel2entity_list = load_data('../' + file_name + '/train.txt', self.entity2id, self.rel2id)

        self.entity_len = len(self.entity2id)
        self.rel_len = len(self.rel2id)

        self.id2e1_test, self.id2e2_test, self.id2rel_test, self.id2flag_test = load_test_data('../' + file_name + '/' + test_name, self.entity2id, self.rel2id)
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
        flag = self.id2flag_test[index]

        data_r, data_e = self.get_data(e1, rel, e2)

        flag = int(flag)

        return data_r, data_e, rel, e2, flag

    def __len__(self):
        return self.len

def test(file_name, net_path, test_name, out_path, epoch, margin_dict):
    net = torch.load(net_path)

    test_dataset = TestDataset(file_name, test_name)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=1, batch_size=1000)


    dict_right1 = defaultdict(int)
    dict_right2 = defaultdict(int)
    dict_count = defaultdict(int)
    right1 = 0
    right2 = 0
    count = 0

    pdist = nn.PairwiseDistance(p=2)

    for i, data in enumerate(test_dataloader, 0):
        data_r, data_e, rel, t, flag = data
        if torch.cuda.is_available() == True:
            data_r = data_r.cuda()
            data_e = data_e.cuda()
            rel = rel.cuda()
            t = t.cuda()
            flag = flag.cuda()

        vh = net.get_vh(data_r, data_e)
        vr = net.get_vr(rel)
        vt = net.get_vt(t)

        dist = pdist(vh + vr, vt)

        if torch.cuda.is_available() == True:
            rel = rel.cpu()
        rel = rel.numpy()

        data_len = t.size(0)
        for idx in range(data_len): 
            r = rel[idx]
            r_str = test_dataset.id2rel[r]  
            dict_count[r] += 1
            if dist[idx] < margin_dict[r_str] and flag[idx] == 1:
                right1 += 1
                dict_right1[r] += 1
            if dist[idx] >= margin_dict[r_str] and flag[idx] == -1:
                right2 += 1
                dict_right2[r] += 1
        
        count += data_len
        if count % 10000 == 0:
            line = '%d %d %d : %.4f' % (right1, right2, count, (right1 + right2 ) / count)
            print(line)
    line = '%d : %d %d %d : %.4f' % (epoch, right1, right2, count, (right1 + right2 ) / count)
    print(line)
    write(out_path + str(epoch) + '.txt', line)
    write(out_path + 'all.txt', line)
    for rel in test_dataset.id2rel:
        if dict_count[rel] == 0:
            continue
        line = '%d: %d %d %d : %.4f' % (rel, dict_right1[rel], dict_right2[rel], dict_count[rel], (dict_right1[rel] + dict_right2[rel] ) / dict_count[rel])
        print(line)
        write(out_path + str(epoch) + '.txt', line)

if __name__ == '__main__':
    file_name = 'WN11'
    net_name = '20-1-10(0.01-100)-bern'
    epoch = 750

    # file_name = 'WN11'
    # net_name = '20-1-10(0.01-100)-unif'
    # epoch = 600

    # file_name = 'FB13'
    # net_name = '50-1-100(0.0001-1000)-bern'
    # epoch = 300

    # file_name = 'FB13'
    # net_name = '50-1-100(0.0001-1000)-unif'
    # epoch = 80

    test_name = 'test.txt'

    net_path = 'out/' + file_name + '/' + net_name + '/net-' + str(epoch) + '.pt'
    margin_path = 'out/' + file_name + '/' + net_name + '/valid/margin-' + str(epoch) + '.txt'
    out_path = 'out/' + file_name + '/' + net_name + '/test/' 

    if not os.path.exists(out_path):
        os.makedirs(out_path)


    margin_dict, _ = load_dict(margin_path)
    
    test(file_name, net_path, test_name, out_path, epoch, margin_dict)
    
