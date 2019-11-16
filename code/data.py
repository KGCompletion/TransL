from _collections import defaultdict
import math
import time

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)


def load_dict(file_path):
    str2id = defaultdict(int)
    id2str = defaultdict(str)

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            string = line[0]
            idx = int(line[1])
            str2id[string] = idx
            id2str[idx] = string

    return str2id, id2str

def load_data(file_path, entity2id, rel2id):
    entity2data_list = defaultdict(list)
    rel2entity_list = defaultdict(list)

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            e1, rel, e2 = line[0], line[1], line[2]
            e1 = entity2id[e1]
            rel = rel2id[rel]
            e2 = entity2id[e2]

            entity2data_list[e1].append((rel, e2))
            rel2entity_list[rel].append(e2)

    return entity2data_list, rel2entity_list

def load_allData(file_name, entity2id, rel2id):
    hr2t_list = defaultdict(list)
    tr2h_list = defaultdict(list)

    with open('../' + file_name + '/train.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            e1, rel, e2 = line[0], line[1], line[2]
            e1 = entity2id[e1]
            rel = rel2id[rel]
            e2 = entity2id[e2]

            hr2t_list[(e1, rel)].append(e2)
            tr2h_list[(e2, rel)].append(e1)

    with open('../' + file_name + '/valid.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            e1, rel, e2 = line[0], line[1], line[2]
            e1 = entity2id[e1]
            rel = rel2id[rel]
            e2 = entity2id[e2]

            hr2t_list[(e1, rel)].append(e2)
            tr2h_list[(e2, rel)].append(e1)

    with open('../' + file_name + '/test.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            e1, rel, e2 = line[0], line[1], line[2]
            e1 = entity2id[e1]
            rel = rel2id[rel]
            e2 = entity2id[e2]

            hr2t_list[(e1, rel)].append(e2)
            tr2h_list[(e2, rel)].append(e1)
    
    return hr2t_list, tr2h_list

def load_train_data(file_path, entity2id, rel2id):
    id2e1 = []
    id2e2 = []
    id2rel = []

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            e1, rel, e2 = line[0], line[1], line[2]

            e1 = entity2id[e1]
            rel = rel2id[rel]
            e2 = entity2id[e2]

            id2e1.append(e1)
            id2e2.append(e2)
            id2rel.append(rel)

    return id2e1, id2e2, id2rel


def load_test_data(file_path, entity2id, rel2id):
    id2e1 = []
    id2e2 = []
    id2rel = []
    id2flag = []

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if len(line) == 4:
                e1, rel, e2, flag = line[0], line[1], line[2], line[3]
            else:
                e1, rel, e2, flag = line[0], line[1], line[2], 1

            e1 = entity2id[e1]
            rel = rel2id[rel]
            e2 = entity2id[e2]
            
            id2e1.append(e1)
            id2e2.append(e2)
            id2rel.append(rel)
            id2flag.append(flag)

    return id2e1, id2e2, id2rel, id2flag


def write(file_path, result):
    with open(file_path, 'a+') as f:
        f.write(result + '\n')

