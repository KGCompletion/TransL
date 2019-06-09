import torch
import torch.nn as nn

torch.manual_seed(1)

class Network(nn.Module):

    def __init__(self, dimension, entity_size, rel_size):
        super(Network, self).__init__()
        self.eneity_embedding = nn.Embedding(entity_size + 1, dimension)
        self.edge_weight = nn.Embedding(rel_size * (rel_size + 1) + 1, 1)
        self.rel_embedding = nn.Embedding(rel_size, dimension)
        
        self.softmax = nn.Softmax(dim=1)

    def get_vh(self, data_r, data_e):
        data_r = self.edge_weight(data_r.long())
        data_r = data_r.view(data_e.size(0), -1)
        data_w = self.softmax(data_r)
        data_w = data_w.view(data_e.size(0), -1, 1)

        data_e = self.eneity_embedding(data_e.long())

        Eh = torch.mul(data_e, data_w)
        Eh = torch.transpose(Eh, 1, 2)
        vh = torch.sum(Eh, dim=2)

        return vh

    def get_vr(self, rel):
        vr = self.rel_embedding(rel.long())
        return vr

    def get_vt(self, t):
        vt = self.eneity_embedding(t.long())
        return vt

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin1, margin2):
        super(ContrastiveLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, vh, vr, vt, vh_neg, vr_neg, vt_neg):
        pdist = nn.PairwiseDistance(p=2)
        pos_dist = pdist(vh + vr, vt)
        neg_dist = pdist(vh_neg + vr_neg, vt_neg)
        loss = torch.mean(torch.clamp(pos_dist - self.margin1, min=0.0) + torch.clamp(self.margin2 - neg_dist, min=0.0))

        return loss
