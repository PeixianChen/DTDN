import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math


class InvNet(nn.Module):
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01):
        super(InvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  
        self.beta = beta  
        self.knn = knn  

        self.em = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)
        self.untouched_targets = set(range(num_classes))

    def forward(self, inputs, label, epoch=None, fnames_target=None):
        '''
        inputs: [128, 2048], each t's 2048-d feature
        label: [128], each t's label
        '''
        alpha = self.alpha * epoch
        tgt_feature = inputs.mm(self.em.t())
        tgt_feature /= self.beta

        loss = self.smooth_loss(tgt_feature, label)

        for x, y in zip(inputs, label):
            self.em.data[y] = alpha * self.em.data[y]  + (1. - alpha) * x.data
            self.em.data[y] /= self.em.data[y].norm()
        return loss

    def smooth_loss(self, tgt_feature, tgt_label):
        '''
        tgt_feature: [128, 16522], similarity of batch & targets
        tgt_label: see forward
        '''
        mask = self.smooth_hot(tgt_feature.detach().clone(), tgt_label.detach().clone(), self.knn)
        outputs = F.log_softmax(tgt_feature, dim=1)
        loss = - (mask * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, tgt_feature, targets, k=6):
        '''
        see smooth_loss
        '''
        mask = torch.zeros(tgt_feature.size()).to(self.device)

        _, topk = tgt_feature.topk(k, dim=1)
        mask.scatter_(1, topk, 2)

        index_2d = targets[..., None]
        mask.scatter_(1, index_2d, 3)

        return mask
