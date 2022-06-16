import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math

TARGET = dict()

class InvNet_office(nn.Module):
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01):
        super(InvNet_office, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  
        self.beta = beta 
        self.knn = knn  

        self.em = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)
        self.untouched_targets = set(range(num_classes))

    def forward(self, inputs, label, epoch=None, fnames_target=None, predict_class=None, model=None):
        alpha = self.alpha * epoch
        if model is not None:
            tgt_feature = inputs.mm(self.em.t())
            tgt_class = model.module.classifier(self.em)
        else:
            tgt_feature = inputs.mm(self.em.t())
            tgt_class = None
        tgt_feature /= self.beta

        loss = self.smooth_loss(tgt_feature, tgt_label,predict_class,tgt_class)

        for x, y in zip(inputs, label):
            self.em.data[y] = alpha * self.em.data[y]  + (1. - alpha) * x.data
            self.em.data[y] /= self.em.data[y].norm()
            
        return loss

    def smooth_loss(self, tgt_feature, tgt_label,predict_class, tgt_class):
        '''
        tgt_feature: [128, 16522], similarity of batch & targets
        tgt_label: see forward
        '''        
        if tgt_class is not None:
            outputs_class = torch.abs(predict_class.argmax(dim=1).unsqueeze(1).repeat(1,tgt_class.shape[0])-tgt_class.argmax(dim=1))
            # from IPython import embed; embed();exit(0)
            outputs_class[outputs_class != 0] = 1
            outputs_class = 1.0 - outputs_class
        else:
            outputs_class = None
        mask = self.smooth_hot(tgt_feature.detach().clone(), tgt_label.detach().clone(), outputs_class, predict_class,self.knn)

        outputs = F.log_softmax(tgt_feature, dim=1)
        loss = - (mask * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, tgt_feature, targets, outputs_class,predict_class, k=6):
        '''
        see smooth_loss
        '''

        mask = torch.zeros(tgt_feature.size()).to(self.device)
        if outputs_class is not None:
            print (tgt_feature.min())
            tgt_mask = tgt_feature.detach().clone()
            tgt_mask[outputs_class[:,0],outputs_class[:,1]] = -1e5
            _, topk = tgt_mask.topk(k, dim=1)
        else:
            _, topk = tgt_feature.topk(k, dim=1)
        mask.scatter_(1, topk, 1.0)

        index_2d = targets[..., None]
        mask.scatter_(1, index_2d, 2.0)

        return mask
