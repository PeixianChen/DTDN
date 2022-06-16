from __future__ import print_function, absolute_import
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import pdb
import random
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import time


class BaseTrainer(object):
    def __init__(self, model, criterion, criterion_trip=None, InvNet=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.criterion_trip = criterion_trip
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.write = SummaryWriter(log_dir="./")
        self.InvNet = InvNet
        self.index = []

    def train(self, epoch, data_loader, optimizer, batch_size, print_freq=1):
        self.model[0].train()
        self.model[1].train()
        self.model[2].train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_IDE_s = AverageMeter()
        losses_IDE_t = AverageMeter()
        losses_triplet = AverageMeter()
        losses_neightboor = AverageMeter()
        losses_agreement = AverageMeter()


        # To make sure the longest loader is consumed, we cycle the other one.

        
        src_loader, tgt_loader = data_loader
        from itertools import cycle, tee
        if len(src_loader) < len(tgt_loader):
            src_loader = cycle(src_loader)
        elif len(src_loader) > len(tgt_loader):
            tgt_loader = cycle(tgt_loader)
        src_loader, src_pad = tee(src_loader)
        tgt_loader, tgt_pad = tee(tgt_loader)

        end = time.time()
        for i, (src_inputs, tgt_inputs) in enumerate(zip(src_loader, tgt_loader)):

            inputs_source, pids_source, pindexs_source = self._parse_data(src_inputs)
            inputs_target, pids_target, pindexs_target = self._parse_data(tgt_inputs)

            # inputs_source_tri, pids_source_tri, _ = self._parse_data(src_tri_inputs)

            data_time.update(time.time() - end)
            end = time.time()

            if inputs_source.size(0) < batch_size:
                new_inputs = next(src_pad)
                x, y, z = self._parse_data(new_inputs)
                inputs_source = torch.cat([inputs_source, x])[:batch_size]
                pids_source = torch.cat([pids_source, y])[:batch_size]
                pindexs_source = torch.cat([pindexs_source, z])[:batch_size]
            if inputs_target.size(0) < batch_size:
                new_inputs = next(tgt_pad)
                x, y, z = self._parse_data(new_inputs)
                inputs_target = torch.cat([inputs_target, x])[:batch_size]
                pids_target = torch.cat([pids_target, y])[:batch_size]
                pindexs_target = torch.cat([pindexs_target, z])[:batch_size]


            # loss_sc_sa, loss_sc_ta, neightboor, target_agreement = self._forward([inputs_source, inputs_target, inputs_source_tri], pids_source, pids_source_tri, pindexs_target, epoch)
            loss_sc_sa, loss_sc_ta, neightboor, target_agreement = self._forward([inputs_source, inputs_target], pids_source, pids_source, pindexs_target, epoch)


            if epoch > 12:
                print ("neightboor")
                loss = 0.7 * 0.5 * (loss_sc_sa + loss_sc_ta) + 0.3 * neightboor + target_agreement
            else:
                loss = (loss_sc_sa + loss_sc_ta) + 0 * neightboor +  0 * target_agreement            
            # from IPython import embed; embed()
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()
            loss.backward()
            optimizer[1].step()
            optimizer[2].step()
            optimizer[0].step()

                
            losses_IDE_s.update(loss_sc_sa.item(), pids_source.size(0))
            losses_IDE_t.update(loss_sc_ta.item(), pids_source.size(0))
            losses_neightboor.update(neightboor.item(), pids_source.size(0))
            losses_agreement.update(target_agreement.item(), pids_source.size(0))

            batch_time.update(time.time() - end)
            end = time.time()


            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}] \t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'task_1 {:.3f} ({:.3f})\t'
                      'task_2 {:.3f} ({:.3f})\t'
                      'triplet {:.3f} ({:.3f})\t'
                      'neightboor {:.3f} ({:.3f})\t'
                      'agreement {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, max(map(len, data_loader)),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_IDE_s.val, losses_IDE_s.avg,
                              losses_IDE_t.val, losses_IDE_t.avg,
                              losses_triplet.val, losses_triplet.avg,
                              losses_neightboor.val, losses_neightboor.avg,
                              losses_agreement.val, losses_agreement.avg))
    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _, pindexs = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        pindexs = pindexs.to(self.device)

        # print (pids)

        return inputs, pids, pindexs

    def _forward(self, inputs, pids_source, pids_source_tri, index_target, epoch, update_only=False):
        outputs_source = self.model[0](inputs[0])
        outputs_target = self.model[0](inputs[1])


        # mask
        source_mask = torch.mean(self.model[2](outputs_source), dim=0)
        target_mask = torch.mean(self.model[2](outputs_target), dim=0)
        outputs_source_c = source_mask * outputs_source
        outputs_source_a = (1-source_mask) * outputs_source
        outputs_target_c = target_mask * outputs_target
        outputs_target_a = (1-target_mask) * outputs_target

        # source ide
        index = torch.randperm(outputs_source_a.size(0))
        outputs_source_a = outputs_source_a[index, :, :, :]
        outputs_target_a = outputs_target_a[index, :, :, :]
        
        inputs_scsa = outputs_source_c + outputs_source_a
        inputs_scta = outputs_source_c + outputs_target_a
        inputs_tcsa = outputs_target_c + outputs_source_a
        inputs_tcta = outputs_target_c + outputs_target_a

        outputs_scsa, _ = self.model[1](inputs_scsa)
        outputs_scta, _ = self.model[1](inputs_scta)

        # _, triplet_sc = self.model[1](outputs_source_tri)
        # source_triplet = self.criterion[1](triplet_sc, pids_source_tri)[0]

        ws = wt = 5e-5
        if source_mask.mean() == 1:
            ws = wt = 1e-4
        
        loss_sc_sa = self.criterion[0](outputs_scsa, pids_source) #+ 0.5 * self.criterion[1](triplet_sc, pids_source)[0]
        loss_sc_ta = self.criterion[0](outputs_scta, pids_source) + ws * torch.norm(source_mask, 1)

        classify_tcsa, _ = self.model[1](inputs_tcsa, domain='target')
        classify_tcta, _ = self.model[1](inputs_tcta, domain='target')

        # print (wt * torch.norm(source_mask, 1))
        # print (wt * torch.norm(target_mask, 1))

        # if toche__version == '1.1.0':
        target_agreement = F.l1_loss(F.log_softmax(classify_tcsa, dim=1), F.log_softmax(classify_tcta, dim=1)) + wt * torch.norm(target_mask, 1)
        # if torch__version == '0.4.0':
        # target_agreement = F.l1_loss(F.log_softmax(classify_tcsa, dim=1), F.log_softmax(classify_tcta, dim=1))/(batch_size * target_numclass)        

        # # source & target invariance
        _, tgt_feature = self.model[1](outputs_target_c, tgt_output_feature='pool5')
        target_loss = self.InvNet(tgt_feature, index_target, epoch=epoch)

        return loss_sc_sa, loss_sc_ta, target_loss, target_agreement#, source_triplet
