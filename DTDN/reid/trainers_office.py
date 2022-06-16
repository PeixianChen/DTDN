from __future__ import print_function, absolute_import
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import pdb
<<<<<<< HEAD
=======
import random
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class BaseTrainer(object):
    def __init__(self, model, criterion, criterion_trip=None, InvNet=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.criterion_trip = criterion_trip
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.write = SummaryWriter(log_dir="./")
        self.InvNet = InvNet

    def train(self, epoch, data_loader, optimizer, targetname, target_num_classes, batch_size=128, print_freq=1):
        self.model[0].train()
        self.model[1].train()
        self.model[2].train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_IDE_s = AverageMeter()
        losses_IDE_t = AverageMeter()
        losses_triplet = AverageMeter()
        losses_ag = AverageMeter()

        # To make sure the longest loader is consumed, we cycle the other one.
        src_loader, tgt_loader = data_loader
<<<<<<< HEAD

        from itertools import cycle, tee
=======
        from itertools import cycle
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
        if len(src_loader) < len(tgt_loader):
            src_loader = cycle(src_loader)
        elif len(src_loader) > len(tgt_loader):
            tgt_loader = cycle(tgt_loader)
<<<<<<< HEAD
        src_loader, src_pad = tee(src_loader)
        tgt_loader, tgt_pad = tee(tgt_loader)
=======
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32

        end = time.time()
        for i, (src_inputs, tgt_inputs) in enumerate(zip(src_loader, tgt_loader)):
            data_time.update(time.time() - end)

            inputs_source, pids_source, pindexs_source = self._parse_data(src_inputs)
            inputs_target, pids_target, pindexs_target, fnames_target = self._parse_data(tgt_inputs, True)
<<<<<<< HEAD
            # print (pindexs_target)


            # print(inputs_source.shape, pids_source.shape, pindexs_source.shape)
            if inputs_source.size(0) < batch_size:
                new_inputs = next(src_pad)
=======

            if inputs_source.size(0) < batch_size:
                new_inputs = next(iter(src_loader))
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
                x, y, z = self._parse_data(new_inputs)
                inputs_source = torch.cat([inputs_source, x])[:batch_size]
                pids_source = torch.cat([pids_source, y])[:batch_size]
                pindexs_source = torch.cat([pindexs_source, z])[:batch_size]
            if inputs_target.size(0) < batch_size:
<<<<<<< HEAD
                new_inputs = next(tgt_pad)
=======
                new_inputs = next(iter(tgt_loader))
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
                x, y, z = self._parse_data(new_inputs)
                inputs_target = torch.cat([inputs_target, x])[:batch_size]
                pids_target = torch.cat([pids_target, y])[:batch_size]
                pindexs_target = torch.cat([pindexs_target, z])[:batch_size]

<<<<<<< HEAD
            # print(pindexs_source)
            # input('...')
            loss_sc_sa, loss_sc_ta, loss_neightboor, loss_agreement = self._forward([inputs_source, inputs_target], pids_source, pindexs_target, epoch, targetname, target_num_classes, fnames_target)

            
            # W-A: 
            # if epoch > 15:
            #     print ("neightbor")
            #     loss =  (loss_sc_sa + loss_sc_ta) + 30 * loss_neightboor + loss_agreement
            # W-A:

            #  D-A: 
            if epoch > 7: 
                print("neightbor")
                loss =  (loss_sc_sa + loss_sc_ta) + 50 * loss_neightboor + loss_agreement
            # D-A

            # # A-D
            # if epoch > 5: 
            #     print("neightbor")
            #     loss =  (loss_sc_sa + loss_sc_ta) + 30 * loss_neightboor + loss_agreement
            # # A-D

            # A-W
            # if epoch > 6: 
            #     print("neightbor")
            #     loss =  (loss_sc_sa + loss_sc_ta) + 20 * loss_neightboor + loss_agreement
            # A-W

            # W-D
            # if epoch > 25: 
            #     print("neightbor")
            #     loss =  (loss_sc_sa + loss_sc_ta) + 20 * loss_neightboor + loss_agreement
            # D-W
            # if epoch > 13: 
            #     print("neightbor")
            #     loss =  (loss_sc_sa + loss_sc_ta) + 20 * loss_neightboor + loss_agreement
            else:
                loss = (loss_sc_sa + loss_sc_ta) +  0 * loss_neightboor + loss_agreement
=======
            loss_sc_sa, loss_sc_ta, loss_neightboor, loss_agreement = self._forward([inputs_source, inputs_target], pids_source, pindexs_target, epoch, targetname, target_num_classes, fnames_target)

            optimizer[2].zero_grad()
            loss_agreement.backward(retain_graph=True)
            optimizer[2].step()
            # d-w,w-d:
            # if epoch > 9:
            # others:
            if epoch > 3:
                # D-W:0.3 * loss_neightboor
                # others:0.5 * loss_neightboor
                loss = (loss_sc_sa + loss_sc_ta) + 0.5 * loss_neightboor
            else:
                loss = (loss_sc_sa + loss_sc_ta) + 0 * loss_neightboor
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()
            loss.backward()
            optimizer[2].step()
            optimizer[1].step()
            optimizer[0].step()
                
            losses_IDE_s.update(loss_sc_sa.item(), pids_source.size(0))
            losses_IDE_t.update(loss_sc_ta.item(), pids_source.size(0))
            losses_triplet.update(loss_neightboor.item(), pids_source.size(0))
            losses_ag.update(loss_agreement.item(), pids_source.size(0))


            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 1 == 0:
                print('Epoch: [{}][{}/{}] \t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'task_1 {:.3f} ({:.3f})\t'
                      'task_2 {:.3f} ({:.3f})\t'
                      'neigtboor {:.3f} ({:.3f})\t'
                      'agreement {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, max(map(len, data_loader)),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_IDE_s.val, losses_IDE_s.avg,
                              losses_IDE_t.val, losses_IDE_t.avg,
                              losses_triplet.val, losses_triplet.avg,
                              losses_ag.val, losses_ag.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs, getname=False):
        pindexs, imgs, pids, fnames = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        pindexs = pindexs.to(self.device)
        if getname:
            return inputs, pids, pindexs, fnames
        return inputs, pids, pindexs

    def _forward(self, inputs, targets, index_target, epoch, targetname, target_num_classes, fnames_target=None, update_only=False):
        outputs_source = self.model[0](inputs[0])
        outputs_target = self.model[0](inputs[1])

        # mask
        source_mask = self.model[2](outputs_source)
        target_mask = self.model[2](outputs_target)
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

        
        outputs_scsa,_ = self.model[1](inputs_scsa)
        outputs_scta,_ = self.model[1](inputs_scta)

<<<<<<< HEAD
        outputs_tcsa, _ = self.model[1](inputs_tcsa)
        outputs_tcta, _ = self.model[1](inputs_tcta)

        loss_agreement = F.l1_loss(F.log_softmax(outputs_tcsa, dim=1),F.log_softmax(outputs_tcta, dim=1))
=======
        outputs_tcsa, _ = self.model[1](inputs_tcsa, domain="target")
        outputs_tcta, _ = self.model[1](inputs_tcta, domain="target")

        loss_agreement = F.l1_loss(F.log_softmax(outputs_tcsa, dim=1),F.log_softmax(outputs_tcta, dim=1))


>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
        loss_sc_sa = self.criterion[0](outputs_scsa, targets)
        loss_sc_ta = self.criterion[0](outputs_scta, targets)


        # # source & target invariance
<<<<<<< HEAD
        x = self.model[1](outputs_target_c)[0]
        _, tgt_feature = self.model[1](outputs_target, tgt_output_feature='pool5')
        loss_neightboor = self.InvNet(tgt_feature, index_target, epoch=epoch, fnames_target=fnames_target,predict_class=x.detach().clone(),model=self.model[1])
        return loss_sc_sa, loss_sc_ta, loss_neightboor, loss_agreement
        
=======
        # if targetname == 'amazon':
        tgt_class_feat, tgt_feature = self.model[1](outputs_target_c, tgt_output_feature='pool5')
        # else:
        # tgt_class_feat, tgt_feature = self.model[1](outputs_target, tgt_output_feature='pool5')
        loss_neightboor = self.InvNet(tgt_feature, index_target, epoch=epoch, fnames_target=fnames_target)

        return loss_sc_sa, loss_sc_ta, loss_neightboor, loss_agreement
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
