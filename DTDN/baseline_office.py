from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import models
<<<<<<< HEAD
from reid.loss import TripletLoss, InvNet
=======
from reid.loss import TripletLoss, InvNet_office
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
from reid.trainers_office import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
#from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

import os
from itertools import groupby
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from torch.utils.data.sampler import Sampler, SequentialSampler

class Samplerrandom(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        import random 
        self.randomlist = [i for i in range(len(self.data_source))]
        random.shuffle(self.randomlist)

    def __iter__(self):
        return iter(self.randomlist)

    def __len__(self):
        return len(self.data_source)

class Office(ImageFolder):
<<<<<<< HEAD

=======
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
    def __getitem__(self, index):
        item = super(Office, self).__getitem__(index)
        fname = self.imgs[index][0]
        # print (index,":", fname)
        return (index, *item, fname)

    def __init__(self, root, partition, known_ids, unknown_ids, target=None, transform=None):
        root = os.path.join(root, partition, 'images')
        super(Office, self).__init__(root, transform)

        # 设置已知、未知 id
        data = []
<<<<<<< HEAD
        for (f1, c1), (f2, c2), c in zip(self.imgs, self.samples, self.targets):
            if c in known_ids:
                c = known_ids.index(c)
            # elif c in unknown_ids:
                # c = len(known_ids)
=======
        # from collections import defaultdict
        # cnt_unknown = defaultdict(int)
        for (f1, c1), (f2, c2), c in zip(self.imgs, self.samples, self.targets):
            if c in known_ids:
                c = known_ids.index(c)
            elif c in unknown_ids:
                # cnt_unknown[c] += 1
                # if cnt_unknown[c] > 8:
                #     continue
                c = len(known_ids)
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
            else:
                continue
            data.append([(f1, c), (f2, c), c, ])
        self.imgs, self.samples, self.targets = zip(*data)

        pindex = []
        if target is not None:
            # 区分每个样本是 train 还是 test
            mask, is_train = [], target == 'train'
<<<<<<< HEAD
            # TODO: 考虑随机划分
=======
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
            for c, samples_of_c in groupby(self.targets):
                size = len(list(samples_of_c))
                bk = round(size * 0.8)
                select = lambda i: i < bk if is_train else i >= bk
                mask.extend(map(select, range(size)))

            # 根据 mask 划分数据集
            old_data = zip(self.imgs, self.samples, self.targets)
            new_data = [data for chosen, data in zip(mask, old_data) if chosen]
            self.imgs, self.samples, self.targets = zip(*new_data)

def get_data(data_dir, source, target, height, width, batch_size, re=0, workers=8):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
<<<<<<< HEAD
            T.RandomHorizontalFlip(),
=======
            # T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
            T.ToTensor(),
            normalizer,
    ])

    test_transformer = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalizer,
    ])

    # source_train_dataset = Office(data_dir, source, range(0,10), range(10,20), None, train_transformer)
<<<<<<< HEAD
    # target_train_dataset = Office(data_dir, target, range(0,10), range(20,31), 'train',train_transformer)
    # target_test_dataset = Office(data_dir, target, range(0,10), range(20,31), 'test', test_transformer)

    source_train_dataset = Office(data_dir, source, range(0,31), range(10,20), None, train_transformer)
    target_train_dataset = Office(data_dir, target, range(0,31), range(20,31), 'train',train_transformer)
    target_test_dataset = Office(data_dir, target, range(0,31), range(20,31), 'test', test_transformer)

    from reid.loss.neightboor import TARGET
    for pindex, _, plabel, _ in target_train_dataset:
        TARGET[pindex] = plabel
    # from IPython import embed; embed();exit(0)
    

    source_sampler = Samplerrandom(source_train_dataset)
    target_sampler = Samplerrandom(target_train_dataset)
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=False, sampler=source_sampler, num_workers=workers, pin_memory=True, drop_last=False)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=False, sampler=target_sampler, num_workers=workers, pin_memory=True, drop_last=False)

    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    num_classes = 31
=======
    source_train_dataset = Office(data_dir, source, range(0,10), [], None, train_transformer)
    target_train_dataset = Office(data_dir, target, range(0,10), range(20,31), 'train',train_transformer)
    target_test_dataset = Office(data_dir, target, range(0,10), range(20,31), 'test', test_transformer)

    from reid.loss.neightboor_office import TARGET
    for pindex, _, plabel, _ in target_train_dataset:
        TARGET[pindex] = plabel

    source_sampler = Samplerrandom(source_train_dataset)
    target_sampler = Samplerrandom(target_train_dataset)
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False)

    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    num_classes = 11
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
    target_num_classes = len(target_train_dataset)


    return num_classes, target_num_classes, source_train_loader, target_train_loader, target_test_loader

def main(args):
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    num_classes, target_num_classes, source_train_loader, target_train_loader, target_test_loader = \
        get_data(args.data_dir, args.source, args.target, args.height, args.width, args.batch_size, args.re, args.workers)

    # Create model
    Encoder, TaskNet, DynamicNet = models.create(args.arch, num_features=args.features,
<<<<<<< HEAD
                          dropout=args.dropout, num_classes=num_classes, target_num = num_classes, cut_layer='layer3')
=======
                          dropout=args.dropout, num_classes=num_classes, target_num = target_num_classes, cut_layer='layer3')
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32

    # Load from checkpoint
    start_epoch = 0

    if target_num_classes > 0:
<<<<<<< HEAD
        # W->A,D->A: knn=18
        # other: knn=9
        # D->W: knn=12
        k = 10
        print ("knn:",k)
        invNet = InvNet(args.features, target_num_classes, beta=0.5, knn=k , alpha=0.01)
=======
        # W->A,D->A: k=6
        # k = 7
        k = 7
        print ("knn:",k)
        invNet = InvNet_office(2048, target_num_classes, beta=0.8, knn=k , alpha=0.01)
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
        invNet.cuda()

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        Encoder.load_state_dict(checkpoint['Encoder'])
        DynamicNet.load_state_dict(checkpoint['DynamicNet'])
        TaskNet.load_state_dict(checkpoint['TaskNet'])
        invNet.load_state_dict(checkpoint['InvNet'])
        start_epoch = checkpoint['epoch']
<<<<<<< HEAD
    
=======

    print ("start_epoch:", start_epoch)
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
    Encoder = nn.DataParallel(Encoder).cuda()
    TaskNet = nn.DataParallel(TaskNet).cuda()
    DynamicNet = nn.DataParallel(DynamicNet).cuda()

    model = [Encoder, TaskNet, DynamicNet]


    # Criterion
    criterion = []
    criterion.append(nn.CrossEntropyLoss().cuda())
<<<<<<< HEAD
    # criterion.append(TripletLoss(margin=args.margin).cuda())
=======
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32

    # Evaluator
    evaluator = Evaluator([Encoder, TaskNet, DynamicNet])
    if args.evaluate:
        print("Test:")
        evaluator.evaluator_classification(target_test_loader, criterion, args.target)
        return

    # Optimizer Encoder
    base_param_ids = set(map(id, Encoder.module.base.parameters()))
    new_params = [p for p in Encoder.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': Encoder.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}
    ]
    optimizer_Encoder = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Optimizer Ide
    base_param_ids = set(map(id, TaskNet.module.base.parameters()))
    new_params = [p for p in TaskNet.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': TaskNet.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}
    ] 
    optimizer_Ide = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    param_groups = [
        {'params':DynamicNet.module.parameters(), 'lr_mult':1.0},
    ]
    optimizer_Att = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = Trainer([Encoder, TaskNet, DynamicNet], criterion, InvNet=invNet)

    # Schedule learning rate
    def adjust_lr(epoch):
<<<<<<< HEAD

        ## AW
        # if epoch <= 5:
        #     lr = 0.01
        # elif epoch <= 10:
        #     lr = 0.0008
        # elif epoch <= 15:
        #     lr = 0.0001
        # else:
        #     lr = 0.00001

        # if epoch <= 9:
        #     lr = 1e-5 * (epoch/10.0)
        # elif epoch <= 20:
        #     lr = 0.001
        # elif epoch <= 29:
        #     lr = 0.0001
        # elif epoch <= 40:
        #     lr = 0.00001
        # else:
        #     lr = 0.000001

        # if epoch <= 15:
        #     lr = 0.001
        # elif epoch <= 16:
        #     lr = 0.0001
        # elif epoch <= 40:
        #     lr = 0.00001
        # else:
        #     lr = 0.000001

        #GY
        if epoch <= 5:
            lr = 0.01
        elif epoch <= 15:
            lr = 0.001
        elif epoch <= 20:
            lr = 0.0001
        elif epoch <= 40:
            lr = 0.00001
        else:
            lr = 0.000001
        ## WD
        # if epoch <= 15:
        #     lr = 0.01
        # elif epoch <= 30:
        #     lr = 0.0008

=======
        # # D-A: total_epoch = 60
        # if epoch <= 25:
        #     lr = 5e-5
        # elif epoch <= 35:
        #     lr = 1e-7
        # elif epoch <= 45:
        #     lr = 1e-7
        # else:
        #     lr = 1e-8

        # W-A total_epoch = 60
        # if epoch <= 18:
        #     lr = 1e-4
        # elif epoch <= 40:
        #     lr = 1e-5
        # elif epoch <= 50:
        #     lr = 1e-6
        # else:
        #     lr = 1e-7

        if epoch <= 20:
            lr = 1e-4
        elif epoch <= 40:
            lr = 1e-5
        elif epoch <= 50:
            lr = 1e-6
        else:
            lr = 1e-7

        # if epoch <= 3:
        #     lr = 1e-3
        # elif epoch <= 20:
        #     lr = 1e-4
        # else:
        #     lr = 1e-6

        # # W-A test:
        # if epoch <= 37:
        #     lr = 1e-4
        # elif epoch <= 20:
        #     lr = 1e-5
        # else:
        #     lr = 1e-6

        # A-W A-D total_epoch=60 W-D D-W  total_epoch=40
        # if epoch <= 100:
        #     lr = 1e-4
        # elif epoch <=45:
        #     lr = 1e-6
        # else:
        #     lr = 1e-8
 
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32

        print('Note: lr = {}'.format(lr))

        for g in optimizer_Encoder.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        for g in optimizer_Ide.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        for g in optimizer_Att.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    tmp=best=0
    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, [source_train_loader, target_train_loader], [optimizer_Encoder, optimizer_Ide, optimizer_Att], args.target, target_num_classes, batch_size=args.batch_size)
        save_checkpoint({
            'Encoder': Encoder.module.state_dict(),
            'TaskNet': TaskNet.module.state_dict(),
            'DynamicNet': DynamicNet.module.state_dict(),
            'InvNet': invNet.state_dict(), 
            'epoch': epoch + 1,
        }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        evaluate = Evaluator(model)
        tmp = evaluator.evaluator_classification(target_test_loader, criterion, args.target)
        if(tmp>best):
            save_checkpoint({
            'Encoder': Encoder.module.state_dict(),
            'TaskNet': TaskNet.module.state_dict(),
            'DynamicNet': DynamicNet.module.state_dict(),
            'InvNet': invNet.state_dict(), 
            'epoch': epoch + 1,
            }, fpath=osp.join(args.logs_dir, 'best_checkpoint.pth.tar'))
            best=tmp
        print('Best Rank-1:{:.1f}%'.format(best*100))

        print('\n * Finished epoch {:3d} \n'.
              format(epoch))
        

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluator_classification(target_test_loader, criterion, args.target)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='market1501',
                        choices=['market1501', 'DukeMTMC-reID', 'msmt', 'cuhk03_detected', 'VeRi', 'VehicleID_V1.0','amazon', 'dslr', 'webcam'])
    # target
    parser.add_argument('-t', '--target', type=str, default='market1501',
                        choices=['market1501', 'DukeMTMC-reID', 'msmt', 'viper', 'VeRi', 'VehicleID_V1.0', 'amazon', 'dslr', 'webcam'])
    # images
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="batch size for source")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)

    main(parser.parse_args())
