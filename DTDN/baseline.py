from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid.datasets.domain_adaptation import DA
from reid import datasets
from reid import models
from reid.loss import TripletLoss, InvNet
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler, IdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from reid.utils.CrossEntropyLabelSmooth import CrossEntropyLabelSmooth
import time


def get_data(data_dir, source, target, height, width, batch_size, re=0, workers=8):

    dataset = DA(data_dir, source, target)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids
    target_num_classes = dataset.target_pindex

    source_train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.Pad(10),
        T.RandomCrop((256,128)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5), 
        T.ColorJitter(brightness=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=0.5),
    ])
    target_train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.Pad(10),
        T.RandomCrop((256,128)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        normalizer,
    ])
    test_transformer = T.Compose([
        T.Resize((height,width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    source_train_loader = DataLoader(
        Preprocessor(dataset.source_train, root=osp.join(dataset.source_images_dir, dataset.source_train_path),
                     transform=source_train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)
    target_train_loader = DataLoader(
        Preprocessor(dataset.target_train, root=osp.join(dataset.target_images_dir, dataset.target_train_path),
                     transform=target_train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)
    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.target_images_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.target_images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, target_num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader

def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    dataset, num_classes, target_num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader = \
        get_data(args.data_dir, args.source, args.target, args.height,
                 args.width, args.batch_size, args.re, args.workers)

    if args.target == 'market1501_':
        target_num_classes += 1
    # Create model
    Encoder, TaskNet, DynamicNet = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes, target_num=target_num_classes, cut_layer='layer3')

    # d-m:knn=6
    # m-d:knn=8
    invNet = InvNet(args.features, target_num_classes, beta=0.05, knn=10, alpha=0.01).cuda()

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        Encoder.load_state_dict(checkpoint['Encoder'])
        DynamicNet.load_state_dict(checkpoint['DynamicNet'])
        TaskNet.load_state_dict(checkpoint['TaskNet'])
        invNet.load_state_dict(checkpoint['InvNet'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))


    Encoder = nn.DataParallel(Encoder).cuda()
    TaskNet = nn.DataParallel(TaskNet).cuda()
    DynamicNet = nn.DataParallel(DynamicNet).cuda()

    model = [Encoder, TaskNet, DynamicNet]

    # Evaluator
    evaluator = Evaluator([Encoder, TaskNet, DynamicNet])
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature)
        return

    # Criterion
    criterion = []
    criterion.append(nn.CrossEntropyLoss().cuda())
    # criterion.append(TripletLoss(margin=0.3,num_instances=4).cuda())

    # Optimizer
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
        step_size = 10
        if epoch <= 9:
            lr = 0.0008 * (epoch/10.0)
        elif epoch <= 14:
            lr = 0.1
        elif epoch <=20:
            lr = args.lr * (0.1 ** ((epoch-10) // step_size))
        elif epoch <=30:
            lr = 1e-5
        else:
            lr = 1e-6

        # if epoch <= 5:
        #     lr = 0.1
        # elif epoch <=11:
        #     lr = args.lr * (0.1 ** ((epoch-10) // step_size))
        # elif epoch <=21:
        #     lr = 1e-5
        # else:
        #     lr = 1e-6

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
        trainer.train(epoch, [source_train_loader, target_train_loader], [optimizer_Encoder, optimizer_Ide, optimizer_Att], args.batch_size)

        save_checkpoint({
            'Encoder': Encoder.module.state_dict(),
            'TaskNet': TaskNet.module.state_dict(),
            'DynamicNet': DynamicNet.module.state_dict(),
            'InvNet': invNet.state_dict(), 
            'epoch': epoch + 1,
        }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        evaluator = Evaluator(model)
        if epoch > 9 and epoch % 1 == 0:tmp=evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature)
       
        if(tmp>best):
            save_checkpoint({
            'Encoder': Encoder.module.state_dict(),
            'TaskNet': TaskNet.module.state_dict(),
            'DynamicNet': DynamicNet.module.state_dict(),
            'InvNet': invNet.state_dict(), 
            'epoch': epoch + 1,
            }, fpath=osp.join(args.logs_dir, 'best_checkpoint.pth.tar'))
            best=tmp
        print ("best:", best)
        print('\n * Finished epoch {:3d} \n'.
              format(epoch))

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='market1501',
                        choices=['market1501_', 'DukeMTMC-reID_', 'msmt', 'cuhk03_detected', 'VehicleID_V1.0', 'VeRi','SYSU','SYSU_IR'])
    # target
    parser.add_argument('-t', '--target', type=str, default='market1501',
                        choices=['market1501_', 'DukeMTMC-reID_', 'msmt', 'viper','VehicleID_V1.0', 'VeRi', 'PRID', 'GRID', 'VIPeR'])
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
