from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import pdb

import torch
import numpy as np

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter

from torch.autograd import Variable
from .utils import to_torch
from .utils import to_numpy
import pdb

import os, math
from torchvision.utils import make_grid, save_image

import os.path as osp
import shutil
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from PIL import Image
from .models import resnet
import torch.nn as nn

import cv2
import torch.nn.functional as F
from .models import upsample






def visualize_ranked_results(distmat, query, gallery, save_dir, topk=20):
    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))
    
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            if not osp.exists(dst):
                os.makedirs(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            if not osp.exists(dst):
                os.makedirs(dst)
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid, _ = query[q_idx]
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))
        _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid, _ = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
                if rank_idx > topk:
                    break

    print("Done")

def save_feature_image(feature_batch, path, name):
    for i in range(feature_batch.size(0)):
        feature_map = feature_batch[i].detach().cpu().unsqueeze(dim=1)
        num_channels, height = feature_map.size(0), feature_map.size(2)
        nrow, padding = round(math.sqrt(num_channels)), height // 2
        for j in range(512):
            n = osp.basename(name[i])
            if not os.path.exists(path + "/" + n):
                os.makedirs(path + "/" + n)
            image = make_grid(feature_map[j], nrow=1, padding=padding, normalize=False, pad_value=1)
            filename = os. path.join(path, '%s/%d-%d.png' % (n, j, i))
            save_image(image, filename)
        print(filename)


def extract_cnn_feature(model, inputs, fnames, output_feature=None, name=None, pids=None):
    model[0].eval()
    model[1].eval()
    model[2].eval()

    inputs = to_torch(inputs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model[0](inputs)
        outputs *= (model[2](outputs))
        outputs = model[1](outputs, output_feature='pool5')
        outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=1, output_feature=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs, fnames, output_feature, 'source', pids)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        # if (i + 1) % print_freq == 0:
        #     print('Extract Features: [{}/{}]\t'
        #           'Time {:.3f} ({:.3f})\t'
        #           'Data {:.3f} ({:.3f})\t'
        #           .format(i + 1, len(data_loader),
        #                   batch_time.val, batch_time.avg,
        #                   data_time.val, data_time.avg))

    return features, labels

def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    print ("dist:", dist)
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _ in query]
        gallery_ids = [pid for _, pid, _, _ in gallery]
        query_cams = [cam for _, _, cam, _ in query]
        gallery_cams = [cam for _, _, cam, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))

    return cmc_scores['market1501'][0]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    # retrieve
    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature=None, rerank=False, save_dir=None):
        query_features, _ = extract_features(self.model, query_loader, 1, output_feature)
        gallery_features, _ = extract_features(self.model, gallery_loader, 1, output_feature)
        if rerank:
            distmat = reranking(query_features, gallery_features, query, gallery)
        else:
            distmat = pairwise_distance(query_features, gallery_features, query, gallery)
            if save_dir is not None:
                visualize_ranked_results(distmat, query, gallery, save_dir)
        return evaluate_all(distmat, query=query, gallery=gallery)
    
    # classification
    def evaluator_classification(self, inputss, criterion, target, epoch=0):
        self.model[0].eval()
        self.model[1].eval()
        self.model[2].eval()

        with torch.no_grad():
            print("=> Evaluating...")

            allnum = [0 for _ in range(11)]
            os_acc = [0 for _ in range(11)]
            acc = [0 for _ in range(11)]
            acc_score = [0 for _ in range(11)]
            soft_logits = []
            soft_logits_all = []
            pre_class = []
            groundtruth = []
            score = [0 for _ in range(10)]
            score_num = [0 for _ in range(10)]
            for i, (_, inputs, targets,_) in enumerate(inputss):

                inputs = inputs.cuda()
                targets = targets.cuda()

                with torch.no_grad():
                    outputs = self.model[0](inputs)
                    # if target == 'amazon':
                    outputs *=  self.model[2](outputs)
                    logits, _ = self.model[1](outputs)

                # loss = criterion[0](logits, targets)

                #measure accuracy and record loss
                # a = logits.argmax(dim=1)
                # for i in range(len(a)):
                #     if a[i] != targets[i]:
                #         print (logits[i])
                #         print (a[i],targets[i])

                #measure accuracy and record loss

                pre_class += logits.argmax(dim=-1).tolist()
                soft_logits += torch.nn.Softmax(dim=-1)(logits).max(dim=-1).values.tolist()
                soft_logits_all += torch.nn.Softmax(dim=-1)(logits).tolist()
                groundtruth += targets.tolist()

                for i in range(len(targets)):
                    allnum[targets[i]] += 1
                    acc_score[targets[i]] += torch.nn.Softmax(dim=-1)(logits)[i].max()

            for i in range(len(pre_class)):
                if pre_class[i] == groundtruth[i]:
                    acc[groundtruth[i]] += 1
                # if groundtruth[i] == 8:
                #     print (pre_class[i], ":", soft_logits[i], soft_logits_all[i][8])
                if soft_logits[i] < 0.3 and pre_class[i]!=8:
                # if soft_logits[i] < 0.25:
                    pre_class[i] = 10
                if pre_class[i] == groundtruth[i]:
                    os_acc[groundtruth[i]] += 1

            # print (score)
            # print (score_num)
            print(allnum)
            print (acc)
            print (os_acc)
            num = 0
            for i in range(10):
                num += os_acc[i]*1.0/allnum[i]
            print ("OS*:",num/10)
            print ("OS:", (num+os_acc[10]/allnum[10])/11)
            # print ("all:", sum([acc[i]/allnum[i] for i in range(11)])/11)
            # print ("all:", sum([acc[i]/allnum[i] for i in range(10)])/10)
            
            # print(
                # 'Test: '
                # 'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f} ({n:d})'.format(
                # top1=top1, top5=top5, n=top1.count))

        return (num+os_acc[10]/allnum[10])/11 * 100
    
