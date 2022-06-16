from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import random
import pdb
from glob import glob
import re



class DA(object):

    def __init__(self, data_dir, source, target):

        # source / target image root
        self.source_images_dir = osp.join(data_dir, source)
        self.target_images_dir = osp.join(data_dir, target)
        # training image dir
        self.source_train_path = 'bounding_box_train'
        self.target_train_path = 'bounding_box_train'
        self.target_train_fake_path = 'output'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'

        self.source_train, self.target_train, self.query, self.gallery = [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids, self.num_generate_ids, self.num_target_ids= 0, 0, 0, 0, 0
        self.target_num_cam = 6 if 'market' in target else 8
        self.source_num_cam = 6 if 'market' in source else 8
        self.pid_num = 0
        self.load()

    def preprocess(self, images_dir, path, relabel=True):
        self.pid_num = 0
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        # pattern = re.compile(r'([-\d]+)_c(\d)\w+([-\d]+)?\.jpg')
        pattern = re.compile(r'([-\d]+)_c?(\d+)(\w+)(?:-(\d+))?\.jpg')

        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        if fpaths == []:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.bmp')))
        for fpath in fpaths:
            # fname = osp.basename(fpath)
            fname = fpath
            pid, cam, _, pindex = map(str, pattern.search(fname).groups())
            # print (pid,cam, x,pindex)
            cam = int(cam)
            pid = int(pid)
            if not (pindex == 'None'): 
                pindex = int(pindex)
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = self.pid_num
                    self.pid_num += 1
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
                # else:
                #     continue
            pid = all_pids[pid]
            # assert 0 <= pid < 13164, f'{pid} {fpath}'
            if pid > self.pid_num: self.pid_num = pid
            cam -= 1
            ret.append((fname, pid, cam, pindex))
        
        # if relabel:
        #     for fname, pid, _, _ in ret:
        #         assert 0 <= pid < len(all_pids), f'{fname} {pid} {len(all_pids)}'
        return ret, int(len(all_pids)), len(ret)

    def randompreprocess(self, images_dir, path, relabel=True, isDuke=False):
        # # duke_pattern
        # pattern = re.compile(r'([-\d]+)_c?(\d)_f(\d+)_(\d+)')
        # # market_pattern
        # pattern = re.compile(r'([-\d]+)_c?(\d)s(\d)_(\d+)_(\d+)_(\d+)')        
        pattern = re.compile(r'([-\d]+)_c(\d)_?\w+(?:-(\d+))?\.jpg')

        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))

        random.shuffle(fpaths)
        for fpath in fpaths:
            fname = osp.basename(fpath)

            pid, cam, pindex = map(int, pattern.search(fname).groups())  
            # duke
            # _, cam, _, pid= map(int, pattern.search(fname).groups())
            # market
            # _, cam, _,_,_,pid= map(int, pattern.search(fname).groups())

            # pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = self.pid_num
                    self.pid_num += 1
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
                # else:
                #     continue
            pid = all_pids[pid]
            if pid > self.pid_num: self.pid_num = pid
            cam -= 1
            ret.append((fname, pid, cam, pindex))
            #if int(len(all_pids))>=200: break
        return ret, int(len(all_pids))    

    def domainpreprocess(self, images_dir, path, label=1):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        random.shuffle(fpaths)
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            pid = label
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))  

    def load(self):
        self.source_train, self.num_train_ids, self.source_pindex = self.preprocess(self.source_images_dir, self.source_train_path)
        self.target_train, self.num_target_ids, self.target_pindex = self.preprocess(self.target_images_dir, self.target_train_path)
        self.gallery, self.num_gallery_ids, _ = self.preprocess(self.target_images_dir, self.gallery_path, False)
        self.query, self.num_query_ids, _ = self.preprocess(self.target_images_dir, self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset          |  # ids  | # images")
        print("  ------------------------------------")
        print("  source train    |  {:5d}  | {:8d}"
              .format(self.num_train_ids, len(self.source_train)))
        print("  target train    | Unknown | {:8d}"
              .format(len(self.target_train)))
        print("  query           |  {:5d}  | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery         |  {:5d}  | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
