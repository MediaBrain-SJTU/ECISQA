# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self, __C):

        # vqav2 dataset root path
        # self.DATASET_PATH = '/GPFS/public/MCAN/datasets/vqa/'

        # # bottom up features root path
        # self.FEATURE_PATH = '/GPFS/public/MCAN/datasets/coco_extract/'

        # self.IMG_PATH = '/GPFS/public/coco2014'

        self.DATASET_PATH = __C['DATASET_PATH']

        # bottom up features root path
        self.FEATURE_PATH = __C['FEATURE_PATH']

        self.IMG_PATH = __C['IMG_PATH']

        self.USE_DEPTH = False
        if __C['SENDER']['USE_DEPTH']:
            self.USE_DEPTH = True
            self.DEPTH_PATH = __C['DEPTH_PATH']


        self.init_path()


    def init_path(self):

        # self.IMG_FEAT_PATH = {
        #     'train': self.FEATURE_PATH + 'train2014/',
        #     'val': self.FEATURE_PATH + 'val2014/',
        #     'test': self.FEATURE_PATH + 'test2015/',
        # }

        self.IMGS_PATH = {
            'train': self.IMG_PATH + 'train2014/',
            'val': self.IMG_PATH + 'val2014/',
            'test': self.IMG_PATH + 'test2015/',
        }

        if self.USE_DEPTH:
            self.DEPTHS_PATH = {
            'train': self.DEPTH_PATH + 'train2014_depth/',
            'val': self.DEPTH_PATH + 'val2014_depth/',
            'test': self.DEPTH_PATH + 'test2015_depth/',
        }

        self.QUESTION_PATH = {
            'train': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'val': self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
            'test': self.DATASET_PATH + 'v2_OpenEnded_mscoco_test2015_questions.json',
            'vg': self.DATASET_PATH + 'VG_questions.json',
        }

        self.ANSWER_PATH = {
            'train': self.DATASET_PATH + 'v2_mscoco_train2014_annotations.json',
            'val': self.DATASET_PATH + 'v2_mscoco_val2014_annotations.json',
            'vg': self.DATASET_PATH + 'VG_annotations.json',
        }

        self.RESULT_PATH = './logs/log_test/'
        self.PRED_PATH = './logs/pred/'
        self.CACHE_PATH = './logs/cache/'
        self.LOG_PATH = './logs/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'log_test' not in os.listdir('./logs'):
            os.mkdir('./logs/log_test')

        if 'pred' not in os.listdir('./logs'):
            os.mkdir('./logs/pred')

        if 'cache' not in os.listdir('./logs'):
            os.mkdir('./logs/cache')

        if 'log' not in os.listdir('./logs'):
            os.mkdir('./logs/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self):
        print('Checking dataset ...')

        # for mode in self.IMG_FEAT_PATH:
        #     if not os.path.exists(self.IMG_FEAT_PATH[mode]):
        #         print(self.IMG_FEAT_PATH[mode] + 'NOT EXIST')
        #         exit(-1)

        for mode in self.QUESTION_PATH:
            if not os.path.exists(self.QUESTION_PATH[mode]):
                print(self.QUESTION_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.ANSWER_PATH:
            if not os.path.exists(self.ANSWER_PATH[mode]):
                print(self.ANSWER_PATH[mode] + 'NOT EXIST')
                exit(-1)

        print('Finished')
        print('')

