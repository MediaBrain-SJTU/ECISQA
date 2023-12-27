# --------------------------------------------------------
# VQA-V2/loading images
# Licensed under The MIT License [see LICENSE for details]
# Last edited by Zixing 01.09
# --------------------------------------------------------

from .data_utils import img_feat_path_load, img_feat_load, ques_load, ans_load, tokenize, ans_stat
from .data_utils import proc_img_feat, proc_ques, proc_ans

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image
import cv2

class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C


        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # if self.__C.PRELOAD:
        # self.img_feat_path_list = []
        # split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        # for split in split_list:
        #     if split in ['train', 'val', 'test']:
        #         self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        # self.imgs_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        # for split in split_list:
        #     if split in ['train', 'val', 'test']:
        #         self.imgs_path_list += glob.glob(__C.IMGS_PATH[split] + '*.jpg')
        self.coco_image_path = __C.IMGS_PATH[split_list[0]]
        if self.__C.SENDER['USE_DEPTH']:
            self.coco_depth_path = __C.DEPTHS_PATH[split_list[0]]
        # if __C.EVAL_EVERY_EPOCH and __C.RUN_MODE in ['train']:
        #     self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz')

        # else:
        #     self.img_feat_path_list = \
        #         glob.glob(__C.IMG_FEAT_PATH['train'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['test'] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        # Loading answer word list
        # self.stat_ans_list = \
        #     json.load(open(__C.ANSWER_PATH['train'], 'r'))['annotations'] + \
        #     json.load(open(__C.ANSWER_PATH['val'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            # if __C.RUN_MODE in ['train']:
            self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']

        # Define run data size
        if __C.RUN_MODE in ['train']:
            # self.data_size = self.ans_list.__len__()
            self.data_size = self.ques_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        # if self.__C.PRELOAD:
        #     print('==== Pre-Loading features ...')
        #     time_start = time.time()
        #     self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
        #     time_end = time.time()
        #     print('==== Finished in {}s'.format(int(time_end-time_start)))
        # else:
        #     self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)
        self.aid_to_ques = ans_load(self.ans_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers statistic
        # Make answer dict during training does not guarantee
        # the same order of {ans_to_ix}, so we published our
        # answer dict to ensure that our pre-trained model
        # can be adapted on each machine.

        # Thanks to Licheng Yu (https://github.com/lichengunc)
        # for finding this bug and providing the solutions.

        # self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = ans_stat('dataloader/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')
        self.img_transform = transforms.Compose([transforms.Resize((224,224), Image.BICUBIC),
                transforms.ToTensor()])
        self.depth_transform = transforms.Compose([transforms.Resize((256,256), Image.BICUBIC, antialias=None),
                # transforms.Lambda(lambda img: self.__crop(img, (27,20), 256)),
                _crop((27,20), 256),
                transforms.ToTensor()])

    def coco_img_loading(self, img_id, mode):
        if mode == 'train':
            image_path = self.coco_image_path + 'COCO_train2014_' + str(img_id).zfill(12) + '.jpg'
        elif mode == 'val':
            image_path = self.coco_image_path + 'COCO_val2014_' + str(img_id).zfill(12) + '.jpg'
        elif mode == 'test':
            image_path = self.coco_image_path + 'COCO_test2015_' + str(img_id).zfill(12) + '.jpg'
        image = Image.open(image_path)
        image = self.img_transform(image)
        if image.shape[0]==1:  
            image = image.repeat(3,1,1)
        return image
    
    def coco_depth_image_loading(self, img_id, mode):
        if mode == 'train':
            image_path = self.coco_depth_path + 'COCO_train2014_' + str(img_id).zfill(12) + '-dpt_beit_large_512.png'
        elif mode == 'val':
            image_path = self.coco_depth_path + 'COCO_val2014_' + str(img_id).zfill(12) + '-dpt_beit_large_512.png'
        elif mode == 'test':
            image_path = self.coco_depth_path + 'COCO_test2015_' + str(img_id).zfill(12) + '-dpt_beit_large_512.png'
        # image = Image.open(image_path)
        img_depth = cv2.imread(image_path)
        img_depth = self.depth_transform(Image.fromarray(img_depth.astype(np.uint8)).convert('RGB'))
        # if image.shape[0]==1:  
        #     image = image.repeat(3,1,1)
        return img_depth

    def __getitem__(self, idx):

        # For code safety
        time_1 = time.time()
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            # ans = self.ans_list[idx]
            # ques = self.qid_to_ques[str(ans['question_id'])]
            # modification for modified dataset
            ques = self.ques_list[idx]
            ans = self.aid_to_ques[str(ques['question_id'])]

            # Process image feature from (.npz) file
            # if self.__C.PRELOAD:
            #     img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            # else:
            #     img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
            #     img_feat_x = img_feat['x'].transpose((1, 0))
            if self.__C.USE_FEAT:
                # img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
                img_feat_iter = np.zeros(1)
            else:
                img_feat_iter = np.zeros(1)
            # Process question
            time_2 = time.time()
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)
            time_3 = time.time()
            # print('load_question cost {}s'.format(time_3 - time_2))
            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)
            time_4 = time.time()
            # print('load_answer cost {}s'.format(time_4 - time_3))
            image = self.coco_img_loading(ans['image_id'], self.__C.RUN_MODE)
            image_dict = {}
            image_dict['rgb'] = image
            if self.__C.SENDER['USE_DEPTH']:
                depth_image = self.coco_depth_image_loading(ans['image_id'], self.__C.RUN_MODE)
                image_dict['depth'] = depth_image
            
                time_5 = time.time()
            # print('load_image cost {}s'.format(time_5 - time_4))

        else:
            # Load the run data from list
            ques = self.ques_list[idx]
            ans = self.aid_to_ques[str(ques['question_id'])]

            # # Process image feature from (.npz) file
            # img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
            # img_feat_x = img_feat['x'].transpose((1, 0))
            # Process image feature from (.npz) file
            # if self.__C.PRELOAD:
            #     img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            # else:
            #     img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                # img_feat_x = img_feat['x'].transpose((1, 0))
            # img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
            if self.__C.USE_FEAT:
                # img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
                img_feat_iter = np.zeros(1)
            else:
                img_feat_iter = np.zeros(1)

            # Process question
            ans_iter = proc_ans(ans, self.ans_to_ix)
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)
            image = self.coco_img_loading(ques['image_id'], self.__C.RUN_MODE)
            image_dict = {}
            image_dict['rgb'] = image
            # if self.__C.SENDER['USE_DEPTH']:
            #     depth_image = self.coco_depth_image_loading(ques['image_id'], self.__C.RUN_MODE)
            #     image_dict['depth'] = depth_image
            

        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter),\
               image_dict
            #    torch.from_numpy(image)
               


    def __len__(self):
        return self.data_size


    def __crop(self, img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        color = (255, 255, 255)
        if img.mode == 'L':
            color = (255)
        elif img.mode == 'RGBA':
            color = (255, 255, 255, 255)

        if (ow > tw and oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        elif ow > tw:
            ww = img.crop((x1, 0, x1 + tw, oh))
            return self.add_margin(ww, size, 0, (th-oh)//2, color)
        elif oh > th:
            hh = img.crop((0, y1, ow, y1 + th))
            return self.add_margin(hh, size, (tw-ow)//2, 0, color)
        return img
    

    def add_margin(self, pil_img, newsize, left, top, color=(255, 255, 255)):
        width, height = pil_img.size
        result = Image.new(pil_img.mode, (newsize, newsize), color)
        result.paste(pil_img, (left, top))
        return result

class _crop(torch.nn.Module):
    def __init__(self, pos, size):
        super().__init__()
        self.pos = pos
        self.size = size

    def forward(self, img):
        ow, oh = img.size
        x1, y1 = self.pos
        tw = th = self.size
        color = (255, 255, 255)
        if img.mode == 'L':
            color = (255)
        elif img.mode == 'RGBA':
            color = (255, 255, 255, 255)

        if (ow > tw and oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        elif ow > tw:
            ww = img.crop((x1, 0, x1 + tw, oh))
            return self.add_margin(ww, self.size, 0, (th-oh)//2, color)
        elif oh > th:
            hh = img.crop((0, y1, ow, y1 + th))
            return self.add_margin(hh, self.size, (tw-ow)//2, 0, color)
        return img