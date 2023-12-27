from dataloader.vqav2_dataset import DataSet
# from cam.net import Net
from models.net import Net
import torchvision
from models.optim import get_optim, adjust_lr
from dataloader.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from models.sender import Sender
from models.receiver import Receiver
from collections import OrderedDict
from utils.loss.sender_loss import Sender_loss
from utils.AverageMeter import AverageMeter
from utils.BalancedDataParallel import BalancedDataParallel
import torch.distributed as dist
from models.sender import bandwidth_list
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '3615'
from dataloader.data_utils import my_collate
import torch.multiprocessing as mp
import torch.nn.functional as F
import math
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
from draft import visual_detector, detectron2_vis

class Game:
    def __init__(self, __C, DDP=False):
        self.__C = __C

        if not DDP:
            print('Loading training set ........')
            self.dataset = DataSet(__C)

            self.dataset_eval = None
            if __C.EVAL_EVERY_EPOCH:
                __C_eval = copy.deepcopy(__C)
                setattr(__C_eval, 'RUN_MODE', 'val')

                print('Loading validation set for per-epoch evaluation ........')
                self.dataset_eval = DataSet(__C_eval)
        

    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):

        # Load parameters
        if self.__C.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__C.CKPT_PATH
        else:
            path = self.__C.CKPTS_PATH + \
                   'ckpt_' + self.__C.CKPT_VERSION + \
                   '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'
            self.VERSION = self.__C.CKPT_VERSION
        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            checkpoint = torch.load(path, map_location='cuda:0')
            state_dict = checkpoint['state_dict']
            sender_dict = checkpoint['sender']
            receiver_dict = checkpoint['receiver']
            print('Finish!')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.eval()
        sender = Sender(self.__C)
        sender.cuda()
        sender.eval()
        receiver = Receiver(self.__C)
        receiver.cuda()
        if self.__C.N_GPU > 1:
            center_gpu_bs = self.__C.C_GPU_BS
            net = BalancedDataParallel(center_gpu_bs, net, device_ids=self.__C.DEVICES)
            sender = BalancedDataParallel(center_gpu_bs, sender, device_ids=self.__C.DEVICES)
            receiver = BalancedDataParallel(center_gpu_bs, receiver, device_ids=self.__C.DEVICES)

        if len(self.__C.DEVICES) <= 1:
            state_dict = key_matching(state_dict)
            sender_dict = key_matching(sender_dict)
            receiver_dict = key_matching(receiver_dict)
        net.load_state_dict(state_dict, strict=False)
        sender.load_state_dict(sender_dict, strict=False)
        receiver.load_state_dict(receiver_dict, strict=False)
        print('Finish!')

        for v in receiver.parameters():
            v.requires_grad = False

        receiver.eval()

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )
        if self.__C.SENDER['USE_CLIP']:
            _sender_loss = Sender_loss(self.__C)
            _sender_loss = _sender_loss.cuda()
            clip_distance = 0
        try:
            pretrained_vqa_path = self.__C.VQA_CKPT_PATH
            checkpoint_vqa = torch.load(pretrained_vqa_path, map_location='cuda:0')
            vqa_state_dict = checkpoint_vqa['state_dict']
            try:
                del vqa_state_dict['module.embedding.weight']
            except:
                pass
            try:
                del vqa_state_dict['embedding.weight']
            except:
                pass
            for k,v in net.named_parameters():
                if k in vqa_state_dict.keys():
                    print(k, ' is in the vqa_state_dict')
                    
            net.load_state_dict(vqa_state_dict, strict = False)
        except:
            pass

        try:
            pretrained_sender_path = self.__C.SENDER_CKPT_PATH
            checkpoint_sender = torch.load(pretrained_sender_path, map_location='cuda')
            sender_state_dict = checkpoint_sender['sender']
            if len(self.__C.DEVICES) <= 1:
                sender_state_dict = key_matching(sender_state_dict)
            sender.load_state_dict(sender_state_dict)
            for v in sender.parameters():
                v.requires_grad = False
        except:
            pass
        comm_ratio_sum = 0
        abstract_level = self.__C.SENDER['AB_FIXED']

        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter,
                img_dict_iter,
                ques,
        ) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')
            comm_loss = 0
            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()
            img_iter = img_dict_iter['rgb'].cuda()
            # if self.__C.SENDER['USE_DEPTH']:
            #     depth_iter = img_dict_iter['depth'].cuda() 
            
            if self.__C.SENDER['BW_LIMIT']:
                bandwidth_thre_level = self.__C.SENDER['BW_FIXED']
            else:
                bandwidth_thre_level = 0
            
            sketches, comm_mask, sketches_raw = sender(img_iter, b = bandwidth_thre_level, a = abstract_level)

            for round in range(self.__C.ROUND):
                if round != 0:
                    sketches, comm_ratio = self.ROIs_Sketch_Select(sketches, sketches_raw, ROIs_mask.to(sketches.device), bandwidth_list[bandwidth_thre_level])
                    comm_ratio_sum += comm_ratio.item()
                if self.__C.SENDER['PRAGMATIC']:
                    v_encoding, proposals, sketches_reco = receiver(sketches)
                    del sketches_reco
                else:
                    v_encoding, proposals = receiver(sketches)
                v_encoding.requires_grad = True
                time_3 = time.time()
                pred = net(
                    v_encoding,
                    ques_ix_iter
                )
                if self.__C.SENDER['USE_CLIP'] and round == self.__C.ROUND-1:
                    [_, sender_clip_loss, _, _] = _sender_loss(img_iter, sketches, sketches_raw, 0, comm_mask, comm_ratio = bandwidth_list[bandwidth_thre_level])
                    clip_distance += (sender_clip_loss.detach() / img_iter.shape[0])
                    del sender_clip_loss
                if self.__C.ROUND != 1 and round < self.__C.ROUND-1:
                    pred_temp = pred.argmax(1).detach()
                    v_encoding.requires_grad = True
                    pred_max = torch.cat([torch.gather(pred, 1, pred.argsort(dim=-1)[:,-1].unsqueeze(1)) for i in range(1)],dim=-1)
                    grad_matrix = torch.autograd.grad(pred_max.sum(), v_encoding, retain_graph=True)[0]
                    CAM_per_proposal_test = cal_cam(grad_matrix, v_encoding, pred_max)
                    CAM_per_proposal_test = torch.softmax(CAM_per_proposal_test, dim=1)
                    ROIs_mask = self.select_ROIs(CAM_per_proposal_test, proposals)


            torchvision.utils.save_image(sketches, './debug_imgs/qwe.jpg')
            torchvision.utils.save_image(ROIs_mask, './debug_imgs/ROIs_mask.jpg')   
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            if pred_argmax.shape[0] != self.__C.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.__C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)

            if self.__C.TEST_SAVE_PRED:
                if pred_np.shape[0] != self.__C.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.__C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)
        print('')
        ans_ix_list = np.array(ans_ix_list).reshape(-1)

        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix])
        }for qix in range(qid_list.__len__())]

        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.VERSION + \
                    '.json'

        else:
            if self.__C.CKPT_PATH is not None:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # Save the whole prediction vector
        if self.__C.TEST_SAVE_PRED:

            if self.__C.CKPT_PATH is not None:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            }for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)
            
        
        # Run validation script
        if valid:
            # create vqa object and vqaRes object
            ques_file_path = self.__C.QUESTION_PATH['val']
            ans_file_path = self.__C.ANSWER_PATH['val']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            # vqaEval.evaluate()
            vqaEval.evaluate_from_ans()
            # print accuracies
            print("\n")
            print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            # print("Per Question Type Accuracy is the following:")
            # for quesType in vqaEval.accuracy['perQuestionType']:
            #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            # print("\n")
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")

            if val_ckpt_flag:
                print('Write to log file: {}'.format(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.CKPT_VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.CKPT_VERSION + '.txt',
                    'a+'
                )

            else:
                print('Write to log file: {}'.format(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+'
                )

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            try:
                logfile.write("\n")
                logfile.write("BW Level: {}\n".format(str(self.__C.SENDER['BW_FIXED'])))
                try:
                    logfile.write("Extra Bandwidth Comsumption is: {}\n".format(str(comm_ratio_sum / step)))
                except:
                    pass
                try:
                    logfile.write("Average CLIP distance is: {}\n".format(str(clip_distance / step)))
                except:
                    pass
            except:
                pass
            logfile.write("\n\n")
            logfile.close()


    def run(self, run_mode, DDP_flag=False, rank=0, world_size=0):
        if DDP_flag:
            print('DDP enabled, Rank: ', rank)
            print('Loading training set ........')
            dataset = DataSet(self.__C)
            dataset_eval = None
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            self.train(dataset, dataset_eval, rank=rank, world_size=world_size)
        elif run_mode == 'train':
            self.empty_log(self.__C.VERSION)
            self.train(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')

    def get_bandwidth_thre_level(self, epoch):
        try:
            bandwidth_thre_level = self.__C.SENDER['BW_FIXED']
        except:    
            if epoch < 12:
                bandwidth_thre_level = 6 - int(epoch % 7) 
            else:
                bandwidth_thre_level = int(np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], p = [0.06, 0.08, 0.1, 0.1, 0.1, 0.12, 0.11, 0.09, 0.08, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02]))
        return bandwidth_thre_level
    
    def get_abstract_level(self, epoch):
        try:
            abstract_level = self.__C.SENDER['AB_FIXED']
        except:
            if epoch <= 2:
                abstract_level = 0
            elif epoch < 24:
                abstract_level = int((epoch-2) % 11) 
            else:
                abstract_level = int(np.round(np.random.uniform(0,10,1)))
        return abstract_level

    def select_ROIs(self, att_maps, proposals):
        multiround_mask = [self.select_ROIs_per_batch(att_map=att_maps[i], proposals=proposals[i]) for i in range(len(proposals))]
        return torch.stack(multiround_mask, 0)

    def select_ROIs_per_batch(self, att_map, proposals, select_num = 5, mask_shape = (224,224)):
        roi_mask = torch.zeros(mask_shape).to(att_map.device)
        if att_map.sum() == 0:
            return roi_mask.unsqueeze(0)

        else:
            average_attn = self.calculate_attn_per_pixel(proposals, att_map)
            att_map_sort = average_attn[:len(proposals)].argsort(descending=True)
            for i in range(min(len(proposals), select_num)):
                proposal = proposals[int(att_map_sort[i])]
                y_ratio = 224 / proposal.image_size[0]
                x_ratio = 224 / proposal.image_size[1]
                y_start = int(proposal._fields['proposal_boxes'].tensor[0][0] * x_ratio)
                y_end = int(proposal._fields['proposal_boxes'].tensor[0][2] * x_ratio)
                x_start = int(proposal._fields['proposal_boxes'].tensor[0][1] * y_ratio)
                x_end = int(proposal._fields['proposal_boxes'].tensor[0][3] * y_ratio)
                roi_mask[x_start:x_end, y_start:y_end] += att_map[int(att_map_sort[i])]/(math.pow((x_end - x_start)*(y_end - y_start), 0.25)+0.00001)
            return roi_mask.unsqueeze(0)/(roi_mask.mean()+0.00001)

    def calculate_attn_per_pixel(self, proposals, att_map):
        average_attn = []
        area_list = []
        if len(proposals) > 0:
            for i in range(len(proposals)):
                proposal = proposals[i]
                y_ratio = 224 / proposal.image_size[0]
                x_ratio = 224 / proposal.image_size[1]
                y_start = int(proposal._fields['proposal_boxes'].tensor[0][0] * x_ratio)
                y_end = int(proposal._fields['proposal_boxes'].tensor[0][2] * x_ratio)
                x_start = int(proposal._fields['proposal_boxes'].tensor[0][1] * y_ratio)
                x_end = int(proposal._fields['proposal_boxes'].tensor[0][3] * y_ratio)
                area = (x_end - x_start)*(y_end - y_start)
                area_list.append(area)                
                average_attn.append((200*att_map[i]/(math.pow((x_end - x_start)*(y_end - y_start), 0.5)+0.00001)).detach())
            average_attn = torch.tensor(average_attn,device = att_map.device)
            average_attn = torch.tensor(average_attn,device = att_map.device) - min(average_attn)
            for i, area in enumerate(area_list):
                if area < 500:
                    average_attn[i] = 0.00001

        return average_attn

    def ROIs_Sketch_Select(self, sketches, sketches_raw, ROIs_mask, comm_thre=0.02):
        sketches = 1 - sketches
        sketches_raw = 1 - sketches_raw
        ROIs_mask = ROIs_mask * torch.where(sketches>0, 0, 1)
        if ROIs_mask.max() > 0:
            comm_mask = sketches_raw * ROIs_mask / ROIs_mask.mean()
            thre_digit = 0.01 / comm_mask.max()
        else:
            comm_mask = sketches_raw * ROIs_mask
            thre_digit = comm_mask.max()+0.0001
        
        comm_mask = comm_mask/(comm_mask.max()+0.0001)
        rank_tensor = torch.sort(comm_mask.view(comm_mask.shape[0], -1))[0]
        select_index = int(rank_tensor.shape[-1] * (1 - comm_thre))
        select_thre = rank_tensor[:, select_index]

        select_thre = torch.where(select_thre>thre_digit, select_thre, torch.tensor(thre_digit).to(select_thre.device))
        comm_mask = comm_mask + (torch.round(comm_mask+(0.5 - select_thre[:,None,None,None].repeat(1,1,int(sketches.shape[-2]), int(sketches.shape[-1])))) - comm_mask).detach()
        sketches = (sketches_raw * comm_mask) + sketches
        sketches = 1 - sketches
        comm_ratio = torch.where((sketches_raw * comm_mask)>0, 1, 0).sum()/sketches.numel()
        return sketches, comm_ratio
    
    
    def ROI_filter(self, pred, ans_index, ROI_mask):
        mask_flag = []
        for i in range(ans_index.shape[0]):
            if ans_index[i] not in [3, 9]:
                certain_thres = 0.80
            else:
                certain_thres = 0.90
            if pred[i,ans_index[i]] > certain_thres:
                mask_flag.append(True)
            else:
                mask_flag.append(False)
        ROI_mask[mask_flag] = 0
        return ROI_mask


def key_matching(weights_dict):
    del_keys_list = []
    new_dict = OrderedDict()
    for k, v in weights_dict.items():
        if k.startswith('module.'):
            new_k = k.replace('module.', '') if 'module' in k else k
            new_dict[new_k] = v
            del_keys_list.append(k)
        else:
            new_dict[k] = v

    return new_dict

def key_matching_add(weights_dict):
    del_keys_list = []
    new_dict = OrderedDict()
    for k, v in weights_dict.items():
        new_k = 'module.' + k
        new_dict[new_k] = v
        del_keys_list.append(k)

    return new_dict

class net_receiver_package(nn.Module):
    def __init__(self, net, receiver):
        super(net_receiver_package, self).__init__()
        self.net = net
        self.receiver = receiver

    def forward(self, input_tensor):
        x, _ = self.receiver(input_tensor)
        x = self.net(x)
        return x

def cal_cam(gradients, activations, score):
    b, h, k = gradients.size()

    alpha_num = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, h, k).sum(-1, keepdim=True).view(b,h,1)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    # alpha = alpha_num.div(alpha_denom+1e-7)
    alpha = alpha_num.div(alpha_denom)
    positive_gradients = F.relu(score.unsqueeze(-1).exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
    weights = (alpha*positive_gradients).sum(-2, keepdim=True)
    saliency_map = (weights*activations).sum(-1, keepdim=True)
    saliency_map = F.relu(saliency_map).squeeze(-1)
    return saliency_map