# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from dataloader.vqav2_dataset import DataSet
from models.net import Net
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
        

    def train(self, dataset, dataset_eval=None, rank=0, world_size=0):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb
        sender = Sender(self.__C)
        sender.cuda(device=rank)
        receiver = Receiver(self.__C)
        receiver.cuda(device=rank)
        # Define the MCAN model
        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda(device=rank)
        net.train()

        if self.__C.RUN_MODE == "train" and (self.__C.SENDER['USE_DEPTH'] or self.__C.SENDER['BW_LIMIT']):
            _sender_loss = Sender_loss(self.__C)
        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1 and world_size==0:
            center_gpu_bs = self.__C.C_GPU_BS
            # net = nn.DataParallel(net, device_ids=self.__C.DEVICES)
            # sender = nn.DataParallel(sender, device_ids=self.__C.DEVICES)
            # receiver = nn.DataParallel(receiver, device_ids=self.__C.DEVICES)
            net = BalancedDataParallel(center_gpu_bs, net, device_ids=self.__C.DEVICES)
            sender = BalancedDataParallel(center_gpu_bs, sender, device_ids=self.__C.DEVICES)
            receiver = BalancedDataParallel(center_gpu_bs, receiver, device_ids=self.__C.DEVICES)
        elif world_size>0:
            distributed_train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank],find_unused_parameters=False)
            sender = torch.nn.parallel.DistributedDataParallel(sender, device_ids=[rank], find_unused_parameters=False)
            receiver = torch.nn.parallel.DistributedDataParallel(receiver, device_ids=[rank], find_unused_parameters=False)
            if self.__C.RUN_MODE == "train" and (self.__C.SENDER['USE_DEPTH'] or self.__C.SENDER['BW_LIMIT']):
            #     _sender_loss = torch.nn.parallel.DistributedDataParallel(_sender_loss, device_ids=[rank])
                _sender_loss = _sender_loss.to(receiver.device)
        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda(device=rank)

        # Load checkpoint if resume training
        if self.__C.RESUME:
            print(' ========== Resume training')

            if self.__C.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.__C.CKPT_PATH
            else:
                path = self.__C.CKPTS_PATH + \
                       'ckpt_' + self.__C.CKPT_VERSION + \
                       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'
                self.__C.VERSION = self.__C.CKPT_VERSION
                
            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            checkpoint = torch.load(path, map_location='cuda:0')
            state_dict = checkpoint['state_dict']
            sender_dict = checkpoint['sender']
            receiver_dict = checkpoint['receiver']
            if len(self.__C.DEVICES) <= 1:
                state_dict = key_matching(state_dict)
                sender_dict = key_matching(sender_dict)
                receiver_dict = key_matching(receiver_dict)
            net.load_state_dict(state_dict, strict=False)
            sender.load_state_dict(sender_dict, strict=False)
            receiver.load_state_dict(receiver_dict, strict=False)
            print('Finish!')
            # print('Finish!')
            # net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, [net, sender, receiver], data_size, checkpoint['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(checkpoint['optimizer'])

            start_epoch = self.__C.CKPT_EPOCH

        else:
            if rank == 0:
                if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                    shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

                os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            optim = get_optim(self.__C, [net, sender, receiver], data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        if self.__C.SHUFFLE_MODE in ['external']:
            if world_size == 0:
                dataloader = Data.DataLoader(
                    dataset,
                    batch_size=self.__C.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.__C.NUM_WORKERS,
                    pin_memory=self.__C.PIN_MEM,
                    drop_last=True
                )
            else:
                if rank != 0:
                    bs_r = self.__C.BATCH_SIZE
                else:
                    bs_r = self.__C.C_GPU_BS
                dataloader = Data.DataLoader(
                    dataset,
                    batch_size=bs_r,
                    shuffle=False,
                    num_workers=self.__C.NUM_WORKERS,
                    pin_memory=self.__C.PIN_MEM,
                    drop_last=True,
                    sampler=distributed_train_sampler
                )
        else:
            if world_size == 0:
                dataloader = Data.DataLoader(
                    dataset,
                    batch_size=self.__C.BATCH_SIZE,
                    shuffle=True,
                    num_workers=self.__C.NUM_WORKERS,
                    pin_memory=self.__C.PIN_MEM,
                    drop_last=True
                )
            else:
                dataloader = Data.DataLoader(
                    dataset,
                    batch_size=self.__C.BATCH_SIZE,
                    shuffle=True,
                    num_workers=self.__C.NUM_WORKERS,
                    pin_memory=self.__C.PIN_MEM,
                    drop_last=True,
                    sampler=distributed_train_sampler
                )

        running_loss_disp = AverageMeter('Total loss', ':.6f')
        if self.__C.USE_DEPTH:
            running_draw_loss_disp = AverageMeter('Draw loss', ':.6f')
        running_game_loss_disp = AverageMeter('Game loss', ':.6f')
        running_comm_loss_disp = AverageMeter('Comm loss', ':.6f')
        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            if world_size > 0:
                dataloader.sampler.set_epoch(epoch)
            running_loss_disp.reset()
            if self.__C.USE_DEPTH:
                running_draw_loss_disp.reset()
            running_game_loss_disp.reset()
            running_comm_loss_disp.reset()
                    # Save log information
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            # Externally shuffle
            if self.__C.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            time_1 = time.time()
            time_5 = time.time()
            comm_loss = torch.zeros(1).cuda(device=rank)
            # Iteration
            for step, (
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter,
                    img_dict_iter
            ) in enumerate(dataloader):
                # print("load time {}s".format(time.time()-time_5))
                optim.zero_grad()
                if self.__C.SENDER['BW_LIMIT']:
                    bandwidth_thre_level = self.get_bandwidth_thre_level(epoch-start_epoch)
                else:
                    bandwidth_thre_level = 0
                if self.__C.SENDER['USE_DEPTH']:
                    abstract_level = self.get_abstract_level(epoch)
                else:
                    abstract_level = 0
                img_feat_iter = img_feat_iter.cuda(device=rank)
                ques_ix_iter = ques_ix_iter.cuda(device=rank)
                ans_iter = ans_iter.cuda(device=rank)
                img_iter = img_dict_iter['rgb'].cuda(device=rank)
                if self.__C.SENDER['USE_DEPTH']:
                    depth_iter = img_dict_iter['depth'].cuda(device=rank) 
                else:
                    depth_iter = 0
                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    time_0 = time.time()
                    sketches, comm_mask = sender(img_iter, b = bandwidth_thre_level, a = abstract_level)
                    # sketches = torch.rand(sketches.shape).to(sketches.device)
                    time_1 = time.time()
                    # print("Sender Cost {}s".format(str(time_1 - time_0)))

                    if self.__C.SENDER['USE_DEPTH'] or self.__C.SENDER['BW_LIMIT']:
                        [sender_loss, sender_clip_loss, sender_geo_loss, comm_loss] = _sender_loss(img_iter, sketches, depth_iter, comm_mask, comm_ratio = bandwidth_list[bandwidth_thre_level])
                    # else:
                    #     [sender_loss, sender_clip_loss, sender_geo_loss, comm_loss] = _sender_loss(img_iter, sketches)
                        draw_loss = sender_clip_loss + sender_geo_loss
                    time_2 = time.time()
                    # print("Sender Loss Cost {}s".format(str(time_2 - time_1)))
                    v_encoding = receiver(sketches)
                    time_3 = time.time()
                    # print("Receiver Cost {}s".format(str(time_3 - time_2)))
                    pred = net(
                        v_encoding,
                        sub_ques_ix_iter
                    )
                    time_4 = time.time()
                    # print("VQA Cost {}s".format(str(time_4 - time_3)))
                    # pred = net(
                    #     sub_img_feat_iter,
                    #     sub_ques_ix_iter
                    # )

                    game_loss = loss_fn(pred, sub_ans_iter) / self.__C.SUB_BATCH_SIZE 
                    if self.__C.SENDER['USE_DEPTH']:
                        loss = game_loss + draw_loss * abstract_level * 0.1
                    else:
                        loss = game_loss
                    if self.__C.SENDER['BW_LIMIT']:
                        loss = loss + comm_loss
                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.__C.GRAD_ACCU_STEPS
                    # torch.autograd.set_detect_anomaly(True)
                    loss.backward()
                    time_5 = time.time()
                    # print("BackWard Cost {}s".format(str(time_5 - time_4)))
                    loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                    if self.__C.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['val']
                        else:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['test']
                        time_now = time.time()
                        running_loss_disp.update(loss)
                        if self.__C.USE_DEPTH:
                            running_draw_loss_disp.update(draw_loss)
                        running_game_loss_disp.update(game_loss)
                        running_comm_loss_disp.update(comm_loss)
                        if world_size == 0:
                            if self.__C.SENDER['USE_DEPTH']:
                                print("\r[version %s][epoch %2d][step %4d/%4d][%s] game_loss: %.4f, draw_loss: %.4f, comm_loss: %.4f, loss: %.4f, lr: %.2e, costtime: %10s" % (
                                    self.__C.VERSION,
                                    epoch + 1,
                                    step,
                                    int(data_size / self.__C.BATCH_SIZE),
                                    mode_str,
                                    running_game_loss_disp.avg.item(),
                                    running_draw_loss_disp.avg.item(),
                                    running_comm_loss_disp.avg.item(),
                                    running_loss_disp.avg.item(),
                                    # game_loss.cpu().data.numpy(),
                                    # draw_loss.cpu().data.numpy(),
                                    # comm_loss.cpu().data.numpy(),
                                    # loss.cpu().data.numpy(),
                                    # game_loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                                    optim._rate,
                                    str(time_now - time_start)
                                ), end='          ')
                            else: 
                                print("\r[version %s][epoch %2d][step %4d/%4d][%s] game_loss: %.4f, loss: %.4f, lr: %.2e, costtime: %10s" % (
                                    self.__C.VERSION,
                                    epoch + 1,
                                    step,
                                    int(data_size / self.__C.BATCH_SIZE),
                                    mode_str,
                                    running_game_loss_disp.avg.item(),
                                    running_loss_disp.avg.item(),
                                    # game_loss.cpu().data.numpy(),
                                    # loss.cpu().data.numpy(),
                                    # game_loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                                    optim._rate,
                                    str(time_now - time_start)
                                ), end='          ')
                        elif rank == 1:
                            if self.__C.SENDER['USE_DEPTH']:
                                print("\r[version %s][epoch %2d][step %4d/%4d][%s] game_loss: %.4f, draw_loss: %.4f, comm_loss: %.4f, loss: %.4f, lr: %.2e, costtime: %10s" % (
                                    self.__C.VERSION,
                                    epoch + 1,
                                    step,
                                    int(data_size / self.__C.BATCH_SIZE),
                                    mode_str,
                                    running_game_loss_disp.avg.item(),
                                    running_draw_loss_disp.avg.item(),
                                    running_comm_loss_disp.avg.item(),
                                    running_loss_disp.avg.item(),
                                    # game_loss.cpu().data.numpy(),
                                    # draw_loss.cpu().data.numpy(),
                                    # comm_loss.cpu().data.numpy(),
                                    # loss.cpu().data.numpy(),
                                    # game_loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                                    optim._rate,
                                    str(time_now - time_start)
                                ), end='          ')
                            else: 
                                print("\r[version %s][epoch %2d][step %4d/%4d][%s] game_loss: %.4f, loss: %.4f, lr: %.2e, costtime: %10s" % (
                                    self.__C.VERSION,
                                    epoch + 1,
                                    step,
                                    int(data_size / self.__C.BATCH_SIZE),
                                    mode_str,
                                    running_game_loss_disp.avg.item(),
                                    running_loss_disp.avg.item(),
                                    # game_loss.cpu().data.numpy(),
                                    # loss.cpu().data.numpy(),
                                    # game_loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                                    optim._rate,
                                    str(time_now - time_start)
                                ), end='          ')

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            if world_size == 0:
                state = {
                    'state_dict': net.state_dict(),
                    'receiver': receiver.state_dict(),
                    'sender': sender.state_dict(),
                    'optimizer': optim.optimizer.state_dict(),
                    'lr_base': optim.lr_base
                }
                torch.save(
                    state,
                    self.__C.CKPTS_PATH +
                    'ckpt_' + self.__C.VERSION +
                    '/epoch' + str(epoch_finish) +
                    '.pkl'
                )
            elif rank == 1:
                state = {
                    'state_dict': net.state_dict(),
                    'receiver': receiver.state_dict(),
                    'sender': sender.state_dict(),
                    'optimizer': optim.optimizer.state_dict(),
                    'lr_base': optim.lr_base
                }
                torch.save(
                    state,
                    self.__C.CKPTS_PATH +
                    'ckpt_' + self.__C.VERSION +
                    '/epoch' + str(epoch_finish) +
                    '.pkl'
                )
            # Logging
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )

            # if self.__C.VERBOSE:
            #     logfile = open(
            #         self.__C.LOG_PATH +
            #         'log_run_' + self.__C.VERSION + '.txt',
            #         'a+'
            #     )
            #     for name in range(len(named_params)):
            #         logfile.write(
            #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
            #                 str(name),
            #                 named_params[name][0],
            #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
            #             )
            #         )
            #     logfile.write('\n')
            #     logfile.close()

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


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
        receiver.eval()
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        state_dict = key_matching(state_dict)
        net.load_state_dict(state_dict, strict=False)
        sender_dict = key_matching(sender_dict)
        sender.load_state_dict(sender_dict, strict=False)
        receiver_dict = key_matching(receiver_dict)
        receiver.load_state_dict(receiver_dict, strict=False)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter,
                img_dict_iter
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
                bandwidth_thre_level = self.get_bandwidth_thre_level(9)
            else:
                bandwidth_thre_level = 0
            if self.__C.SENDER['USE_DEPTH']:
                abstract_level = self.get_abstract_level(0)
            else:
                abstract_level = 0
            sketches, comm_mask = sender(img_iter, b=bandwidth_thre_level, a=abstract_level)
            # sketches = torch.ones(sketches.shape).to(sketches.device)
            v_encoding = receiver(sketches)
            # v_encoding = torch.zeros(v_encoding.shape).cuda()
            pred = net(
                v_encoding,
                ques_ix_iter
            )
            # pred = net(
            #     img_feat_iter,
            #     ques_ix_iter
            # )
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            # Save the answer index
            if pred_argmax.shape[0] != self.__C.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.__C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)

            # Save the whole prediction vector
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

        # Write the results to result file
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
            if epoch < 18:
                bandwidth_thre_level = 8 - int(epoch % 9) 
            else:
                bandwidth_thre_level = int(np.round(np.random.uniform(0,9,1)))
        return bandwidth_thre_level
    
    def get_abstract_level(self, epoch):
        try:
            abstract_level = self.__C.SENDER['BW_FIXED']
        except:
            if epoch <= 2:
                abstract_level = 0
            elif epoch < 24:
                abstract_level = int((epoch-2) % 11) 
            else:
                abstract_level = int(np.round(np.random.uniform(0,10,1)))
        return abstract_level
            
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