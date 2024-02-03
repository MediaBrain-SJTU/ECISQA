import sys
import torch
from cfgs.base_cfgs import Cfgs
from tools.game import Game
import argparse, yaml
import os
import torch.distributed as dist
import torch.utils.data.distributed
from torch.multiprocessing import Process

os.environ['MASTER_ADDR'] = '127.0.0.1' 
os.environ['MASTER_PORT'] = '3615'

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--yaml', dest='cfg',
                      help='path of the cfg files',
                      type=str, required=True)
    parser.add_argument('--DDP', action='store_true',
                      help='Whether to use Distributed Data Parallel')
    parser.add_argument('--port', default='9910', type=str, help='DDP port')
    args = parser.parse_args()
    return args


if __name__ == '__main__': 
    args = parse_args()
    with open(args.cfg, 'r') as f:
        config_yaml = yaml.safe_load(f)

    if args.DDP:
        if len(config_yaml['SETTING']['GPU']) > 1:
            config_yaml['SETTING']['BATCH_SIZE'] = int((config_yaml['SETTING']['BATCH_SIZE']-config_yaml['SETTING']['C_GPU_BS']) / (len(config_yaml['SETTING']['GPU']) - 1))
        else:
            config_yaml['SETTING']['BATCH_SIZE'] = 1
    __C = Cfgs(config_yaml['SETTING'])

    args_dict = {**config_yaml['MODEL'],**config_yaml['SETTING']}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)
  
    __C.check_path()

    if args.DDP:
        torch.multiprocessing.set_start_method('spawn')
        game = Game(__C, args.DDP)
        world_size = len(__C.GPU.split(','))
        __C.NUM_WORKERS = int(__C.NUM_WORKERS / world_size)
        
        processes = []
        for rank in range(world_size):
            p = Process(target=game.run, args=(__C.RUN_MODE, args.DDP, rank, world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        game = Game(__C)
        game.run(__C.RUN_MODE)