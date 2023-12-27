# import os
# import torch

# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

# F = "/GPFS/rhome/yimingzhang/my_projects/bottom-up-attention.pytorch/output/model_final.pth"

# net_load = torch.load(F)

# print("----------ckpt0-----------")

# print(type(net_load['model']))

# print(net_load['model']['pixel_mean'])
# print(net_load['model']['pixel_std'])

# print(net_load['model'].keys())

# # model = GeneralizedRCNN()

# # model.load_state_dict(net_load['model'])

# # model.load

# print("----------ckpt1-----------")

import os
import sys
import time
import argparse
import torch
sys.path.append('detectron2')

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.utils.registry import Registry
from bua import add_config


REGISTRY = Registry("META_ARCH")

os.environ["CUDA_VISIBLE_DEVICES"]="5"

def parse_opt():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="BottomUpAttention Training")
    parser.add_argument("--mode", default="d2", type=str, help="'caffe' and 'd2' indicates \
                        'use caffe model' and 'use detectron2 model'respectively")
    parser.add_argument("--config-file", default="/GPFS/rhome/yimingzhang/my_projects/RCNN_test/rcnn/configs/d2/test-d2-r101.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODE = args.mode
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def RCNN():
    args = parse_opt().parse_args()
    # print(args)
    cfg = setup(args)
    print(cfg)

    model = DefaultTrainer.build_model(cfg)

    # exit()
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

    print(type(model))

    return model


# print(cfg)
