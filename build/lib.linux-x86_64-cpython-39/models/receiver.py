import torch
from torch import nn
import clip
from collections import OrderedDict
from models.basicmodels.unet import U_Net_S
from models.basicmodels.VisionTransformer import ViT
from models.RCNN import RCNN
import detectron2

CLIP_MODEL_BASES = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
# TRANFORMER_RESBLOCKS_KEYS = ['resblock0', 'resblock1', 'resblock2', 'resblock3', 'resblock4','resblock5', 'resblock6', 'resblock7', 'resblock8', 'resblock9','resblock10', 'resblock11']
# TRANFORMER_RESBLOCKS_KEYS = ['resblock0', 'resblock1', 'resblock3', 'resblock4', 'resblock6',  'resblock8', 'resblock9', 'resblock11']
TRANFORMER_RESBLOCKS_KEYS = ['resblock4', 'resblock6', 'resblock8', 'resblock11']
CLIP_TRANFORMER_RESBLOCKS = {'resblock0': 0,
                'resblock1': 1,
                'resblock2': 2,
                'resblock3': 3,
                'resblock4': 4,
                'resblock5': 5,
                'resblock6': 6,       
                'resblock7': 7,
                'resblock8': 8,            
                'resblock9': 9,             
                'resblock10': 10,             
                'resblock11': 11}

class Receiver(nn.Module):
    def __init__(self,__C):
        super().__init__()
        if __C.RECEIVER_V_ENC_ARC == 'clip':
            self.clip_arc = __C.CLIP['CLIP_MODEL']
            self.clipmodel, _ = clip.load(self.clip_arc, device='cuda', jit=False)
            self.freeze_model = __C.CLIP['CLIP_FREEZE_MODEL']
            if self.freeze_model:
                for p in self.clipmodel.parameters():
                    p.requires_grad = False
                self.clipmodel.eval()
            self.get_clip_hook()
            self.feature_fusion = CLIP_Feats_Fusion()
            # self.v_enc = self.clipmodel.visual
            self.enc = self.clip_enc
            self.enc = self.clip_enc()
            
        
        if __C.RECEIVER_V_ENC_ARC == 'vit':
            self.dim = __C.RECEIVER['DIM']
            self.depth = __C.RECEIVER['DEPTH']
            self.enc = ViT(image_size=224, patch_size=32, num_classes=100, dim=self.dim, depth=self.depth, heads=32, mlp_dim=1024, pool = 'cls', channels = 1, dim_head = 32, dropout = 0.1, emb_dropout = 0.1)

        if __C.RECEIVER_V_ENC_ARC == 'rcnn':
            self.enc = RCNN()

        self.fstate = OrderedDict()
        for gpu in __C.GPU.split(','):
            self.fstate[gpu] = OrderedDict()

    def get_clip_hook(self):
        if 'ViT' in self.clip_arc:
            for i in range(len(TRANFORMER_RESBLOCKS_KEYS)):
                key = TRANFORMER_RESBLOCKS_KEYS[i]
                lay = CLIP_TRANFORMER_RESBLOCKS[key]
                self.clipmodel.visual.transformer.resblocks[lay].register_forward_hook(self.make_hook(key))
                if key == 'resblock11':
                    break


        elif 'RN' in self.clip_model:
            self.clipmodel.visual.layer1.register_forward_hook(self.make_hook('layer1'))
            self.clipmodel.visual.layer2.register_forward_hook(self.make_hook('layer2'))
            self.clipmodel.visual.layer3.register_forward_hook(self.make_hook('layer3'))
            self.clipmodel.visual.layer4.register_forward_hook(self.make_hook('layer4'))
            print('pause')

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape)==3:
                # print(name, output.shape)
                self.fstate[str(output.device.index)][name] = output.permute(1, 0, 2) #LND -> NLD bs, smth, 768
            else:
                # print(name, output.shape)
                self.fstate[str(output.device.index)][name] = output

        return hook 

    def clip_enc(self, images):
        v_encoding = self.clipmodel.visual(images.repeat(1,3,1,1).to(torch.float16))
        v_feats = self.feature_fusion(self.fstate[str(images.device.index)])
        return v_feats

    def forward(self, images):
        v_feats = self.enc(images)
        return v_feats

class CLIP_Feats_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_feats_proj_list = []
        linear = nn.Linear(768,128)
        for i in range(len(TRANFORMER_RESBLOCKS_KEYS)):
            self.__setattr__('linear_'+str(i), linear)
        self.channels =  len(TRANFORMER_RESBLOCKS_KEYS) * 128
        self.deconv_1 = nn.ConvTranspose2d(128, 128, 2, stride=1)
        self.bn_1 = nn.BatchNorm2d(128)
        self.deconv_2 = nn.ConvTranspose2d(self.channels, 2048, 3, stride=1)
        self.bn_2 = nn.BatchNorm2d(2048)
        self.feature_fusion = U_Net_S(in_ch=self.channels, out_ch=self.channels)

    def forward(self, fstate):
        feat_list = []
        cls_feat_list = []
        for i in range(len(TRANFORMER_RESBLOCKS_KEYS)):
            feat_list.append(eval('self.linear_' + str(i))(fstate[TRANFORMER_RESBLOCKS_KEYS[i]].to(torch.float32)))
        for i in range(len(feat_list)):
            cls_feat_list.append([feat_list[i][0]])
            feat_list[i] = self.deconv_1(feat_list[i][:,1:].view((feat_list[i][:,1:].shape[0], feat_list[i][:,1:].shape[2], 7, 7)))
        feat_tensor = torch.cat(feat_list, 1)
        feat_tensor = self.feature_fusion(feat_tensor)
        feat_tensor = torch.relu(self.bn_2(self.deconv_2(feat_tensor)))
        feat_tensor = feat_tensor.permute(0,2,3,1).view(feat_tensor.shape[0], -1, feat_tensor.shape[1])
        return feat_tensor