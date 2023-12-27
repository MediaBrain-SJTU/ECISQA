import torch
import torch.nn as nn
import clip
from collections import OrderedDict
from models.drawer import GlobalGenerator2
from models.receiver import TRANFORMER_RESBLOCKS_KEYS, CLIP_TRANFORMER_RESBLOCKS
from models.basicmodels.InceptionV3 import InceptionV3
from torchvision.transforms import Resize 
import copy


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, reduceme=False):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.reduceme = reduceme
        if use_lsgan:
            self.loss = nn.MSELoss(reduce=False)
        else:
            self.loss = nn.BCELoss(reduce=False)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        loss = self.loss(input, target_tensor).view(input.size(0), -1).mean(1)
        if self.reduceme:
            return torch.mean(loss)
        return loss

class Sender_loss(nn.Module):
    def __init__(self, __C):
        super(Sender_loss, self).__init__()
        self.USE_DEPTH = __C.SENDER['USE_DEPTH']
        self.BW_LIMIT = __C.SENDER['BW_LIMIT']
        if self.USE_DEPTH:
            self.get_clip_model(__C)
            self.clip_loss_func = nn.MSELoss(reduce=True)
            self.clip_sem_loss_func = nn.CosineSimilarity(dim=1, eps=1e-08)
            self.Geoloss_func = nn.BCELoss(reduce=True)
            self.net_recog = InceptionV3(55, False, use_aux=True, pretrain=True, freeze=True, every_feat=True)
            for v in self.net_recog.parameters():
                v.requires_grad=False
            self.net_recog.cuda()
            self.net_recog.eval()
        self.lambda_clip = __C.SENDER['LAMBDA_CLIP']
        self.lambda_geo = __C.SENDER['LAMBDA_GEO']
        self.lambda_comm = __C.SENDER['LAMBDA_COMM']

    def get_clip_model(self, __C):
        # if __C.RECEIVER_V_ENC_ARC == 'clip':
        if 1:
            self.clip_arc = __C.CLIP['CLIP_MODEL']
            self.clipmodel, _ = clip.load(self.clip_arc, device='cuda', jit=False)
            self.freeze_model = __C.CLIP['CLIP_FREEZE_MODEL']
            if self.freeze_model:
                for p in self.clipmodel.parameters():
                    p.requires_grad = False
                self.clipmodel.eval()
    
            self.clip_fstate = OrderedDict()
            self.get_clip_hook()
            # for gpu in __C.GPU.split(','):
                # self.clip_fstate[gpu] = OrderedDict()
            if __C.USE_DEPTH:
                self.geo_up = Resize([256,256])
                # self.geo_down = Resize([224,224])
                self.get_depth_model(__C)
                self.depth_loss_func = torch.nn.BCELoss(reduce=True)

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
                if str(output.device.index) not in self.clip_fstate.keys():
                    self.clip_fstate[str(output.device.index)] = OrderedDict()
                self.clip_fstate[str(output.device.index)][name] = output.permute(1, 0, 2) #LND -> NLD bs, smth, 768
            else:
                # print(name, output.shape)
                self.clip_fstate[str(output.device.index)][name] = output

        return hook 

    def cal_clip_loss(self, imgs, sketches):
        clip_loss = 0
        device_index = str(imgs.device.index)
        imgs_enc = self.clipmodel.visual(imgs.to(torch.float16)).detach()
        imgs_clip_feats = copy.deepcopy(self.clip_fstate[device_index])
        sketches_enc = self.clipmodel.visual(sketches.repeat(1,3,1,1).to(torch.float16))
        sketches_clip_feats = self.clip_fstate[device_index]
        for i in range(len(imgs_clip_feats)):
            clip_loss += self.clip_loss_func(imgs_clip_feats[TRANFORMER_RESBLOCKS_KEYS[i]], sketches_clip_feats[TRANFORMER_RESBLOCKS_KEYS[i]])
        clip_loss += torch.mean((1 - self.clip_sem_loss_func(sketches_enc, imgs_enc.detach())))
        return clip_loss

    def get_depth_model(self, __C):
        self.netGeom = GlobalGenerator2(768, 3, n_downsampling=1, n_UPsampling=3)
        self.netGeom.load_state_dict(torch.load(__C.DEPTH_CKPT))
        for k,v in self.netGeom.named_parameters():
            v.requires_grad = False
        self.netGeom.eval()
        self.netGeom.cuda()

    def cal_depth_loss(self, sketches, depth):
        _, sketches_feat = self.net_recog(self.geo_up(sketches).repeat(1, 3, 1, 1))
        pred_geom = self.netGeom(sketches_feat)
        pred_geom = (pred_geom+1)/2.0   ###[-1, 1] ---> [0, 1]
        geo_loss = self.depth_loss_func(pred_geom, depth)
        return geo_loss
    
    def cal_bw_thre_loss(self, comm_mask, comm_ratio = 0.05):
        bs, _, h, w = comm_mask.shape
        sum_pixels = h * w
        comm_loss_sum = 0.0
        for i in range(bs):
            comm_loss = comm_mask[i].sum()
            # print('comm_loss:', comm_loss)
            # print('comm_ratio:',comm_ratio)
            if comm_loss <= (sum_pixels * comm_ratio):
                # comm_loss = torch.zeros(1).to(comm_loss.device)
                comm_loss_sum += comm_loss * 0.05
            else:
                comm_loss_sum += comm_loss * 100
            return comm_loss_sum * self.lambda_comm

    def forward(self, imgs, sketches, depth=0, comm_mask=0, comm_ratio=0.05, comm_thre=0.05):
        sketches_temp = 1 - sketches
        # sketches_temp = sketches - (0.5-comm_thre)
        geo_loss = torch.zeros(1).to(sketches.device)
        clip_loss = torch.zeros(1).to(sketches.device)
        comm_loss = torch.zeros(1).to(sketches.device)
        if self.USE_DEPTH:
            clip_loss = self.cal_clip_loss(imgs, sketches) * self.lambda_clip
            if self.USE_DEPTH:
                geo_loss = self.cal_depth_loss(sketches, depth)
            geo_loss = geo_loss * self.lambda_geo
        if self.BW_LIMIT:
            comm_loss = self.cal_bw_thre_loss(comm_mask, comm_ratio)
        loss = clip_loss + geo_loss + comm_loss
        return [loss, clip_loss, geo_loss, comm_loss]
    

