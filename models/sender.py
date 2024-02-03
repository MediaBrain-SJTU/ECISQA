import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
from models.drawer import Generator, Generator_downsampling, Generator_upsampling
from models.basicmodels.unet import U_Net_S
import time, torchvision

class Sender(nn.Module):
    def __init__(self, __C):
        super().__init__()
        try:
            self.Geometric_Flag = __C.SENDER['GEOMETRIC'] and (__C.SENDER['AB_FIXED'] != 0)
        except:
            self.Geometric_Flag = False
        try:
            self.Pragmatic_Flag = __C.SENDER['PRAGMATIC']
        except:
            self.Pragmatic_Flag = False
        self.drawer_train = __C.DRAWER_TRAIN
        self.draw_geo = self.get_net_G(__C.DRAWER_CKPT_PATH)


        self.bw_enc_length = __C.SENDER['BW_ENC_LENGTH']
        self.bw_linear_1 = nn.Linear(1, 224)
        self.abstract_enc_init(11, 256, __C.SENDER['SP_AB_ENCODING_FIXED'])
        try:
            self.linear_encoder = __C.SENDER['LINEAR_ENCODE']
        except:
            self.linear_encoder = False

        self.bandwidth_enc_init(self.bw_enc_length, __C.SENDER['SP_BW_ENCODING_FIXED'])
        self.BW_LIMIT = __C.SENDER['BW_LIMIT']
        self.BW_decoder = U_Net_S(65,64)
        self.mode = __C.RUN_MODE
        self.draw_mode = 0
        # if 'Sender_Only' in __C.INDEX:
        if __C.RUN_MODE == 'train':
            self.draw_mode = 1

            
        if self.Pragmatic_Flag:
            self.draw_prag = Generator_downsampling(3,1,3)
            self.draw_prag.model0.load_state_dict(self.draw_geo.model0.state_dict())
            self.draw_prag.model1.load_state_dict(self.draw_geo.model1.state_dict())
            self.draw_prag.model2.load_state_dict(self.draw_geo.model2.state_dict())


    def abstract_enc_init(self, length = 10, channels = 256, fixed = True):
        enc_list = []
        for i in range(length):
            enc_list.append((i) * 0.1 * torch.ones(channels))
        self.abstract_enc = torch.stack(enc_list, 0)
        # if not fixed:
        #     self.abstract_enc = nn.parameter.Parameter(self.abstract_enc, requires_grad=True)
    
    
    def bandwidth_enc_init(self, channels = 224, fixed = True):
        if self.linear_encoder:
            self.linear_encoder_1 = nn.Linear(1,56)
            self.linear_encoder_2 = nn.Linear(56,224)
        else:
            enc_list = []
            for i in range(10):
                enc_list.append(torch.ones(channels))
            self.bandwidth_enc = torch.stack(enc_list,0)
            if not fixed:
                self.bandwidth_enc = nn.parameter.Parameter(self.bandwidth_enc)


    def get_net_G(self, checkpoints_dir=''):
        net_G = 0
        net_G = Generator(3,1,3)
        net_G.cuda()
        # Load state dicts
        if checkpoints_dir != '':
            net_G.load_state_dict(torch.load(checkpoints_dir), strict=False)
        print('loaded', checkpoints_dir)
        # Set model's test mode
        # net_G.eval()
        
        net_G.model0.eval()
        net_G.model1.eval()
        net_G.model2.eval()
        # net_G.requires_grad = False
        if not self.drawer_train:
            for k in net_G.model0.parameters():
                k.requires_grad = False
            for k in net_G.model1.parameters():
                k.requires_grad = False
            for k in net_G.model2.parameters():
                k.requires_grad = False

        for v in net_G.model3.parameters():
            v.requires_grad = False
        net_G.model3.eval()

        for v in net_G.model4.parameters():
            v.requires_grad = False
        net_G.model4.eval()
        return net_G

    def draw_geo_downsampling(self, x):
        out = self.draw_geo.model0(x)
        out = self.draw_geo.model1(out)
        out = self.draw_geo.model2(out)
        return out

    def draw_geo_upsampling(self, x):
        out = self.draw_geo.model3(x)
        out = self.draw_geo.model4(out)
        return out


    def draw_upsampling(self, feat_geo, feat_prag, bw_enc, ab_enc):
        # time_1 = time.time()
        out = self.feature_fusion(feat_geo, feat_prag, ab_enc)
        out = self.draw_geo.model3(out)
        if self.BW_LIMIT:
            self.bw_enc_matrix = self.bw_linear_1(bw_enc.unsqueeze(-1)).repeat(out.shape[0],1,1,1)
            self.bw_enc_matrix = F.interpolate(self.bw_enc_matrix, out.shape[-2:])
        else:
            self.bw_enc_matrix = self.bw_linear_1(self.bandwidth_enc[-1].unsqueeze(-1)).repeat(out.shape[0],1,1,1)
        out = self.BW_decoder(torch.cat([out, self.bw_enc_matrix], dim=1))
        out = torch.relu(out)
        out = self.draw_geo.model4(out)
        return out

    def feature_fusion(self, feat_geo, feat_prag, weight):
        if self.Geometric_Flag:
            feat = feat_geo
        else:
            feat = torch.einsum('b,abcd->abcd',weight.to(feat_geo.device),feat_geo) + torch.einsum('b,abcd->abcd',(1-weight.to(feat_geo.device)),feat_prag)
        return feat
    
    def feature_selection(self, feat, weight):
        out = torch.einsum('b,abcd->abcd',weight.to(feat.device),feat)
        return out

    def sketches_bw_filter(self, sketch, full_pixels, trans_pixels):
        threshold = sketch.flatten()[sketch.flatten().argsort()[full_pixels-trans_pixels]]
        comm_mask = torch.round(sketch + (0.5 - threshold))
        return comm_mask
    

    def forward(self, images, bandwidth_thre=0, a = 0):
        b = int(bandwidth_thre / 0.1)
        if self.linear_encoder:
            bw_enc = torch.sigmoid(self.linear_encoder_1(math.sin(torch.tensor(bandwidth_thre*math.pi/2)) * torch.ones(1,device=images.device)))
            bw_enc = torch.sigmoid(self.linear_encoder_2(bw_enc))
        else:
            bw_enc = self.bandwidth_enc[b].to(images.device)
        ab_enc = self.abstract_enc[a].to(images.device)

        feat_geo = self.draw_geo_downsampling(images)
        
        if self.Pragmatic_Flag:
            feat_prag = self.draw_prag(images)
        else:
            feat_prag = 0
        sketches_raw = self.draw_upsampling(feat_geo, feat_prag, bw_enc, ab_enc)
        
        sketches = 1 - sketches_raw
          
        if self.BW_LIMIT:
            pooled_mask = F.adaptive_avg_pool2d(sketches, (int(sketches.shape[-2]), int(sketches.shape[-1])))
            rank_tensor = torch.sort(pooled_mask.view(pooled_mask.shape[0], -1))[0]
            select_index = int(rank_tensor.shape[-1] * (1 - bandwidth_thre))
            select_thre = rank_tensor[:, select_index]
            pooled_mask = pooled_mask + (torch.round(pooled_mask+(0.5 - select_thre[:,None,None,None].repeat(1,1,int(sketches.shape[-2]), int(sketches.shape[-1])))) - pooled_mask).detach()
            # comm_mask = sketches + (torch.round(sketches+(bbandwidth_thre/2)) - sketches).detach()
            comm_mask = F.interpolate(pooled_mask, sketches.shape[-2:], mode='nearest')
            sketches = sketches * comm_mask
            sketches = 1 - sketches
        
            if self.draw_mode == 1:
                sketches_informative = self.draw_geo_upsampling(feat_geo)
                return sketches, comm_mask, sketches_raw, sketches_informative

            return sketches, comm_mask, sketches_raw
        else:
            return sketches, torch.zeros(1).to(sketches.device), sketches_raw
