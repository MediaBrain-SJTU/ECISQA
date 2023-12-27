import torch
from torch import nn
import copy
from models.drawer import Generator, Generator_downsampling, Generator_upsampling
from models.basicmodels.unet import U_Net_S
import time, torchvision

bandwidth_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]

class Sender(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.drawer_train = __C.DRAWER_TRAIN
        self.draw_geo = self.get_net_G(__C.DRAWER_CKPT_PATH)
        try:
            self.Geometric_Flag = __C.SENDER['GEOMETRIC']
        except:
            self.Geometric_Flag = False
        if not self.Geometric_Flag:
            self.draw_prag = Generator_downsampling(3,1,3)
        self.bw_enc_length = __C.SENDER['BW_ENC_LENGTH']
        self.bw_linear_1 = nn.Linear(1, 224)
        # self.abstract_enc = nn.parameter.Parameter(torch.ones(10,256))
        self.abstract_enc_init(11, 256, __C.SENDER['SP_AB_ENCODING_FIXED'])
        # self.bandwidth_enc = nn.parameter.Parameter(torch.ones(10,64))
        # self.bandwidth_enc_init(11, 64, __C.SENDER['SP_BW_ENCODING_FIXED'])
        self.bandwidth_enc_init(10, self.bw_enc_length, 1, __C.SENDER['SP_BW_ENCODING_FIXED'])
        self.BW_LIMIT = __C.SENDER['BW_LIMIT']
        self.BW_decoder = U_Net_S(65,64)
        # if self.BW_LIMIT :
        #     self.mask_making = mask_making(2,1)
        

    def abstract_enc_init(self, length = 10, channels = 256, fixed = True):
        enc_list = []
        for i in range(length):
            enc_list.append((i) * 0.1 * torch.ones(channels))
        self.abstract_enc = torch.stack(enc_list, 0)
        if not fixed:
            self.abstract_enc = nn.parameter.Parameter(self.abstract_enc)
    
    
    def bandwidth_enc_init(self, length = 10, channels = 224, fixed = True, thre_unit = 0.05):
        enc_list = []
        for i in range(length):
            enc_list.append(bandwidth_list[i] * torch.ones(channels))
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
            v.requires_grad = True
        net_G.model3.train()

        for v in net_G.model4.parameters():
            v.requires_grad = True
        net_G.model4.train()
        return net_G

    def draw_geo_downsampling(self, x):
        out = self.draw_geo.model0(x)
        out = self.draw_geo.model1(out)
        out = self.draw_geo.model2(out)
        return out

    def draw_upsampling(self, feat_geo, feat_prag, bw_enc, ab_enc):
        # time_1 = time.time()
        out = self.feature_fusion(feat_geo, feat_prag, ab_enc)
        # time_2 = time.time()
        # print("Feature Fusion Cost {}s".format(str(time_2 - time_1)))
        out = self.draw_geo.model3(out)
        if self.BW_LIMIT:
            bw_enc_matrix = self.bw_linear_1(bw_enc.unsqueeze(-1)).repeat(out.shape[0],1,1,1)
            # out = self.BW_decoder(torch.cat([out, bw_enc_matrix], dim=1))
            # out = torch.relu(out)
        else:
            bw_enc_matrix = self.bw_linear_1(self.bandwidth_enc[-1].unsqueeze(-1)).repeat(out.shape[0],1,1,1)
        out = self.BW_decoder(torch.cat([out, bw_enc_matrix], dim=1))
        out = torch.relu(out)
        out = self.draw_geo.model4(out)
        return out

    def feature_fusion(self, feat_geo, feat_prag, weight):
        # feat = torch.zeros(feat1.shape).to(feat1.device)
        # for i in range(len(weight)):
        #     feat[:, i] = feat1[:, i] * weight[i] + feat2[:, i] * (1 - weight[i])
        if self.Geometric_Flag:
            feat = feat_geo
        else:
            feat = torch.einsum('b,abcd->abcd',weight.to(feat_geo.device),feat_geo) + torch.einsum('b,abcd->abcd',(1-weight.to(feat_geo.device)),feat_prag)
        return feat
    
    def feature_selection(self, feat, weight):
        # out = torch.zeros(feat.shape).to(feat.device)
        # for i in range(len(weight)):
        #     out[:,i] = feat[:,1] * weight[i]
        out = torch.einsum('b,abcd->abcd',weight.to(feat.device),feat)
        return out

    def sketches_bw_filter(self, sketch, full_pixels, trans_pixels):
        threshold = sketch.flatten()[sketch.flatten().argsort()[full_pixels-trans_pixels]]
        comm_mask = torch.round(sketch + (0.5 - threshold))
        return comm_mask
    

    def forward(self, images, b = 1, a = 0):
        # para_enc = self.sp_enc(b, a)
        # bw_enc = torch.sigmoid(self.bandwidth_enc[b])
        # ab_enc = torch.sigmoid(self.abstract_enc[a])
        bw_enc = self.bandwidth_enc[b].to(images.device)
        ab_enc = self.abstract_enc[a].to(images.device)

        feat_geo = self.draw_geo_downsampling(images)
        
        if not self.Geometric_Flag:
            feat_prag = self.draw_prag(images)
        else:
            feat_prag = 0
        sketches = self.draw_upsampling(feat_geo, feat_prag, bw_enc, ab_enc)
        sketches = 1 - sketches

        # if self.BW_LIMIT:
        #     full_pixels = sketches[0].shape[-1] * sketches[0].shape[-2]
        #     transmission_pixels = bandwidth_list[b] * full_pixels
        #     for i in sketches.shape[0]:
        #         sketches[i] = self.sketches_bw_filter(sketches[i], full_pixels, transmission_pixels)
        if self.BW_LIMIT:
            # comm_mask = self.mask_making(sketches, bw_enc)
            # sketches = sketches * comm_mask
            comm_mask = sketches + (torch.round(sketches+(0.5-(bandwidth_list[b]/2))) - sketches).detach()
            sketches = sketches * comm_mask
            sketches = 1 - sketches
            return sketches, comm_mask
        else:
            return sketches, torch.zeros(1).to(sketches.device)
    # def forward(self, images, b = 1, a = 1):
    #     # sketches = self.draw_geo(images)
    #     feat_geo = self.draw_geo_downsampling(images)
    #     sketches = self.draw_geo.model3(feat_geo)
    #     sketches = self.draw_geo.model4(sketches)
    #     return sketches

class super_parameter_encoding(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.enc_length = __C.ENC_LENGTH
        self.output_channel = __C.ENC_CHANNEL
        # self.linear1 = nn.Linear(1, 224)
        # self.conv1 = nn.Conv2d(1,self.output_channel,3,stride=1,padding=1)
        # self.conv2 = nn.Conv2d(self.output_channel + 3,self.output_channel, 3, stride=1, padding=1)
        self.parameters_encoding_matrix = nn.parameter.Parameter(torch.ones(10,10,self.enc_length))
    

    def forward(self, x, p = 1, a = 1):
        para_enc = self.parameters_encoding_matrix[p,a].unsqueeze(0).unsqueeze(-1)
        # para_enc = self.linear1(para_enc)
        # para_enc = torch.self.conv1(para_enc)
        # para_enc = self.softplus(para_enc)
        # # image_enc = self.conv2(torch.cat([para_enc, x]))
        # return para_enc
        return para_enc

    def softplus(self, x):
        return torch.log(1+torch.exp(x))


class bandlimited_encoding(super_parameter_encoding):
    def __init__(self, __C):
        super().__init__()


class Abstract_encoding(super_parameter_encoding):
    def __init__(self, __C):
        super().__init__()
        self.enc_length = __C.ENC_LENGTH
        self.output_channel = __C.ENC_CHANNEL
        self.onehot_encoding = nn.parameter.Parameter(torch.ones(10, 512))
        # self.linear1 = nn.Linear(self.enc_length, 512)
        # self.conv1 = nn.Conv2d(1,self.output_channel,3,stride=1,padding=1)
        # self.conv2 = nn.Conv2d(self.output_channel + 3,self.output_channel, 3, stride=1, padding=1)
        # self.init_encoding = nn.parameter
    def onehot_encoding(self, p, a):
        # TO DO
        onehot_encoding = copy.deepcopy(self.init_encoding)
        onehot_encoding[int(p)] = 0
        return onehot_encoding

    def forward(self, x, a):

        x = self.onehot_encoding

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()       
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         #pe.requires_grad = False
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

class mask_making(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, enc_length=224):
        super(mask_making, self).__init__()
        self.mask_enc = U_Net_S(in_ch, out_ch)
        self.linear1 = nn.Linear(1, enc_length)
        # self.linear2 = nn.Linear(64, 224)
        # self.mask_enc = U_Net(in_ch=in_ch, out_ch=out_ch)

    def forward(self, sketches, bw_enc):
        bw_enc_matrix = self.linear1(bw_enc.unsqueeze(-1))
        # bw_enc_matrix = self.linear2(bw_enc_matrix.permute(1,0)).unsqueeze(0).unsqueeze(0)
        comm_mask = self.mask_enc(torch.cat([sketches, bw_enc_matrix.repeat(sketches.shape[0],1,1,1)], dim=1))
        comm_mask = torch.sigmoid(comm_mask)
        comm_mask = comm_mask + (torch.round(comm_mask) - comm_mask).detach()
        comm_mask = nn.functional.interpolate(comm_mask, (sketches.shape[-2],sketches.shape[-1]), mode='nearest')
        comm_mask = comm_mask * torch.where(sketches>0.5, 1, 0)
        return comm_mask