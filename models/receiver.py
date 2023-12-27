import torch
from torch import nn
import torch.nn.functional as F
import clip
from collections import OrderedDict
from models.basicmodels.unet import U_Net_S, U_Net
from models.basicmodels.VisionTransformer import ViT
from models.RCNN import RCNN
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import postprocessing
from detectron2.structures import Instances
from detectron2.structures import Boxes
from typing import List, Tuple, Union
import numpy as np

CLIP_MODEL_BASES = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

TRANFORMER_RESBLOCKS_KEYS = ['resblock2', 'resblock5', 'resblock8', 'resblock11']
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

TEST_SCALES = (600,)
TEST_MAX_SIZE = 1000


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
            self.enc = self.clip_enc
            self.enc = self.clip_enc()
            self.fstate = OrderedDict()
            for gpu in __C.GPU.split(','):
                self.fstate[gpu] = OrderedDict()
            
        self.v_enc_arc = __C.RECEIVER_V_ENC_ARC
        if self.v_enc_arc == 'vit':
            self.dim = __C.RECEIVER['DIM']
            self.depth = __C.RECEIVER['DEPTH']
            self.enc = ViT(image_size=224, patch_size=32, num_classes=100, dim=self.dim, depth=self.depth, heads=32, mlp_dim=1024, pool = 'cls', channels = 1, dim_head = 32, dropout = 0.1, emb_dropout = 0.1)

        if self.v_enc_arc == 'rcnn':
            self.enc = RCNN(__C)
            self.extract_mode = __C.RECEIVER['EXTRACT_MODE']
            self.enc.eval()
            for k, v in self.enc.named_parameters():
                v.requires_grad = False
        try:
            self.Pragmatic_Flag = __C.SENDER['PRAGMATIC']
            if 'DEC_FIRST' in __C.RECEIVER.keys():
                self.Pragmatic_Flag = (__C.SENDER['PRAGMATIC'] or __C.RECEIVER['DEC_FIRST'])
        except:
            self.Pragmatic_Flag = False
        if self.Pragmatic_Flag:
            self.Prag_Dec = U_Net(1,1)

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
                self.fstate[str(output.device.index)][name] = output.permute(1, 0, 2) #LND -> NLD bs, smth, 768
            else:
                self.fstate[str(output.device.index)][name] = output

        return hook 

    def clip_enc(self, images):
        v_encoding = self.clipmodel.visual(images.repeat(1,3,1,1).to(torch.float16))
        v_feats = self.feature_fusion(self.fstate[str(images.device.index)])
        return v_feats

    def model_inference(self, batched_inputs, proposals=0):
        
        images = self.enc.preprocess_image(batched_inputs)
        features = self.enc.backbone(images.tensor)
        bs = images.tensor.shape[0]
        proposals, _ = self.enc.proposal_generator(images, features, None)
           
        _, pooled_features, _ = self.enc.roi_heads.get_roi_features(features, proposals)  # fc7 feats
        predictions = self.enc.roi_heads.box_predictor(pooled_features)
        
        predictions, r_indices = self.enc.roi_heads.box_predictor.inference(predictions, proposals)
        height = 224
        width = 224
        r = [postprocessing.detector_postprocess(predictions[i], height, width) for i in range(len(predictions))] # image   
        
        if self.extract_mode == 1:
            return get_out_features(bs, proposals, pooled_features, r_indices)

        if self.extract_mode != 1:
            bboxes = [r[i].get("pred_boxes").tensor for i in range(len(predictions))]  # box
            height = images[0].shape[1]
            width = images[0].shape[2]



            proposals_bboxes = [Instances((height, width)) for i in range(len(predictions))]
            for i in range(len(predictions)):
                proposals_bboxes[i].proposal_boxes = BUABoxes(bboxes[i])
            _, pooled_features_bboxes, _ = self.enc.roi_heads.get_roi_features(features, proposals_bboxes)  # fc7 feats

            return get_out_features(bs, proposals_bboxes, pooled_features_bboxes)

            
        


    def forward(self, images):
        if self.v_enc_arc == 'rcnn':
            if self.Pragmatic_Flag:
                images = self.Prag_Dec(1 - images)
                images = torch.sigmoid(images)
                images = 1 - images
            with torch.no_grad():
                input_list = []
                for i in range(images.shape[0]):
                    input_list.append(get_image_blob(255*images[i]))
                v_feats, proposals = self.model_inference(input_list)
        else:
            v_feats = self.enc(images)
        if self.Pragmatic_Flag:
            return v_feats, proposals, images
        else:
            return v_feats, proposals

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



class BUABoxes(Boxes):
    """
        This structure stores a list of boxes as a Nx4 torch.Tensor.
        It supports some common methods about boxes
        (`area`, `clip`, `nonempty`, etc),
        and also behaves like a Tensor
        (support indexing, `to(device)`, `.device`, and iteration over all boxes)

        Attributes:
            tensor: float matrix of Nx4.
        """

    BoxSizeType = Union[List[int], Tuple[int, int]]
    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor)

    def clip(self, box_size: BoxSizeType) -> None:
        """
        NOTE: In order to be the same as bottom-up-attention network, we have
        defined the new clip function.

        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        TO_REMOVE = 1
        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w - TO_REMOVE)
        self.tensor[:, 1].clamp_(min=0, max=h - TO_REMOVE)
        self.tensor[:, 2].clamp_(min=0, max=w - TO_REMOVE)
        self.tensor[:, 3].clamp_(min=0, max=h - TO_REMOVE)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        """
        NOTE: In order to be the same as bottom-up-attention network, we have
        defined the new nonempty function.

        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        TO_REMOVE = 1
        box = self.tensor
        widths = box[:, 2] - box[:, 0] + TO_REMOVE
        heights = box[:, 3] - box[:, 1] + TO_REMOVE
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def filter_boxes(self):
        box = self.tensor
        keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
        Returns:
            BUABoxes: Create a new :class:`BUABoxes` by indexing.

        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BUABoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return BUABoxes(b)

def get_out_features(bs, proposals, pooled_features, r_indices=0):
    output_feature_list = []
    output_proposals_list = []
    start = 0
    if r_indices != 0:
        for i in range(bs):
            if len(proposals[i]) != 0:
                end = start + len(proposals[i])
                output_feature_list.append(img_feat_padding(pooled_features[start:end][r_indices[i]]))
                output_proposals_list.append(proposals[i][r_indices[i]])
                start = end
            else:
                output_feature_list.append(img_feat_padding(torch.zeros((1,2048)).to(pooled_features.device)))
    else:
        for i in range(bs):
            end = start + len(proposals[i])
            output_feature_list.append(img_feat_padding(pooled_features[start:end]))
            start = end
    output_features = torch.stack(output_feature_list, dim=0)
    return output_features, output_proposals_list

def img_feat_padding(img_feat, img_feat_pad_size=100):
    return F.pad(
        img_feat,
        (0, 0, 0, img_feat_pad_size - img_feat.shape[0]),
        "constant",
        0
    )

def get_image_blob(im, pixel_means=0):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    pixel_means = torch.tensor([[pixel_means]]).to(im.device)
    dataset_dict = {}
    im_orig = im - pixel_means

    im_shape = im_orig.shape
    im_size_min = min(im_shape[-2:])
    im_size_max = max(im_shape[-2:])

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = torch.nn.functional.interpolate(im_orig.unsqueeze(0), scale_factor=[im_scale,im_scale], mode='bilinear').squeeze(0)
    dataset_dict["image"] = im
    dataset_dict["im_scale"] = im_scale

    return dataset_dict