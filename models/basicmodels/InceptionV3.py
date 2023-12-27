import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models


class InceptionV3(nn.Module): #avg pool
    def __init__(self, num_classes, isTrain, use_aux=True, pretrain=False, freeze=True, every_feat=False):
        super(InceptionV3, self).__init__()
        """ Inception v3 expects (299,299) sized images for training and has auxiliary output
        """

        self.every_feat = every_feat

        self.model_ft = models.inception_v3(pretrained=pretrain)
        stop = 0
        if freeze and pretrain:
            for child in self.model_ft.children():
                if stop < 17:
                    for param in child.parameters():
                        param.requires_grad = False
                stop += 1

        num_ftrs = self.model_ft.AuxLogits.fc.in_features #768
        self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Handle the primary net
        num_ftrs = self.model_ft.fc.in_features #2048
        self.model_ft.fc = nn.Linear(num_ftrs,num_classes)

        self.model_ft.input_size = 299

        self.isTrain = isTrain
        self.use_aux = use_aux

        if self.isTrain:
            self.model_ft.train()
        else:
            self.model_ft.eval()


    def forward(self, x, cond=None, catch_gates=False):
        # N x 3 x 299 x 299
        x = self.model_ft.Conv2d_1a_3x3(x)

        # N x 32 x 149 x 149
        x = self.model_ft.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model_ft.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.model_ft.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model_ft.Conv2d_4a_3x3(x)

        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.model_ft.Mixed_5b(x)
        feat1 = x
        # N x 256 x 35 x 35
        x = self.model_ft.Mixed_5c(x)
        feat11 = x
        # N x 288 x 35 x 35
        x = self.model_ft.Mixed_5d(x)
        feat12 = x
        # N x 288 x 35 x 35
        x = self.model_ft.Mixed_6a(x)
        feat2 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6b(x)
        feat21 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6c(x)
        feat22 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6d(x)
        feat23 = x
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_6e(x)

        feat3 = x

        # N x 768 x 17 x 17
        aux_defined = self.isTrain and self.use_aux
        if aux_defined:
            aux = self.model_ft.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.model_ft.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model_ft.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model_ft.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        feats = F.dropout(x, training=self.isTrain)
        # N x 2048 x 1 x 1
        x = torch.flatten(feats, 1)
        # N x 2048
        x = self.model_ft.fc(x)
        # N x 1000 (num_classes)

        if self.every_feat:
            # return feat21, feats, x
            return x, feat21

        return x, aux