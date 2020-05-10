# deep self-attention network based on VGG
import torch.nn as nn
import torch.nn.functional as F

from net.module.modules import weight_init
from net.module.seg_hrnet import get_seg_model


class hrnet_v3(nn.Module):
    def __init__(self, channel=256):
        super(hrnet_v3, self).__init__()
        self.hrnet = get_seg_model()
        weight_init(self)

    def forward(self, x):
        size = x.size()[2:]

        pred_s = self.hrnet(x)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)

        return pred_s
