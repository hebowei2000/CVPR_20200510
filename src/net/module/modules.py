import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        else:
            pass


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        weight_init(self)

    def forward(self, x):
        return self.conv_bn(x)


class Reduction(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        weight_init(self)

    def forward(self, x):
        return self.reduce(x)


class ConvUpsample(nn.Module):
    def __init__(self, channel=32):
        super(ConvUpsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, kernel_size=1)
        weight_init(self)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class PPM(nn.Module):
    # pyramid pooling module
    def __init__(self, channel):
        super(PPM, self).__init__()
        self.scales = [1, 2, 4, 8, 16]
        self.poolings = [nn.AdaptiveAvgPool2d((s, s)) for s in self.scales]
        self.convs = nn.ModuleList([BasicConv2d(channel, channel, kernel_size=3, padding=1)
                                    for i in range(len(self.scales))])
        self.cat = BasicConv2d(len(self.scales) * channel, channel, 1)
        weight_init(self)

    def forward(self, x):
        pool_x = []
        for i, pooling in enumerate(self.poolings):
            pool_x.append(self.convs[i](pooling(x)))

        inp_x = []
        for i in range(len(self.scales)):
            inp_x.append(F.upsample(pool_x[i], size=x.size()[2:], mode='bilinear', align_corners=True))
        return x + self.cat(torch.cat(inp_x, dim=1))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        weight_init(self)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        weight_init(self)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConcatOutput(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_upsample0 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat0 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        weight_init(self)

    def forward(self, x0, x1, x2, x3, x4, x5):
        x4 = torch.cat((x4, self.conv_upsample0(self.upsample(x5))), 1)
        x4 = self.conv_cat0(x4)

        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x0 = torch.cat((x0, self.conv_upsample4(self.upsample(x1))), 1)
        x0 = self.conv_cat4(x0)

        x = self.output(x0)
        return x


class AddCoords(nn.Module):
    def __init__(self, radius_channel=False):
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel

    def forward(self, in_tensor):
        """
        in_tensor: (batch_size, channels, x_dim, y_dim)
        [0,0,0,0]   [0,1,2,3]
        [1,1,1,1]   [0,1,2,3]    << (i,j)th coordinates of pixels added as separate channels
        [2,2,2,2]   [0,1,2,3]
        taken from mkocabas.
        """
        batch_size_tensor = in_tensor.shape[0]

        xx_ones = torch.ones([1, in_tensor.shape[2]], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(in_tensor.shape[2], dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, in_tensor.shape[3]], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(in_tensor.shape[3], dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 1, 2)
        yy_channel = yy_channel.permute(0, 3, 1, 2)

        xx_channel = xx_channel.float() / (in_tensor.shape[2] - 1)
        yy_channel = yy_channel.float() / (in_tensor.shape[3] - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        out = torch.cat([in_tensor.cuda(), xx_channel.cuda(), yy_channel.cuda()], dim=1)

        if self.radius_channel:
            radius_calc = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, radius_calc], dim=1).cuda()

        return out


class CoordConv(nn.Module):
    """ add any additional coordinate channels to the input tensor """

    def __init__(self, radius_channel, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=radius_channel)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose(nn.Module):
    """CoordConvTranspose layer for segmentation tasks."""

    def __init__(self, radius_channel, *args, **kwargs):
        super(CoordConvTranspose, self).__init__()
        self.addcoord = AddCoords(radius_channel=radius_channel)
        self.convT = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.convT(out)
        return out


class ConcatFusion(nn.Module):
    def __init__(self, channel):
        super(ConcatFusion, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_upsample0 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat0 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )

        weight_init(self)

    def forward(self, x0, x1, x2, x3, x4, x5):
        x4 = torch.cat((x4, self.conv_upsample0(self.upsample(x5))), 1)
        x4 = self.conv_cat0(x4)

        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x0 = torch.cat((x0, self.conv_upsample4(self.upsample(x1))), 1)
        x0 = self.conv_cat4(x0)

        return x0, x1, x2, x3, x4, x5


class encoder(nn.Module):
    def __init__(self):
        self.inplanes = 32
        super(encoder, self).__init__()
        self.conv = BasicConv2d(1, 32, 3, padding=1)
        self.layer0 = self._make_layer(Bottleneck, 16, 3, stride=2)  # 256
        self.layer1 = self._make_layer(Bottleneck, 32, 3, stride=2)  # 128
        self.layer2 = self._make_layer(Bottleneck, 64, 4, stride=2)  # 64
        self.layer3 = self._make_layer(Bottleneck, 128, 6, stride=2)  # 32
        self.layer4 = self._make_layer(Bottleneck, 256, 3, stride=2)  # 16
        self.layer5 = self._make_layer(Bottleneck, 512, 3, stride=2)  # 8
        weight_init(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        out1 = self.layer0(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)  # 1/8
        out4 = self.layer3(out3)  # 1/ 16
        out5 = self.layer4(out4)  # 1/32
        out6 = self.layer5(out5)

        return out1, out2, out3, out4, out5, out6


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class encoder_w(nn.Module):
    def __init__(self):
        self.inplanes = 32
        super(encoder_w, self).__init__()
        self.conv = BasicConv2d(1, 32, 3, padding=1)
        self.coord_0 = CoordConv(radius_channel=True, in_planes=32, out_planes=32, kernel_size=3, padding=1)
        self.layer0 = self._make_layer(Bottle2neck, 16, 3, stride=2)  # 256
        self.coord_1 = CoordConv(radius_channel=True, in_planes=64, out_planes=64, kernel_size=3, padding=1)
        self.layer1 = self._make_layer(Bottle2neck, 32, 3, stride=2)  # 128
        self.coord_2 = CoordConv(radius_channel=True, in_planes=128, out_planes=128, kernel_size=3, padding=1)
        self.layer2 = self._make_layer(Bottle2neck, 64, 4, stride=2)  # 64
        self.coord_3 = CoordConv(radius_channel=True, in_planes=256, out_planes=256, kernel_size=3, padding=1)
        self.layer3 = self._make_layer(Bottle2neck, 128, 6, stride=2)  # 32
        self.coord_4 = CoordConv(radius_channel=True, in_planes=512, out_planes=512, kernel_size=3, padding=1)
        self.layer4 = self._make_layer(Bottle2neck, 256, 3, stride=2)  # 16
        self.coord_5 = CoordConv(radius_channel=True, in_planes=1024, out_planes=1024, kernel_size=3, padding=1)
        self.layer5 = self._make_layer(Bottle2neck, 512, 3, stride=2)  # 8
        weight_init(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=6, scale=4))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=6, scale=4))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        out1 = self.layer0(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)  # 1/8
        out4 = self.layer3(out3)  # 1/ 16
        out5 = self.layer4(out4)  # 1/32
        out6 = self.layer5(out5)

        return out1, out2, out3, out4, out5, out6


class encoder_new(nn.Module):
    def __init__(self):
        self.inplanes = 32
        super(encoder_new, self).__init__()
        self.conv = BasicConv2d(1, 32, 3, padding=1)
        self.layer0 = self._make_layer(Bottle2neck, 16, 3, stride=2)  # 256
        self.layer1 = self._make_layer(Bottle2neck, 32, 3, stride=2)  # 128
        self.layer2 = self._make_layer(Bottle2neck, 64, 4, stride=2)  # 64
        self.layer3 = self._make_layer(Bottle2neck, 128, 6, stride=2)  # 32
        self.layer4 = self._make_layer(Bottle2neck, 256, 3, stride=2)  # 16
        self.layer5 = self._make_layer(Bottle2neck, 512, 3, stride=2)  # 8
        rate = 0.5
        self.dropout_0 = nn.Dropout2d(rate)
        self.dropout_1 = nn.Dropout2d(rate)
        self.dropout_2 = nn.Dropout2d(rate)
        self.dropout_3 = nn.Dropout2d(rate)
        self.dropout_4 = nn.Dropout2d(rate)
        self.dropout_5 = nn.Dropout2d(rate)
        weight_init(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=6, scale=4))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=6, scale=4))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        out1 = self.layer0(x)
        out1 = self.dropout_0(out1)
        out2 = self.layer1(out1)
        out2 = self.dropout_1(out2)
        out3 = self.layer2(out2)  # 1/8
        out3 = self.dropout_2(out3)
        out4 = self.layer3(out3)  # 1/ 16
        out4 = self.dropout_3(out4)
        out5 = self.layer4(out4)  # 1/32
        out5 = self.dropout_4(out5)
        out6 = self.layer5(out5)
        out6 = self.dropout_5(out6)

        return out1, out2, out3, out4, out5, out6


class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = BasicConv2d(in_planes=out_channels, out_planes=1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        # print('spatial',x.size())
        x = F.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = BasicConv2d(in_planes=out_channels, out_planes=int(out_channels / 2), kernel_size=1, padding=0)
        self.conv2 = BasicConv2d(in_planes=int(out_channels / 2), out_planes=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        # print('channel',x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = BasicConv2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        # print('x',x.size())
        # print('e',e.size())
        if e is not None:
            x = torch.cat([x, e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        # print('x_new',x.size())
        g1 = self.spatial_gate(x)
        # print('g1',g1.size())
        g2 = self.channel_gate(x)
        # print('g2',g2.size())
        x = g1 * x + g2 * x
        return x


class FusionOutput(nn.Module):
    def __init__(self, channel):
        super(FusionOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsampled1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_catd1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_catd2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_catd3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.outputd = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.att_1 = Attention(channel)
        self.att_2 = Attention(channel)
        self.att_3 = Attention(channel)
        self.att_4 = Attention(channel)

        weight_init(self)

    def forward(self, s1, s2, s3, s4, d1, d2, d3, d4):
        s4 = s4 * self.att_4(d4) + s4

        d3 = torch.cat((d3, self.conv_upsampled1(self.upsample(d4))), 1)
        d3 = self.conv_catd1(d3)

        s3 = torch.cat((s3, self.conv_upsample1(self.upsample(s4))), 1)
        s3 = self.conv_cat1(s3)
        s3 = s3 * self.att_3(d3) + s3

        d2 = torch.cat((d2, self.conv_upsampled2(self.upsample(d3))), 1)
        d2 = self.conv_catd2(d2)

        s2 = torch.cat((s2, self.conv_upsample2(self.upsample(s3))), 1)
        s2 = self.conv_cat2(s2)
        s2 = s2 * self.att_2(d2) + s2

        d1 = torch.cat((d1, self.conv_upsampled3(self.upsample(d2))), 1)
        d1 = self.conv_catd3(d1)

        s1 = torch.cat((s1, self.conv_upsample3(self.upsample(s2))), 1)
        s1 = self.conv_cat3(s1)
        s1 = s1 * self.att_1(d1) + s1

        d = self.outputd(d1)
        s = self.output(s1)
        return s, d


class Attention(nn.Module):
    def __init__(self, channel=32):
        super(Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.activation = nn.Sigmoid()
        weight_init(self)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class FusionOutput_v2(nn.Module):
    def __init__(self, channel):
        super(FusionOutput_v2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.output_s1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_s2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_s3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_s4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsampled1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_catd1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_catd2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_catd3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.output_d1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_d2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_d3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_d4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsamplee1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamplee2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamplee3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cate1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_cate2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_cate3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.output_e1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_e2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_e3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_e4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.att_1_s = Attention(channel)
        self.att_2_s = Attention(channel)
        self.att_3_s = Attention(channel)
        self.att_4_s = Attention(channel)
        self.att_1_e = Attention(channel)
        self.att_2_e = Attention(channel)
        self.att_3_e = Attention(channel)
        self.att_4_e = Attention(channel)
        self.att_1_d = Attention(channel)
        self.att_2_d = Attention(channel)
        self.att_3_d = Attention(channel)
        self.att_4_d = Attention(channel)

        weight_init(self)

    def forward(self, s1, s2, s3, s4, d1, d2, d3, d4, e1, e2, e3, e4):
        tmp_s4, tmp_d4, tmp_e4 = s4, d4, e4
        s4 = tmp_s4 * tmp_e4 + tmp_s4 * self.att_4_s(tmp_d4) + tmp_s4
        e4 = tmp_e4 * tmp_s4 * self.att_4_e(tmp_d4)
        d4 = tmp_d4 * tmp_e4 + tmp_d4 * self.att_4_d(tmp_s4) + tmp_d4

        d3 = torch.cat((d3, self.conv_upsampled1(self.upsample(d4))), 1)
        d3 = self.conv_catd1(d3)

        e3 = torch.cat((e3, self.conv_upsamplee1(self.upsample(e4))), 1)
        e3 = self.conv_cate1(e3)

        s3 = torch.cat((s3, self.conv_upsample1(self.upsample(s4))), 1)
        s3 = self.conv_cat1(s3)
        tmp_s3, tmp_d3, tmp_e3 = s3, d3, e3
        s3 = tmp_s3 * tmp_e3 + tmp_s3 * self.att_3_s(tmp_d3) + tmp_s3
        e3 = tmp_e3 * tmp_s3 * self.att_3_e(tmp_d3)
        d3 = tmp_d3 * tmp_e3 + tmp_d3 * self.att_3_d(tmp_s3) + tmp_d3

        d2 = torch.cat((d2, self.conv_upsampled2(self.upsample(d3))), 1)
        d2 = self.conv_catd2(d2)

        e2 = torch.cat((e2, self.conv_upsamplee2(self.upsample(e3))), 1)
        e2 = self.conv_cate2(e2)

        s2 = torch.cat((s2, self.conv_upsample2(self.upsample(s3))), 1)
        s2 = self.conv_cat2(s2)
        tmp_s2, tmp_d2, tmp_e2 = s2, d2, e2
        s2 = tmp_s2 * tmp_e2 + tmp_s2 * self.att_2_s(tmp_d2) + tmp_s2
        e2 = tmp_e2 * tmp_s2 * self.att_2_e(tmp_d2)
        d2 = tmp_d2 * tmp_e2 + tmp_d2 * self.att_2_d(tmp_s2) + tmp_d2

        d1 = torch.cat((d1, self.conv_upsampled3(self.upsample(d2))), 1)
        d1 = self.conv_catd3(d1)

        e1 = torch.cat((e1, self.conv_upsamplee3(self.upsample(e2))), 1)
        e1 = self.conv_cate3(e1)

        s1 = torch.cat((s1, self.conv_upsample3(self.upsample(s2))), 1)
        s1 = self.conv_cat3(s1)
        tmp_s1, tmp_d1, tmp_e1 = s1, d1, e1
        s1 = tmp_s1 * tmp_e1 + tmp_s1 * self.att_1_s(tmp_d1) + tmp_s1
        e1 = tmp_e1 * tmp_s1 * self.att_1_e(tmp_d1)
        d1 = tmp_d1 * tmp_e1 + tmp_d1 * self.att_1_d(tmp_s4) + tmp_d1

        s1 = self.output_s1(s1)
        s2 = self.output_s2(s2)
        s3 = self.output_s3(s3)
        s4 = self.output_s4(s4)
        e1 = self.output_e1(e1)
        e2 = self.output_e2(e2)
        e3 = self.output_e3(e3)
        e4 = self.output_e4(e4)
        d1 = self.output_d1(d1)
        d2 = self.output_d2(d2)
        d3 = self.output_d3(d3)
        d4 = self.output_d4(d4)
        return s1, s2, s3, s4, e1, e2, e3, e4, d1, d2, d3, d4


class MultiFusionOutput(nn.Module):
    def __init__(self, channel):
        super(MultiFusionOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsamples1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamples2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamples3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_downsamples4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsamples2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsamples3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cats1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cats2 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cats3 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cats4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.outputs = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsampled1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_downsampled4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsampled2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsampled3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_catd1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_catd2 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_catd3 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_catd4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.outputd = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsamplee1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamplee2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamplee3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_downsamplee4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsamplee2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsamplee3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cate1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cate2 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cate3 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cate4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.outpute = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.att_1 = Attention(channel)
        self.att_2 = Attention(channel)
        self.att_3 = Attention(channel)
        self.att_4 = Attention(channel)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        weight_init(self)

    def forward(self, s1, s2, s3, s4, d1, d2, d3, d4, e1, e2, e3, e4):
        d4 = torch.cat((d4, self.conv_downsampled4(self.pooling(d3))), 1)
        d4 = self.conv_catd4(d4)

        e4 = torch.cat((e4, self.conv_downsamplee4(self.pooling(e3))), 1)
        e4 = self.conv_cate4(e4)

        s4 = torch.cat((s4, self.conv_downsamples4(self.pooling(s3))), 1)
        s4 = self.conv_cats4(s4)
        s4 = s4 * e4 + s4 * self.att_4(d4) + s4

        d3 = torch.cat((d3, self.conv_upsampled3(self.upsample(d4)), self.conv_downsampled3(self.pooling(d2))), 1)
        d3 = self.conv_catd3(d3)

        e3 = torch.cat((e3, self.conv_upsamplee3(self.upsample(e4)), self.conv_downsamplee3(self.pooling(e2))), 1)
        e3 = self.conv_cate3(e3)

        s3 = torch.cat((s3, self.conv_upsamples3(self.upsample(s4)), self.conv_downsamples3(self.pooling(s2))), 1)
        s3 = self.conv_cats3(s3)
        s3 = s3 * e3 + s3 * self.att_3(d3) + s3

        d2 = torch.cat((d2, self.conv_upsampled2(self.upsample(d3)), self.conv_downsampled2(self.pooling(d1))), 1)
        d2 = self.conv_catd2(d2)

        e2 = torch.cat((e2, self.conv_upsamplee2(self.upsample(e3)), self.conv_downsamplee2(self.pooling(e1))), 1)
        e2 = self.conv_cate2(e2)

        s2 = torch.cat((s2, self.conv_upsamples2(self.upsample(s3)), self.conv_downsamples2(self.pooling(s1))), 1)
        s2 = self.conv_cats2(s2)
        s2 = s2 * e2 + s2 * self.att_2(d2) + s2

        d1 = torch.cat((d1, self.conv_upsampled1(self.upsample(d2))), 1)
        d1 = self.conv_catd1(d1)

        e1 = torch.cat((e1, self.conv_upsamplee1(self.upsample(e2))), 1)
        e1 = self.conv_cate1(e1)

        s1 = torch.cat((s1, self.conv_upsamples1(self.upsample(s2))), 1)
        s1 = self.conv_cats1(s1)
        s1 = s1 * e1 + s1 * self.att_1(d1) + s1

        e = self.outpute(e1)
        s = self.outputs(s1)
        return s, e
