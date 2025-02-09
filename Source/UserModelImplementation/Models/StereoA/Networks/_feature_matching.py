from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad,
                                   dilation = dilation,
                                   bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                                   padding=pad, stride=stride, bias=False),
                         nn.BatchNorm3d(out_planes))


class DisparityRegression(nn.Module):
    def __init__(self, start_disp, maxdisp, win_size):
        super().__init__()
        self.max_disp = maxdisp
        self.win_size = win_size
        self.start_disp = start_disp

    def forward(self, x):
        if self.win_size > 0:
            max_d = torch.argmax(x, dim=1, keepdim=True)
            d_value = []
            prob_value = []
            for d in range(-self.win_size, self.win_size + 1):
                index = max_d + d
                index[index < 0] = 0
                index[index > x.shape[1] - 1] = x.shape[1] - 1
                d_value.append(index)
                prob = torch.gather(x, dim=1, index=index)
                prob_value.append(prob)
            part_x = torch.cat(prob_value, dim=1)
            part_x = part_x / (torch.sum(part_x, dim=1, keepdim=True) + 1e-8)
            part_d = torch.cat(d_value, dim=1).float()
            out = torch.sum(part_x * part_d, dim=1) + self.start_disp
        else:
            disp = torch.arange(self.start_disp,
                                self.start_disp + self.max_disp).view(1, -1, 1, 1).float().to(x.device)
            out = torch.sum(x * disp, 1)

        return out


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes * 2,
                                                      kernel_size=3, padding=1,
                                                      output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes,
                                                      kernel_size=3, padding=1,
                                                      output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8

        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4
        return out, pre, post


class hourglass_gwcnet(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 4, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes * 4, inplanes * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes * 4, inplanes * 2, kernel_size=3, padding=1,
                                                      output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1,
                                                      output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes))

        self.redir1 = convbn_3d(inplanes, inplanes, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


class PSMNet(nn.Module):
    def __init__(self, in_channels, start_disp, maxdisp, udc):
        super().__init__()
        self.maxdisp, self.start_disp, self.udc = maxdisp, start_disp, udc

        self.dres0 = nn.Sequential(convbn_3d(in_channels, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
        self.dres2, self.dres3, self.dres4 = self._make_hourglass_gwcnet(32)
        self.classif1 = self._make_classif(32)
        self.classif2 = self._make_classif(32)
        self.classif3 = self._make_classif(32)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_hourglass_gwcnet(self, in_channels) -> None:
        return hourglass_gwcnet(in_channels), hourglass_gwcnet(in_channels), hourglass_gwcnet(in_channels)

    def _make_classif(self, in_channels) -> None:
        return nn.Sequential(convbn_3d(in_channels, 32, 3, 1, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    def _regress(self, classif_fn, size, cost_feat, win_s, probability=0.4) -> tuple:
        cost = classif_fn(cost_feat)
        cost = F.interpolate(cost, size, mode='trilinear', align_corners=True)
        cost = torch.squeeze(cost, 1)
        distribute = F.softmax(cost, dim=1)
        mask = torch.sum((distribute > probability).float(), dim = 1, keepdim = True).detach()
        valid_num = torch.sum(mask, dim=(2, 3), keepdim=False)
        pred = DisparityRegression(self.start_disp, self.maxdisp, win_size = win_s)(distribute)
        return pred, distribute, mask, valid_num

    def forward(self, cost, size):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        win_s = 5 if self.udc else 0
        if self.training:
            pred1, distribute1, _, _ = self._regress(self.classif1, size, out1, win_s)
            pred2, distribute2, _, _ = self._regress(self.classif2, size, out2, win_s)
        pred3, distribute3, mask, valid_num = self._regress(self.classif3, size, out3, win_s)

        if self.training:
            res = [pred1, pred2, pred3]
            if self.udc:
                res += [distribute1, distribute2, distribute3, mask, valid_num]
            return res
        return [pred3, mask]
