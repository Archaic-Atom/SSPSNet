import torch
from torch import nn
import torch.nn.functional as F
from featup.upsamplers import JBULearnedRange


class JBUStack(torch.nn.Module):

    def __init__(self, feat_dim, out_dim, * args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up1 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up2 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1))
        self.proj = nn.Conv2d(in_channels=feat_dim, out_channels=out_dim,
                              kernel_size=1, stride=1, padding=0,)

    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled

    def forward(self, source, guidance):
        source_2 = self.upsample(source, guidance, self.up1)
        source_4 = self.upsample(source_2, guidance, self.up2)
        source_4 = self.fixup_proj(source_4) * 0.1 + source_4
        return self.proj(source_4)


img = torch.rand(2, 3, 224, 224).cuda()
feat = torch.rand(2, 784, 64, 64).cuda()
upsampler = JBUStack(784, 784 // 16).cuda()

res = upsampler(feat, img)
print(res.shape)
