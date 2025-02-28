"""
@author: Xinjing Cheng & Peng Wang

"""
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
try:
    from ._offset import Refinement, RefinementLayer, MLP
except ImportError:
    from _offset import Refinement, RefinementLayer, MLP


class Prompt(nn.Module):
    EPS = 1e-9

    def __init__(self, prop_time, in_channels=64, prop_kernel=3, norm_type='8sum'):
        """

        Inputs:
            prop_time: how many steps for CSPN to perform
            prop_kernel: the size of kernel (current only support 3x3)
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super().__init__()
        assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'
        assert norm_type in ['8sum', '8sum_abs']
        self.norm_type, self.prop_time = norm_type, prop_time
        self.prop_kernel, self.out_feature = prop_kernel, 1
        self.sum_conv = self._get_sum_conv()
        self.concatconv_refine, self.gw_refine = \
            self._concatconv(in_channels, 8), self._gw(in_channels, 32)
        self.guidance, self.guidance_head = self._refine_layers(8)

    def _get_sum_conv(self):
        sum_conv = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=(1, 1, 1),
                             stride=1, padding=0, bias=False)
        weight = torch.ones(1, 8, 1, 1, 1).cuda()
        sum_conv.weight = nn.Parameter(weight)
        for param in sum_conv.parameters():
            param.requires_grad = False
        return sum_conv

    def _concatconv(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)
        )

    def _gw(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 1, 1, 0, bias=False)
        )

    def _refine_layers(self, out_channels: int, infer_embed_dim: int = 16,
                       num_refine_layers: int = 5, mlp_ratio: float = 4,
                       refine_window_size: int = 6, infer_n_heads: int = 4,
                       activation: str = "gelu", attn_drop: float = 0.,
                       proj_drop: float = 0., dropout: float = 0.,
                       drop_path: float = 0., normalize_before: bool = False,
                       return_intermediate=False) -> nn.Module:
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_refine_layers)]
        refine_layers = nn.ModuleList([
            RefinementLayer(
                infer_embed_dim, mlp_ratio=mlp_ratio, window_size=refine_window_size,
                shift_size=0 if i % 2 == 0 else refine_window_size // 2, n_heads=infer_n_heads,
                activation=activation,
                attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], dropout=dropout,
                normalize_before=normalize_before,
            )
            for i in range(num_refine_layers)]
        )
        refine_norm = nn.LayerNorm(infer_embed_dim)
        refinement = Refinement(32, infer_embed_dim, layers=refine_layers, norm=refine_norm,
                                return_intermediate=return_intermediate)
        refine_head = MLP(infer_embed_dim, infer_embed_dim, out_channels, 3)
        return refinement, refine_head

    def forward(self, left_feat, right_feat, blur_depth, sparse_depth, size):
        if len(blur_depth.shape) == 3:
            blur_depth = blur_depth.unsqueeze(1)
        h, w = size
        left_feat_gw = self.gw_refine(
            F.interpolate(left_feat, size, mode='bilinear', align_corners=True))
        right_feat_gw = self.gw_refine(
            F.interpolate(right_feat, size, mode='bilinear', align_corners=True))

        left_feat = self.concatconv_refine(
            F.interpolate(left_feat, size, mode='bilinear', align_corners=True))
        right_feat = self.concatconv_refine(
            F.interpolate(right_feat, size, mode='bilinear', align_corners=True))

        blur_depth = blur_depth.detach()
        tgt = self.guidance(blur_depth, left_feat, right_feat, left_feat_gw, right_feat_gw)
        guidance = self.guidance_head(tgt)
        guidance = rearrange(guidance, 'a (b h w) c -> (a b) c h w ', h=h, w=w)

        gate_wb, gate_sum = self.affinity_normalization(guidance)

        # pad input and convert to 8 channel 3D features
        raw_depth_input = blur_depth

        # blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
        result_depth = blur_depth

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()

        for _ in range(self.prop_time):
            # one propagation
            result_depth = self.pad_blur_depth(result_depth)

            neigbor_weighted_sum = self.sum_conv(gate_wb * result_depth)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum

            if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input

        return result_depth.squeeze(1)

    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)

        gate1_wb_cmb = guidance.narrow(1, 0, self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb, gate2_wb_cmb, gate3_wb_cmb, gate4_wb_cmb,
                             gate5_wb_cmb, gate6_wb_cmb, gate7_wb_cmb, gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)

        gate_wb = torch.div(gate_wb, abs_weight + self.EPS)
        gate_sum = self.sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum

    def pad_blur_depth(self, blur_depth):
        # top pad
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        blur_depth_1 = left_top_pad(blur_depth).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        blur_depth_2 = center_top_pad(blur_depth).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        blur_depth_3 = right_top_pad(blur_depth).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        blur_depth_4 = left_center_pad(blur_depth).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        blur_depth_5 = right_center_pad(blur_depth).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        blur_depth_6 = left_bottom_pad(blur_depth).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        blur_depth_7 = center_bottom_pad(blur_depth).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        blur_depth_8 = right_bottm_pad(blur_depth).unsqueeze(1)

        result_depth = torch.cat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4,
                                  blur_depth_5, blur_depth_6, blur_depth_7, blur_depth_8), 1)
        return result_depth

    def normalize_gate(self, guidance):
        elesum_gate1_x1 = torch.add(torch.abs(guidance.narrow(1, 0, 1)),
                                    torch.abs(guidance.narrow(1, 1, 1)))
        gate1_x1_g1_cmb = torch.div(guidance.narrow(1, 0, 1), elesum_gate1_x1 + self.EPS)
        gate1_x1_g2_cmb = torch.div(guidance.narrow(1, 1, 1), elesum_gate1_x1 + self.EPS)
        return gate1_x1_g1_cmb, gate1_x1_g2_cmb

    def max_of_4_tensor(self, element1, element2, element3, element4):
        return torch.max(torch.max(element1, element2),
                         torch.max(element3, element4))

    def max_of_8_tensor(self, element1, element2, element3,
                        element4, element5, element6, element7, element8):
        return torch.max(self.max_of_4_tensor(element1, element2, element3, element4),
                         self.max_of_4_tensor(element5, element6, element7, element8))
