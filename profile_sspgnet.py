# -*- coding: utf-8 -*-
"""
profile_sspgnet.py
==================
Profile the SSPGNet (StereoA) model:
  - parameter count (total / trainable / frozen)
  - FLOPs at a typical KITTI input size
  - wall-clock inference latency on the host GPU

Run from the SAStereo project root on a CUDA-capable machine:

    cd /path/to/SAStereo
    pip install thop
    python profile_sspgnet.py

The DINOv2 backbone is loaded via torch.hub with source='local' (see
Source/UserModelImplementation/Models/StereoA/Networks/model.py:50).
If your TORCH_HOME / torch.hub directory layout is different from the
training setup, set TORCH_HOME (or torch.hub.set_dir) before running.
"""

import os
import sys
import time

import torch


def _setup_paths():
    """Mirror the import paths used when JackFramework launches training."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, 'Source')
    user = os.path.join(src, 'UserModelImplementation')
    for p in (src, user, here):
        if p not in sys.path:
            sys.path.insert(0, p)


def main():
    _setup_paths()

    if not torch.cuda.is_available():
        print('!! No CUDA device detected. This script needs a GPU '
              '(parameter counting alone could run on CPU, but DINOv2 '
              'instantiation and the latency benchmark expect cuda()). '
              'Run on the 3090 server.', file=sys.stderr)
        sys.exit(1)

    # Lazy import after sys.path is set up.
    from UserModelImplementation.Models.StereoA.Networks import StereoA

    # Match the inference configuration in
    # Source/UserModelImplementation/Models/StereoA/inference.py:42-44
    # and the defaults in user_define.py.
    print('Building SSPGNet (StereoA) ...', flush=True)
    model = StereoA(
        in_channles=3,        # original signature has the typo "in_channles"
        start_disp=1,
        disp_num=196,
        backbone='dinov2',
        pre_train_opt=False,
        confidence_level=0.10,
    ).cuda().eval()

    # Freeze the DINOv2 backbone, mirroring inference.py.
    for name, p in model.named_parameters():
        if 'pretrained' in name:
            p.requires_grad = False

    # ------------------------------------------------------------------
    # 1) Parameter counts
    # ------------------------------------------------------------------
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print()
    print('========== SSPGNet parameter count ==========')
    print(f'  total      : {total/1e6:8.2f} M')
    print(f'  trainable  : {trainable/1e6:8.2f} M')
    print(f'  frozen VFM : {frozen/1e6:8.2f} M')

    # KITTI envelope, multiples of 14 (DINOv2 patch size)
    H, W = 378, 1246            # 27*14, 89*14, covers KITTI raw 376x1241
    left = torch.randn(1, 3, H, W).cuda()
    right = torch.randn(1, 3, H, W).cuda()

    # ------------------------------------------------------------------
    # 2) FLOPs
    # ------------------------------------------------------------------
    try:
        from thop import profile
    except ImportError:
        print()
        print('!! `thop` is not installed; skipping FLOPs.')
        print('   Install with `pip install thop` and re-run.')
    else:
        with torch.no_grad():
            flops, _ = profile(model, inputs=(left, right), verbose=False)

        print()
        print('========== SSPGNet FLOPs (one stereo pair) ==')
        print(f'  input         : 1 x 3 x {H} x {W}  (KITTI envelope, multiples of 14)')
        print(f'  forward FLOPs : {flops/1e9:8.2f} GFLOPs   ({flops/1e12:5.2f} TFLOPs)')

    # ------------------------------------------------------------------
    # 3) Wall-clock latency
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    for _ in range(3):                           # warmup
        with torch.no_grad():
            _ = model(left, right)
    torch.cuda.synchronize()

    N = 10
    t0 = time.time()
    for _ in range(N):
        with torch.no_grad():
            _ = model(left, right)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N

    print()
    print('========== SSPGNet wall-clock latency =======')
    print(f'  device     : {torch.cuda.get_device_name(0)}')
    print(f'  resolution : {H} x {W}')
    print(f'  avg over {N:>2}: {dt*1000:8.1f} ms  ({dt:5.3f} s)')


if __name__ == '__main__':
    main()
