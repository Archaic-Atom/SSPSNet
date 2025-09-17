#!/usr/bin/env python3
# ckpt2pth.py
# Robust converter: .ckpt (e.g., PyTorch Lightning or generic torch checkpoints) -> .pth (plain state_dict)
# Usage:
#   python ckpt2pth.py input.ckpt output.pth [--strip-common-prefixes] [--replace old=new ...]
#
# Notes:
# - Loads with torch.load; no dependency on PyTorch Lightning.
# - Auto-detects 'state_dict' / 'model' / 'net' keys; falls back to the first dict-of-tensors found.
# - Can strip common prefixes like 'module.' (from DataParallel), 'model.', 'net.', etc.
# - You can add custom key replacements via --replace old=new.
#
import argparse
import sys
import re
import json
from typing import Dict, Any
import torch

COMMON_PREFIXES = ["module.", "model.", "net.", "network.", "student.", "generator.", "discriminator.", "encoder.", "decoder."]


def is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    # Must have at least one Tensor value
    for k, v in obj.items():
        if torch.is_tensor(v):
            return True
    return False


def autodetect_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # Priority order
    candidates = []
    for key in ["state_dict", "model", "model_state", "model_state_dict", "net", "network"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            candidates.append((key, ckpt[key]))
    for key, cand in candidates:
        if is_state_dict_like(cand):
            print(f"[ckpt2pth] Using dict under key '{key}'", file=sys.stderr)
            return cand

    # Directly a state_dict?
    if is_state_dict_like(ckpt):
        print("[ckpt2pth] Using top-level dict as state_dict", file=sys.stderr)
        return ckpt  # type: ignore[return-value]

    # Search recursively (shallow)
    for k, v in ckpt.items():
        if is_state_dict_like(v):
            print(f"[ckpt2pth] Using nested state_dict at '{k}'", file=sys.stderr)
            return v

    raise RuntimeError("Could not find a state_dict-like dict inside checkpoint. "
                       "Inspect keys or pass through a different saving pipeline.")


def strip_common_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def strip_one(k: str) -> str:
        for p in COMMON_PREFIXES:
            if k.startswith(p):
                return k[len(p):]
        return k
    out = {}
    for k, v in sd.items():
        nk = strip_one(k)
        out[nk] = v
    return out


def apply_replacements(sd: Dict[str, torch.Tensor], rules):
    # rules is a list of (old, new) exact-prefix or regex substitutions.
    out = {}
    for k, v in sd.items():
        nk = k
        for old, new in rules:
            if old.startswith("^") or ("(" in old and ")") in old:
                nk = re.sub(old, new, nk)
            else:
                if nk.startswith(old):
                    nk = new + nk[len(old):]
        out[nk] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert .ckpt (Lightning or generic) to .pth (plain state_dict).")
    ap.add_argument("input", type=str, help="Path to input .ckpt/.pth")
    ap.add_argument("output", type=str, help="Path to output .pth (state_dict)")
    ap.add_argument("--strip-common-prefixes", action="store_true",
                    help="Strip common prefixes like 'module.', 'model.', 'net.' from keys.")
    ap.add_argument("--replace", type=str, nargs="*", default=[],
                    help="Key replacements, e.g., old=new or regex_old=regex_new (applied in order).")
    ap.add_argument("--print-keys", action="store_true", help="Print first 50 keys before/after for inspection.")
    args = ap.parse_args()

    print(f"[ckpt2pth] Loading: {args.input}", file=sys.stderr)
    ckpt = torch.load(args.input, map_location="cpu")
    sd = autodetect_state_dict(ckpt)

    if args.print_keys:
        all_keys = list(sd.keys())
        print("[ckpt2pth] First 50 original keys:", file=sys.stderr)
        print("\n".join(all_keys[:50]), file=sys.stderr)

    if args.strip_common_prefixes:
        sd = strip_common_prefixes(sd)

    rules = []
    for r in args.replace:
        if "=" not in r:
            print(f"[ckpt2pth] Ignoring malformed --replace rule: {r}", file=sys.stderr)
            continue
        old, new = r.split("=", 1)
        rules.append((old, new))
    if rules:
        sd = apply_replacements(sd, rules)

    if args.print_keys:
        all_keys = list(sd.keys())
        print("[ckpt2pth] First 50 final keys:", file=sys.stderr)
        print("\n".join(all_keys[:50]), file=sys.stderr)

    torch.save(sd, args.output)
    print(f"[ckpt2pth] Saved state_dict to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
