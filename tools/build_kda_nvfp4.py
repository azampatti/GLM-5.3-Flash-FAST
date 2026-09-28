#!/usr/bin/env python3
"""Offline 4-bit KDA view (runs inside the vLLM image, CPU only): KDA q/k/v/o -> NVFP4 (W4A16), b/f_a/g_a -> MXFP8.

Needs the campaign patch with GLM53_KDA_SPLIT=1 (in_proj_qkvbfg_a split into in_proj_qkv [NVFP4, Marlin needs N % 64 == 0,
12288 per rank] + in_proj_bfg_a [MXFP8, 288 per rank]), GLM53_NVFP4_SERIALIZED=kda and GLM53_MXFP8_SERIALIZED=kdabfg.

NVFP4 on disk (ModelOpt convention, same as the checkpoint's dense MLPs):
  .weight         U8 [N, K/2]   e2m1 codes, LOW nibble = even column (vLLM break_fp4_bytes order)
  .weight_scale   F8_E4M3 [N, K/16]  per-16 block scales (linear layout)
  .weight_scale_2 F32 []         = amax / (6 * 448); dequant = e2m1 * weight_scale * weight_scale_2
q/k/v of one layer share ONE weight_scale_2 (they are fused into in_proj_qkv; ModelOpt syncs amax across fused shards).
Quantization math = vLLM's ref_nvfp4_quant (round-to-nearest e2m1, e4m3 block scales); every tensor is round-trip checked.

usage: build_kda_nvfp4.py <snapshot dir> <out dir>
"""
import json, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant, break_fp4_bytes

snap, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
NEW = "model-kda-nvfp4.safetensors"
MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43}
KDA = [i for i in range(45) if i not in MLA]
FP4, MX = ["q_proj", "k_proj", "v_proj", "o_proj"], ["b_proj", "f_a_proj", "g_a_proj"]
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

idx = json.load(open(f"{snap}/model.safetensors.index.json"))
wm = dict(idx["weight_map"])
P = lambda i, p: f"model.language_model.layers.{i}.self_attn.{p}.weight"  # noqa: E731


def load(k):
    with safe_open(f"{snap}/{wm[k]}", framework="pt") as fh:
        return fh.get_tensor(k)


def nvfp4(w, gscale):
    """-> (packed U8 [N,K/2], F8 scales [N,K/16], dequantized f32) using vLLM's ref quantizer."""
    q, s = ref_nvfp4_quant(w.float(), gscale, 16)            # q: f32 values on the e2m1 grid (signed), s: f32 [N, K/16]
    mag = q.abs()
    code = torch.searchsorted(E2M1, mag.contiguous())         # exact grid values -> index 0..7
    if not torch.equal(E2M1[code], mag):
        raise ValueError("values off the e2m1 grid")
    code = code.to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    s8 = s.to(torch.float8_e4m3fn)
    deq = break_fp4_bytes(packed, torch.float32).view(w.shape[0], -1, 16) * (s8.float() / gscale).unsqueeze(-1)
    return packed, s8.contiguous(), deq.view_as(w)


t0 = time.time()
tensors, bf16_bytes, worst = {}, 0, 0.0
for i in KDA:
    ws = {p: load(P(i, p)) for p in FP4 + MX}
    for p, w in ws.items():
        if w.dtype != torch.bfloat16 or w.shape[1] % 32:
            sys.exit(f"{P(i, p)}: unexpected {w.dtype} {tuple(w.shape)}")
        bf16_bytes += w.numel() * 2
    groups = [["q_proj", "k_proj", "v_proj"], ["o_proj"]]      # shared global scale per fused group
    for g in groups:
        amax = max(ws[p].abs().max().float().item() for p in g)
        gscale = torch.tensor(448.0 * 6.0 / amax, dtype=torch.float32)
        for p in g:
            packed, s8, deq = nvfp4(ws[p], gscale)
            err = ((deq - ws[p].float()).norm() / ws[p].float().norm()).item()
            worst = max(worst, err)
            if err > 0.2:
                sys.exit(f"{P(i, p)}: NVFP4 round-trip error {err:.3f}")
            k = P(i, p)
            tensors[k] = packed
            tensors[k[:-len(".weight")] + ".weight_scale"] = s8
            tensors[k[:-len(".weight")] + ".weight_scale_2"] = (1.0 / gscale).reshape(())
    for p in MX:
        q, s = mxfp8_e4m3_quantize(ws[p].contiguous())
        k = P(i, p)
        tensors[k] = q.contiguous()
        tensors[k[:-len(".weight")] + ".weight_scale"] = s.contiguous()

os.makedirs(out, exist_ok=True)
for name in os.listdir(out):
    pth = os.path.join(out, name)
    if os.path.islink(pth) or name in (NEW, "model.safetensors.index.json"):
        os.remove(pth)
save_file(tensors, f"{out}/{NEW}", metadata={"format": "pt", "producer": "glm53-lever build_kda_nvfp4.py (vllm ref_nvfp4_quant)"})
rel = os.path.relpath(snap, out)
for name in sorted(os.listdir(snap)):
    if name != "model.safetensors.index.json":
        os.symlink(os.path.join(rel, name), os.path.join(out, name))
for k in tensors:
    wm[k] = NEW
json.dump({"metadata": idx.get("metadata", {}), "weight_map": wm}, open(f"{out}/model.safetensors.index.json", "w"), indent=1)
print(f"KDA 4-bit view: {len(KDA)} layers, BF16 {bf16_bytes / 1e9:.2f} GB -> {os.path.getsize(f'{out}/{NEW}') / 1e9:.2f} GB "
      f"({len(tensors)} tensors), worst NVFP4 rel err {worst:.3f}, {time.time() - t0:.0f} s; view {out}")
