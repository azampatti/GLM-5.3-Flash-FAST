#!/usr/bin/env python3
"""In-container patches for the GLM-5.3-Flash lever campaign (pilcothink/vllm_spark_glm53:0.28).

Applied by serve_wrap.sh inside EACH rank's container before `vllm serve`; the image on disk is never touched (a new
container starts clean). Every edit is idempotent (marker) and checked (anchor must match exactly once, file must parse).
All behaviour is env-driven, so with no GLM53_* env set the patched code is a no-op.

  GLM53_TOPK=<n>          routed experts per token override (config num_experts_per_token), e.g. 7 or 6
  GLM53_KDA_SPLIT=1       split KDA in_proj into in_proj_qkv + in_proj_bfg_a; GLM53_NVFP4_SERIALIZED=kda -> q/k/v/o NVFP4 W4A16,
                          GLM53_MXFP8_SERIALIZED=kdabfg -> b/f_a/g_a MXFP8 (view: build_kda_nvfp4.py)
  GLM53_MXFP8=<groups>    comma list of layer groups served as ONLINE MXFP8 (BF16 on disk -> e4m3 + e8m0 block-32 at load,
                          weight-only Marlin on GB10): se (shared experts), mla (MLA q_b_proj + o_proj), head (lm_head),
                          kda (KDA in_proj_qkvbfg_a + o_proj)
"""
import ast, sys

V = "/usr/local/lib/python3.12/dist-packages/vllm"
MARK = "glm53-lever"
done, fatal = [], []


def edit(path, fn, label):
    try:
        src = open(path).read()
        if f"{MARK}:{label}" in src:
            done.append(f"{label}(already)")
            return
        new = fn(src)
        if new is None or new == src:
            raise ValueError("anchor not found")
        ast.parse(new)
        open(path, "w").write(new)
        done.append(label)
    except Exception as e:  # noqa: BLE001
        fatal.append(f"{label}: {e} [{path}]")


def once(src, old, new):
    return src.replace(old, new) if src.count(old) == 1 else None


# 1. top-k override ------------------------------------------------------------------------------------------------
edit(f"{V}/transformers_utils/configs/glm5_next.py",
     lambda s: once(s, "        self.num_experts_per_token = num_experts_per_token\n",
                    "        self.num_experts_per_token = num_experts_per_token\n"
                    f"        import os as _os  # {MARK}:topk\n"
                    "        if _os.environ.get('GLM53_TOPK'):\n"
                    "            if int(_os.environ['GLM53_TOPK']) != self.num_experts_per_token:\n"
                    "                print(f\"[glm53-lever] top-k {self.num_experts_per_token} -> {_os.environ['GLM53_TOPK']}\", flush=True)\n"
                    "            self.num_experts_per_token = int(_os.environ['GLM53_TOPK'])\n"),
     "topk")

# 2. online MXFP8 for selected BF16 linears -------------------------------------------------------------------------
HOOK = f'''

# --- {MARK}:mxfp8-hook ---
def _glm53_mxfp8_patterns():
    import os as _os
    groups = {{g.strip() for g in _os.environ.get("GLM53_MXFP8", "").split(",") if g.strip()}}
    mla = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    kda = [i for i in range(45) if i not in mla]
    pats = []
    if "se" in groups:
        pats += ["*.layers.*.mlp.shared_experts.gate_up_proj", "*.layers.*.mlp.shared_experts.down_proj"]
    if "mla" in groups:
        pats += [f"*.layers.{{i}}.self_attn.q_b_proj" for i in mla] + [f"*.layers.{{i}}.self_attn.o_proj" for i in mla]
    if "kda" in groups:
        pats += [f"*.layers.{{i}}.self_attn.in_proj_qkvbfg_a" for i in kda] + [f"*.layers.{{i}}.self_attn.o_proj" for i in kda]
    if "head" in groups:
        pats += ["lm_head", "*.lm_head"]
    return pats


def _glm53_serialized_patterns():
    import os as _os
    groups = {{g.strip() for g in _os.environ.get("GLM53_MXFP8_SERIALIZED", "").split(",") if g.strip()}}
    mla = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    kda = [i for i in range(45) if i not in mla]
    pats = []
    if "kda" in groups:
        pats += [f"*.layers.{{i}}.self_attn.in_proj_qkvbfg_a" for i in kda] + [f"*.layers.{{i}}.self_attn.o_proj" for i in kda]
    if "kdabfg" in groups:
        pats += [f"*.layers.{{i}}.self_attn.in_proj_bfg_a" for i in kda]
    return pats


_GLM53_SER_PATS = _glm53_serialized_patterns()


def _glm53_nvfp4_patterns():
    import os as _os
    groups = {{g.strip() for g in _os.environ.get("GLM53_NVFP4_SERIALIZED", "").split(",") if g.strip()}}
    mla = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    kda = [i for i in range(45) if i not in mla]
    pats = []
    if "kda" in groups:
        pats += [f"*.layers.{{i}}.self_attn.in_proj_qkv" for i in kda] + [f"*.layers.{{i}}.self_attn.o_proj" for i in kda]
    return pats


_GLM53_NVFP4_PATS = _glm53_nvfp4_patterns()


_GLM53_PATS = _glm53_mxfp8_patterns()
_GLM53_HITS = []


def _glm53_match(prefix):
    import fnmatch as _fn
    return bool(prefix) and any(_fn.fnmatchcase(prefix, p) for p in _GLM53_PATS)


def _glm53_note(prefix):
    _GLM53_HITS.append(prefix)
    n = len(_GLM53_HITS)
    if n <= 2 or n % 25 == 0:
        print(f"[glm53-lever] online MXFP8 #{{n}}: {{prefix}}", flush=True)
'''

LINEAR_WRAP = HOOK + f'''
# --- {MARK}:mxfp8-linear ---

def _glm53_wrap_linear_base():
    _orig_init = LinearBase.__init__

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        _p = getattr(self, "prefix", "")
        if _GLM53_NVFP4_PATS and _p and isinstance(self.quant_method, UnquantizedLinearMethod) \\
                and any(__import__("fnmatch").fnmatchcase(_p, q) for q in _GLM53_NVFP4_PATS):
            from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
            _cfg = ModelOptNvFp4Config(quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True,
                                       kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
            self.quant_method = _cfg.get_quant_method(self, _p)
            _GLM53_HITS.append(_p)
            if len(_GLM53_HITS) <= 2 or len(_GLM53_HITS) % 25 == 0:
                print(f"[glm53-lever] serialized NVFP4-W4A16 #{{len(_GLM53_HITS)}}: {{_p}} -> {{type(self.quant_method).__name__}}", flush=True)
        elif _GLM53_SER_PATS and _p and isinstance(self.quant_method, UnquantizedLinearMethod) \\
                and any(__import__("fnmatch").fnmatchcase(_p, q) for q in _GLM53_SER_PATS):
            from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
            _cfg = ModelOptMxFp8Config(is_checkpoint_mxfp8_serialized=True, kv_cache_quant_algo=None, exclude_modules=[])
            self.quant_method = _cfg.get_quant_method(self, _p)
            _GLM53_HITS.append(_p)
            if len(_GLM53_HITS) <= 2 or len(_GLM53_HITS) % 25 == 0:
                print(f"[glm53-lever] serialized MXFP8 #{{len(_GLM53_HITS)}}: {{_p}} -> {{type(self.quant_method).__name__}}", flush=True)
        elif _GLM53_PATS and _glm53_match(getattr(self, "prefix", "")) and isinstance(self.quant_method, UnquantizedLinearMethod):
            from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineLinearMethod
            self.quant_method = Mxfp8OnlineLinearMethod()
            _glm53_note(self.prefix)

    LinearBase.__init__ = __init__


_glm53_wrap_linear_base()
'''
edit(f"{V}/model_executor/layers/linear.py", lambda s: s + LINEAR_WRAP, "mxfp8-linear")

HEAD_WRAP = HOOK + f'''
# --- {MARK}:mxfp8-head ---

def _glm53_wrap_head_resolver():
    global resolve_quant_method
    _orig = resolve_quant_method

    def _resolve(quant_config, layer, prefix=""):
        if _GLM53_PATS and isinstance(layer, ParallelLMHead) and _glm53_match(prefix):
            from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineLinearMethod
            _glm53_note(prefix)
            return Mxfp8OnlineLinearMethod()
        return _orig(quant_config, layer, prefix=prefix)

    resolve_quant_method = _resolve


_glm53_wrap_head_resolver()
'''
edit(f"{V}/model_executor/layers/vocab_parallel_embedding.py", lambda s: s + HEAD_WRAP, "mxfp8-head")

# 3. allow an online-quantized DFlash2 draft (pilcothink's adapter pins it to BF16) -- only when GLM53_DRAFT_QUANT_OK=1.
#    Lossless for outputs (the target verifies every token); everything else in the guard stays enforced.
edit(f"{V}/models/glm5next/nvidia/model.py",
     lambda s: once(s, "            or sc.draft_model_config.quantization is not None):\n",
                    "            or (sc.draft_model_config.quantization is not None  # glm53-lever:draft-quant\n"
                    "                and __import__('os').environ.get('GLM53_DRAFT_QUANT_OK') != '1')):\n"),
     "draft-quant")

# 4. the FP8-MLA dequant hook in load_weights must not swallow MXFP8-serialized o_proj weights (they carry .weight_scale)
edit(f"{V}/models/glm5next/nvidia/model.py",
     lambda s: once(s, "    if target_s in params_dict:\n        return False\n",
                    "    if target_s in params_dict:\n        return False\n"
                    "    if f\"{layer_prefix}.{target_base}.weight_scale\" in params_dict:  # glm53-lever:fp8-attn-guard\n"
                    "        return False\n"),
     "fp8-attn-guard")

# 5. index filter: for model views under .../glm53-builds/, read a tensor only from the file the view's index names for it
FILTER = f'''

# --- {MARK}:index-filter ---
def _glm53_index_filter():
    import json as _json
    from pathlib import Path as _Path
    global safetensors_weights_iterator
    _orig = safetensors_weights_iterator
    _maps = {{}}

    def _filtered(hf_weights_files, *args, **kwargs):
        files = list(hf_weights_files)
        folders = {{_Path(f).parent for f in files}}
        if len(folders) != 1 or "/glm53-builds/" not in str(next(iter(folders))) + "/":
            yield from _orig(files, *args, **kwargs)
            return
        folder = next(iter(folders))
        if folder not in _maps:
            _maps[folder] = _json.load(open(folder / "model.safetensors.index.json"))["weight_map"]
            print(f"[glm53-lever] index filter active for {{folder}}", flush=True)
        wm = _maps[folder]
        for f in files:
            here = _Path(f).name
            for name, tensor in _orig([f], *args, **kwargs):
                if wm.get(name) == here:
                    yield name, tensor

    safetensors_weights_iterator = _filtered


_glm53_index_filter()
'''
edit(f"{V}/model_executor/model_loader/default_loader.py", lambda s: s + FILTER, "index-filter")

# 6. KDA split (GLM53_KDA_SPLIT=1): in_proj_qkvbfg_a -> in_proj_qkv (q|k|v, N % 64 == 0 for Marlin FP4) + in_proj_bfg_a (b|f_a|g_a)
KDA_CTOR_START = "        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(\n"
KDA_CTOR_END = '            prefix=f"{prefix}.in_proj_qkvbfg_a",\n        )\n'
def _kda_ctor(s):
    if s.count(KDA_CTOR_START) != 1:
        return None
    a = s.index(KDA_CTOR_START); b = s.index(KDA_CTOR_END, a) + len(KDA_CTOR_END)
    orig = s[a:b]
    new = ("        import os as _glm53_os  # glm53-lever:kda-split\n"
           "        self._glm53_split = _glm53_os.environ.get('GLM53_KDA_SPLIT') == '1'\n"
           "        if self._glm53_split:\n"
           "            self.in_proj_qkv = MergedColumnParallelLinear(\n"
           "                self.hidden_size, [projection_size] * 3, bias=False,\n"
           "                quant_config=self.quant_config, prefix=f\"{prefix}.in_proj_qkv\")\n"
           "            self.in_proj_bfg_a = _Glm5NextMergedColumnParallelLinear(\n"
           "                self.hidden_size, [self.num_heads, self.head_dim, self.head_dim],\n"
           "                replicated_shard_ids=(1, 2), tp_size=self.tp_size, bias=False,\n"
           "                quant_config=self.quant_config, prefix=f\"{prefix}.in_proj_bfg_a\")\n"
           "        else:\n" + "".join("    " + l if l.strip() else l for l in orig.splitlines(True)))
    return s[:a] + new + s[b:]
edit(f"{V}/models/glm5next/nvidia/kda.py", _kda_ctor, "kda-split")

KDA_FWD_START = "        projected = self.in_proj_qkvbfg_a(hidden_states)[0]\n"
KDA_FWD_END = "            dim=-1,\n        )\n"
def _kda_fwd(s):
    if s.count(KDA_FWD_START) != 1:
        return None
    a = s.index(KDA_FWD_START); b = s.index(KDA_FWD_END, a) + len(KDA_FWD_END)
    orig = s[a:b]
    new = ("        if self._glm53_split:  # glm53-lever:kda-split-fwd\n"
           "            qkv = self.in_proj_qkv(hidden_states)[0]\n"
           "            beta_raw, f_a, g_a = self.in_proj_bfg_a(hidden_states)[0].split(\n"
           "                [self.local_num_heads, self.head_dim, self.head_dim], dim=-1)\n"
           "        else:\n" + "".join("    " + l if l.strip() else l for l in orig.splitlines(True)))
    return s[:a] + new + s[b:]
edit(f"{V}/models/glm5next/nvidia/kda.py", _kda_fwd, "kda-split-fwd")

# 7. checkpoint-name mapping for the split
MAP_OLD = ('            (".in_proj_qkvbfg_a", ".q_proj", 0),\n            (".in_proj_qkvbfg_a", ".k_proj", 1),\n'
           '            (".in_proj_qkvbfg_a", ".v_proj", 2),\n            (".in_proj_qkvbfg_a", ".b_proj", 3),\n'
           '            (".in_proj_qkvbfg_a", ".f_a_proj", 4),\n            (".in_proj_qkvbfg_a", ".g_a_proj", 5),\n')
MAP_NEW = ('            *([(".in_proj_qkv", ".q_proj", 0), (".in_proj_qkv", ".k_proj", 1), (".in_proj_qkv", ".v_proj", 2),\n'
           '               (".in_proj_bfg_a", ".b_proj", 0), (".in_proj_bfg_a", ".f_a_proj", 1), (".in_proj_bfg_a", ".g_a_proj", 2)]\n'
           '              if __import__("os").environ.get("GLM53_KDA_SPLIT") == "1" else [  # glm53-lever:kda-split-map\n'
           + MAP_OLD + '            ]),\n')
edit(f"{V}/models/glm5next/nvidia/model.py", lambda s: once(s, MAP_OLD, MAP_NEW), "kda-split-map")

# 9. W4A16 NVFP4 dense layers under --linear-backend b12x -> Marlin (b12x has no W4A16 dense kernel; nothing else is W4A16)
edit(f"{V}/model_executor/kernels/linear/__init__.py",
     lambda s: once(s, '    elif linear_backend == "auto" and use_a16:\n',
                    '    elif linear_backend in ("auto", "b12x") and use_a16:  # glm53-lever:a16-marlin\n'),
     "a16-marlin")

print(f"[glm53-lever] patches: {', '.join(done) or '-'}", flush=True)
if fatal:
    print("[glm53-lever] FATAL: " + " | ".join(fatal), flush=True)
    sys.exit(1)
