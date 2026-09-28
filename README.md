# GLM-5.3-Flash FAST · 2× DGX Spark

> **Requires [0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks](https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks),
> cloned and with its `.env` filled in for your two nodes.** This repo uses its `.env` (IPs, NICs, HF cache) and its launcher.

A faster build of `nvidia/GLM-5.3-Flash-NVFP4` on two DGX Sparks: draft depth 7, MXFP8 for the BF16 shared experts,
MLA projections and `lm_head`, and 4-bit KDA attention. **53.0 tok/s vs 40.7** for the upstream recipe (run.py),
Tool-Eval-Bench 95, same 900K context.

## Run it (on the head node)

```bash
git clone https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks ~/GLM53F-NvidiaNVFP4-DFlash
git -C ~/GLM53F-NvidiaNVFP4-DFlash checkout 4ec03bf
cp ~/GLM53F-NvidiaNVFP4-DFlash/.env.sample ~/GLM53F-NvidiaNVFP4-DFlash/.env    # fill in your site values

git clone https://github.com/azampatti/GLM-5.3-Flash-FAST ~/GLM-5.3-Flash-FAST
cd ~/GLM-5.3-Flash-FAST
./setup.sh     # pinned image, model and draft on both nodes; builds the 4-bit KDA weights; copies this repo to the worker
./serve.sh     # boots both nodes (~16 min), verifies the build, API on http://<head>:8000/v1
./stop.sh
```

`setup.sh` writes the rebuilt KDA weights into `glm53-builds/` here (2.6 GB, checked byte for byte); everything else
in that folder links to the model in your HF cache. This repo must sit at the same path on both nodes.

## Options

In `config.env`, or for one run: `PORT=8100 ./serve.sh`.

| Setting | Default | |
|---|---|---|
| `UPSTREAM_DIR` | `~/GLM53F-NvidiaNVFP4-DFlash` | where the required repo is cloned |
| `PORT` | 8000 | API port |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | the name clients send as `"model"` |
| `CTX` | 900096 | longest request |
| `KV_CACHE_MEMORY` | 9663676416 | KV pin per node (9 GiB = 1,043,808 tokens) |
| `SEQS` | 4 | requests at once |
| `K` | 7 | draft depth |

`./serve.sh --dry-run` prints the launch command. `OFFLINE=1 ./setup.sh` skips the hub check.

## Credits

- **Z.ai** for GLM-5.3-Flash, **NVIDIA** for the NVFP4 checkpoint, **incoai** for the DFlash2 draft
- **[0rand](https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks)** for the two-node recipe and launcher
- **pilcothink** for the `vllm_spark_glm53` image, and **[lukealonso](https://github.com/lukealonso/b12x)** for b12x inside it
