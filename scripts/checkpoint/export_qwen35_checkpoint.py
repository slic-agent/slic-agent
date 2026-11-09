#!/usr/bin/env python3
"""Export a Qwen3.5 FSDP actor checkpoint written by `slic train` to an HF directory that vLLM / transformers load.

Qwen3.5 checkpoints are saved by the trainer under the text-only key layout (`model.<x>`), while the released HF directories
of Qwen3.5 (and the `vllm serve` loader) use the composite layout `model.language_model.<x>`. This tool
  1. merges the FSDP shards of `--src` (a `global_step_N` directory with `actor/`, or the `actor/` directory itself),
  2. renames `model.<x>` -> `model.language_model.<x>` (lm_head unchanged) and writes ~5 GB safetensors shards with an index,
  3. copies config.json, chat_template.jinja, tokenizer*, generation_config.json and the preprocessor configs from `--base`
     (the HF directory of the model the run started from, e.g. your SFT checkpoint), and
  4. checks that the exported key set equals the key set of `--base` when that directory carries safetensors weights.

    python scripts/checkpoint/export_qwen35_checkpoint.py --src outputs/checkpoints/<experiment>/best/global_step_360 \\
        --base models/my-sft-qwen3.5-4b --dst models/my-slic-random

For other model families use `scripts/checkpoint/merge_checkpoint.py` directly.
"""
import argparse, hashlib, json, os, re, shutil, struct, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="global_step_<N> directory (with actor/) or the actor/ directory")
ap.add_argument("--base", required=True, help="HF directory of the model the run started from (config, tokenizer, chat template)")
ap.add_argument("--dst", required=True, help="output HF directory (must not exist)")
a = ap.parse_args()
src = Path(a.src)
if (src / "actor").is_dir():
    src = src / "actor"
if not src.is_dir() or not any(f.startswith("model_world_size_") for f in os.listdir(src)):
    raise SystemExit(f"{src} holds no FSDP actor shards (model_world_size_*_rank_*.pt)")
base = Path(a.base)
if not (base / "config.json").is_file():
    raise SystemExit(f"{base} is not an HF model directory (no config.json)")
dst = Path(a.dst)
if dst.exists():
    raise SystemExit(f"{dst} exists")
tmp = dst.with_name(dst.name + ".merging"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)

import torch
from torch.distributed.tensor import DTensor
world = max(int(re.match(r"model_world_size_(\d+)_rank", f).group(1)) for f in os.listdir(src) if f.startswith("model_world_size_"))
shards = [torch.load(src / f"model_world_size_{world}_rank_{r}.pt", map_location="cpu", weights_only=False) for r in range(world)]
merged = {}
for k in sorted(shards[0]):
    parts = [sd.pop(k) for sd in shards]
    if isinstance(parts[0], DTensor):
        pl = parts[0].placements
        names = parts[0].device_mesh.mesh_dim_names or ("fsdp",)
        pl = pl[1:] if names[0] in ("dp", "ddp") else pl
        if len(pl) != 1:
            raise SystemExit(f"unsupported placements {pl} for {k}")
        locs = [t._local_tensor.to(torch.bfloat16) for t in parts]
        merged[k] = locs[0].contiguous() if pl[0].is_replicate() else torch.cat(locs, dim=pl[0].dim).contiguous()
    else:
        merged[k] = torch.cat([t.to(torch.bfloat16) for t in parts], dim=0) if parts[0].dim() > 0 else parts[0].to(torch.bfloat16)
del shards

def rename(k):
    return "model.language_model." + k[len("model."):] if k.startswith("model.") and not k.startswith("model.language_model.") else k

merged = {rename(k): v for k, v in merged.items()}
from safetensors.torch import save_file
files, cur, size, i = {}, {}, 0, 1
def flush():
    global cur, size, i
    if not cur:
        return
    name = f"model-{i:05d}.safetensors"; save_file(cur, str(tmp / name), metadata={"format": "pt"}); files[name] = list(cur); cur, size, i = {}, 0, i + 1
for k, v in merged.items():
    if size + v.numel() * 2 > 5_000_000_000:
        flush()
    cur[k] = v; size += v.numel() * 2
flush()
total = len(files)
for j, name in enumerate(list(files), 1):
    new = f"model-{j:05d}-of-{total:05d}.safetensors"; os.rename(tmp / name, tmp / new); files[new] = files.pop(name)
(tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": sum(v.numel() * 2 for v in merged.values())},
                                                            "weight_map": {k: f for f, ks in files.items() for k in ks}}, indent=2))
n_keys = len(merged); del merged

def st_keys(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
    return {k for k in hdr if k != "__metadata__"}

base_keys = set()
if (base / "model.safetensors.index.json").is_file():
    base_keys = set(json.loads((base / "model.safetensors.index.json").read_text())["weight_map"])
elif (base / "model.safetensors").is_file():
    base_keys = st_keys(base / "model.safetensors")
out_keys = set(json.loads((tmp / "model.safetensors.index.json").read_text())["weight_map"])
if base_keys and out_keys != base_keys:
    raise SystemExit(f"key set mismatch vs base: missing={sorted(base_keys - out_keys)[:5]} extra={sorted(out_keys - base_keys)[:5]}")
for name in ("config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "preprocessor_config.json", "video_preprocessor_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
    if (base / name).is_file():
        shutil.copy2(base / name, tmp / name)
prov = dict(source_fsdp=str(src), base=str(base), exported=time.strftime("%Y-%m-%d %H:%M %Z"), n_keys=n_keys,
            key_check="against base safetensors" if base_keys else "skipped (base carries no safetensors)",
            tool="scripts/checkpoint/export_qwen35_checkpoint.py (FSDP merge + key rename model.*->model.language_model.* + base aux files)",
            safetensors={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tmp.glob("*.safetensors"))})
(tmp / "export_provenance.json").write_text(json.dumps(prov, indent=2))
tmp.rename(dst)
print(json.dumps({k: v for k, v in prov.items() if k != "safetensors"}, indent=1)); print("OK", dst)
