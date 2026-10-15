"""Miles FSDP checkpoint (``<out>/dcp/iter_XXXXXXX/model``, PyTorch DCP) -> a HF directory that
``training/serve_jev.py`` / ``eval_decisions`` / ``calibrate`` load unchanged (``--adapter <dir>``: a full checkpoint).

  python -m jevharness.training.miles_port.export --dcp checkpoints/r6/sft/dcp --iter last --origin <init HF dir> \
      --out checkpoints/r6/sft_v2

The weights are loaded into the SAME transformers class Miles' FSDP actor trained (the image-text-to-text class for
Qwen3.5-family checkpoints that carry a vision config, else the native causal-LM class), strictly: every parameter of
the class must come from the checkpoint (a tied output head excepted) and no checkpoint tensor may be left over, so a
key-mapping error cannot silently leave random weights (the stock ``tools/convert_fsdp_to_hf.py`` loads non-strictly).
Saved in bf16 (the trainer's compute dtype) with the origin's tokenizer, chat template and processor files, plus
``export.json`` (source, iteration, sha256 of every weight file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

PREFIXES = ("model_state.model.", "model_state.", "model.", "")
COPY = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
        "chat_template.jinja", "chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json",
        "generation_config.json", "added_tokens.json")
EXPORT_FILES = set(COPY) | {"config.json", "export.json", "decision_config.json"}


def model_class(origin: str) -> Any:
    """Miles' ``FSDPTrainRayActor._get_model_cls`` rule."""
    import transformers
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    cfg = AutoConfig.from_pretrained(origin, trust_remote_code=True)
    if hasattr(cfg, "vision_config"):
        auto_map = getattr(cfg, "auto_map", None)
        if not auto_map or "AutoModelForImageTextToText" in auto_map:
            return transformers.AutoModelForImageTextToText
    name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(getattr(cfg, "model_type", ""))
    return getattr(transformers, name) if name else transformers.AutoModelForCausalLM


def latest_iter(dcp_root: Path) -> Path:
    tracker = dcp_root / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        return dcp_root / f"iter_{int(tracker.read_text().strip()):07d}"
    its = sorted(dcp_root.glob("iter_*"))
    if not its:
        raise FileNotFoundError(f"no iter_* under {dcp_root}")
    return its[-1]


def load_dcp_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner

    sd: dict[str, Any] = {}
    dcp.state_dict_loader._load_state_dict(sd, storage_reader=dcp.FileSystemReader(str(model_dir)),
                                           planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    flat: dict[str, torch.Tensor] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            flat[prefix] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)

    walk("", sd)
    return flat


def map_keys(tensors: dict[str, torch.Tensor], target: set[str]) -> tuple[dict[str, torch.Tensor], str]:
    best, best_n = "", -1
    for p in PREFIXES:
        n = sum(1 for k in tensors if k.startswith(p) and k[len(p):] in target)
        if n > best_n:
            best, best_n = p, n
    return {k[len(best):]: v for k, v in tensors.items() if k.startswith(best)}, best


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def export(dcp_root: str, origin: str, out: str, iteration: str = "last", extra: dict[str, Any] | None = None
           ) -> dict[str, Any]:
    root = Path(dcp_root)
    it_dir = latest_iter(root) if iteration == "last" else root / f"iter_{int(iteration):07d}"
    tensors = load_dcp_tensors(it_dir / "model")
    cls = model_class(origin)
    model = cls.from_pretrained(origin, dtype=torch.bfloat16)
    target = model.state_dict()
    mapped, prefix = map_keys(tensors, set(target))
    tied = bool(getattr(model.config, "tie_word_embeddings", False) or
                getattr(getattr(model.config, "text_config", None), "tie_word_embeddings", False))
    missing = [k for k in target if k not in mapped and not (tied and k.endswith("lm_head.weight"))]
    unexpected = [k for k in mapped if k not in target]
    if missing or unexpected:
        raise RuntimeError(f"strict export failed ({cls.__name__}, prefix {prefix!r}): missing {missing[:8]} "
                           f"(+{max(0, len(missing) - 8)}), unexpected {unexpected[:8]} (+{max(0, len(unexpected) - 8)})")
    with torch.no_grad():
        for k, v in mapped.items():
            if target[k].shape != v.shape:
                raise RuntimeError(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(target[k].shape)}")
            target[k].copy_(v.to(target[k].dtype))
    o = Path(out)
    # The last continuation of a stage is exported into the stage's own directory, which also holds its records
    # (rollout_metrics.jsonl, rollouts.jsonl, teacher_ledger.jsonl, stage_config.yaml, dcp/): remove only what an
    # export writes (stale weights / configs), never the records.
    if o.exists():
        for p in o.iterdir():
            if p.is_file() and (p.name.endswith((".safetensors", ".safetensors.index.json")) or p.name in EXPORT_FILES):
                p.unlink()
    o.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(o, safe_serialization=True)
    for name in COPY:
        src = Path(origin) / name
        if src.exists():
            shutil.copy2(src, o / name)
    files = {p.name: sha256(p) for p in sorted(o.glob("*.safetensors"))}
    info = {"source": str(it_dir), "origin": origin, "class": cls.__name__, "prefix": prefix,
            "tensors": len(mapped), "dtype": "bfloat16", "weights_sha256": files, **(extra or {})}
    meta = json.loads((it_dir / "meta.json").read_text()) if (it_dir / "meta.json").exists() else {}
    info["miles_meta"] = meta
    (o / "export.json").write_text(json.dumps(info, indent=1))
    (o / "decision_config.json").write_text(json.dumps(
        {"base_model": origin, "lora": False, "max_len": 4096, "trained_by": "miles_port", **(extra or {})}, indent=1))
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", required=True, help="the stage's --save directory (iter_* inside)")
    ap.add_argument("--iter", default="last", help="iteration number (step id) or 'last'")
    ap.add_argument("--origin", required=True, help="the HF directory the stage started from (config + tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="{}", help="JSON merged into export.json / decision_config.json")
    a = ap.parse_args(argv)
    print(json.dumps(export(a.dcp, a.origin, a.out, a.iter, json.loads(a.extra)), indent=1))


if __name__ == "__main__":
    main()
