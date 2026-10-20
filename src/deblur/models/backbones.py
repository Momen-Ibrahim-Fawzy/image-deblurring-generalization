"""Loading restoration backbones from their official repositories.

FFTformer and SAM-Deblur each ship their own copy of a package called
``basicsr``. Importing both into one process normally collides, so this module
imports an architecture file from a given repository root while temporarily
isolating ``basicsr`` in ``sys.modules``.

Repositories are cloned into ``external/`` by ``scripts/setup_external.sh``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXTERNAL = Path(__file__).resolve().parents[3] / "external"

ARCHS = {
    # name: (repo folder, module path, class name, default kwargs)
    "fftformer": ("FFTformer", "basicsr.models.archs.fftformer_arch", "fftformer", {}),
    "nafnet": ("SAM-Deblur", "basicsr.models.archs.NAFNet_arch", "NAFNet",
               {"width": 64, "enc_blk_nums": [1, 1, 1, 28], "middle_blk_num": 1,
                "dec_blk_nums": [1, 1, 1, 1]}),
}


@contextmanager
def _isolated_basicsr(repo_root: Path):
    saved = {k: v for k, v in sys.modules.items() if k == "basicsr" or k.startswith("basicsr.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(repo_root))
    try:
        yield
    finally:
        sys.path.remove(str(repo_root))
        for k in [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def build_backbone(name: str, external_dir: str | Path | None = None, **kwargs) -> nn.Module:
    """Instantiate a backbone architecture from its official repository."""
    key = name.lower()
    if key not in ARCHS:
        raise KeyError(f"unknown backbone '{name}', choose from {sorted(ARCHS)}")
    repo, module_path, cls_name, defaults = ARCHS[key]
    repo_root = Path(external_dir or EXTERNAL) / repo
    if not repo_root.exists():
        raise FileNotFoundError(f"{repo_root} not found, run scripts/setup_external.sh first")
    arch_file = repo_root / (module_path.replace(".", "/") + ".py")
    with _isolated_basicsr(repo_root):
        # Load the architecture file directly so that basicsr/__init__.py (which pulls in
        # OpenCV, lmdb, ...) is only imported if the architecture itself needs it.
        spec = importlib.util.spec_from_file_location(f"_deblur_ext_{key}", arch_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, cls_name)
    return cls(**{**defaults, **kwargs})


def load_checkpoint(model: nn.Module, path: str | Path, strict: bool = True) -> nn.Module:
    """Load BasicSR-style checkpoints ({'params': ...} or {'params_ema': ...}) or plain state dicts."""
    state = torch.load(path, map_location="cpu")
    for key in ("params_ema", "params", "state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict and (missing or unexpected):
        print(f"[load_checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    return model


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad H and W up to a multiple (FFTformer needs multiples of 32)."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


def crop_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return x[..., : size[0], : size[1]]
