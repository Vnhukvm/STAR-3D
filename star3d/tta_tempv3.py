"""STAR-3D online retrieval adaptation, distilled from RoMa/BERT/lib/tta_tempv3.py.

The caller supplies an encoder for the query modality and frozen gallery
embeddings. This module owns the CRSI/REM adaptation and final score matrix.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    steps: int = 3
    lr: float = 3e-5
    weight_decay: float = 1e-4
    temperature: float = 0.1
    reliable_ratio: float = 0.3
    rem_capacity: int = 128
    history_batches: int = 32
    sinkhorn_tau: float = 0.05
    sinkhorn_iterations: int = 5
    entropy_weight: float = 1.0
    anchor_weight: float = 1.0
    adapt: str = "norm"
    rerank: str = "none"

    def validate(self) -> None:
        if self.steps < 1 or self.rem_capacity < 1 or self.history_batches < 1:
            raise ValueError("steps, rem_capacity and history_batches must be positive")
        if self.sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive")
        if self.temperature <= 0 or self.sinkhorn_tau <= 0:
            raise ValueError("temperatures must be positive")
        if not 0 < self.reliable_ratio <= 1:
            raise ValueError("reliable_ratio must lie in (0, 1]")
        if self.entropy_weight <= 0 or self.anchor_weight < 0:
            raise ValueError("invalid loss weights")
        if self.adapt not in {"norm", "adapter"}:
            raise ValueError("adapt must be norm or adapter")
        if self.rerank not in {"none", "sinkhorn"}:
            raise ValueError("rerank must be none or sinkhorn")


class FeatureEncoder(nn.Module):
    """Small executable example for already extracted query feature vectors.

    Real CrossOver/Mosaic3D use should pass its checkpoint-loaded query encoder
    to ``adapt_stream`` instead of substituting this example encoder.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.adapter = nn.Linear(dim, dim)
        nn.init.eye_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.adapter(self.norm(features)), dim=-1)


def _adapt_parameters(encoder: nn.Module, mode: str,
                      explicit: Iterable[nn.Parameter] | None) -> list[nn.Parameter]:
    chosen_explicit = list(explicit) if explicit is not None else None
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    selected: list[nn.Parameter] = []
    seen: set[int] = set()
    if chosen_explicit is not None:
        valid_ids = {id(p) for p in encoder.parameters()}
        if any(id(p) not in valid_ids for p in chosen_explicit):
            raise ValueError("adaptation_parameters must belong to encoder")
        selected = list(dict((id(p), p) for p in chosen_explicit).values())
    else:
        norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        for name, module in encoder.named_modules():
            chosen = isinstance(module, norm_types) if mode == "norm" else name.split(".")[-1] in {"adapter", "input_adapter", "_pre_l2_scale"}
            if chosen:
                for parameter in module.parameters(recurse=mode == "adapter"):
                    if id(parameter) not in seen:
                        selected.append(parameter)
                        seen.add(id(parameter))
    if not selected:
        raise ValueError(f"encoder has no {mode} parameters to adapt")
    for parameter in selected:
        parameter.requires_grad_(True)
    return selected


def _sinkhorn(scores: torch.Tensor, tau: float, iterations: int) -> torch.Tensor:
    log_plan = (scores - scores.amax(dim=1, keepdim=True)) / tau
    for _ in range(iterations):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=0, keepdim=True)
    return log_plan.exp()


def _crsi(query: torch.Tensor, paired_gallery: torch.Tensor) -> torch.Tensor:
    q = query - query.mean(dim=0, keepdim=True)
    g = paired_gallery - paired_gallery.mean(dim=0, keepdim=True)
    return 2 * (q - g).norm(dim=1) - q.norm(dim=1) - g.norm(dim=1)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _encode(encoder: nn.Module, inputs: Any) -> torch.Tensor:
    if isinstance(inputs, dict):
        result = encoder(**inputs)
    elif isinstance(inputs, tuple):
        result = encoder(*inputs)
    else:
        result = encoder(inputs)
    if not isinstance(result, torch.Tensor) or result.ndim != 2:
        raise ValueError("encoder must return a [B,D] tensor")
    return result


@torch.no_grad()
def _source_encoding(encoder: nn.Module, inputs: Any,
                     parameters: list[nn.Parameter], source: list[torch.Tensor]) -> torch.Tensor:
    current = [p.detach().clone() for p in parameters]
    try:
        for parameter, value in zip(parameters, source):
            parameter.copy_(value)
        return _encode(encoder, inputs).detach()
    finally:
        for parameter, value in zip(parameters, current):
            parameter.copy_(value)


def adapt_stream(
    encoder: nn.Module,
    batches: Iterable[tuple[torch.Tensor, Any]],
    gallery: torch.Tensor,
    num_queries: int,
    config: Config = Config(),
    adaptation_parameters: Iterable[nn.Parameter] | None = None,
) -> torch.Tensor:
    """Return [query, gallery] scores for a stream of (query_ids, encoder_inputs).

    The gallery must be encoded by the source model, ordered as desired by the
    caller, and kept fixed. Every query ID must occur exactly once. The encoder
    is adapted in place; construct a fresh source encoder for each new run.
    ``adaptation_parameters`` lets backbone integrations select exactly the
    query parameters specified by their original adaptation protocol.
    """
    config.validate()
    if gallery.ndim != 2 or not gallery.is_floating_point() or gallery.shape[0] < 1:
        raise ValueError("gallery must be a nonempty floating [G,D] tensor")
    if num_queries < 1:
        raise ValueError("num_queries must be positive")
    if not torch.isfinite(gallery).all():
        raise ValueError("gallery must contain finite embeddings")
    if any(p.device != gallery.device for p in encoder.parameters()):
        raise ValueError("encoder and gallery must be on the same device")
    parameters = _adapt_parameters(encoder, config.adapt, adaptation_parameters)
    source = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    matrix = gallery.new_empty((num_queries, gallery.shape[0]))
    seen = torch.zeros(num_queries, dtype=torch.bool)
    history: deque[torch.Tensor] = deque(maxlen=config.history_batches)
    rem: deque[torch.Tensor] = deque(maxlen=config.rem_capacity)

    for ids, inputs in batches:
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            raise ValueError("batch IDs must be nonempty")
        if (ids < 0).any() or (ids >= num_queries).any() or seen[ids].any() or ids.unique().numel() != ids.numel():
            raise ValueError("query IDs must be unique and within range")
        inputs = _to_device(inputs, gallery.device)
        with torch.no_grad():
            initial = _encode(encoder, inputs).detach()
            if initial.shape != (ids.numel(), gallery.shape[1]):
                raise ValueError("encoder output and gallery embedding dimensions differ")
            frozen = _source_encoding(encoder, inputs, parameters, source)
            raw = initial @ gallery.T
            if history:
                calibrated = _sinkhorn(torch.cat([*history, raw]), config.sinkhorn_tau,
                                        config.sinkhorn_iterations)[-ids.numel():]
            else:
                calibrated = raw
            selected = gallery[calibrated.argmax(dim=1)].detach()
            count = max(1, int(config.reliable_ratio * ids.numel()))
            reliable = torch.zeros(ids.numel(), dtype=torch.bool, device=gallery.device)
            reliable[_crsi(initial, selected).argsort()[:count]] = True
            first_entropy = _entropy(initial @ selected.T / config.temperature)
            rem.extend(first_entropy[reliable].unbind())

        for _ in range(config.steps):
            online = _encode(encoder, inputs)
            entropies = _entropy(online @ selected.T / config.temperature)
            threshold = torch.stack(tuple(rem)).max().detach()
            weights = (1 - entropies.detach() / (threshold + 1e-3)).clamp_min(0)
            eligible = reliable & (entropies <= threshold)
            entropy_loss = (entropies * weights)[eligible].mean() if eligible.any() else online.sum() * 0
            anchored = ~eligible
            anchor_loss = (1 - (F.normalize(online[anchored], dim=1) *
                                F.normalize(frozen[anchored], dim=1)).sum(dim=1)).mean() if anchored.any() else online.sum() * 0
            loss = config.entropy_weight * entropy_loss + config.anchor_weight * anchor_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite adaptation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError("non-finite adaptation gradient")
            optimizer.step()

        with torch.no_grad():
            scores = _encode(encoder, inputs) @ gallery.T
            matrix[ids.to(gallery.device)] = scores
            history.append(scores.detach().clone())
            seen[ids] = True

    if not seen.all():
        raise ValueError(f"missing {int((~seen).sum())} query IDs")
    if config.rerank == "sinkhorn":
        matrix = _sinkhorn(matrix.to(torch.float64), config.sinkhorn_tau,
                           config.sinkhorn_iterations).to(matrix.dtype)
    return matrix.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STAR-3D on extracted feature tensors")
    parser.add_argument("--query", type=Path, required=True, help="torch [Q,D] tensor")
    parser.add_argument("--gallery", type=Path, required=True, help="torch [G,D] tensor")
    parser.add_argument("--output", type=Path, required=True, help="output .npy score matrix")
    parser.add_argument("--config", type=Path, help="optional JSON Config fields")
    parser.add_argument("--checkpoint", type=Path, help="FeatureEncoder state_dict")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = Config(**json.loads(args.config.read_text())) if args.config else Config()
    query = torch.load(args.query, map_location="cpu", weights_only=True)
    gallery = torch.load(args.gallery, map_location="cpu", weights_only=True)
    if (not isinstance(query, torch.Tensor) or not isinstance(gallery, torch.Tensor)
            or query.ndim != 2 or gallery.ndim != 2 or not query.is_floating_point()
            or not gallery.is_floating_point() or query.shape[1] != gallery.shape[1]):
        raise ValueError("query and gallery must be floating [N,D] tensors with the same D")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    encoder = FeatureEncoder(query.shape[1]).to(device)
    if args.checkpoint:
        encoder.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    gallery = gallery.to(device)
    def batches():
        for start in range(0, query.shape[0], args.batch_size):
            stop = min(start + args.batch_size, query.shape[0])
            yield torch.arange(start, stop), query[start:stop]
    result = adapt_stream(encoder, batches(), gallery, query.shape[0], config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result.cpu().numpy())
    print(f"saved {result.shape[0]} x {result.shape[1]} scores to {args.output}")


if __name__ == "__main__":
    main()
