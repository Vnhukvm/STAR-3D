# STAR-3D

Minimal, dataset-independent runtime for the online adaptation method in
`RoMa/BERT/lib/tta_tempv3.py` (source snapshot: 2026-09-18). This repository
contains the adaptation algorithm and a small executable feature encoder.
It contains no datasets, trained weights, experiments, or other TTA methods.
Source SHA-256: `454b3e800d43061eeebcee25ac48a81279098eba760984b7fd70899c237dbe23`.

## What runs

`star3d.tta_tempv3.adapt_stream` takes a PyTorch query encoder, batches of
`(query_ids, encoder_inputs)`, and **fixed source-model gallery embeddings**.
Inputs may be a tensor, positional-argument tuple, or keyword-argument dict.
It performs Sinkhorn-assisted pseudo-pair selection, CRSI routing, online REM
entropy and frozen-source anchoring, then returns a retrieval score matrix.
`Config(rerank="sinkhorn")` applies the full-matrix Sinkhorn rerank after the
stream; use it for text-to-3D configurations. For 3D-to-text, use `none`.

The included `FeatureEncoder` and CLI run on extracted `[N,D]` tensors. This
is a runnable example of the method, **not** a replacement for the original
CrossOver or Mosaic3D backbone. To reproduce those experiments, load the
appropriate source checkpoint and pass its differentiable query encoder to
`adapt_stream`; encode the gallery with the same source model and keep it
fixed. By default, `adapt="norm"` selects LayerNorm and BatchNorm affine
parameters; `adapt="adapter"` selects modules named `adapter`,
`input_adapter`, or `_pre_l2_scale`. For exact backbone-specific selection,
pass `adaptation_parameters` explicitly. These are extension points rather
than copies of the large backbone training stacks.

## Install and run

```bash
python -m pip install -e .
python -m star3d.tta_tempv3 --query query.pt --gallery gallery.pt --output scores.npy
```

`query.pt` and `gallery.pt` are floating PyTorch tensors of shapes `[Q,D]`
and `[G,D]`. `--checkpoint` accepts a `FeatureEncoder.state_dict()`.
`--config` accepts a JSON object using `Config` field names. The encoder
starts from its initialization when no checkpoint is given, so scores from
that example are only a runtime check, not a trained-model result.

For a real encoder:

```python
from star3d import Config, adapt_stream

# encoder: checkpoint-loaded nn.Module mapping input batches to [B,D]
# batches: iterable of (unique query IDs, input batch) covering 0..Q-1
# gallery: source-model embeddings [G,D], fixed throughout adaptation
scores = adapt_stream(encoder, batches, gallery, num_queries=Q,
                      config=Config(adapt="norm", rerank="sinkhorn"),
                      adaptation_parameters=selected_query_parameters)
```

The encoder is updated in place. Start each evaluation condition with a fresh
checkpoint-loaded encoder. Put the encoder and gallery on the same target
device; the function moves tensor inputs to that device.

## Source and scope

The algorithm follows the CRSI, REM, source-anchor, and queued Sinkhorn path
of `tta_tempv3.py`. The original file is kept untouched. This package uses a
small direct encoder interface instead of the RoMa dataset loaders and the
large `tta_experiment.py` dispatch tree. It does not bundle raw point-cloud
or text backbones, so it cannot claim to reproduce pretrained CrossOver or
Mosaic3D scores without those checkpoints and encoder implementations.
