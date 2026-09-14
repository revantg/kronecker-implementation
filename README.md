# Kronecker Embeddings - Paper Implementation

This repository implements the input-embedding method proposed in
**“Kronecker Embeddings: Byte-Level Structured Token Representations for
Parameter-Efficient Language Models”** by Rohan Shravan.

- [Paper on arXiv](https://arxiv.org/abs/2605.29459)
- [Paper HTML](https://arxiv.org/html/2605.29459v1)
- [Bundled paper PDF](kronecker-embeddings-paper.pdf)

The implementation is presented as an executable
[marimo](https://marimo.io/) notebook. It constructs the same small
decoder-only transformer with two interchangeable input pathways:

1. a conventional learned token embedding table; and
2. the paper's deterministic byte-position Kronecker codec followed by a
   learned projection.

## What is implemented

- GPT-2 token IDs mapped to fixed-width UTF-8 byte buffers.
- The paper's byte-value × byte-position codec from Equation 1.
- Length normalization and per-token z-normalization.
- A learned `codec_dim -> d_model` projection replacing `nn.Embedding`.
- A readable loop implementation and a vectorized `scatter_add_` version.
- Numerical verification that both codec implementations agree.
- A controlled standard-vs.-Kronecker comparison using the same transformer
  body, positional embeddings, normalization, and output head.
- Explanations of the mathematics, tensor shapes, parameter counts, and model
  data flow, supported by SVG diagrams.

## Scope

This is a focused implementation and teaching companion for the paper's core
embedding method. The included tiny models are intentionally untrained, so the
repository does not reproduce the paper's multi-billion-token training runs or
benchmark claims. It isolates the architectural change and demonstrates that
both embedding pathways execute end to end.

## Run locally

Install dependencies and open the notebook:

```bash
uv sync
uv run marimo edit kronecker-implementation.py
```

The SVG diagrams used by the notebook are stored in [`assets/`](assets/).
