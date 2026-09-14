# From a Normal LLM to Kronecker Embeddings

An educational [marimo](https://marimo.io/) notebook that implements the same
small decoder-only language model with two input pathways:

1. a conventional learned token embedding table; and
2. the fixed byte-position codec plus learned projection proposed in
   [Kronecker Embeddings](https://arxiv.org/abs/2605.29459).

The notebook explains the notation, tensor shapes, Kronecker product, causal
transformer, and parameter trade-off with diagrams and executable PyTorch code.
It also includes a readable loop implementation of the codec alongside its
vectorized equivalent and verifies that their outputs agree.

The models are intentionally untrained. Their purpose is to isolate and explain
the architectural change, not to produce meaningful text.

## Run locally

Install dependencies and open the notebook:

```bash
uv sync
uv run marimo edit kronecker-implementation.py
```

The SVG diagrams used by the notebook are stored in [`assets/`](assets/).
