# Assignment 7: Byte-Structured Language Models

This submission extends
[Kronecker Embeddings: Byte-Level Structured Token Representations for
Parameter-Efficient Language Models](https://arxiv.org/abs/2605.29459)
with a controlled comparison of three language-model interfaces:

1. standard BPE tokens with a learned embedding and vocabulary output;
2. Kronecker byte-position input embeddings with a BPE output; and
3. a vocabulary-free model that predicts a parallel chunk of raw bytes.

The work includes the architecture changes, training implementations, small
language-model runs, generated samples, and validation comparisons.

## Artifacts

- [Rendered notebook with outputs](kronecker-implementation.html)
- [Markdown export with code and outputs](kronecker-implementation.md)
- [Runnable marimo notebook](kronecker-implementation.py)
- [Flat Python script](kronecker-implementation.script.py)
- [Original paper PDF](kronecker-embeddings-paper.pdf)
- [Paper on arXiv](https://arxiv.org/abs/2605.29459)

The rendered HTML is the easiest way to review the completed work without
rerunning training.

## Architecture Comparison

All three interfaces use the same causal Transformer body, learned
sentence-position embeddings, normalization, and training objective where
applicable.

| Model | Input representation | Prediction |
|---|---|---|
| Standard BPE | Learned BPE token embedding | BPE vocabulary softmax |
| Kronecker input | Byte-value by byte-position codec, then learned projection | BPE vocabulary softmax |
| Vocabulary-free byte model | Codec of UTF-8 byte chunks, then learned projection | Parallel byte and chunk-length distributions |

The Kronecker codec marks the coordinate `byte_value * P + byte_position`,
scales active positions by `1 / sqrt(length)`, and normalizes each code before
the learned projection.

The byte model predicts every byte position in the next chunk from the same
previous-chunk state. Its loss combines byte cross entropy with a chunk-length
cross entropy, then divides by the number of real target bytes.

## Training Results

Lower validation bits per raw byte is better.

### Tiny Shakespeare Pilot

Three seeds, 10 MB sampled training bytes per run, 3 layers, width 128.

| Model | Parameters | Mean validation bits/byte |
|---|---:|---:|
| Standard BPE | 1.13M | 2.244 |
| Kronecker input + BPE head | 1.52M | 2.238 |
| Parallel byte head | 1.39M | 3.878 |

### FineWeb-Edu Comparison

Three seeds, 20 MB distinct training documents, 60 MB sampled training bytes,
2 MB held out, 8 layers, width 384.

| Model | Parameters | Mean validation bits/byte |
|---|---:|---:|
| Standard BPE | 17.39M | 1.717 |
| Kronecker input + BPE head | 18.96M | 1.729 |
| Parallel byte head | 17.40M | 3.899 |

### Capacity and Data Scale-Up

The scale-up uses 100 MB distinct training documents and the same 2 MB
validation set. The BPE controls stop at 60 MB; the byte models continue to
300 MB. This scale-up uses seed 7.

| Model | Parameters | 60 MB seen | 300 MB seen |
|---|---:|---:|---:|
| Standard BPE | 17.39M | 1.699 | not run |
| Kronecker input + BPE head | 18.96M | 1.683 | not run |
| Parallel byte head | 17.40M | 3.883 | 3.678 |
| Parallel byte head, wider and deeper | 91.46M | 3.913 | **3.640** |

Kronecker input remains close to standard BPE in these runs. The
vocabulary-free byte model is substantially worse because it predicts a whole
chunk in parallel and cannot condition later bytes in that chunk on earlier
ones. Increasing capacity from 17.40M to 91.46M improves the 300 MB byte
result by only 0.038 bits/byte.

These are small, exploratory training runs. They do not reproduce the paper's
large-scale training regime, and the byte experiment changes segmentation as
well as the output head.

## Checkpoints

The final four scale-up models, enlarged-data tokenizer, model code, and
manifest were archived at OCI object key
`kronecker-comparison/2026-09-21/scaleup-final-4-models.zip`.

Archive SHA-256:
`d21622b8fb54db98052b52b0638cc9bcfffb34902191f400f8cafd16d48f283f`

The upload returned HTTP 200 with a matching MD5. The supplied endpoint did
not permit a read-back check. Optimizer states are not included.

## Run Locally

The training cells expect a CUDA GPU and download SmolLM FineWeb-Edu through
Hugging Face Datasets.

```bash
uv sync
uv run marimo edit kronecker-implementation.py
```

The recorded result tables can be read from the HTML or Markdown exports
without retraining.
