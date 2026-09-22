import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import torch
    import torch.nn as nn
    import marimo as mo
    import math

    return math, mo, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From positions to byte-structured language models

    We start with two small embedding experiments, then compare three language models on the same text. The result tables are saved observations; the training cells are recipes for new runs.

    ## Contents

    1. [Token and sequence positions](#token-and-sequence-positions) — why an embedding also needs a location in the sentence.
    2. [Byte-position Kronecker codec](#byte-position-kronecker-codec) — how a byte and its location inside a token become one coordinate.
    3. [Three model interfaces](#three-model-interfaces) — shared Transformer body, different inputs and prediction heads.
    4. [Tiny Shakespeare pilot](#tiny-shakespeare-pilot) — a quick learning check.
    5. [FineWeb-Edu comparison](#fineweb-edu-comparison) — a larger three-seed comparison.
    6. [Capacity and data scale-up](#capacity-and-data-scale-up) — test whether more data and parameters help the byte model.
    7. [Saved checkpoints](#saved-checkpoints) — where the trained weights were backed up.

    ## Token and sequence positions

    **Objective:** See how token identity and sentence position combine. A token embedding says *what* a token is; a positional vector says *where* it appears. The next cell preserves the sine/cosine formula, followed by your `RegularEmbedding` and calculation scratchpad.

    This sentence position is different from the byte position used later. The comparison models below use a **learned** sentence-position table; their Kronecker codec uses fixed byte positions inside a token or chunk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this work, we use sine and cosine functions of different frequencies:

    $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$

    $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

    where $pos$ is the position and $i$ is the dimension.  That is, each
    dimension of the positional encoding corresponds to a sinusoid.  The
    wavelengths form a geometric progression from $2\pi$ to $10000 \cdot
    2\pi$.  We chose this function because we hypothesized it would
    allow the model to easily learn to attend by relative positions,
    since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a
    linear function of $PE_{pos}$.

    In addition, we apply dropout to the sums of the embeddings and the
    positional encodings in both the encoder and decoder stacks.  For
    the base model, we use a rate of $P_{drop}=0.1$.
    """)
    return


@app.cell
def _(math, nn, torch):
    class RegularEmbedding(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, dropout=0.1):
            super().__init__()

            # d_model will always be a power of 2

            self.dropout = nn.Dropout(p=dropout)
            self.d_model = d_model
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        def forward(self, inp: torch.Tensor):
            """
                inp: (seq-len,) <- token inputs
            """
            seq_len = inp.shape[-1]

            # converting tokens into embeddings
            x = self.embed(inp)

            """
            idea is to create a matrix of pos which would look like
               [0, 1, 2, 3.. seq_len]

            and create it in the shape of (seq_len, 1)

            and have the second fractional matrix in the shape of (1, d_model)
            so that we can do a dot product of both to produce (seq_len, d_model)
            """

            pos = torch.arange(
                seq_len,
                dtype=x.dtype,
                device=x.device
            ).unsqueeze(1)

            """
            for computing fractional part
            fraction = 1 / 10,000 ^ (2i / d) fraction is in terms of 2i here
            fraction = 10000 ^ -(2i / d)
            ln(fraction) = -(2i/d) ln(10,000)
            fraction = e ^ -(2i * ln(10,000) / d)
            """
            ind = torch.arange(0, self.d_model, 2, dtype=x.dtype, device=x.device)
            frac_2i = torch.exp(                      # defining fraction in terms of 2i
                ind * -1 * math.log(10_000) / self.d_model
            )
            # since every 2i, 2i+1 will be the same
            frac = torch.zeros(self.d_model, dtype=x.dtype, device=x.device)
            frac[0::2] = frac_2i
            frac[1::2] = frac_2i
            frac = frac.unsqueeze(0)  # to convert to a shape of (1, d_model)

            # now we do a dot product
            positional_embed = pos @ frac # this will give the term inside sin/cos for each element
            # will be of shape (seq_len, d_model)

            # we do sin and cos on each of the elements
            positional_embed[:, 0::2] = torch.sin(positional_embed[:, 0::2])
            positional_embed[:, 1::2] = torch.cos(positional_embed[:, 1::2])

            # adding position_embed to the embeddings
            return self.dropout(x * math.sqrt(self.d_model) + positional_embed)


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Your original scratch derivation (kept as text because the last line is unfinished):

    ```python
    "\""
    fraction = 1 / 1e4 ^ (2i / d)
    fraction = 1e4 ^ -(2i / d)
    log(fraction) = (-2i /d * 4) ln(10)
    fraction(2i) = e ^ -(8i * ln(10) / d)
    fraction(i) = e ^ -(4i * ln(10) / d)
    "\""

    torch.exp(torch.arange(0, d_model)
    ```
    """)
    return


@app.cell
def _(math, torch):
    seq_len = 10
    """
    0 -> [0, 0, 0, 0, ..](d=16)
    ..
    4 -> [4, 4, 4, 4, ..](d=16)
    """
    pos = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
    print("pos part", pos.shape, pos)

    d_model = 16
    temp = torch.exp(
        torch.arange(0, d_model//2, dtype=float) * 8 * math.log(10) * -1 / d_model # used 8 instead of 4 because i only want even indices
    )
    print(temp.shape, temp)
    frac = torch.zeros(d_model, dtype=float)
    frac[0::2] = temp
    frac[1::2] = temp
    frac = frac.unsqueeze(0)
    print("fractional part", frac.shape, frac)

    matrix = pos @ frac

    matrix[:, 0::2] = torch.sin(matrix[:, 0::2])
    matrix[:, 1::2] = torch.cos(matrix[:, 1::2])

    print("matrix", matrix.shape, matrix)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Byte-position Kronecker codec

    **Objective:** Represent a short byte sequence without a learned vocabulary lookup. For each real byte, combine its value and its position *inside this token*: coordinate `byte * P + position`. The sparse `256P` vector then goes through a learned projection. Your draft class is kept below; its `forward` is still a work in progress.
    """)
    return


@app.cell
def _(nn, torch):
    class KroneckerV1Embedding(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, dropout=0.1):
            super().__init__()
            self.d_model = d_model

        def forward(self, inp: torch.Tensor):
            """
                inp: (seq-len,) <- token inputs
            """
            seq_len = inp.shape[-1]

        @staticmethod
        def kronecker_codec(
            byte_seq,  # shape: (batch, seq-len, max-bytes)
            lengths,  # shape: (batch, seq-len)
            char_dim=256,
            eps=1e-6,
        ):
            d_p = byte_seq.shape[-1]
            device = byte_seq.device

            batch_tokens = byte_seq.shape[0] * byte_seq.shape[1]
            flat_byte_seq = byte_seq.reshape(batch_tokens, d_p).long()
            flat_lengths = lengths.reshape(batch_tokens).to(device)

            positions = torch.arange(d_p, device=device)
            # (batch_tokens, 16) + (1, 16)
            # clever use of broadcasting
            indexes = flat_byte_seq * d_p + positions.unsqueeze(0)

            #clever use of broadcasting
            active = positions.unsqueeze(0) < flat_lengths.unsqueeze(1)

            codec = torch.zeros((batch_tokens, d_p * char_dim), device=device, dtype=torch.float32)
            source = torch.rsqrt(flat_lengths).unsqueeze(1) * active.float()
            codec.scatter_add_(1, indexes.to(int), source)

            mean = codec.mean(dim=-1, keepdim=True)
            std = codec.std(dim=-1, keepdim=True) + eps
            normalized = (codec - mean) / std
            return normalized.reshape(*byte_seq.shape[:-1], char_dim * d_p)


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    First, turn example words into padded byte rows and record their true lengths. The length tells us which zeros are padding; a zero-length row can also mean end of document.
    """)
    return


@app.cell
def _(torch):
    scratch_d_p = 8
    scratch_char_dim = 256
    scratch_seq_length = 4

    scratch_str_seq = [
        ["hello", "world", "from", "revant"],
        ["some", "other", "string"],
    ]

    scratch_empty_byte = 0

    scratch_byte_seq = torch.Tensor([
        (
            [
                (list(token.encode()) + [scratch_empty_byte] * scratch_d_p)[:scratch_d_p]
                for token in seq
            ] + [[scratch_empty_byte] * scratch_d_p] * scratch_seq_length
        )[:scratch_seq_length]
        for seq in scratch_str_seq
    ])

    scratch_lengths = torch.Tensor([
        ([len(token) for token in seq] + [0] * scratch_seq_length)[:scratch_seq_length]
        for seq in scratch_str_seq
    ])

    print(scratch_byte_seq.shape)
    print(scratch_byte_seq)
    print(scratch_lengths.shape)
    print(scratch_lengths)
    return scratch_byte_seq, scratch_char_dim, scratch_d_p, scratch_lengths


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Next, flatten the batch to one row per token. Broadcasting adds the same position row to every byte row, giving each real `(byte, position)` pair its coordinate. The following shape check shows what `unsqueeze(0)` and `unsqueeze(1)` change.
    """)
    return


@app.cell
def _(scratch_byte_seq, scratch_d_p, scratch_lengths, torch):
    scratch_batch_tokens = scratch_byte_seq.shape[0] * scratch_byte_seq.shape[1]
    scratch_flat_byte_seq = scratch_byte_seq.reshape(scratch_batch_tokens, scratch_d_p)
    scratch_flat_lengths = scratch_lengths.reshape(scratch_batch_tokens)

    scratch_positions = torch.arange(scratch_d_p)
    # (batch_tokens, d_p) + (1, d_p): broadcasting across tokens
    scratch_indexes = scratch_flat_byte_seq * scratch_d_p + scratch_positions.unsqueeze(0)
    scratch_indexes
    return (
        scratch_batch_tokens,
        scratch_flat_lengths,
        scratch_indexes,
        scratch_positions,
    )


@app.cell
def _(scratch_d_p, torch):
    scratch_positions1 = torch.arange(scratch_d_p)
    print(scratch_positions1.shape, scratch_positions1.unsqueeze(0).shape, scratch_positions1.unsqueeze(1).shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Compare each position with its token's length to build an **active mask**. A padded byte still has an index, but it must contribute zero to the codec.
    """)
    return


@app.cell
def _(scratch_flat_lengths, scratch_positions):
    # Positions have shape (1, d_p); lengths have shape (batch_tokens, 1).
    scratch_active = scratch_positions.unsqueeze(0) < scratch_flat_lengths.unsqueeze(1)
    scratch_active
    return (scratch_active,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Finally, `scatter_add_` puts each active contribution at its coordinate in the `256P` vector. This preserved scratch cell uses `sqrt(length)`; the class and experiment codec use `1/sqrt(length)`.
    """)
    return


@app.cell
def _(
    scratch_active,
    scratch_batch_tokens,
    scratch_char_dim,
    scratch_d_p,
    scratch_flat_lengths,
    scratch_indexes,
    torch,
):
    scratch_final = torch.zeros((scratch_batch_tokens, scratch_d_p * scratch_char_dim))
    scratch_source = torch.sqrt(scratch_flat_lengths).unsqueeze(1) * scratch_active
    scratch_final.scatter_add_(1, scratch_indexes.to(int), scratch_source)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reusing your handwritten class

    Your `KroneckerV1Embedding` preserves the core byte-position calculation, but its `forward` has not yet been completed. The comparison's `byte_codec_cmp` uses the same idea while supporting both token tables and batches of chunks, handling zero-length end markers, and using population standard deviation. That keeps the recorded experiments numerically consistent. To reuse your class in a new run, align those details and pass its codec output through the model's linear input projection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A tiny byte-index example

    **Objective:** Check the broadcast rule before using it in a model. With at most `P` bytes per token, each `(byte value, position)` pair owns one coordinate in a `256 × P` grid. Its flat index is `byte * P + position`. The short cell below prints indices and the mask that ignores padding.
    """)
    return


@app.cell
def _(torch):
    codec_demo_bytes = torch.tensor([[97, 98, 0], [99, 0, 0]], dtype=torch.uint8)
    codec_demo_lengths = torch.tensor([2, 1])
    codec_demo_positions = torch.arange(codec_demo_bytes.shape[-1])
    codec_demo_indices = codec_demo_bytes.long() * 3 + codec_demo_positions
    codec_demo_active = codec_demo_positions < codec_demo_lengths[:, None]
    print("coordinate indices:", codec_demo_indices.tolist())
    print("active positions:", codec_demo_active.tolist())
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo_cmp
    mo_cmp.md("""
    ## Three model interfaces

    **Objective:** Hold the causal Transformer body fixed and change how text enters and leaves it.

    | Model | Input | Next-step prediction |
    |---|---|---|
    | Standard | Learned BPE token embedding | BPE vocabulary softmax, tied to the input table |
    | Kronecker input | Fixed byte-position codec, then learned projection | BPE vocabulary softmax |
    | Vocabulary-free byte model | Fixed codec of UTF-8 chunks, then learned projection | 256 choices at each byte position, plus a length/end choice |

    The data cell first makes both BPE tokens and byte chunks from the same documents. The model cell then defines the shared attention body and the three input/output branches. The batch and loss cells shift inputs by one step and score predictions in **bits per raw byte**, so lower is better.

    The byte model predicts all bytes of the next chunk **at once**. It has no vocabulary-sized matrices, but an early byte of that new chunk cannot help predict a later byte. The result tables record completed runs; running a training cell starts a new run.
    """)
    return (mo_cmp,)


@app.cell
def _():

    def build_comparison_data(vocab_size=4096, chunk_bytes=12):
        """Build matching BPE and byte-chunk streams from Tiny Shakespeare."""
        import re
        from urllib.request import urlopen
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with urlopen(url, timeout=30) as response:
            text = response.read(2_000_000).decode("utf-8")
        # This corpus is overwhelmingly ASCII. Keeping ASCII guarantees that each
        # BPE token decodes independently to exactly the bytes used by the codec.
        text = text.encode("ascii", errors="ignore").decode("ascii")
        documents = [part + "\n\n" for part in text.split("\n\n") if part]
        split = int(len(documents) * 0.9)
        train_docs, val_docs = documents[:split], documents[split:]

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(train_docs, trainer=trainer)
        eos_id = tokenizer.token_to_id("<eos>")

        def bpe_stream(docs):
            ids = []
            for doc in docs:
                ids.extend(tokenizer.encode(doc, add_special_tokens=False).ids)
                ids.append(eos_id)
            return ids

        def chunk_stream(docs):
            chunks = []
            for doc in docs:
                # Whitespace stays attached to the following word; long segments
                # are split into consecutive chunks with no bytes discarded.
                for segment in re.findall(r"\s*\S+|\s+", doc):
                    raw = segment.encode("ascii")
                    chunks.extend(raw[i:i + chunk_bytes] for i in range(0, len(raw), chunk_bytes))
                chunks.append(b"")  # length zero is the end-of-document signal
            return chunks

        token_bytes = [
            b"<eos>" if i == eos_id else tokenizer.decode([i], skip_special_tokens=False).encode("ascii", errors="replace")
            for i in range(tokenizer.get_vocab_size())
        ]
        train_bpe = bpe_stream(train_docs)
        val_bpe = bpe_stream(val_docs)
        used_ids = set(train_bpe + val_bpe)
        assert all(
            tokenizer.decode([i], skip_special_tokens=False).isascii()
            for i in used_ids if i != eos_id
        )
        assert tokenizer.decode(train_bpe[:-1], skip_special_tokens=True) == "".join(train_docs)

        return {
            "url": url,
            "train_docs": train_docs,
            "val_docs": val_docs,
            "tokenizer": tokenizer,
            "eos_id": eos_id,
            "token_bytes": token_bytes,
            "train_bpe": train_bpe,
            "val_bpe": val_bpe,
            "train_chunks": chunk_stream(train_docs),
            "val_chunks": chunk_stream(val_docs),
            "chunk_bytes": chunk_bytes,
            "vocab_size": tokenizer.get_vocab_size(),
        }

    cmp_data = build_comparison_data()
    print(
        "Tiny Shakespeare:",
        len("".join(cmp_data["train_docs"]).encode("ascii")), "train bytes,",
        len("".join(cmp_data["val_docs"]).encode("ascii")), "validation bytes;",
        cmp_data["vocab_size"], "BPE tokens;"
        , cmp_data["chunk_bytes"], "bytes per chunk"
    )
    return (cmp_data,)


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### One Transformer, three ways in and out

    `byte_codec_cmp` marks one coordinate for each real `(byte, position)` pair, scales by `1/sqrt(length)`, then normalizes the vector. The Kronecker-input model precomputes codes for BPE tokens; the byte model makes codes from chunks on demand. `SmallLanguageModelCmp` uses the same attention blocks for all modes and selects the matching output head. Your handwritten draft remains above; this experiment codec handles zero-length end markers and uses population standard deviation.
    """)
    return


@app.cell
def _():
    import math as math_cmp
    import torch as torch_cmp
    from torch import nn as nn_cmp
    from torch.nn import functional as F_cmp


    def byte_codec_cmp(byte_values, lengths):
        """Map (..., P) byte values and (...) lengths to (..., 256*P) codes."""
        # Byte b at position p occupies coordinate b*P + p.
        p = byte_values.shape[-1]
        positions = torch_cmp.arange(p, device=byte_values.device)
        indices = byte_values.long() * p + positions
        active = positions < lengths[..., None]
        weights = active.float() * lengths.clamp_min(1).float().rsqrt()[..., None]

        codec = torch_cmp.zeros(
            *byte_values.shape[:-1], 256 * p, device=byte_values.device
        )
        codec.scatter_add_(-1, indices, weights)
        # The paper applies per-token z-normalization after the fixed codec.
        mean = codec.mean(dim=-1, keepdim=True)
        std = codec.std(dim=-1, keepdim=True, correction=0)
        return (codec - mean) / (std + 1e-6)


    def build_codec_table_cmp(token_bytes, chunk_bytes):
        """Precompute the fixed input code for every BPE token."""
        rows = torch_cmp.zeros(len(token_bytes), chunk_bytes, dtype=torch_cmp.uint8)
        lengths = torch_cmp.zeros(len(token_bytes), dtype=torch_cmp.long)
        for token_id, raw in enumerate(token_bytes):
            clipped = raw[:chunk_bytes]
            rows[token_id, :len(clipped)] = torch_cmp.tensor(list(clipped), dtype=torch_cmp.uint8)
            lengths[token_id] = len(clipped)
        return byte_codec_cmp(rows, lengths)


    class CausalBlockCmp(nn_cmp.Module):
        def __init__(self, width, heads):
            super().__init__()
            self.heads = heads
            self.norm1 = nn_cmp.LayerNorm(width)
            self.qkv = nn_cmp.Linear(width, 3 * width)
            self.attn_out = nn_cmp.Linear(width, width)
            self.norm2 = nn_cmp.LayerNorm(width)
            self.mlp = nn_cmp.Sequential(
                nn_cmp.Linear(width, 4 * width),
                nn_cmp.GELU(),
                nn_cmp.Linear(4 * width, width),
            )

        def forward(self, x):
            batch, time, width = x.shape
            q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
            shape = (batch, time, self.heads, width // self.heads)
            q, k, v = (item.reshape(shape).transpose(1, 2) for item in (q, k, v))
            attention = F_cmp.scaled_dot_product_attention(q, k, v, is_causal=True)
            attention = attention.transpose(1, 2).reshape(batch, time, width)
            x = x + self.attn_out(attention)
            return x + self.mlp(self.norm2(x))


    class SmallLanguageModelCmp(nn_cmp.Module):
        """One causal body with a selectable input representation and output head."""

        def __init__(
            self, mode, vocab_size, codec_table=None,
            width=128, layers=3, heads=4, context=64, chunk_bytes=12,
        ):
            super().__init__()
            assert mode in {"standard", "kronecker", "byte"}
            assert width % heads == 0
            self.mode = mode
            self.chunk_bytes = chunk_bytes
            self.context = context

            if mode == "standard":
                self.input = nn_cmp.Embedding(vocab_size, width)
            else:
                self.input = nn_cmp.Linear(256 * chunk_bytes, width, bias=False)
                if mode == "kronecker":
                    self.register_buffer("codec_table", codec_table)

            self.positions = nn_cmp.Embedding(context, width)
            self.blocks = nn_cmp.ModuleList(
                CausalBlockCmp(width, heads) for _ in range(layers)
            )
            self.final_norm = nn_cmp.LayerNorm(width)
            if mode == "standard":
                self.output = None  # reuse the input matrix for vocabulary logits
            elif mode == "kronecker":
                self.output = nn_cmp.Linear(width, vocab_size, bias=False)
            else:
                self.output = nn_cmp.Linear(width, 256 * chunk_bytes)
                self.length_output = nn_cmp.Linear(width, chunk_bytes + 1)

            self.apply(self._init_weights)
            if mode != "standard":
                # z-normalized codec has variance about one per coordinate.
                nn_cmp.init.normal_(
                    self.input.weight, std=0.02 / math_cmp.sqrt(256 * chunk_bytes)
                )

        @staticmethod
        def _init_weights(module):
            if isinstance(module, (nn_cmp.Linear, nn_cmp.Embedding)):
                nn_cmp.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn_cmp.Linear) and module.bias is not None:
                    nn_cmp.init.zeros_(module.bias)

        def forward(self, values, lengths=None):
            if self.mode == "standard":
                x = self.input(values)
            elif self.mode == "kronecker":
                x = self.input(F_cmp.embedding(values, self.codec_table))
            else:
                x = self.input(byte_codec_cmp(values, lengths))

            time = x.shape[1]
            x = x + self.positions(torch_cmp.arange(time, device=x.device))
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)

            if self.mode == "standard":
                return F_cmp.linear(x, self.input.weight), None
            if self.mode == "kronecker":
                return self.output(x), None

            byte_logits = self.output(x).reshape(
                *x.shape[:-1], 256, self.chunk_bytes
            )
            length_logits = self.length_output(x)
            return byte_logits, length_logits

    return (
        F_cmp,
        SmallLanguageModelCmp,
        build_codec_table_cmp,
        math_cmp,
        torch_cmp,
    )


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### Batches, training, and a common score

    **Objective:** Make the comparisons use the same training and validation rules. A window of positions `0..T-1` predicts positions `1..T`. For BPE, the loss scores the next token; for bytes, it scores active bytes of the next chunk plus its length (zero means end of document). Both losses are divided by the number of **raw target bytes** and reported as bits/byte.

    Random windows provide training batches. Nonoverlapping held-out windows provide validation batches. A shared `train_until_cmp` helper performs optimizer steps for the pilot and both larger experiments; each experiment chooses its own model size and byte budget.
    """)
    return


@app.cell
def _(
    F_cmp,
    SmallLanguageModelCmp,
    build_codec_table_cmp,
    cmp_data,
    math_cmp,
    torch_cmp,
):

    import time as time_cmp


    def prepare_streams_cmp(data, device):
        """Move BPE IDs, byte chunks, lengths, and the fixed codec table to device."""
        p = data["chunk_bytes"]
        token_bytes = data["token_bytes"]

        # Input codec may truncate very rare BPE tokens; target byte counts do not.
        codec_table = build_codec_table_cmp(token_bytes, p).to(device)
        target_byte_lengths = torch_cmp.tensor(
            [0 if i == data["eos_id"] else len(raw)
             for i, raw in enumerate(token_bytes)],
            device=device, dtype=torch_cmp.long,
        )

        def pack_chunks(chunks):
            raw = b"".join(chunk.ljust(p, b"\0") for chunk in chunks)
            values = torch_cmp.frombuffer(bytearray(raw), dtype=torch_cmp.uint8)
            values = values.reshape(-1, p).to(device)
            lengths = torch_cmp.tensor(
                [len(chunk) for chunk in chunks], device=device, dtype=torch_cmp.long
            )
            return values, lengths

        return {
            "codec_table": codec_table,
            "bpe_byte_lengths": target_byte_lengths,
            "train_bpe": torch_cmp.tensor(data["train_bpe"], device=device),
            "val_bpe": torch_cmp.tensor(data["val_bpe"], device=device),
            "train_byte": pack_chunks(data["train_chunks"]),
            "val_byte": pack_chunks(data["val_chunks"]),
        }


    def batch_cmp(streams, mode, split, starts, context):
        positions = starts[:, None] + torch_cmp.arange(
            context + 1, device=starts.device
        )
        if mode == "byte":
            values, lengths = streams[split + "_byte"]
            sampled = values[positions]
            sampled_lengths = lengths[positions]
            return sampled[:, :-1], sampled_lengths[:, :-1], sampled[:, 1:], sampled_lengths[:, 1:]

        sampled = streams[split + "_bpe"][positions]
        return sampled[:, :-1], None, sampled[:, 1:], None


    def loss_cmp(model, batch, streams):
        """Return summed predictive loss divided by target raw-byte count."""
        inputs, input_lengths, targets, target_lengths = batch
        logits, length_logits = model(inputs, input_lengths)

        if model.mode != "byte":
            nll = F_cmp.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            byte_count = streams["bpe_byte_lengths"][targets].sum()
            return nll / byte_count, byte_count

        p = model.chunk_bytes
        byte_nll = F_cmp.cross_entropy(
            logits.permute(0, 2, 1, 3), targets.long(), reduction="none"
        )
        active = torch_cmp.arange(p, device=targets.device) < target_lengths[..., None]
        length_nll = F_cmp.cross_entropy(
            length_logits.reshape(-1, p + 1),
            target_lengths.reshape(-1),
            reduction="sum",
        )
        byte_count = target_lengths.sum()
        nll = (byte_nll * active).sum() + length_nll
        return nll / byte_count, byte_count


    @torch_cmp.no_grad()
    def evaluate_cmp(model, streams, context=64, batch_size=32):
        """Evaluate nonoverlapping windows of the held-out stream."""
        model.eval()
        device = next(model.parameters()).device
        values = streams["val_byte"][0] if model.mode == "byte" else streams["val_bpe"]
        starts = torch_cmp.arange(
            0, len(values) - context - 1, context, device=device
        )
        total_nll = 0.0
        total_bytes = 0
        for group in starts.split(batch_size):
            batch = batch_cmp(streams, model.mode, "val", group, context)
            with torch_cmp.autocast("cuda", dtype=torch_cmp.bfloat16):
                nll_per_byte, byte_count = loss_cmp(model, batch, streams)
            count = int(byte_count)
            total_nll += float(nll_per_byte) * count
            total_bytes += count
        model.train()
        return total_nll / total_bytes, total_bytes


    def train_until_cmp(
        model, optimizer, streams, *, context, batch_size,
        target_bytes, bytes_seen=0, steps=0, capture_first_loss=False,
    ):
        """Take optimizer steps until the requested raw-byte exposure is reached."""
        mode = model.mode
        device = streams["codec_table"].device
        values = streams["train_byte"][0] if mode == "byte" else streams["train_bpe"]
        first_loss = None

        while bytes_seen < target_bytes:
            starts = torch_cmp.randint(
                0, len(values) - context - 1, (batch_size,), device=device
            )
            batch = batch_cmp(streams, mode, "train", starts, context)
            optimizer.zero_grad(set_to_none=True)
            with torch_cmp.autocast("cuda", dtype=torch_cmp.bfloat16):
                loss, batch_bytes = loss_cmp(model, batch, streams)
            loss.backward()
            torch_cmp.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if capture_first_loss and first_loss is None:
                first_loss = float(loss.detach())
            bytes_seen += int(batch_bytes)
            steps += 1

        return bytes_seen, steps, first_loss


    def train_one_cmp(
        mode, streams, vocab_size, chunk_bytes,
        byte_budget=2_000_000, context=64, batch_size=32, seed=7,
    ):
        torch_cmp.manual_seed(seed)
        torch_cmp.cuda.manual_seed_all(seed)
        device = streams["codec_table"].device
        model = SmallLanguageModelCmp(
            mode=mode, vocab_size=vocab_size,
            codec_table=streams["codec_table"] if mode == "kronecker" else None,
            chunk_bytes=chunk_bytes, context=context,
        ).to(device)
        optimizer = torch_cmp.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        parameter_count = sum(p.numel() for p in model.parameters())

        torch_cmp.cuda.synchronize()
        start_time = time_cmp.perf_counter()
        bytes_seen = 0
        steps = 0
        bytes_seen, steps, first_loss = train_until_cmp(
            model, optimizer, streams, context=context, batch_size=batch_size,
            target_bytes=byte_budget, capture_first_loss=True,
        )
        torch_cmp.cuda.synchronize()
        seconds = time_cmp.perf_counter() - start_time

        val_nats, val_bytes = evaluate_cmp(model, streams, context, batch_size)
        return {
            "mode": mode,
            "parameters": parameter_count,
            "steps": steps,
            "train_bytes": bytes_seen,
            "train_seconds": seconds,
            "train_bytes_per_second": bytes_seen / seconds,
            "first_train_bpb": first_loss / math_cmp.log(2),
            "val_bpb": val_nats / math_cmp.log(2),
            "val_bytes": val_bytes,
            "model": model,
        }


    cmp_streams = prepare_streams_cmp(cmp_data, "cuda")
    print("Prepared GPU data:", cmp_streams["train_bpe"].shape,
          cmp_streams["train_byte"][0].shape,
          cmp_streams["codec_table"].shape)
    return (
        cmp_streams,
        evaluate_cmp,
        prepare_streams_cmp,
        time_cmp,
        train_one_cmp,
        train_until_cmp,
    )


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ## Tiny Shakespeare pilot

    **Objective:** Quickly check that all three models can learn before using a larger corpus. Build short BPE and byte-chunk streams from Tiny Shakespeare, train three seeds to the same 10 MB raw-byte exposure, then inspect losses and a few generated samples. This is a sanity check, not a reproduction of the paper.
    """)
    return


@app.cell
def _(cmp_data, cmp_streams, train_one_cmp):

    import statistics as stats_cmp

    # A longer, three-seed pilot. The budget is raw bytes seen, not token count.
    cmp_repeats = []
    cmp_models = {}
    for _seed in (7, 17, 29):
        for _mode in ("standard", "kronecker", "byte"):
            _result = train_one_cmp(
                _mode, cmp_streams,
                vocab_size=cmp_data["vocab_size"],
                chunk_bytes=cmp_data["chunk_bytes"],
                byte_budget=10_000_000,
                context=64,
                batch_size=32,
                seed=_seed,
            )
            cmp_models[_mode, _seed] = _result.pop("model")
            _result["seed"] = _seed
            cmp_repeats.append(_result)
            print(
                f'seed={_seed:2d}  {_mode:10s}'
                f'  val={_result["val_bpb"]:.3f} bits/byte'
                f'  train={_result["train_seconds"]:.1f}s'
            )

    print("\nMean validation bits/byte (lower is better):")
    for _mode in ("standard", "kronecker", "byte"):
        _values = [r["val_bpb"] for r in cmp_repeats if r["mode"] == _mode]
        print(
            f'{_mode:10s}  {stats_cmp.mean(_values):.3f} ± '
            f'{stats_cmp.stdev(_values):.3f}  (3 seeds)'
        )
    return (cmp_models,)


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### Read a few generated samples

    Loss is the quantitative check. Sampling gives a quick, intuitive check for repetition, spelling, and whether the model stops. Both BPE modes choose a next token; the byte mode chooses a length and all bytes of the next chunk.
    """)
    return


@app.cell
def _(cmp_data, cmp_models, torch_cmp):

    def generate_cmp(model, prompt, data, max_steps=60):
        import re

        model.eval()
        device = next(model.parameters()).device
        p = data["chunk_bytes"]

        if model.mode != "byte":
            ids = data["tokenizer"].encode(prompt, add_special_tokens=False).ids
            for _ in range(max_steps):
                x = torch_cmp.tensor([ids[-model.context:]], device=device)
                with torch_cmp.inference_mode():
                    logits, _ = model(x)
                next_id = int(logits[0, -1].argmax())
                if next_id == data["eos_id"]:
                    break
                ids.append(next_id)
            return data["tokenizer"].decode(ids, skip_special_tokens=True)

        chunks = []
        for segment in re.findall(r"\s*\S+|\s+", prompt):
            raw = segment.encode("utf-8")
            chunks.extend(raw[i:i + p] for i in range(0, len(raw), p))

        for _ in range(max_steps):
            recent = chunks[-model.context:]
            x = torch_cmp.zeros(1, len(recent), p, device=device, dtype=torch_cmp.uint8)
            lengths = torch_cmp.tensor(
                [[len(chunk) for chunk in recent]], device=device
            )
            for j, chunk in enumerate(recent):
                x[0, j, :len(chunk)] = torch_cmp.tensor(
                    list(chunk), device=device, dtype=torch_cmp.uint8
                )
            with torch_cmp.inference_mode():
                byte_logits, length_logits = model(x, lengths)
            next_length = int(length_logits[0, -1].argmax())
            if next_length == 0:
                break
            next_bytes = byte_logits[0, -1].argmax(dim=0)[:next_length]
            chunks.append(bytes(next_bytes.tolist()))
        return b"".join(chunks).decode("utf-8", errors="replace")


    cmp_prompt = "First Citizen:\n"
    cmp_samples = {
        mode: generate_cmp(cmp_models[mode, 7], cmp_prompt, cmp_data)
        for mode in ("standard", "kronecker", "byte")
    }
    for _mode, _sample in cmp_samples.items():
        print(_mode, repr(_sample[:350]))
    return (generate_cmp,)


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### Tiny Shakespeare: recorded pilot

    Three seeds, 10 MB sampled training bytes per run, a 3-layer width-128 body,
    4,096 BPE tokens, and 12-byte chunks for the vocabulary-free model.
    The values below come from the completed run; reading them does not retrain it.

    | Model | Parameters | Mean validation bits/byte ↓ |
    |---|---:|---:|
    | Standard BPE | 1.13M | 2.244 |
    | Kronecker input + BPE head | 1.52M | 2.238 |
    | Parallel byte head | 1.39M | 3.878 |

    The first two are close at this scale. The byte model is much worse, but it
    also changes how text is segmented. This pilot only motivated the larger,
    more varied FineWeb-Edu experiment below.
    """)
    return


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ## FineWeb-Edu comparison

    **Objective:** Compare the same three designs on more varied text. We take disjoint training and validation documents from SmolLM FineWeb-Edu, train an 8,192-token BPE tokenizer on training documents only, and also make UTF-8 chunks from those documents. The next cells prepare both views on the GPU, train the three modes at matched raw-byte exposure across three seeds, and sample their outputs. The recorded table follows the code.
    """)
    return


@app.cell
def _():

    def build_large_data_cmp(train_limit=20_000_000, val_limit=2_000_000,
                             vocab_size=8192, chunk_bytes=16):
        """Stream disjoint documents, then build BPE and byte-chunk views."""
        import re
        from datasets import load_dataset
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

        # Hash IDs so training and validation use disjoint documents, even
        # though the source dataset only exposes a training split.
        import hashlib
        train_docs, val_docs = [], []
        train_bytes = val_bytes = 0
        rows = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "fineweb-edu-dedup",
            split="train",
            streaming=True,
        )
        for row in rows:
            document = row["text"].strip() + "\n\n"
            size = len(document.encode("utf-8"))
            bucket = int.from_bytes(
                hashlib.blake2b(row["id"].encode(), digest_size=8).digest(), "big"
            ) % 10
            if bucket == 0 and val_bytes + size <= val_limit:
                val_docs.append(document)
                val_bytes += size
            elif bucket != 0 and train_bytes + size <= train_limit:
                train_docs.append(document)
                train_bytes += size
            if train_bytes >= train_limit - 20_000 and val_bytes >= val_limit - 20_000:
                break

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(train_docs, trainer=trainer)
        eos_id = tokenizer.token_to_id("<eos>")

        # Invert ByteLevel's reversible byte-to-character alphabet. This handles
        # BPE pieces that contain only part of a UTF-8 character.
        printable = (list(range(ord("!"), ord("~") + 1))
                     + list(range(ord("¡"), ord("¬") + 1))
                     + list(range(ord("®"), ord("ÿ") + 1)))
        byte_values = printable.copy()
        unicode_values = printable.copy()
        for byte in range(256):
            if byte not in printable:
                byte_values.append(byte)
                unicode_values.append(256 + len(unicode_values) - len(printable))
        inverse_alphabet = {
            chr(codepoint): byte for byte, codepoint in zip(byte_values, unicode_values)
        }

        token_bytes = []
        for token_id in range(tokenizer.get_vocab_size()):
            if token_id in (tokenizer.token_to_id("<unk>"), eos_id):
                raw = tokenizer.id_to_token(token_id).encode("utf-8")
            else:
                raw = bytes(inverse_alphabet[character]
                            for character in tokenizer.id_to_token(token_id))
            token_bytes.append(raw)

        def bpe_stream(documents):
            result = []
            for document in documents:
                result.extend(tokenizer.encode(document, add_special_tokens=False).ids)
                result.append(eos_id)
            return result

        def byte_stream(documents):
            result = []
            for document in documents:
                for segment in re.findall(r"\s*\S+|\s+", document):
                    raw = segment.encode("utf-8")
                    result.extend(raw[i:i + chunk_bytes]
                                  for i in range(0, len(raw), chunk_bytes))
                result.append(b"")  # zero length means end of document
            return result

        train_bpe = bpe_stream(train_docs)
        val_bpe = bpe_stream(val_docs)
        train_chunks = byte_stream(train_docs)
        val_chunks = byte_stream(val_docs)
        assert b"".join(train_chunks) == "".join(train_docs).encode("utf-8")
        assert b"".join(val_chunks) == "".join(val_docs).encode("utf-8")

        return {
            "source": "HuggingFaceTB/smollm-corpus/fineweb-edu-dedup",
            "train_docs": train_docs,
            "val_docs": val_docs,
            "tokenizer": tokenizer,
            "eos_id": eos_id,
            "token_bytes": token_bytes,
            "train_bpe": train_bpe,
            "val_bpe": val_bpe,
            "train_chunks": train_chunks,
            "val_chunks": val_chunks,
            "chunk_bytes": chunk_bytes,
            "vocab_size": tokenizer.get_vocab_size(),
        }


    large_data_cmp = build_large_data_cmp()
    print(
        f"FineWeb-Edu: {len(large_data_cmp['train_docs'])} train documents, "
        f"{len(large_data_cmp['val_docs'])} validation documents, "
        f"{large_data_cmp['vocab_size']} BPE tokens, "
        f"{len(large_data_cmp['train_chunks'])} training byte chunks"
    )
    return build_large_data_cmp, large_data_cmp


@app.cell
def _(large_data_cmp, prepare_streams_cmp):

    large_streams_cmp = prepare_streams_cmp(large_data_cmp, "cuda")
    print(
        "GPU streams:",
        tuple(large_streams_cmp["train_bpe"].shape),
        tuple(large_streams_cmp["train_byte"][0].shape),
        tuple(large_streams_cmp["codec_table"].shape),
    )
    return (large_streams_cmp,)


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### Train at a matched byte budget

    The training function below evaluates at 10, 30, and 60 MB of sampled raw-byte exposure. The next two cells run seed 7, then seeds 17 and 29 with the same model settings; a final cell prints example completions.
    """)
    return


@app.cell
def _(
    SmallLanguageModelCmp,
    evaluate_cmp,
    math_cmp,
    time_cmp,
    torch_cmp,
    train_until_cmp,
):

    def train_scaled_cmp(mode, streams, data, seed=7,
                         checkpoints=(10_000_000, 30_000_000, 60_000_000)):
        """Train one larger model; record validation loss as raw bytes accumulate."""
        torch_cmp.manual_seed(seed)
        torch_cmp.cuda.manual_seed_all(seed)
        device = streams["codec_table"].device
        context, batch_size = 128, 16
        model = SmallLanguageModelCmp(
            mode=mode,
            vocab_size=data["vocab_size"],
            codec_table=streams["codec_table"] if mode == "kronecker" else None,
            width=384, layers=8, heads=6, context=context,
            chunk_bytes=data["chunk_bytes"],
        ).to(device)
        optimizer = torch_cmp.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.1
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())

        bytes_seen, steps, evaluation_seconds = 0, 0, 0.0
        trace = []
        torch_cmp.cuda.synchronize()
        start_time = time_cmp.perf_counter()

        for target_bytes in checkpoints:
            bytes_seen, steps, _ = train_until_cmp(
                model, optimizer, streams, context=context, batch_size=batch_size,
                target_bytes=target_bytes, bytes_seen=bytes_seen, steps=steps,
            )

            torch_cmp.cuda.synchronize()
            eval_start = time_cmp.perf_counter()
            val_nats, val_bytes = evaluate_cmp(
                model, streams, context=context, batch_size=batch_size
            )
            evaluation_seconds += time_cmp.perf_counter() - eval_start
            train_seconds = time_cmp.perf_counter() - start_time - evaluation_seconds
            point = {
                "bytes_seen": bytes_seen,
                "steps": steps,
                "val_bpb": val_nats / math_cmp.log(2),
                "val_bytes": val_bytes,
                "train_seconds": train_seconds,
            }
            trace.append(point)
            print(
                f'{mode:10s}  seen={bytes_seen / 1e6:.1f} MB'
                f'  steps={steps:5d}'
                f'  val={point["val_bpb"]:.3f} bits/byte'
                f'  train={train_seconds:.1f}s',
                flush=True,
            )

        return {
            "mode": mode,
            "seed": seed,
            "parameters": parameters,
            "trace": trace,
            "model": model,
        }


    return (train_scaled_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_standard_seed7_cmp = train_scaled_cmp(
        "standard", large_streams_cmp, large_data_cmp,
        seed=7, checkpoints=(10_000_000, 30_000_000, 60_000_000),
    )
    return (large_standard_seed7_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_kronecker_seed7_cmp = train_scaled_cmp(
        "kronecker", large_streams_cmp, large_data_cmp,
        seed=7, checkpoints=(10_000_000, 30_000_000, 60_000_000),
    )
    return (large_kronecker_seed7_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_byte_seed7_cmp = train_scaled_cmp(
        "byte", large_streams_cmp, large_data_cmp,
        seed=7, checkpoints=(10_000_000, 30_000_000, 60_000_000),
    )
    return (large_byte_seed7_cmp,)


@app.cell
def _(
    large_byte_seed7_cmp,
    large_kronecker_seed7_cmp,
    large_standard_seed7_cmp,
):
    large_results_cmp = {
        "standard": large_standard_seed7_cmp,
        "kronecker": large_kronecker_seed7_cmp,
        "byte": large_byte_seed7_cmp,
    }
    print("Seed 7: all three model runs available for sampling.")
    return (large_results_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_standard_seed17_cmp = train_scaled_cmp(
        "standard", large_streams_cmp, large_data_cmp,
        seed=17, checkpoints=(60_000_000,),
    )
    return (large_standard_seed17_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_kronecker_seed17_cmp = train_scaled_cmp(
        "kronecker", large_streams_cmp, large_data_cmp,
        seed=17, checkpoints=(60_000_000,),
    )
    return (large_kronecker_seed17_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_byte_seed17_cmp = train_scaled_cmp(
        "byte", large_streams_cmp, large_data_cmp,
        seed=17, checkpoints=(60_000_000,),
    )
    return (large_byte_seed17_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_standard_seed29_cmp = train_scaled_cmp(
        "standard", large_streams_cmp, large_data_cmp,
        seed=29, checkpoints=(60_000_000,),
    )
    return (large_standard_seed29_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_kronecker_seed29_cmp = train_scaled_cmp(
        "kronecker", large_streams_cmp, large_data_cmp,
        seed=29, checkpoints=(60_000_000,),
    )
    return (large_kronecker_seed29_cmp,)


@app.cell
def _(large_data_cmp, large_streams_cmp, train_scaled_cmp):
    large_byte_seed29_cmp = train_scaled_cmp(
        "byte", large_streams_cmp, large_data_cmp,
        seed=29, checkpoints=(60_000_000,),
    )
    return (large_byte_seed29_cmp,)


@app.cell
def _(
    large_byte_seed17_cmp,
    large_byte_seed29_cmp,
    large_kronecker_seed17_cmp,
    large_kronecker_seed29_cmp,
    large_results_cmp,
    large_standard_seed17_cmp,
    large_standard_seed29_cmp,
):
    large_repeat_cmp = [
        large_standard_seed17_cmp, large_kronecker_seed17_cmp, large_byte_seed17_cmp,
        large_standard_seed29_cmp, large_kronecker_seed29_cmp, large_byte_seed29_cmp,
    ]
    for _mode in ("standard", "kronecker", "byte"):
        _runs = [large_results_cmp[_mode]] + [
            _run for _run in large_repeat_cmp if _run["mode"] == _mode
        ]
        _values = [_run["trace"][-1]["val_bpb"] for _run in _runs]
        print(f"{_mode:10s} mean={sum(_values) / len(_values):.4f} bits/byte; "
              f"seeds 7/17/29={[round(_value, 4) for _value in _values]}")
    return


@app.cell
def _(generate_cmp, large_data_cmp, large_results_cmp):

    large_samples_cmp = {
        _mode: generate_cmp(
            large_results_cmp[_mode]["model"],
            "The history of science",
            large_data_cmp,
            max_steps=40,
        )
        for _mode in ("standard", "kronecker", "byte")
    }
    for _mode, _sample in large_samples_cmp.items():
        print(_mode, repr(_sample[:260]))
    return


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### FineWeb-Edu: recorded three-seed comparison

    The models share 20 MB of distinct training documents and 2 MB of held-out
    documents. Each sees 60 MB of sampled raw-byte training exposure; the body is
    8 layers wide 384 with a 128-position context. Lower bits/byte is better.

    | Model | Parameters | Mean validation bits/byte ↓ |
    |---|---:|---:|
    | Standard BPE | 17.39M | 1.717 |
    | Kronecker input + BPE head | 18.96M | 1.729 |
    | Parallel byte head | 17.40M | 3.899 |

    For seed 7, validation changed as training continued:

    | MB seen | Standard | Kronecker input | Byte head |
    |---:|---:|---:|---:|
    | 10 | 2.177 | 2.081 | 4.073 |
    | 30 | 1.870 | 1.835 | 3.975 |
    | 60 | 1.723 | 1.730 | 3.893 |

    The byte head predicts every position of the next 16-byte chunk from the
    same previous-chunk state. It cannot condition a later byte on an earlier
    byte of that new chunk. These are short, exploratory runs, not a reproduction
    of the paper; segmentation differs as well as the head.
    """)
    return


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ## Capacity and data scale-up

    **Objective:** Test whether the byte model's gap shrinks with more distinct text, more training exposure, and a larger Transformer. We grow training text from 20 MB to 100 MB while keeping the **same validation documents**. A 17M-parameter byte model and a 91M-parameter byte model are checked at matched byte budgets; the BPE-head models are rerun to 60 MB as references.

    The first cells rebuild the data streams. The shorter control runs use `train_scaleup_cmp`; the large byte run uses a session object so successive cells can advance the **same** model and optimizer without one long notebook execution.
    """)
    return


@app.cell
def _(build_large_data_cmp, large_data_cmp):
    # Expand distinct training text while preserving the original held-out sample.
    extended_data_cmp = build_large_data_cmp(
        train_limit=100_000_000, val_limit=2_000_000,
        vocab_size=8192, chunk_bytes=16,
    )
    assert extended_data_cmp["val_docs"] == large_data_cmp["val_docs"]
    print(
        "Extended FineWeb-Edu:", len(extended_data_cmp["train_docs"]), "train documents,",
        len(extended_data_cmp["train_chunks"]), "byte chunks; validation unchanged."
    )
    return (extended_data_cmp,)


@app.cell
def _(extended_data_cmp, prepare_streams_cmp):
    extended_streams_cmp = prepare_streams_cmp(extended_data_cmp, "cuda")
    print(
        "Extended GPU streams:",
        tuple(extended_streams_cmp["train_bpe"].shape),
        tuple(extended_streams_cmp["train_byte"][0].shape),
    )
    return (extended_streams_cmp,)


@app.cell
def _(
    SmallLanguageModelCmp,
    evaluate_cmp,
    math_cmp,
    time_cmp,
    torch_cmp,
    train_until_cmp,
):
    def train_scaleup_cmp(
        mode, streams, data, *, width, layers, heads,
        checkpoints=(60_000_000, 150_000_000, 300_000_000),
        seed=7, context=128, batch_size=16,
    ):
        """Compare model capacity on a shared corpus and held-out set."""
        from pathlib import Path
        from safetensors.torch import save_file

        torch_cmp.manual_seed(seed)
        torch_cmp.cuda.manual_seed_all(seed)
        device = streams["codec_table"].device
        model = SmallLanguageModelCmp(
            mode=mode, vocab_size=data["vocab_size"],
            codec_table=streams["codec_table"] if mode == "kronecker" else None,
            width=width, layers=layers, heads=heads,
            context=context, chunk_bytes=data["chunk_bytes"],
        ).to(device)
        optimizer = torch_cmp.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.1
        )
        output_dir = Path("/marimo/kronecker-scaleup-2026-09-21")
        output_dir.mkdir(parents=True, exist_ok=True)
        name = f"{mode}-w{width}-l{layers}-seed{seed}"
        trace = []
        bytes_seen = steps = 0
        evaluation_seconds = 0.0
        torch_cmp.cuda.synchronize()
        started = time_cmp.perf_counter()

        for target_bytes in checkpoints:
            bytes_seen, steps, _ = train_until_cmp(
                model, optimizer, streams, context=context, batch_size=batch_size,
                target_bytes=target_bytes, bytes_seen=bytes_seen, steps=steps,
            )

            torch_cmp.cuda.synchronize()
            eval_started = time_cmp.perf_counter()
            val_nats, val_bytes = evaluate_cmp(
                model, streams, context=context, batch_size=batch_size
            )
            evaluation_seconds += time_cmp.perf_counter() - eval_started
            point = {
                "bytes_seen": bytes_seen, "steps": steps,
                "val_bpb": val_nats / math_cmp.log(2), "val_bytes": val_bytes,
                "train_seconds": time_cmp.perf_counter() - started - evaluation_seconds,
            }
            trace.append(point)
            weights_path = output_dir / f"{name}-{target_bytes // 1_000_000}MB.safetensors"
            save_file(
                {key: value.detach().cpu().contiguous()
                 for key, value in model.state_dict().items()},
                str(weights_path),
            )
            print(
                f"{name}: {bytes_seen / 1e6:.0f} MB seen, "
                f"{point['val_bpb']:.3f} validation bits/byte, "
                f"{point['train_seconds']:.1f}s training",
                flush=True,
            )

        return {
            "mode": mode, "seed": seed, "parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "width": width, "layers": layers, "heads": heads,
            "trace": trace, "model": model,
        }

    return (train_scaleup_cmp,)


@app.cell
def _(
    SmallLanguageModelCmp,
    evaluate_cmp,
    math_cmp,
    time_cmp,
    torch_cmp,
    train_until_cmp,
):
    class ByteScaleupSessionCmp:
        """Train a byte model in short notebook steps, preserving optimizer state."""

        def __init__(self, streams, data, *, seed=7, width=768, layers=12, heads=12):
            from pathlib import Path

            torch_cmp.manual_seed(seed)
            torch_cmp.cuda.manual_seed_all(seed)
            self.streams = streams
            self.device = streams["codec_table"].device
            self.context = 128
            self.batch_size = 16
            self.model = SmallLanguageModelCmp(
                mode="byte", vocab_size=data["vocab_size"],
                width=width, layers=layers, heads=heads, context=self.context,
                chunk_bytes=data["chunk_bytes"],
            ).to(self.device)
            self.optimizer = torch_cmp.optim.AdamW(
                self.model.parameters(), lr=3e-4, weight_decay=0.1
            )
            self.output_dir = Path("/marimo/kronecker-scaleup-2026-09-21")
            self.checkpoint_name = f"byte-w{width}-l{layers}-seed{seed}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.bytes_seen = 0
            self.steps = 0
            self.train_seconds = 0.0
            self.trace = []

        def advance(self, target_bytes):
            from safetensors.torch import save_file

            self.model.train()
            torch_cmp.cuda.synchronize()
            started = time_cmp.perf_counter()
            self.bytes_seen, self.steps, _ = train_until_cmp(
                self.model, self.optimizer, self.streams,
                context=self.context, batch_size=self.batch_size,
                target_bytes=target_bytes, bytes_seen=self.bytes_seen,
                steps=self.steps,
            )

            torch_cmp.cuda.synchronize()
            self.train_seconds += time_cmp.perf_counter() - started
            val_nats, val_bytes = evaluate_cmp(
                self.model, self.streams,
                context=self.context, batch_size=self.batch_size,
            )
            point = {
                "bytes_seen": self.bytes_seen, "steps": self.steps,
                "val_bpb": val_nats / math_cmp.log(2),
                "val_bytes": val_bytes, "train_seconds": self.train_seconds,
            }
            self.trace.append(point)
            path = self.output_dir / f"{self.checkpoint_name}-{target_bytes // 1_000_000}MB.safetensors"
            save_file(
                {key: value.detach().cpu().contiguous()
                 for key, value in self.model.state_dict().items()},
                str(path),
            )
            print(
                f"{self.checkpoint_name}: {self.bytes_seen / 1e6:.0f} MB seen, "
                f"{point['val_bpb']:.3f} validation bits/byte, "
                f"{self.train_seconds:.1f}s training",
                flush=True,
            )
            return point


    return (ByteScaleupSessionCmp,)


@app.cell
def byte_control_init_cmp(
    ByteScaleupSessionCmp,
    extended_data_cmp,
    extended_streams_cmp,
):
    byte_control_session_cmp = ByteScaleupSessionCmp(
        extended_streams_cmp, extended_data_cmp,
        width=384, layers=8, heads=6,
    )
    byte_control_60_cmp = byte_control_session_cmp.advance(60_000_000)
    return byte_control_60_cmp, byte_control_session_cmp


@app.cell
def _(byte_control_60_cmp, byte_control_session_cmp):
    assert byte_control_60_cmp["bytes_seen"] >= 60_000_000
    byte_control_150_cmp = byte_control_session_cmp.advance(150_000_000)
    return (byte_control_150_cmp,)


@app.cell
def _(byte_control_150_cmp, byte_control_session_cmp):
    assert byte_control_150_cmp["bytes_seen"] >= 150_000_000
    byte_control_225_cmp = byte_control_session_cmp.advance(225_000_000)
    return (byte_control_225_cmp,)


@app.cell
def _(byte_control_225_cmp, byte_control_session_cmp):
    assert byte_control_225_cmp["bytes_seen"] >= 225_000_000
    byte_control_300_cmp = byte_control_session_cmp.advance(300_000_000)
    return (byte_control_300_cmp,)


@app.cell
def _(byte_control_300_cmp, byte_control_session_cmp):
    assert byte_control_300_cmp["bytes_seen"] >= 300_000_000
    byte_control_scaleup_cmp = {
        "mode": "byte",
        "seed": 7,
        "parameters": sum(p.numel() for p in byte_control_session_cmp.model.parameters()),
        "trace": byte_control_session_cmp.trace,
        "model": byte_control_session_cmp.model,
    }
    print("17M byte model complete:",
          [(round(p["bytes_seen"] / 1e6), round(p["val_bpb"], 4))
           for p in byte_control_scaleup_cmp["trace"]])
    return (byte_control_scaleup_cmp,)


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### Advance the large byte model in stages

    **Objective:** Keep one 91M model in memory while training to 60, 120, 150, 210, 270, and 300 MB seen. `ByteScaleupSessionCmp` owns the model and optimizer; each following cell calls `advance` with the next total byte budget, evaluates, and saves weights. Rerunning an `advance` cell may train further; the result table below can be read without running these cells.
    """)
    return


@app.cell
def _(ByteScaleupSessionCmp, extended_data_cmp, extended_streams_cmp):
    byte_big_session_cmp = ByteScaleupSessionCmp(extended_streams_cmp, extended_data_cmp)
    print(
        "Parameters:", sum(p.numel() for p in byte_big_session_cmp.model.parameters())
    )
    return (byte_big_session_cmp,)


@app.cell
def _(byte_big_session_cmp):
    byte_big_60_cmp = byte_big_session_cmp.advance(60_000_000)
    return (byte_big_60_cmp,)


@app.cell
def _(byte_big_60_cmp, byte_big_session_cmp):
    assert byte_big_60_cmp["bytes_seen"] >= 60_000_000
    byte_big_120_cmp = byte_big_session_cmp.advance(120_000_000)
    return (byte_big_120_cmp,)


@app.cell
def _(byte_big_120_cmp, byte_big_session_cmp):
    assert byte_big_120_cmp["bytes_seen"] >= 120_000_000
    byte_big_150_cmp = byte_big_session_cmp.advance(150_000_000)
    return (byte_big_150_cmp,)


@app.cell
def _(byte_big_150_cmp, byte_big_session_cmp):
    assert byte_big_150_cmp["bytes_seen"] >= 150_000_000
    byte_big_210_cmp = byte_big_session_cmp.advance(210_000_000)
    return (byte_big_210_cmp,)


@app.cell
def _(byte_big_210_cmp, byte_big_session_cmp):
    assert byte_big_210_cmp["bytes_seen"] >= 210_000_000
    byte_big_270_cmp = byte_big_session_cmp.advance(270_000_000)
    return (byte_big_270_cmp,)


@app.cell
def _(byte_big_270_cmp, byte_big_session_cmp):
    assert byte_big_270_cmp["bytes_seen"] >= 270_000_000
    byte_big_300_cmp = byte_big_session_cmp.advance(300_000_000)
    return


@app.cell
def _(extended_data_cmp, extended_streams_cmp, train_scaleup_cmp):
    standard_extended_cmp = train_scaleup_cmp(
        "standard", extended_streams_cmp, extended_data_cmp,
        width=384, layers=8, heads=6, checkpoints=(60_000_000,),
    )
    return (standard_extended_cmp,)


@app.cell
def _(extended_data_cmp, extended_streams_cmp, train_scaleup_cmp):
    kronecker_extended_cmp = train_scaleup_cmp(
        "kronecker", extended_streams_cmp, extended_data_cmp,
        width=384, layers=8, heads=6, checkpoints=(60_000_000,),
    )
    return (kronecker_extended_cmp,)


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### Does more data and capacity close the gap? Recorded result

    Training text grew to **100 MB distinct**; the held-out **2 MB documents are
    identical** to the earlier run. All rows below use seed 7, context 128,
    batch size 16, and the same validation bits-per-raw-byte calculation.

    | Model | Parameters | 60 MB seen ↓ | 300 MB seen ↓ |
    |---|---:|---:|---:|
    | Standard BPE | 17.39M | 1.699 | — |
    | Kronecker input + BPE head | 18.96M | 1.683 | — |
    | Parallel byte head | 17.40M | 3.883 | 3.678 |
    | Parallel byte head, wider/deeper | 91.46M | 3.913 | **3.640** |

    At matched 300 MB exposure, **5.3× parameters** improve the byte model by
    only **0.038 bits/byte**. The vocabulary-head models are already below 1.7
    after 60 MB. More capacity helps the byte model's context, but cannot remove
    its independence assumption within a new chunk. This scale-up has one seed;
    the BPE models were not run to 300 MB, and the changed segmentation means
    these numbers do not isolate the output head alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ### Qualitative check

    Generate from the same prompt with the four scale-up models. These snippets help explain what the loss numbers feel like in text, but the held-out bits/byte table above remains the comparison metric.
    """)
    return


@app.cell
def _(
    byte_big_session_cmp,
    byte_control_scaleup_cmp,
    extended_data_cmp,
    generate_cmp,
    kronecker_extended_cmp,
    standard_extended_cmp,
):
    scaleup_prompt_cmp = "The history of science"
    scaleup_samples_cmp = {
        "Standard BPE": generate_cmp(
            standard_extended_cmp["model"], scaleup_prompt_cmp, extended_data_cmp, max_steps=50
        ),
        "Kronecker + BPE": generate_cmp(
            kronecker_extended_cmp["model"], scaleup_prompt_cmp, extended_data_cmp, max_steps=50
        ),
        "Byte head, 17M": generate_cmp(
            byte_control_scaleup_cmp["model"], scaleup_prompt_cmp, extended_data_cmp, max_steps=50
        ),
        "Byte head, 91M": generate_cmp(
            byte_big_session_cmp.model, scaleup_prompt_cmp, extended_data_cmp, max_steps=50
        ),
    }
    for _label, _sample in scaleup_samples_cmp.items():
        print(_label, repr(_sample[:250]))
    return


@app.cell(hide_code=True)
def _(mo_cmp):
    mo_cmp.md("""
    ## Saved checkpoints

    **Objective:** Locate the final scale-up weights without rerunning training. The archive includes the four final models, enlarged-data tokenizer, model code, and a manifest. It was uploaded to OCI object key `kronecker-comparison/2026-09-21/scaleup-final-4-models.zip`.

    Archive SHA-256: `d21622b8fb54db98052b52b0638cc9bcfffb34902191f400f8cafd16d48f283f`. The upload returned HTTP 200 with a matching MD5. The supplied endpoint did not permit a read-back check; restoring it needs read-capable access. Optimizer states are not included.
    """)
    return


if __name__ == "__main__":
    app.run()
