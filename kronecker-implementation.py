import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import html
    import math
    import time
    from dataclasses import dataclass

    import marimo as mo
    import tiktoken
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return F, dataclass, html, math, mo, nn, tiktoken, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From a normal LLM to Kronecker Embeddings

    This notebook builds the **same tiny decoder-only language model twice**:

    1. a normal model using a learned token lookup table, and
    2. a Kronecker model using a fixed byte-position codec followed by one learned projection.

    The tokenizer, positional embeddings, attention blocks, MLPs, normalization, and output head are the same. Only the **input token representation** changes.

    > The models here are intentionally untrained. Their predictions are random; the useful comparison is the data flow, shapes, geometry, gradients, parameter counts, and runtime.
    """)
    return


@app.cell(hide_code=True)
def _(html, mo):
    def html_table(headers, rows):
        header_html = "".join(
            f"<th>{html.escape(str(h))}</th>" for h in headers
        )
        row_html = "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
            + "</tr>"
            for row in rows
        )
        return mo.Html(f"""
        <div style="overflow:auto;border:1px solid #263449;border-radius:14px;background:#0b1220;">
          <table style="border-collapse:collapse;width:100%;font:13px ui-monospace,monospace;color:#dbeafe;">
            <thead><tr style="background:#111c30;color:#93c5fd;">{header_html}</tr></thead>
            <tbody>{row_html}</tbody>
          </table>
          <style>th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #1e2b40;}}</style>
        </div>""")

    def info_card(title, body, accent="#38bdf8"):
        return mo.Html(f"""
        <div style="border:1px solid #263449;border-left:5px solid {accent};border-radius:14px;
                    padding:14px 16px;background:#0b1220;color:#dbeafe;font:14px/1.55 system-ui;">
          <div style="font-weight:750;color:{accent};margin-bottom:4px;">{html.escape(title)}</div>
          <div>{body}</div>
        </div>""")

    return html_table, info_card


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · What a normal autoregressive LLM does

    An LLM repeatedly solves one task: **given all previous tokens, assign a probability to the next token**.

    ### Symbols used throughout

    | Symbol | Meaning |
    |---|---|
    | $B$ | batch size: number of sequences processed together |
    | $T$ | sequence length: number of tokens in each sequence |
    | $V$ | vocabulary size: number of possible token IDs |
    | $d_{model}$ | width of every vector flowing through the transformer |
    | $x \in \mathbb{N}^{B\times T}$ | integer token-ID tensor |
    | $E \in \mathbb{R}^{V\times d_{model}}$ | normal learned embedding table |
    | $D$ | Kronecker codec width, $D=d_c d_p$ |
    | $d_c$ | number of byte values, always 256 |
    | $d_p$ | maximum byte positions kept from one token |

    $\mathbb{N}$ means integers. $\mathbb{R}$ means real-valued numbers. A shape such as $B\times T\times d_{model}$ means “one $d_{model}$-dimensional vector for every token in every sequence.”
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "assets/llm-pipeline.svg",
        alt="Decoder-only LLM pipeline from tokenization through next-token sampling",
        width="100%",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The usual embedding: learned lookup by identity

    A tokenizer turns text into IDs. The normal embedding layer owns a matrix

    $$E\in\mathbb{R}^{V\times d_{model}}.$$

    For token ID $i$, lookup returns row $E[i]$. Initially the rows are random. During training, gradient descent moves the rows whenever their token IDs occur.

    $$e_i = E[i]$$

    - $i$: one integer token ID.
    - $E[i]$: row $i$ of the table.
    - $e_i\in\mathbb{R}^{d_{model}}$: the vector sent into the transformer.

    The lookup knows only identity: before training, `run` and `runs` have no built-in relationship. Any useful geometry has to be learned from data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "assets/normal-embedding.svg",
        alt="A token ID selecting one learned row from the normal embedding matrix",
        width="100%",
    )
    return


@app.cell
def _(html_table, tiktoken):
    BYTE_VALUES = 256
    MAX_TOKEN_BYTES = 16
    CODEC_DIM = BYTE_VALUES * MAX_TOKEN_BYTES
    MODEL_WIDTH = 64
    CONTEXT_LENGTH = 48
    N_HEADS = 4
    N_LAYERS = 2
    MODEL_SEED = 7

    encoding = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = encoding.n_vocab

    config_summary = {
        "vocabulary V": VOCAB_SIZE,
        "model width d_model": MODEL_WIDTH,
        "byte values d_c": BYTE_VALUES,
        "positions d_p": MAX_TOKEN_BYTES,
        "codec width D": CODEC_DIM,
    }
    html_table(["quantity", "value"], config_summary.items())
    return (
        BYTE_VALUES,
        CODEC_DIM,
        CONTEXT_LENGTH,
        MAX_TOKEN_BYTES,
        MODEL_SEED,
        MODEL_WIDTH,
        N_HEADS,
        N_LAYERS,
        VOCAB_SIZE,
        encoding,
    )


@app.cell(hide_code=True)
def _(mo):
    prompt_input = mo.ui.text(
        value="The model learns from tokens.",
        label="Try a prompt",
        full_width=True,
    )
    prompt_input
    return (prompt_input,)


@app.cell
def _(CONTEXT_LENGTH, encoding, html_table, mo, prompt_input, torch):
    prompt_text = prompt_input.value
    prompt_token_ids = encoding.encode(prompt_text)
    print("tokenizer output:", prompt_token_ids)
    prompt_token_rows = [
        (
            position,
            token_id,
            repr(encoding.decode([token_id])),
            list(encoding.decode_single_token_bytes(token_id)),
        )
        for position, token_id in enumerate(prompt_token_ids)
    ]
    print(f"{prompt_token_rows=}")
    input_ids_tensor = torch.tensor(
        [prompt_token_ids[:CONTEXT_LENGTH]], dtype=torch.long
    )

    mo.vstack(
        [
            mo.md(
                f"**Tokenized shape:** `{tuple(input_ids_tensor.shape)}` — one batch containing {input_ids_tensor.shape[1]} tokens."
            ),
            html_table(
                ["position", "token ID", "decoded piece", "UTF-8 bytes"],
                prompt_token_rows,
            ),
        ]
    )
    return (input_ids_tensor,)


@app.cell
def _(torch):
    def build_gpt2_byte_buffers(tokenizer, max_bytes):
        byte_buffer = torch.zeros(
            (tokenizer.n_vocab, max_bytes), dtype=torch.uint8
        )
        print(f"shape of byte_buffer={byte_buffer.shape}")
        length_buffer = torch.zeros(tokenizer.n_vocab, dtype=torch.int16)
        for token_id in range(tokenizer.n_vocab):
            raw = tokenizer.decode_single_token_bytes(token_id)
            clipped = raw[:max_bytes]
            length_buffer[token_id] = len(clipped)
            if clipped:
                byte_buffer[token_id, : len(clipped)] = torch.tensor(
                    list(clipped), dtype=torch.uint8
                )
        return byte_buffer, length_buffer

    return (build_gpt2_byte_buffers,)


@app.cell
def _(encoding):
    encoding.decode_single_token_bytes(50000)[:16]
    return


@app.cell
def _(MAX_TOKEN_BYTES, build_gpt2_byte_buffers, encoding, info_card):
    gpt2_byte_buffer, gpt2_length_buffer = build_gpt2_byte_buffers(
        encoding, MAX_TOKEN_BYTES
    )
    buffer_megabytes = (
        gpt2_byte_buffer.numel() * gpt2_byte_buffer.element_size()
        + gpt2_length_buffer.numel() * gpt2_length_buffer.element_size()
    ) / 1_000_000

    info_card(
        "Fixed tokenizer buffers created",
        f"byte_buffer shape = {tuple(gpt2_byte_buffer.shape)}, length_buffer shape = {tuple(gpt2_length_buffer.shape)}. "
        f"Together they occupy about <b>{buffer_megabytes:.2f} MB</b> and receive no gradients.",
        "#22d3ee",
    )
    return gpt2_byte_buffer, gpt2_length_buffer


@app.cell
def _(gpt2_byte_buffer):
    bytes(gpt2_byte_buffer[500].tolist()).decode("utf-8")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The paper's replacement: byte × position features

    Let a token contain UTF-8 bytes $b=(b_1,\ldots,b_L)$.

    $$
    \kappa(b)=\frac{1}{\sqrt L}\sum_{p=1}^{L} c_{b_p}\otimes p_p
    $$

    Read every symbol as follows:

    - $\kappa$ (“kappa”) is the deterministic codec function.
    - $b_p$ is the numerical byte value at position $p$.
    - $L$ is the number of retained bytes in the token.
    - $c_{b_p}\in\mathbb{R}^{256}$ is a one-hot vector selecting byte value $b_p$.
    - $p_p\in\mathbb{R}^{d_p}$ is a one-hot vector selecting position $p$.
    - $\otimes$ is the **Kronecker product**. For two one-hot vectors it simply selects one cell in a byte-by-position grid.
    - $\sum$ combines the selected cells for all bytes.
    - $1/\sqrt L$ prevents longer tokens from automatically having a larger vector norm.

    The flattened grid has width

    $$D=d_c d_p=256\times16=4096.$$

    After per-token z-normalization, the only learned input-side operation is

    $$e_i=\kappa(b_i)W_{proj},\qquad W_{proj}\in\mathbb{R}^{D\times d_{model}}.$$

    The projection converts the fixed 4096-dimensional byte-position description into the same $d_{model}$-wide vector expected by an ordinary transformer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "assets/kronecker-byte-position.svg",
        alt="The bytes in the word token activating coordinates in a byte-by-position grid",
        width="100%",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the Kronecker product means here

    A Kronecker product combines every entry of one vector with every entry of another. For two short vectors,

    \[
    [a,b]\otimes[c,d]=[ac,ad,bc,bd].
    \]

    The paper uses two **one-hot** vectors for every byte at position $p$:

    - $\mathbf{c}_{b_p}\in\mathbb{R}^{256}$ has a 1 at the byte value $b_p$.
    - $\mathbf{p}_p\in\mathbb{R}^{d_p}$ has a 1 at the byte position $p$.

    Their product $\mathbf{c}_{b_p}\otimes\mathbf{p}_p$ has length $256d_p$ and contains exactly one 1. It identifies one pair: **this byte occurred at this position**.

    For `"cat"`, using zero-based code positions and $d_p=16$:

    ```text
    byte "c" =  99 at position 0  →  c₉₉  ⊗ p₀  → index  99×16 + 0 = 1584
    byte "a" =  97 at position 1  →  c₉₇  ⊗ p₁  → index  97×16 + 1 = 1553
    byte "t" = 116 at position 2  →  c₁₁₆ ⊗ p₂  → index 116×16 + 2 = 1858
    ```

    You can picture the Kronecker output as a $256\times16$ byte-position grid:

    ```text
                     position
                   0  1  2  3 ... 15
    byte 97  (a)   0  1  0  0 ...  0
    byte 99  (c)   1  0  0  0 ...  0
    byte 116 (t)   0  0  1  0 ...  0
    other bytes    0  0  0  0 ...  0
    ```

    The token codec sums these three one-hot products and gives each active coordinate the value $1/\sqrt{3}$.

    The vectorized implementation does not call `torch.kron`: `flat_indices = byte * pos_dim + position` calculates where each Kronecker-product 1 belongs, and `scatter_add_` writes all those values for the complete batch at once.
    """)
    return


@app.cell
def _(
    BYTE_VALUES,
    MAX_TOKEN_BYTES,
    gpt2_byte_buffer,
    gpt2_length_buffer,
    input_ids_tensor,
    torch,
):
    def kronecker_codec(
        byte_sequences,  # shape: (batch, num-tokens, max-bytes)
        lengths,  # shape: (batch, num-tokens)
        char_dim=256,
        eps=1e-6,
    ):
        pos_dim = byte_sequences.shape[-1]

        # preserves the original (batch, num-tokens) from (batch, num-tokens, pos-dim)
        original_shape = byte_sequences.shape[:-1]

        # remove batch dimension and flatten
        #   converted (batch, num-tokens, max-bytes) --to--> (batch * num-tokens, max_bytes)
        flat_bytes = byte_sequences.reshape(-1, pos_dim).long()

        # similarly remove the dimension for flat_lengths
        #   converted (batch, num-tokens) to (batch * num_tokens, )
        flat_lengths = lengths.reshape(-1).long().to(byte_sequences.device)

        # batch_size * num-tokens
        batch_tokens = flat_bytes.shape[0]
        # max-bytes (acc to paper) * 256
        codec_dim = char_dim * pos_dim

        """
            For pos_dim = 4, produces:
            tensor([0, 1, 2, 3])  # shape (4,)
            Then:
            .expand(batch_tokens, -1)
            repeats that row for every token. If batch_tokens = 3:
            positions = tensor([
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ])
        """
        positions = torch.arange(pos_dim, device=byte_sequences.device).expand(
            batch_tokens, -1
        )  # shape: batch_tokens (batch-size * num-of-tokens), pos_dim

        # positions:    (batch * no-of-tokens, pos_dim)
        # flat_lengths: (batch * no-of-tokens, 1)
        # creates a mask for each sequence:
        #    for each row: all the elements will be true till their length
        #                  and all the empty places would be filled with false
        active = positions < flat_lengths.unsqueeze(1)

        # flat_bytes: (batch * num-of-tokens, pos-dim)
        # pos_dim = 16 (for eg)
        # positions:  (batch * num-of-tokens, pos-dim)
        # computes the indexes where we have to put one
        flat_indices = flat_bytes * pos_dim + positions

        scales = torch.rsqrt(flat_lengths.clamp_min(1).float()).unsqueeze(1)
        source = active.float() * scales
        codec = torch.zeros(
            batch_tokens, codec_dim, device=byte_sequences.device
        )

        # writes values from source into positions specified by flat_indices
        codec.scatter_add_(1, flat_indices, source)

        mean = codec.mean(dim=-1, keepdim=True)
        std = codec.std(dim=-1, keepdim=True) + eps
        normalized = (codec - mean) / std
        return normalized.reshape(*original_shape, codec_dim)

    # input_ids_tensor.shape == (batch_size, no_of_tokens)
    byte_sequences = gpt2_byte_buffer[
        input_ids_tensor
    ]  # shape: batch, num_tokens, dp (max bytes -> pos_dim)
    lengths = gpt2_length_buffer[input_ids_tensor]  # shape: batch, num_tokens
    char_dim = BYTE_VALUES
    pos_dim = MAX_TOKEN_BYTES

    (
        f"{input_ids_tensor.shape=}",
        f"{byte_sequences.shape=}",
        f"{lengths.shape=}",
    )
    return byte_sequences, kronecker_codec, lengths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A simple loop version for comparison

    This version implements the same codec literally: visit every token, visit every real byte, calculate its flattened `(byte, position)` coordinate, and write `1 / sqrt(length)` there. It is easier to read; the vectorized version is better suited to large GPU batches.
    """)
    return


@app.cell
def _(torch):
    def simple_kronecker_codec(
        byte_sequences,  # (batch, num_tokens, max_bytes)
        lengths,  # (batch, num_tokens)
        char_dim=256,
        eps=1e-6,
    ):
        batch_size, num_tokens, pos_dim = byte_sequences.shape
        codec_dim = char_dim * pos_dim

        # Start with one empty codec vector for every token.
        codec = torch.zeros(
            batch_size,
            num_tokens,
            codec_dim,
            device=byte_sequences.device,
        )

        for batch_index in range(batch_size):
            for token_index in range(num_tokens):
                token_length = int(lengths[batch_index, token_index].item())

                # Equation (1) places 1 / sqrt(L) at each real byte-position pair.
                scale = 1 / max(token_length, 1) ** 0.5

                for position in range(min(token_length, pos_dim)):
                    byte_value = int(
                        byte_sequences[
                            batch_index, token_index, position
                        ].item()
                    )

                    # Flatten grid coordinate (byte_value, position) into one index.
                    flat_index = byte_value * pos_dim + position
                    codec[batch_index, token_index, flat_index] = scale

        # Apply the same per-token z-normalization as the vectorized function.
        mean = codec.mean(dim=-1, keepdim=True)
        std = codec.std(dim=-1, keepdim=True) + eps
        return (codec - mean) / std

    return (simple_kronecker_codec,)


@app.cell
def _(byte_sequences, kronecker_codec, lengths, mo, simple_kronecker_codec):
    _simple_output = simple_kronecker_codec(byte_sequences, lengths)
    _vectorized_output = kronecker_codec(byte_sequences, lengths)
    _max_difference = (_simple_output - _vectorized_output).abs().max().item()

    mo.md(
        f"""
    **Comparison on the current prompt**

    - Simple output shape: `{tuple(_simple_output.shape)}`
    - Vectorized output shape: `{tuple(_vectorized_output.shape)}`
    - Largest element-wise difference: `{_max_difference:.2e}`

    A difference of zero (or tiny floating-point noise) confirms that both implementations construct the same codec.
    """
    )
    return


@app.cell
def _(kronecker_codec, math, nn):
    class KroneckerEmbedding(nn.Module):
        def __init__(
            self, byte_buffer, length_buffer, d_model, char_dim=256, pos_dim=16
        ):
            super().__init__()
            self.char_dim = char_dim
            self.pos_dim = pos_dim
            self.codec_dim = char_dim * pos_dim
            self.num_embeddings = byte_buffer.shape[0]
            self.embedding_dim = d_model
            self.register_buffer("byte_buffer", byte_buffer, persistent=False)
            self.register_buffer(
                "length_buffer", length_buffer, persistent=False
            )
            self.projection = nn.Linear(self.codec_dim, d_model, bias=False)
            nn.init.normal_(
                self.projection.weight,
                mean=0.0,
                std=1 / math.sqrt(self.codec_dim),
            )

        def forward(self, input_ids):
            token_bytes = self.byte_buffer[input_ids]
            token_lengths = self.length_buffer[input_ids]
            fixed_features = kronecker_codec(
                token_bytes,
                token_lengths,
                char_dim=self.char_dim,
            )
            return self.projection(
                fixed_features.to(self.projection.weight.dtype)
            )

    return (KroneckerEmbedding,)


@app.cell
def _(
    BYTE_VALUES,
    CODEC_DIM,
    gpt2_byte_buffer,
    gpt2_length_buffer,
    html_table,
    input_ids_tensor,
    kronecker_codec,
):
    print(f"raw input tokenized using gpt2 tokenizer: {input_ids_tensor}")

    print(
        f"length of bytes corresponding to each token: {gpt2_length_buffer[input_ids_tensor]}"
    )
    print(f"tensor for each token:", gpt2_byte_buffer[input_ids_tensor])

    sample_codec = kronecker_codec(
        gpt2_byte_buffer[input_ids_tensor],
        gpt2_length_buffer[input_ids_tensor],
        char_dim=BYTE_VALUES,
    )
    codec_check_rows = [
        (
            "codec shape",
            tuple(sample_codec.shape),
            f"[B, T, D] = [1, {input_ids_tensor.shape[1]}, {CODEC_DIM}]",
        ),
        (
            "mean over D",
            f"{sample_codec.mean(dim=-1).abs().max().item():.2e}",
            "approximately 0 after z-normalization",
        ),
        (
            "std over D",
            f"{sample_codec.std(dim=-1).mean().item():.6f}",
            "approximately 1 after z-normalization",
        ),
    ]
    html_table(["check", "observed", "expected"], codec_check_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "assets/embedding-swap.svg",
        alt="Normal embedding and Kronecker embedding feeding the same transformer",
        width="100%",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The rest is a normal GPT-style transformer

    For token representations $X\in\mathbb{R}^{B\times T\times d_{model}}$, each attention head forms

    $$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

    - $Q$ (**queries**) asks what each position is looking for.
    - $K$ (**keys**) describes what each position offers.
    - $V$ (**values**) carries the information that will be mixed.
    - $W_Q,W_K,W_V$ are learned matrices.

    Attention is

    $$\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{head}}}+M\right)V.$$

    $QK^\top$ scores every query-key pair. $\sqrt{d_{head}}$ stabilizes score scale. The causal mask $M$ sets future positions to $-\infty$, so after softmax they receive probability zero. This is why an autoregressive LLM cannot peek at future tokens.

    The MLP transforms each position independently; residual connections preserve an easy information path around attention and the MLP.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "assets/transformer-block.svg",
        alt="A causal transformer block with attention, MLP, and residual paths",
        width="100%",
    )
    return


@app.cell
def _(
    BYTE_VALUES,
    F,
    KroneckerEmbedding,
    MAX_TOKEN_BYTES,
    dataclass,
    math,
    nn,
    torch,
):
    @dataclass(frozen=True)
    class TinyConfig:
        vocab_size: int
        context_length: int = 48
        d_model: int = 64
        n_heads: int = 4
        n_layers: int = 2
        mlp_multiplier: int = 4

    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.d_model % config.n_heads == 0
            self.n_heads = config.n_heads
            self.head_dim = config.d_model // config.n_heads
            self.qkv = nn.Linear(
                config.d_model, 3 * config.d_model, bias=False
            )
            self.output = nn.Linear(config.d_model, config.d_model, bias=False)

        def forward(self, x):
            batch, time_steps, width = x.shape
            query, key, value = self.qkv(x).chunk(3, dim=-1)
            query = query.view(
                batch, time_steps, self.n_heads, self.head_dim
            ).transpose(1, 2)
            key = key.view(
                batch, time_steps, self.n_heads, self.head_dim
            ).transpose(1, 2)
            value = value.view(
                batch, time_steps, self.n_heads, self.head_dim
            ).transpose(1, 2)
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            future_mask = torch.triu(
                torch.ones(
                    time_steps, time_steps, dtype=torch.bool, device=x.device
                ),
                diagonal=1,
            )
            weights = F.softmax(
                scores.masked_fill(future_mask, float("-inf")), dim=-1
            )
            mixed = weights @ value
            mixed = (
                mixed.transpose(1, 2)
                .contiguous()
                .view(batch, time_steps, width)
            )
            return self.output(mixed)

    class FeedForward(nn.Module):
        def __init__(self, config):
            super().__init__()
            hidden = config.mlp_multiplier * config.d_model
            self.layers = nn.Sequential(
                nn.Linear(config.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.d_model),
            )

        def forward(self, x):
            return self.layers(x)

    class TransformerBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attention_norm = nn.LayerNorm(config.d_model)
            self.attention = CausalSelfAttention(config)
            self.mlp_norm = nn.LayerNorm(config.d_model)
            self.mlp = FeedForward(config)

        def forward(self, x):
            x = x + self.attention(self.attention_norm(x))
            return x + self.mlp(self.mlp_norm(x))

    class TinyGPT(nn.Module):
        def __init__(
            self, config, embedding_kind, byte_buffer=None, length_buffer=None
        ):
            super().__init__()
            self.config = config
            self.embedding_kind = embedding_kind
            if embedding_kind == "standard":
                self.token_embedding = nn.Embedding(
                    config.vocab_size, config.d_model
                )
            elif embedding_kind == "kronecker":
                self.token_embedding = KroneckerEmbedding(
                    byte_buffer,
                    length_buffer,
                    d_model=config.d_model,
                    char_dim=BYTE_VALUES,
                    pos_dim=MAX_TOKEN_BYTES,
                )
            else:
                raise ValueError(
                    "embedding_kind must be 'standard' or 'kronecker'"
                )

            self.position_embedding = nn.Embedding(
                config.context_length, config.d_model
            )
            self.blocks = nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.n_layers)]
            )
            self.final_norm = nn.LayerNorm(config.d_model)
            self.lm_head = nn.Linear(
                config.d_model, config.vocab_size, bias=False
            )

        def forward(self, input_ids, return_token_vectors=False):
            batch, time_steps = input_ids.shape
            if time_steps > self.config.context_length:
                raise ValueError("sequence exceeds configured context length")
            positions = torch.arange(time_steps, device=input_ids.device)
            token_vectors = self.token_embedding(input_ids)
            x = token_vectors + self.position_embedding(positions)[None, :, :]
            for block in self.blocks:
                x = block(x)
            hidden = self.final_norm(x)
            logits = self.lm_head(hidden)
            if return_token_vectors:
                return logits, token_vectors
            return logits

    return TinyConfig, TinyGPT


@app.cell
def _(time, torch):
    def count_trainable(module):
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    def copy_shared_weights(source, target):
        target.position_embedding.load_state_dict(
            source.position_embedding.state_dict()
        )
        target.blocks.load_state_dict(source.blocks.state_dict())
        target.final_norm.load_state_dict(source.final_norm.state_dict())
        target.lm_head.load_state_dict(source.lm_head.state_dict())

    def benchmark_forward(model, ids, repeats=5):
        model.eval()
        with torch.no_grad():
            for _ in range(2):
                model(ids)
            start = time.perf_counter()
            for _ in range(repeats):
                model(ids)
            return 1_000 * (time.perf_counter() - start) / repeats

    return benchmark_forward, copy_shared_weights, count_trainable


@app.cell
def _(
    CONTEXT_LENGTH,
    MODEL_SEED,
    MODEL_WIDTH,
    N_HEADS,
    N_LAYERS,
    TinyConfig,
    TinyGPT,
    VOCAB_SIZE,
    copy_shared_weights,
    gpt2_byte_buffer,
    gpt2_length_buffer,
    info_card,
    torch,
):
    tiny_config = TinyConfig(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=MODEL_WIDTH,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )

    torch.manual_seed(MODEL_SEED)
    standard_model = TinyGPT(tiny_config, embedding_kind="standard")
    torch.manual_seed(MODEL_SEED)
    kronecker_model = TinyGPT(
        tiny_config,
        embedding_kind="kronecker",
        byte_buffer=gpt2_byte_buffer,
        length_buffer=gpt2_length_buffer,
    )
    copy_shared_weights(standard_model, kronecker_model)

    info_card(
        "Two comparable models are ready",
        "The transformer body, positional embedding, final normalization, and untied output head have identical weights. "
        "Only the input embedding modules differ.",
        "#4ade80",
    )
    return kronecker_model, standard_model


@app.cell
def _(
    encoding,
    html_table,
    info_card,
    input_ids_tensor,
    kronecker_model,
    mo,
    standard_model,
    torch,
):
    standard_model.eval()
    kronecker_model.eval()
    with torch.no_grad():
        standard_logits, standard_token_vectors = standard_model(
            input_ids_tensor, return_token_vectors=True
        )
        kronecker_logits, kronecker_token_vectors = kronecker_model(
            input_ids_tensor, return_token_vectors=True
        )

    standard_next_token_id = int(standard_logits[0, -1].argmax())
    kronecker_next_token_id = int(kronecker_logits[0, -1].argmax())
    standard_next_piece = repr(encoding.decode([standard_next_token_id]))
    kronecker_next_piece = repr(encoding.decode([kronecker_next_token_id]))

    model_run_rows = [
        (
            "standard",
            tuple(standard_token_vectors.shape),
            tuple(standard_logits.shape),
            standard_next_piece,
        ),
        (
            "Kronecker",
            tuple(kronecker_token_vectors.shape),
            tuple(kronecker_logits.shape),
            kronecker_next_piece,
        ),
    ]

    mo.vstack(
        [
            html_table(
                [
                    "model",
                    "embedding output",
                    "logits output",
                    "random top-1 next piece",
                ],
                model_run_rows,
            ),
            info_card(
                "Do not interpret these predictions",
                "Both networks are untrained. The top-1 pieces merely prove that each complete LLM path executes from token IDs to vocabulary logits.",
                "#f59e0b",
            ),
        ]
    )
    return


@app.cell
def _(
    CODEC_DIM,
    MODEL_WIDTH,
    VOCAB_SIZE,
    benchmark_forward,
    count_trainable,
    html_table,
    input_ids_tensor,
    kronecker_model,
    standard_model,
):
    normal_input_parameters = VOCAB_SIZE * MODEL_WIDTH
    kronecker_input_parameters = CODEC_DIM * MODEL_WIDTH
    input_parameter_reduction = 100 * (
        1 - kronecker_input_parameters / normal_input_parameters
    )
    standard_total_parameters = count_trainable(standard_model)
    kronecker_total_parameters = count_trainable(kronecker_model)
    standard_ms = benchmark_forward(standard_model, input_ids_tensor)
    kronecker_ms = benchmark_forward(kronecker_model, input_ids_tensor)

    parameter_rows = [
        (
            "normal input",
            f"V × d_model = {VOCAB_SIZE:,} × {MODEL_WIDTH}",
            f"{normal_input_parameters:,}",
        ),
        (
            "Kronecker input",
            f"D × d_model = {CODEC_DIM:,} × {MODEL_WIDTH}",
            f"{kronecker_input_parameters:,}",
        ),
        ("input reduction", "1 − D/V", f"{input_parameter_reduction:.1f}%"),
        (
            "normal total",
            "all trainable tensors",
            f"{standard_total_parameters:,}",
        ),
        (
            "Kronecker total",
            "all trainable tensors",
            f"{kronecker_total_parameters:,}",
        ),
    ]

    html_table(["quantity", "formula", "observed"], parameter_rows)
    return kronecker_ms, standard_ms


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "assets/parameter-comparison.svg",
        alt="Normal versus Kronecker trainable input parameter counts",
        width="100%",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "assets/codec-similarity.svg",
        alt="Cosine similarities induced by shared bytes at shared positions",
        width="100%",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · What changed, and what did not?

    | Component | Normal model | Kronecker model |
    |---|---|---|
    | Tokenizer | GPT-2 BPE | **same GPT-2 BPE** |
    | Input representation | learned row per token ID | fixed byte-position features + learned projection |
    | Output shape | $B\times T\times d_{model}$ | **same shape** |
    | Positional embedding | learned | **same** |
    | Attention and MLP | normal causal transformer | **same** |
    | Output head | untied $d_{model}\to V$ | **same and untied** |
    | Input trainable parameters | $Vd_{model}$ | $Dd_{model}$ |
    | Built-in geometry | none before training | byte equality at equal positions |

    ### The conceptual trade

    The normal table gives every token an independent trainable memory slot. It is expressive, but large vocabularies require many parameters and rare rows receive few updates.

    Kronecker replaces those independent slots with a shared rule. Every token is described through the same byte-position coordinate system, so related surface forms share features immediately. The cost is a rigid prior: byte-similar but semantically unrelated strings also start close, and shifted suffixes can look less similar than expected.

    ### Why both output heads are untied here

    Many GPT implementations reuse the input table as the output classifier. Kronecker cannot do that directly because its fixed codec width $D$ generally differs from $d_{model}$. We deliberately use a separate output head in **both** notebook models, ensuring the experiment changes only the input representation.
    """)
    return


@app.cell
def _(html_table, kronecker_ms, mo, standard_ms):
    runtime_rows = [
        (
            "standard forward",
            f"{standard_ms:.3f} ms",
            "row lookup + same transformer",
        ),
        (
            "Kronecker forward",
            f"{kronecker_ms:.3f} ms",
            "construct codec + projection + same transformer",
        ),
    ]
    mo.vstack(
        [
            html_table(
                ["untrained CPU measurement", "mean", "work performed"],
                runtime_rows,
            ),
            mo.md(
                "Runtime here is a tiny CPU demonstration, not a production benchmark. The dynamic codec trades extra arithmetic for much lower fixed-buffer memory."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Final mental model

    ```text
    Normal embedding
    token ID i ──select row──> E[i] ──> transformer
                               ↑
                      V independent learned rows

    Kronecker embedding
    token ID i ──> token bytes ──> fixed κ(bytes) ──> learned W_proj ──> transformer
                                      shared rule         shared map
    ```

    The paper's change happens **before contextual reasoning begins**. It does not alter attention, generation, the causal objective, or tokenization. It changes the prior representation handed to the first transformer block:

    - normal embedding says: “this is vocabulary item 17,”
    - Kronecker embedding says: “this token contains these bytes at these positions.”

    To compare learning rather than mere execution, the next step would be to train both models on identical batches and plot held-out cross-entropy. This notebook intentionally stops at the clean architectural and mathematical comparison.

    **References:** [Kronecker Embeddings paper](https://arxiv.org/html/2605.29459v1) · [reference implementation](https://github.com/theschoolofai/kronecker-embeddings) · [marimo documentation](https://docs.marimo.io/)
    """)
    return


if __name__ == "__main__":
    app.run()
