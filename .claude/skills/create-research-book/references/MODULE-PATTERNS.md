# Neural Network Module Design Patterns

These rules apply to ALL model code in research notebooks. Every module must follow these conventions.

## Core Rules

1. **Version attribute** — every class has `VERSION = "v1"` as a class attribute
2. **No global variables** — all config passed via `__init__` parameters; never reference outer scope
3. **Typed parameters with defaults** — every `__init__` param has a type hint and default value
4. **Composition** — top-level models compose custom sub-modules, not just raw framework layers
5. **Swappability** — each module can be replaced by incrementing its VERSION or subclassing

## MLX Module Pattern

```python
class MLPBlock(nn.Module):
    """Two-layer MLP with residual connection and layer norm."""
    VERSION = "v1"

    def __init__(
        self,
        dims: int = 256,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, dims * expansion)
        self.fc2 = nn.Linear(dims * expansion, dims)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        h = self.dropout(nn.gelu(self.fc1(h)))
        return x + self.fc2(h)


class TransformerBlock(nn.Module):
    """Single transformer block composing attention and MLP."""
    VERSION = "v1"

    def __init__(
        self,
        dims: int = 256,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.mlp = MLPBlock(dims, mlp_expansion, dropout)   # reuses custom sub-module

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, mask=mask)
        return self.mlp(x)


class ResearchModel(nn.Module):
    """Top-level model composing TransformerBlocks."""
    VERSION = "v1"

    def __init__(
        self,
        vocab_size: int = 256,
        dims: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dims)
        self.layers = [
            TransformerBlock(dims, num_heads, mlp_expansion, dropout)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(dims)
        self.head = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))
```

## PyTorch Module Pattern

```python
class ConvBlock(nn.Module):
    """Conv + BN + activation building block."""
    VERSION = "v1"

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block composing two ConvBlocks."""
    VERSION = "v1"

    def __init__(
        self,
        channels: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),           # reuses custom sub-module
            ConvBlock(channels, channels),
        )
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.block(x))


class ResearchCNN(nn.Module):
    """Top-level CNN composing ConvBlock and ResBlock."""
    VERSION = "v1"

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 4,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels, kernel_size=7, stride=2)
        self.blocks = nn.ModuleList([
            ResBlock(base_channels, dropout) for _ in range(num_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.pool(x).flatten(1))
```

## VAE-Specific Pattern

```python
class VAEEncoder(nn.Module):
    VERSION = "v1"

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 32):
        super().__init__()
        self.mlp = MLPBlock(input_dim, hidden_dims=[hidden_dim])  # reuse custom block
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def __call__(self, x):
        h = self.mlp(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    VERSION = "v1"

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 512, output_dim: int = 784):
        super().__init__()
        self.mlp = MLPBlock(latent_dim, hidden_dims=[hidden_dim])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def __call__(self, z):
        return mx.sigmoid(self.fc_out(self.mlp(z)))


class VAE(nn.Module):
    VERSION = "v1"

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 32):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = mx.exp(0.5 * logvar)
        return mu + mx.random.normal(std.shape) * std

    def __call__(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z):
        return self.decoder(z)
```

## Architecture Documentation Cell

Every model definition section must include a markdown cell like this:

```python
mo.md(f"""
### Model Architecture — {ModelClass.__name__} {ModelClass.VERSION}

| Component | Module | Version | Output Shape |
|-----------|--------|---------|--------------|
| Stem | `ConvBlock` | v1 | `(B, 64, H/2, W/2)` |
| Residual blocks | `ResBlock` × 4 | v1 | `(B, 64, H/2, W/2)` |
| Head | `Linear` | — | `(B, num_classes)` |

**Total parameters**: `{{param_count:,}}`
""")
```

## Parameter Count Utility

```python
# MLX
def count_params(model):
    return sum(v.size for _, v in mx.utils.tree_flatten(model.parameters()))

# PyTorch
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## VERSION Upgrade Pattern

When improving a module, increment its VERSION and keep the old one available:

```python
class MLPBlock(nn.Module):
    VERSION = "v1"
    ...

class MLPBlockV2(nn.Module):
    """v2: adds pre-norm and gated linear unit."""
    VERSION = "v2"
    ...

# In the model cell, choose which to use:
block_cls = MLPBlockV2   # swap here without changing downstream code
```
