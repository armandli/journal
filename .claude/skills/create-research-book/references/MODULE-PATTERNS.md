# Neural Network Module Design Patterns

These rules apply to ALL model code in research notebooks. Every module must follow these conventions.

## Core Rules

1. **Version in class name** — embed the version number at the end of the class name: `MultiLayerPerceptronV1`, `ResidualBlockV1`. Never use a separate `VERSION` attribute.
2. **No global variables** — all config passed via `__init__` parameters; never reference outer scope
3. **Typed parameters with defaults** — every `__init__` param has a type hint and default value
4. **Composition** — top-level models compose custom sub-modules, not just raw framework layers
5. **Swappability** — to upgrade a module, define a new class with the incremented suffix: `MultiLayerPerceptronV2`
6. **Snake_case functions** — all standalone functions use snake_case and receive every dependency as an explicit parameter

## MLX Module Pattern

Each class gets its own `@app.class_definition` cell. Each helper function gets its own `@app.function` cell.

```python
@app.class_definition
class MultiLayerPerceptronBlockV1(nn.Module):
    def __init__(self, dims: int = 256, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, dims * expansion)
        self.fc2 = nn.Linear(dims * expansion, dims)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        h = self.dropout(nn.gelu(self.fc1(h)))
        return x + self.fc2(h)


@app.class_definition
class TransformerBlockV1(nn.Module):
    def __init__(self, dims: int = 256, num_heads: int = 8, mlp_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.mlp = MultiLayerPerceptronBlockV1(dims, mlp_expansion, dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, mask=mask)
        return self.mlp(x)


@app.class_definition
class ResearchTransformerV1(nn.Module):
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
        self.layers = [TransformerBlockV1(dims, num_heads, mlp_expansion, dropout) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(dims)
        self.head = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mx.utils.tree_flatten(model.parameters()))


@app.function
def compute_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(logits, y).mean()


@app.function
def run_train_epoch(model: nn.Module, optimizer, train_iter, preprocess_fn) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    epoch_loss = 0.0
    n_batches = 0
    train_iter.reset()
    for batch in train_iter:
        x, y = preprocess_fn(batch)
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n_batches += 1
    return epoch_loss / max(n_batches, 1)


@app.function
def run_evaluate(model: nn.Module, data_iter, preprocess_fn) -> float:
    correct = 0
    total = 0
    data_iter.reset()
    for batch in data_iter:
        x, y = preprocess_fn(batch)
        preds = mx.argmax(model(x), axis=-1)
        mx.eval(preds)
        correct += int(mx.sum(preds == y).item())
        total += y.shape[0]
    return correct / total
```

## PyTorch Module Pattern

Each class gets its own `@app.class_definition` cell. Each helper function gets its own `@app.function` cell.

```python
@app.class_definition
class ConvolutionBlockV1(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3, stride: int = 1, activation: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@app.class_definition
class ResidualBlockV1(nn.Module):
    def __init__(self, channels: int = 64, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(ConvolutionBlockV1(channels, channels), ConvolutionBlockV1(channels, channels))
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.block(x))


@app.class_definition
class ResearchCnnV1(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_blocks: int = 4, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.stem = ConvolutionBlockV1(in_channels, base_channels, kernel_size=7, stride=2)
        self.blocks = nn.ModuleList([ResidualBlockV1(base_channels, dropout) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.pool(x).flatten(1))


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@app.function
def run_train_epoch(model: nn.Module, optimizer, train_loader, loss_fn, device: torch.device) -> float:
    model.train()
    epoch_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


@app.function
def run_evaluate(model: nn.Module, data_loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
```

## VAE-Specific Pattern

Each class in its own `@app.class_definition` cell.

```python
@app.class_definition
class MultiLayerPerceptronBlockV1(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.fc2(nn.relu(self.fc1(x))))


@app.class_definition
class VariationalEncoderV1(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 32):
        super().__init__()
        self.mlp = MultiLayerPerceptronBlockV1(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def __call__(self, x: mx.array):
        h = self.mlp(x)
        return self.fc_mu(h), self.fc_logvar(h)


@app.class_definition
class VariationalDecoderV1(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 512, output_dim: int = 784):
        super().__init__()
        self.mlp = MultiLayerPerceptronBlockV1(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def __call__(self, z: mx.array) -> mx.array:
        return mx.sigmoid(self.fc_out(self.mlp(z)))


@app.class_definition
class VariationalAutoEncoderV1(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 32):
        super().__init__()
        self.encoder = VariationalEncoderV1(input_dim, hidden_dim, latent_dim)
        self.decoder = VariationalDecoderV1(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: mx.array, logvar: mx.array) -> mx.array:
        std = mx.exp(0.5 * logvar)
        return mu + mx.random.normal(std.shape) * std

    def __call__(self, x: mx.array):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z)


@app.function
def compute_vae_loss(model: nn.Module, x: mx.array) -> mx.array:
    recon, mu, logvar = model(x)
    x_flat = x.reshape(x.shape[0], -1)
    recon_flat = recon.reshape(recon.shape[0], -1)
    recon_loss = mx.mean(mx.sum((recon_flat - x_flat) ** 2, axis=-1))
    kl_loss = -0.5 * mx.mean(mx.sum(1 + logvar - mu ** 2 - mx.exp(logvar), axis=-1))
    return recon_loss + kl_loss
```

## Architecture Documentation Cell

Every model definition section must include a markdown cell like this:

```python
mo.md(f"""
### Model Architecture — ResearchCnnV1

| Component | Module | Output Shape |
|-----------|--------|--------------|
| Stem | `ConvolutionBlockV1` | `(B, 64, H/2, W/2)` |
| Residual blocks | `ResidualBlockV1` × 4 | `(B, 64, H/2, W/2)` |
| Head | `Linear` | `(B, num_classes)` |

**Total parameters**: `{count_parameters(model):,}`
""")
```

## Version Upgrade Pattern

When improving a module, define a new class with the incremented suffix and keep the old one:

```python
class MultiLayerPerceptronBlockV1(nn.Module):
    ...

class MultiLayerPerceptronBlockV2(nn.Module):
    """V2: adds pre-norm and gated linear unit."""
    ...

# In the model cell, choose which to use:
block_cls = MultiLayerPerceptronBlockV2   # swap here without changing downstream code
```
