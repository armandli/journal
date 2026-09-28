import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset, random_split

    import torchvision
    from torchvision import datasets
    from torchvision.transforms import v2


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Diffusion Transformer (DiT) + DDPM on Fashion-MNIST

    **Goal.** Train a small class-conditional Diffusion Transformer (DiT) on the
    Fashion-MNIST dataset with the DDPM noise-prediction objective and
    classifier-free guidance (CFG). The notebook auto-detects CUDA and falls
    back to CPU; every helper is written so training, evaluation and sampling
    run unchanged on either device.

    **Sections.**
    1. Title & research goal (this cell)
    2. Data exploration — load Fashion-MNIST, show samples and class distribution, pick device
    3. Dataset creation — 85/15 train/val split from the 60k train images, normalize to [-1, 1]
    4. Model definition — DiT modules (patchify, timestep + label embedders, adaLN-Zero blocks, unpatchify) + a DDPM scheduler
    5. Training — DDPM MSE loss with random label-drop for CFG, AMP-safe on CPU and CUDA
    6. Hyperparameter search (optional) — small grid over lr and model width
    7. Validation & 5-fold cross-validation — test noise-prediction MSE + per-fold stats
    8. Results — training curves, forward-noising demo, and CFG sample grids
    9. Save trained model — persist weights + constructor kwargs to `models/`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 — Data Exploration
    """)
    return


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _(device, mo):
    mo.md(
        f"""
        **Selected device:** `{device}`

        - CUDA available: `{torch.cuda.is_available()}`
        - torch: `{torch.__version__}`
        - torchvision: `{torchvision.__version__}`
        """
    )
    return


@app.cell
def _():
    fashion_mnist_class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    return (fashion_mnist_class_names,)


@app.cell
def _():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "fashion_mnist"
    data_dir.mkdir(parents=True, exist_ok=True)
    return (data_dir,)


@app.function
def load_raw_fashion_mnist(data_dir: Path):
    train_ds = datasets.FashionMNIST(root=str(data_dir), train=True, download=True)
    test_ds = datasets.FashionMNIST(root=str(data_dir), train=False, download=True)
    return train_ds, test_ds


@app.cell
def _(data_dir):
    raw_train_ds, raw_test_ds = load_raw_fashion_mnist(data_dir)
    return raw_test_ds, raw_train_ds


@app.cell
def _(mo, raw_test_ds, raw_train_ds):
    mo.md(
        f"""
        **Fashion-MNIST raw splits**

        | Split | Size |
        |-------|-----:|
        | Train (raw) | {len(raw_train_ds):,} |
        | Test | {len(raw_test_ds):,} |

        Images are 28x28 grayscale in 10 classes. We will split the 60k train
        images 85/15 into a train/val set in Section 3.
        """
    )
    return


@app.function
def plot_sample_grid(
    dataset,
    class_names: list,
    n_show: int = 40,
    rows: int = 5,
    cols: int = 8,
):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    for i in range(n_show):
        img, label = dataset[i]
        arr = np.asarray(img)
        r, c = divmod(i, cols)
        axes[r, c].imshow(arr, cmap="gray")
        axes[r, c].set_title(class_names[int(label)], fontsize=8)
        axes[r, c].axis("off")
    fig.suptitle("Fashion-MNIST sample images", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(fashion_mnist_class_names, raw_train_ds):
    plot_sample_grid(raw_train_ds, fashion_mnist_class_names)
    return


@app.function
def plot_class_distribution(dataset, class_names: list):
    labels = np.asarray(dataset.targets)
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(class_names)), counts, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Fashion-MNIST class distribution (train)")
    for i, c in enumerate(counts):
        ax.text(i, c + 40, str(int(c)), ha="center", fontsize=8)
    fig.tight_layout()
    return fig


@app.cell
def _(fashion_mnist_class_names, raw_train_ds):
    plot_class_distribution(raw_train_ds, fashion_mnist_class_names)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 — Dataset Creation
    """)
    return


@app.function
def build_transform():
    return v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])


@app.function
def make_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    val_fraction: float = 0.15,
):
    tfm = build_transform()
    full_train = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=tfm)
    n_total = len(full_train)
    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "full_train": full_train,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


@app.cell
def _(data_dir):
    dataloaders = make_dataloaders(data_dir=data_dir, batch_size=128, seed=42, num_workers=0)
    train_loader = dataloaders["train_loader"]
    val_loader = dataloaders["val_loader"]
    test_loader = dataloaders["test_loader"]
    full_train_ds = dataloaders["full_train"]
    return dataloaders, full_train_ds, test_loader, train_loader


@app.cell
def _(dataloaders, mo):
    mo.md(
        f"""
        **Splits**

        | Split | Size |
        |-------|-----:|
        | Train | {len(dataloaders['train_ds']):,} |
        | Val | {len(dataloaders['val_ds']):,} |
        | Test | {len(dataloaders['test_ds']):,} |

        Images are normalized to `[-1, 1]` (mean=0.5, std=0.5) so the DDPM
        forward process operates on symmetric-range inputs.
        """
    )
    return


@app.function
def peek_batch(loader) -> dict:
    x, y = next(iter(loader))
    return {
        "image_shape": tuple(x.shape),
        "image_dtype": str(x.dtype),
        "image_min": float(x.min()),
        "image_max": float(x.max()),
        "label_shape": tuple(y.shape),
        "label_dtype": str(y.dtype),
    }


@app.cell
def _(mo, train_loader):
    mo.md(
        "### Batch preview\n\n"
        + "\n".join(f"- `{k}`: `{v}`" for k, v in peek_batch(train_loader).items())
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 — Model Definition
    """)
    return


@app.class_definition
class SinusoidalTimestepEmbeddingV1(nn.Module):
    def __init__(self, dim: int = 256, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=device)
            / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


@app.class_definition
class TimestepEmbedderV1(nn.Module):
    def __init__(self, hidden_size: int = 192, frequency_embedding_size: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbeddingV1(dim=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


@app.class_definition
class LabelEmbedderV1(nn.Module):
    def __init__(self, num_classes: int = 10, hidden_size: int = 192, dropout_prob: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        # +1 slot for the "null" (unconditional) class used by CFG.
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool()
        return torch.where(drop_ids, torch.full_like(labels, self.num_classes), labels)

    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


@app.class_definition
class PatchEmbedV1(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_size: int = 192,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                        # (B, C, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)        # (B, N, C)
        return x


@app.class_definition
class DiTAttentionV1(nn.Module):
    def __init__(self, hidden_size: int = 192, num_heads: int = 6, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@app.class_definition
class DiTMlpV1(nn.Module):
    def __init__(self, hidden_size: int = 192, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(inner, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


@app.class_definition
class AdaLayerNormV1(nn.Module):
    """adaLN-Zero modulation: produces 6 conditioning vectors from `c`."""

    def __init__(self, hidden_size: int = 192):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # adaLN-Zero: initialize modulation output to 0 so blocks start as identity.
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        params = self.mod(c)                                      # (B, 6*H)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)
        x_norm = self.norm(x)                                     # (B, N, H)
        return x_norm, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


@app.class_definition
class DiTBlockV1(nn.Module):
    def __init__(
        self,
        hidden_size: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.adaln = AdaLayerNormV1(hidden_size=hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttentionV1(hidden_size=hidden_size, num_heads=num_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.mlp = DiTMlpV1(hidden_size=hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_attn_norm, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, c)
        x = x + gate_msa.unsqueeze(1) * self.attn(self._modulate(x_attn_norm, shift_msa, scale_msa))
        x_mlp_norm = self.norm_mlp(x)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self._modulate(x_mlp_norm, shift_mlp, scale_mlp))
        return x


@app.class_definition
class DiTFinalLayerV1(nn.Module):
    def __init__(self, hidden_size: int = 192, patch_size: int = 4, out_channels: int = 1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


@app.class_definition
class DiffusionTransformerV1(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        patch_size: int = 4,
        hidden_size: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        class_dropout_prob: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        self.patch_embed = PatchEmbedV1(
            image_size=image_size, patch_size=patch_size,
            in_channels=in_channels, hidden_size=hidden_size,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_embedder = TimestepEmbedderV1(hidden_size=hidden_size)
        self.y_embedder = LabelEmbedderV1(
            num_classes=num_classes, hidden_size=hidden_size, dropout_prob=class_dropout_prob,
        )

        self.blocks = nn.ModuleList([
            DiTBlockV1(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_layer = DiTFinalLayerV1(hidden_size=hidden_size, patch_size=patch_size, out_channels=self.out_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.patch_size
        c = self.out_channels
        h = w = self.patch_embed.grid_size
        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); t: (B,); y: (B,) integer class ids in [0, num_classes]
        h = self.patch_embed(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, train=self.training)
        for block in self.blocks:
            h = block(h, c)
        h = self.final_layer(h, c)
        return self.unpatchify(h)

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, guidance_scale: float = 1.0) -> torch.Tensor:
        """Classifier-free-guided noise prediction.

        Runs one forward pass over the concatenation of a conditional half
        (with true labels `y`) and an unconditional half (with the null class
        index `= num_classes`), then combines them:
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        """
        # Conditional forward (no CFG dropout applied — training=False keeps labels intact).
        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        null_labels = torch.full_like(y, self.num_classes)
        combined_y = torch.cat([y, null_labels], dim=0)
        # Force `training=False` path — no random label drop — but still route through the embedder.
        h = self.patch_embed(combined_x) + self.pos_embed
        c = self.t_embedder(combined_t) + self.y_embedder(combined_y, train=False)
        for block in self.blocks:
            h = block(h, c)
        h = self.final_layer(h, c)
        eps = self.unpatchify(h)
        eps_cond, eps_uncond = eps.chunk(2, dim=0)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@app.class_definition
class DDPMSchedulerV1:
    """Standard DDPM linear-beta schedule with a class-conditional ancestral sampler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = arr.to(t.device)[t]
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return s1 * x0 + s2 * noise

    def training_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, y)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        y: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 1.0,
        num_steps: int | None = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        model.eval()
        img = torch.randn(shape, device=device)
        y = y.to(device)
        total_T = self.num_train_timesteps
        if num_steps is None or num_steps >= total_T:
            step_indices = list(range(total_T - 1, -1, -1))
        else:
            step_indices = list(reversed(np.linspace(0, total_T - 1, num_steps).round().astype(int).tolist()))

        for i, t_int in enumerate(step_indices):
            t = torch.full((shape[0],), int(t_int), device=device, dtype=torch.long)
            if guidance_scale != 1.0:
                eps = model.forward_with_cfg(img, t, y, guidance_scale=guidance_scale)
            else:
                eps = model(img, t, y)

            beta_t = self._extract(self.betas, t, img.shape)
            sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
            sqrt_recip_a = self._extract(self.sqrt_recip_alphas, t, img.shape)
            mean = sqrt_recip_a * (img - beta_t / sqrt_one_minus * eps)

            if clip_denoised:
                # Reconstruct x0 estimate for numerical safety and clip to [-1, 1].
                sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, img.shape)
                x0_pred = (img - sqrt_one_minus * eps) / sqrt_ab
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                # Recompute mean from the clipped x0 estimate.
                coef1 = (self._extract(self.betas, t, img.shape)
                         * torch.sqrt(self._extract(self.alphas_cumprod_prev, t, img.shape))
                         / (1.0 - self._extract(self.alphas_cumprod, t, img.shape)))
                coef2 = ((1.0 - self._extract(self.alphas_cumprod_prev, t, img.shape))
                         * torch.sqrt(self._extract(self.alphas, t, img.shape))
                         / (1.0 - self._extract(self.alphas_cumprod, t, img.shape)))
                mean = coef1 * x0_pred + coef2 * img

            if i == len(step_indices) - 1:
                img = mean
            else:
                var = self._extract(self.posterior_variance, t, img.shape)
                img = mean + torch.sqrt(var) * torch.randn_like(img)
        return img

    def config(self) -> dict:
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@app.function
def default_model_kwargs() -> dict:
    return {
        "image_size": 28,
        "in_channels": 1,
        "patch_size": 4,
        "hidden_size": 192,
        "depth": 6,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "num_classes": 10,
        "class_dropout_prob": 0.1,
        "dropout": 0.0,
    }


@app.function
def build_model(model_kwargs: dict, device: torch.device) -> nn.Module:
    return DiffusionTransformerV1(**model_kwargs).to(device)


@app.function
def build_scheduler(num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> DDPMSchedulerV1:
    return DDPMSchedulerV1(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)


@app.cell
def _(device):
    demo_model_kwargs = default_model_kwargs()
    demo_model = build_model(demo_model_kwargs, device)
    demo_scheduler = build_scheduler()
    demo_param_count = count_parameters(demo_model)
    return demo_model_kwargs, demo_param_count


@app.cell
def _(demo_model_kwargs, demo_param_count, mo):
    mo.md(
        f"""
        ### Model Architecture — `DiffusionTransformerV1`

        | Component | Module | Notes |
        |-----------|--------|-------|
        | Patchify | `PatchEmbedV1` | 28x28 -> 7x7 = 49 tokens (`patch_size=4`) |
        | Positional embedding | `nn.Parameter` | learned, shape `(1, 49, H)` |
        | Timestep embedder | `TimestepEmbedderV1` (uses `SinusoidalTimestepEmbeddingV1`) | -> conditioning vector `c_t` |
        | Class embedder | `LabelEmbedderV1` | `num_classes+1` entries; null class enables CFG |
        | Transformer blocks | `DiTBlockV1` x `depth` | adaLN-Zero mod on attention + MLP |
        | Attention | `DiTAttentionV1` | multi-head self-attn (`num_heads=6`) |
        | Feed-forward | `DiTMlpV1` | GELU, hidden = 4x |
        | Final layer | `DiTFinalLayerV1` | adaLN + linear -> `patch*patch*C` |
        | Unpatchify | inline in `forward` | tokens -> `(B, 1, 28, 28)` noise prediction |

        **Constructor kwargs:** `{demo_model_kwargs}`

        **Total trainable parameters:** `{demo_param_count:,}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 — Training
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "5e-4": 5e-4, "1e-3": 1e-3},
        value="3e-4", label="Learning Rate",
    )
    bs_ui = mo.ui.dropdown(options=[32, 64, 128, 256], value=128, label="Batch Size")
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="0", label="Weight Decay",
    )
    epochs_ui = mo.ui.slider(1, 30, value=2, step=1, label="Epochs")
    class_drop_ui = mo.ui.dropdown(
        options={"0.05": 0.05, "0.10": 0.10, "0.20": 0.20},
        value="0.10", label="CFG label dropout (training)",
    )
    default_guidance_ui = mo.ui.slider(0.0, 8.0, value=3.0, step=0.5, label="Default CFG guidance scale (used later)")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack([
        mo.md("### Training hyperparameters"),
        mo.hstack([lr_ui, epochs_ui]),
        mo.hstack([bs_ui, wd_ui]),
        mo.hstack([class_drop_ui, default_guidance_ui]),
        train_btn,
    ])
    return (
        bs_ui,
        class_drop_ui,
        default_guidance_ui,
        epochs_ui,
        lr_ui,
        train_btn,
        wd_ui,
    )


@app.function
def sample_random_timesteps(batch_size: int, num_train_timesteps: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, num_train_timesteps, (batch_size,), device=device, dtype=torch.long)


@app.function
def run_train_epoch(
    model: nn.Module,
    scheduler: DDPMSchedulerV1,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_loader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n_batches = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        t = sample_random_timesteps(x.shape[0], scheduler.num_train_timesteps, device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss = scheduler.training_losses(model, x, t, y, device=device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


@app.function
def run_evaluate(
    model: nn.Module,
    scheduler: DDPMSchedulerV1,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            t = sample_random_timesteps(x.shape[0], scheduler.num_train_timesteps, device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                loss = scheduler.training_losses(model, x, t, y, device=device)
            total += loss.item()
            n_batches += 1
    return total / max(n_batches, 1)


@app.cell
def _(
    bs_ui,
    class_drop_ui,
    data_dir,
    device,
    epochs_ui,
    lr_ui,
    mo,
    train_btn,
    wd_ui,
):
    train_losses: list[float] = []
    val_losses: list[float] = []
    trained_model: nn.Module | None = None
    trained_scheduler: DDPMSchedulerV1 | None = None
    trained_kwargs: dict | None = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin training."))
    else:
        loaders = make_dataloaders(data_dir=data_dir, batch_size=bs_ui.value, seed=42, num_workers=0)
        run_train_loader = loaders["train_loader"]
        run_val_loader = loaders["val_loader"]

        run_kwargs = default_model_kwargs()
        run_kwargs["class_dropout_prob"] = class_drop_ui.value
        model_run = build_model(run_kwargs, device)
        scheduler_run = build_scheduler()
        optimizer = torch.optim.AdamW(model_run.parameters(), lr=lr_ui.value, weight_decay=wd_ui.value)
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
        n_epochs = epochs_ui.value

        for epoch in range(n_epochs):
            tl = run_train_epoch(model_run, scheduler_run, optimizer, scaler, run_train_loader, device)
            vl = run_evaluate(model_run, scheduler_run, run_val_loader, device)
            train_losses.append(tl)
            val_losses.append(vl)
            mo.output.replace(
                mo.md(f"**Epoch {epoch + 1}/{n_epochs}** — train: `{tl:.4f}` | val: `{vl:.4f}`")
            )

        trained_model = model_run
        trained_scheduler = scheduler_run
        trained_kwargs = run_kwargs
        mo.output.replace(
            mo.md(
                f"**Training complete.** Final train: `{train_losses[-1]:.4f}` | "
                f"final val: `{val_losses[-1]:.4f}`"
            )
        )
    return (
        train_losses,
        trained_kwargs,
        trained_model,
        trained_scheduler,
        val_losses,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 6 — Hyperparameter Search (Optional)
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.function
def run_hp_config(
    lr: float,
    hidden_size: int,
    depth: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 2,
) -> float:
    kwargs = default_model_kwargs()
    kwargs["hidden_size"] = hidden_size
    kwargs["depth"] = depth
    # Ensure num_heads still divides hidden_size.
    if hidden_size % kwargs["num_heads"] != 0:
        kwargs["num_heads"] = 4
    model = build_model(kwargs, device)
    scheduler = build_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    for _ in range(n_epochs):
        run_train_epoch(model, scheduler, optimizer, scaler, train_loader, device)
    return run_evaluate(model, scheduler, val_loader, device)


@app.cell
def _(data_dir, device, full_train_ds, hp_search_cb, mo):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )

    hp_search_space = {
        "lr": [3e-4, 1e-3],
        "hidden_size": [96, 192],
        "depth": [4, 6],
    }
    hp_n_epochs = 2
    hp_subset_size = 4000
    hp_val_size = 800

    hp_generator = torch.Generator().manual_seed(123)
    hp_perm = torch.randperm(len(full_train_ds), generator=hp_generator).tolist()
    hp_train_indices = hp_perm[:hp_subset_size]
    hp_val_indices = hp_perm[hp_subset_size:hp_subset_size + hp_val_size]
    hp_train_subset = Subset(full_train_ds, hp_train_indices)
    hp_val_subset = Subset(full_train_ds, hp_val_indices)
    hp_train_loader = DataLoader(hp_train_subset, batch_size=128, shuffle=True, drop_last=True)
    hp_val_loader = DataLoader(hp_val_subset, batch_size=128, shuffle=False)

    hp_results: list[dict] = []
    for lr_val in hp_search_space["lr"]:
        for h_val in hp_search_space["hidden_size"]:
            for d_val in hp_search_space["depth"]:
                hp_val_mse = run_hp_config(
                    lr=lr_val, hidden_size=h_val, depth=d_val,
                    train_loader=hp_train_loader, val_loader=hp_val_loader,
                    device=device, n_epochs=hp_n_epochs,
                )
                hp_results.append({
                    "lr": lr_val, "hidden_size": h_val, "depth": d_val,
                    "val_loss": round(hp_val_mse, 4),
                })
                mo.output.replace(
                    mo.md(f"tried lr={lr_val}, hidden={h_val}, depth={d_val} -> val=`{hp_val_mse:.4f}`")
                )
    hp_results.sort(key=lambda r: r["val_loss"])
    _ = data_dir
    mo.ui.table(hp_results)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7 — Validation & Cross-Validation
    """)
    return


@app.function
def evaluate_model(
    model: nn.Module,
    scheduler: DDPMSchedulerV1,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    mse = run_evaluate(model, scheduler, data_loader, device)
    # Proxy metric: RMSE of noise prediction (also useful as a monitor).
    return {"noise_mse": mse, "noise_rmse": float(np.sqrt(mse))}


@app.cell
def _(
    device,
    mo,
    test_loader,
    trained_model: nn.Module | None,
    trained_scheduler: DDPMSchedulerV1 | None,
):
    if trained_model is None or trained_scheduler is None:
        _out = mo.md("_Train the model first (Section 5) before running validation._")
    else:
        metrics = evaluate_model(trained_model, trained_scheduler, test_loader, device)
        _out = mo.md(
            "### Test-set metrics\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Noise-prediction MSE | `{metrics['noise_mse']:.4f}` |\n"
            f"| Noise-prediction RMSE | `{metrics['noise_rmse']:.4f}` |\n"
        )
    _out
    return


@app.function
def run_cv_fold(
    train_indices: list,
    val_indices: list,
    full_dataset,
    device: torch.device,
    n_epochs: int = 1,
    batch_size: int = 128,
) -> float:
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    tl = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    kwargs = default_model_kwargs()
    kwargs["depth"] = 4
    kwargs["hidden_size"] = 96
    kwargs["num_heads"] = 4
    model = build_model(kwargs, device)
    scheduler = build_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    for _ in range(n_epochs):
        run_train_epoch(model, scheduler, optimizer, scaler, tl, device)
    return run_evaluate(model, scheduler, vl, device)


@app.cell
def _(mo):
    cv_run_cb = mo.ui.checkbox(label="Run 5-fold cross-validation (slow on CPU)", value=False)
    cv_run_cb
    return (cv_run_cb,)


@app.cell
def _(cv_run_cb, device, full_train_ds, mo):
    mo.stop(
        not cv_run_cb.value,
        mo.md("_Enable 5-fold CV above to run this section (uses a subset for CPU-feasibility)._"),
    )

    k = 5
    cv_subset_size = 3000
    cv_generator = torch.Generator().manual_seed(7)
    cv_perm = torch.randperm(len(full_train_ds), generator=cv_generator).tolist()[:cv_subset_size]
    fold_size = cv_subset_size // k
    fold_metrics: list[float] = []
    for fold in range(k):
        val_ids = cv_perm[fold * fold_size:(fold + 1) * fold_size]
        train_ids = cv_perm[:fold * fold_size] + cv_perm[(fold + 1) * fold_size:]
        mse = run_cv_fold(train_ids, val_ids, full_train_ds, device, n_epochs=1, batch_size=128)
        fold_metrics.append(mse)
        mo.output.replace(mo.md(f"Fold {fold + 1}/{k} — val MSE: `{mse:.4f}`"))

    mean_mse = float(np.mean(fold_metrics))
    std_mse = float(np.std(fold_metrics))
    cv_results = {"folds": fold_metrics, "mean": mean_mse, "std": std_mse}
    cv_table_md = "| Fold | Val MSE |\n|---|---|\n"
    for i_fold, m in enumerate(fold_metrics):
        cv_table_md += f"| {i_fold + 1} | `{m:.4f}` |\n"
    cv_table_md += f"| **mean ± std** | **`{mean_mse:.4f} ± {std_mse:.4f}`** |\n"
    mo.output.replace(mo.md("### 5-fold Cross-Validation\n\n" + cv_table_md))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8 — Results
    """)
    return


@app.function
def plot_loss_curve(train_losses: list, val_losses: list | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", lw=2, ms=4, label="Train")
    if val_losses:
        ax.plot(epochs, val_losses, "r-s", lw=2, ms=4, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise-prediction MSE")
    ax.set_title("DDPM training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses: list[float], val_losses: list[float]):
    if not train_losses:
        _out = mo.md("_Train the model first to see loss curves._")
    else:
        _out = plot_loss_curve(train_losses, val_losses)
    _out
    return


@app.function
def plot_forward_diffusion(
    scheduler: DDPMSchedulerV1,
    image: torch.Tensor,
    device: torch.device,
    num_steps: int = 8,
):
    """Show one image being progressively corrupted at evenly-spaced timesteps."""
    image = image.to(device)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    ts = torch.linspace(0, scheduler.num_train_timesteps - 1, num_steps).long().to(device)
    fig, axes = plt.subplots(1, num_steps, figsize=(2 * num_steps, 2.5))
    noise = torch.randn_like(image)
    for i, t_val in enumerate(ts):
        t_batch = t_val.unsqueeze(0)
        xt = scheduler.q_sample(image, t_batch, noise)
        arr = xt[0, 0].detach().cpu().numpy()
        arr = (arr + 1.0) / 2.0
        arr = np.clip(arr, 0.0, 1.0)
        axes[i].imshow(arr, cmap="gray")
        axes[i].set_title(f"t={int(t_val.item())}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle("Forward diffusion q(x_t | x_0)", fontsize=12)
    fig.tight_layout()
    return fig


@app.cell
def _(device, mo, raw_train_ds, trained_scheduler: DDPMSchedulerV1 | None):
    if trained_scheduler is None:
        _out = mo.md("_Train first to see forward diffusion with the fitted scheduler._")
    else:
        sample_img, _ = raw_train_ds[0]
        img_tensor = torch.from_numpy(np.asarray(sample_img, dtype=np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0)                 # (1, 28, 28)
        img_tensor = img_tensor * 2.0 - 1.0                  # [-1, 1]
        _out = plot_forward_diffusion(trained_scheduler, img_tensor, device, num_steps=8)
    _out
    return


@app.function
def plot_generated_grid(images: torch.Tensor, class_names: list, labels: list | None = None):
    n = images.shape[0]
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 2.0 * rows), squeeze=False)
    imgs = ((images.clamp(-1.0, 1.0) + 1.0) / 2.0).detach().cpu().numpy()
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")
        if i >= n:
            continue
        axes[r][c].imshow(imgs[i, 0], cmap="gray")
        if labels is not None:
            label_id = int(labels[i])
            title = class_names[label_id] if 0 <= label_id < len(class_names) else f"cls={label_id}"
            axes[r][c].set_title(title, fontsize=8)
    fig.suptitle("Generated samples", fontsize=12)
    fig.tight_layout()
    return fig


@app.cell
def _(default_guidance_ui, fashion_mnist_class_names, mo):
    target_class_ui = mo.ui.dropdown(
        options={name: i for i, name in enumerate(fashion_mnist_class_names)},
        value=fashion_mnist_class_names[0],
        label="Target class",
    )
    guidance_ui = mo.ui.slider(
        0.0, 8.0, value=float(default_guidance_ui.value), step=0.5,
        label="Classifier-free-guidance scale",
    )
    num_samples_ui = mo.ui.slider(1, 16, value=8, step=1, label="Number of samples per class")
    num_sample_steps_ui = mo.ui.slider(10, 1000, value=50, step=10, label="DDPM sampling steps")
    sample_all_classes_cb = mo.ui.checkbox(label="Sample one row per class (10 classes)", value=False)
    sample_btn = mo.ui.run_button(label="Generate Samples")
    mo.vstack([
        mo.md("### Sampling controls"),
        mo.hstack([target_class_ui, guidance_ui]),
        mo.hstack([num_samples_ui, num_sample_steps_ui]),
        sample_all_classes_cb,
        sample_btn,
    ])
    return (
        guidance_ui,
        num_sample_steps_ui,
        num_samples_ui,
        sample_all_classes_cb,
        sample_btn,
        target_class_ui,
    )


@app.function
def generate_samples(
    model: nn.Module,
    scheduler: DDPMSchedulerV1,
    class_ids: torch.Tensor,
    device: torch.device,
    guidance_scale: float = 3.0,
    num_steps: int = 50,
    image_size: int = 28,
    in_channels: int = 1,
) -> torch.Tensor:
    shape = (class_ids.shape[0], in_channels, image_size, image_size)
    return scheduler.p_sample_loop(
        model=model,
        shape=shape,
        y=class_ids.to(device),
        device=device,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
    )


@app.cell
def _(
    device,
    fashion_mnist_class_names,
    guidance_ui,
    mo,
    num_sample_steps_ui,
    num_samples_ui,
    sample_all_classes_cb,
    sample_btn,
    target_class_ui,
    trained_model: nn.Module | None,
    trained_scheduler: DDPMSchedulerV1 | None,
):
    if trained_model is None or trained_scheduler is None:
        _out = mo.md("_Train the model first (Section 5) before generating samples._")
    elif not sample_btn.value:
        _out = mo.md("Click **Generate Samples** to draw class-conditional samples with CFG.")
    else:
        if sample_all_classes_cb.value:
            per_class = max(1, num_samples_ui.value)
            all_ids = []
            all_labels = []
            for cid in range(len(fashion_mnist_class_names)):
                all_ids.extend([cid] * per_class)
                all_labels.extend([cid] * per_class)
            id_tensor = torch.tensor(all_ids, dtype=torch.long)
        else:
            per_class = max(1, num_samples_ui.value)
            id_tensor = torch.full((per_class,), int(target_class_ui.value), dtype=torch.long)
            all_labels = [int(target_class_ui.value)] * per_class
        samples = generate_samples(
            model=trained_model,
            scheduler=trained_scheduler,
            class_ids=id_tensor,
            device=device,
            guidance_scale=float(guidance_ui.value),
            num_steps=int(num_sample_steps_ui.value),
        )
        _out = plot_generated_grid(samples, fashion_mnist_class_names, labels=all_labels)
    _out
    return


@app.cell
def _(default_guidance_ui, demo_model_kwargs, demo_param_count, mo):
    mo.md(
        f"""
        ### Model comparison

        Only one model variant (`DiffusionTransformerV1`) is defined in this notebook,
        so there is nothing to compare. A summary of the default configuration:

        | Property | Value |
        |----------|-------|
        | Model | `DiffusionTransformerV1` |
        | Hidden size | `{demo_model_kwargs['hidden_size']}` |
        | Depth | `{demo_model_kwargs['depth']}` |
        | Heads | `{demo_model_kwargs['num_heads']}` |
        | Patch size | `{demo_model_kwargs['patch_size']}` |
        | Trainable params | `{demo_param_count:,}` |
        | Default CFG guidance | `{float(default_guidance_ui.value):.2f}` |

        ### Summary

        - We trained a small class-conditional DiT with the DDPM noise-prediction
          objective and adaLN-Zero modulation from `(timestep + class)` conditioning.
        - The `LabelEmbedderV1` has an extra "null" class index; during training it
          is randomly substituted for the true label with probability `class_dropout_prob`,
          enabling classifier-free guidance at sampling time.
        - `DDPMSchedulerV1.p_sample_loop` runs ancestral sampling and, when
          `guidance_scale != 1.0`, dispatches through `model.forward_with_cfg` to
          combine conditional and unconditional noise predictions.
        - All device-sensitive helpers (`run_train_epoch`, `run_evaluate`,
          `evaluate_model`, `generate_samples`, and sampling) take `device` as an
          explicit parameter, so the same code path runs on CPU and CUDA.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9 — Save Trained Model
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="fashion_mnist_dit_ddpm_v1.pt",
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_model_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_model_btn])
    return save_filename_ui, save_model_btn


@app.cell
def _(
    fashion_mnist_class_names,
    mo,
    save_filename_ui,
    save_model_btn,
    trained_kwargs: dict | None,
    trained_model: nn.Module | None,
    trained_scheduler: DDPMSchedulerV1 | None,
):
    if trained_model is None or trained_scheduler is None:
        _out = mo.md("_Train the model first (Section 5) before saving._")
    elif not save_model_btn.value:
        _out = mo.md(
            "Enter a filename and click **Save Model** to write the trained "
            "weights, model kwargs, scheduler config, and class names to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_filename_ui.value
        _payload = {
            "model_state_dict": trained_model.state_dict(),
            "model_kwargs": trained_kwargs,
            "scheduler_config": trained_scheduler.config(),
            "class_names": fashion_mnist_class_names,
        }
        torch.save(_payload, _save_path)
        _out = mo.md(f"**Saved.** Wrote model + config to `{_save_path}`.")
    _out
    return


if __name__ == "__main__":
    app.run()
