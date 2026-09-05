import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    from typing import Optional, Tuple, Literal, List
    from collections.abc import Callable
    from dataclasses import dataclass

    import math

    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.distributions as D
    from torch.nn.utils import clip_grad_norm_

    from geomloss import SamplesLoss

    import ot as pot_lib


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Optimal Transport Pairing in Flow Matching

    In the baseline flow matching approach, we randomly pair latent samples with data samples. This notebook explores how optimal transport (OT) pairings can provide more structured couplings that improve training efficiency and sample quality.
    Motivation

    When training flow matching models, we need to pair samples from the source distribution u1 (latents) with samples from the target distribution u0 (data). The choice of pairing affects:

        Training dynamics: How quickly the model learns
        Flow straightness: Whether trajectories are straight or curved
        Sample quality: The final distribution match
    """)
    return


@app.cell
def _():
    plt.style.use("seaborn-v0_8")
    torch.set_default_dtype(torch.float32)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (DEVICE,)


@app.function
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.cell
def _():
    # optimal transport theory
    return


@app.function
def minibatch_ot_pairing(x0, x1):
    """
    Compute minibatch OT pairing between data (x0) and latents (x1).

    Uses the POT library to solve the optimal transport problem and returns
    the optimal permutation indices.

    Args:
        x0: Data batch (B, D) 
        x1: Latent batch (B, D)

    Returns:
        indices: (B,) permutation indices so x1[indices] is optimally paired with x0
        transport_plan: (B, B) transport matrix from the OT solution
    """

    if x0.shape[0] != x1.shape[0]:
        raise ValueError(f"x0 and x1 must have same batch size. Got {x0.shape[0]} and {x1.shape[0]}")

    device = x0.device

    # Compute cost matrix: squared Euclidean distances
    with torch.no_grad():
        C = torch.cdist(x0, x1, p=2).pow(2).cpu().numpy()

        # Uniform weights for source and target
        a = pot_lib.unif(C.shape[0])
        b = pot_lib.unif(C.shape[1])

        # Solve optimal transport with exact EMD (Earth Mover's Distance)
        transport_plan = pot_lib.emd(a, b, C)

        # Convert back to torch
        P = torch.tensor(transport_plan, dtype=torch.float32, device=device)

        # Extract indices to reorder x1 to match x0
        # Hint: For each x0[i], find which x1[j] should be paired with it
        # P[i,j] represents transport from x0[i] to x1[j]
        # Use argmax on the appropriate dimension of P
        indices = torch.argmax(P, dim=1)

    return indices, P


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualize random vs OT coupling

    To understand the difference between random and OT pairings, we construct a simple toy problem:

        Source: Two Gaussias with modes at (-2, -1) and (2, -1)
        Target: Two Gaussians with modes at (2, 1) and (-2, 1)
    """)
    return


@app.cell
def _(DEVICE):
    set_seed(1)
    np.random.seed(1)

    source_means = torch.tensor([[-2.0, -1.0], [2.0, -1.0]], device=DEVICE, dtype=torch.float32)
    target_means = torch.tensor([[2.0, 1.0], [-2.0, 1.0]], device=DEVICE, dtype=torch.float32)
    cov_matrix = torch.eye(2, device=DEVICE, dtype=torch.float32) * 0.03

    component_distribution = D.MultivariateNormal(
        loc=source_means,
        covariance_matrix=cov_matrix.expand(source_means.shape[0], -1, -1),
    )
    mixture_distribution = D.Categorical(torch.ones(source_means.shape[0], device=DEVICE))
    base_sampler = D.MixtureSameFamily(mixture_distribution, component_distribution)

    component_distribution_target = D.MultivariateNormal(
        loc=target_means,
        covariance_matrix=cov_matrix.expand(target_means.shape[0], -1, -1),
    )
    target_mixture = D.Categorical(torch.ones(target_means.shape[0], device=DEVICE))
    target_sampler = D.MixtureSameFamily(target_mixture, component_distribution_target)

    num_points = 4096
    source_tensor = base_sampler.sample((num_points,))
    target_tensor = target_sampler.sample((num_points,))

    source_np = source_tensor.detach().cpu().numpy()
    target_np = target_tensor.detach().cpu().numpy()

    source_color = '#000000'
    target_color = '#E74C3C'
    line_color = '#0E2ECC'

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(source_np[:, 0], source_np[:, 1], s=6, alpha=0.6, color=source_color, label='source')
    ax.scatter(target_np[:, 0], target_np[:, 1], s=6, alpha=0.6, color=target_color, label='target')

    x_min = min(source_np[:, 0].min(), target_np[:, 0].min())
    x_max = max(source_np[:, 0].max(), target_np[:, 0].max())
    y_min = min(source_np[:, 1].min(), target_np[:, 1].min())
    y_max = max(source_np[:, 1].max(), target_np[:, 1].max())
    range_x = max(x_max - x_min, 1.0)
    range_y = max(y_max - y_min, 1.0)
    pad_x = 0.08 * range_x
    pad_y = 0.08 * range_y

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_title('Two-Gaussian Source and Target')

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return (
        base_sampler,
        line_color,
        source_color,
        source_np,
        source_tensor,
        target_color,
        target_np,
        target_sampler,
        target_tensor,
    )


@app.cell
def _():
    # comparing coupling strategies
    return


@app.function
def to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or array to numpy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


@app.function
def compute_limits(
    arrays: tuple[np.ndarray, ...],
    margin: float = 0.02,
    min_extent: float = 0.1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute axis limits with padding for multiple arrays."""
    data = np.concatenate(arrays, axis=0)
    x_min, y_min = data.min(axis=0)
    x_max, y_max = data.max(axis=0)
    range_x = max(x_max - x_min, min_extent)
    range_y = max(y_max - y_min, min_extent)
    pad_x = margin * range_x
    pad_y = margin * range_y
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)


@app.function
def plot_transport_lines(
    source: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    assignment: np.ndarray,
    *,
    ax=None,
    title: Optional[str] = None,
    source_color: str = "#9467bd",
    target_color: str = "#1f77b4",
    line_color: str = "#ff7f0e",
    alpha: float = 0.35,
) -> plt.Axes:
    """Plot straight-line transport couplings between source and target samples."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    src = to_numpy(source)
    tgt = to_numpy(target)
    ax.scatter(src[:, 0], src[:, 1], s=12, alpha=0.6, color=source_color, label="source")
    ax.scatter(tgt[:, 0], tgt[:, 1], s=12, alpha=0.6, color=target_color, label="target")

    for i, j in enumerate(assignment):
        xs = [src[i, 0], tgt[int(j), 0]]
        ys = [src[i, 1], tgt[int(j), 1]]
        ax.plot(xs, ys, color=line_color, alpha=alpha, linewidth=0.8)

    limits = compute_limits((src, tgt))
    (x_min, x_max), (y_min, y_max) = limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    return ax


@app.cell
def _(
    line_color,
    source_color,
    source_np,
    source_tensor,
    target_color,
    target_np,
    target_tensor,
):
    # Compute OT pairing using our function
    indices_ot, _ = minibatch_ot_pairing(source_tensor, target_tensor)
    assignment_ot = indices_ot.cpu().numpy()

    # Random pairing for comparison
    perm_random = torch.randperm(target_tensor.shape[0]).cpu().numpy()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    plot_transport_lines(
        source_np, target_np, perm_random, 
        ax=axes2[0], 
        title='Random Pairing', 
        source_color=source_color, 
        target_color=target_color, 
        line_color=line_color
    )
    plot_transport_lines(
        source_np, target_np, assignment_ot, 
        ax=axes2[1], 
        title='Optimal Transport Pairing', 
        source_color=source_color, 
        target_color=target_color, 
        line_color=line_color
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### training with different pairings

    First, let's set up our samplers for training.

    🎯 Exercise: Implement OT Pairing in Training

    Before we use the full training utilities, let's understand how OT pairing works by implementing a simplified training loop for the toy example. In the cell below, you'll complete a training function that uses OT pairing - the key difference from random pairing is just 2 lines of code!
    """)
    return


@app.function
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA update helper for model parameters."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


@app.class_definition
class SinusoidalTimeEmbedding(nn.Module):
    """Fourier-style embedding for scalar time inputs."""

    def __init__(self, embed_dim: int, max_freq: float = 1000.0) -> None:
        super().__init__()
        half = embed_dim // 2
        freq = torch.logspace(0, math.log10(max_freq), steps=half)
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1).float()
        angles = t * self.freq.view(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


@app.class_definition
class VelocityMLP(nn.Module):
    """Time-conditioned MLP that predicts the velocity field."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Build simple feedforward network
        layers = []
        layers.append(nn.Linear(input_dim + time_embed_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.expand(x.shape[0], -1)

        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


@app.cell
def _():
    BatchSampler = Callable[[int, torch.device, torch.dtype], torch.Tensor]
    return (BatchSampler,)


@app.cell
def _(BatchSampler, DEVICE):
    def make_batch_sampler(sampler_like) -> BatchSampler:
        """Adapt a distribution or callable into a batch sampler returning fresh draws."""

        def sample(n: int, *, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            target_device = torch.device(device)
            sample_attr = getattr(sampler_like, "sample", None)
            if sample_attr is None:
                try:
                    result = sampler_like(n, device=target_device, dtype=dtype)
                except TypeError:
                    result = sampler_like(n)
            else:
                try:
                    result = sample_attr(n, device=target_device, dtype=dtype)
                except TypeError:
                    try:
                        result = sample_attr((n,))
                    except TypeError:
                        result = sample_attr(n)
            if not isinstance(result, torch.Tensor):
                result = torch.as_tensor(result, dtype=dtype)
            return result.to(device=target_device, dtype=dtype)

        return sample

    return (make_batch_sampler,)


@app.cell
def _(base_sampler, make_batch_sampler, target_sampler):
    toy_source_sampler = make_batch_sampler(base_sampler)
    toy_target_sampler = make_batch_sampler(target_sampler)
    return toy_source_sampler, toy_target_sampler


@app.cell
def _(DEVICE, toy_source_sampler):
    def toy_latent_sampler(shape: tuple[int, ...]) -> torch.Tensor:
        if len(shape) != 2 or shape[1] != 2:
            raise ValueError('Expected latent shape (batch, 2) for the toy example.')
        return toy_source_sampler(shape[0], device=DEVICE, dtype=torch.float32)

    return (toy_latent_sampler,)


@app.cell
def _(DEVICE, toy_latent_sampler, toy_target_sampler):
    def train_toy_with_ot(num_steps=5000, batch_size=128, lr=5e-4):
        """
        Simplified training loop with OT pairing for the toy two-Gaussian example.

        This function is almost identical to random pairing training, except for
        the OT pairing step. Your task: complete the 2 TODO sections below.
        """
        set_seed(42)

        # Setup model and optimizer (provided)
        model = VelocityMLP(input_dim=2, hidden_dim=64).to(DEVICE)
        ema_model = VelocityMLP(input_dim=2, hidden_dim=64).to(DEVICE)
        ema_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []

        for step in range(num_steps):
            # Sample from target (x0) and source/latent (x1) distributions
            x0 = toy_target_sampler(batch_size, device=DEVICE, dtype=torch.float32)
            x1 = toy_latent_sampler((batch_size, 2))

            # TODO 1: Compute OT pairing between x0 and x1
            # Review the minibatch_ot_pairing function signature above
            # What does it return?
            indices, _ = minibatch_ot_pairing(x0, x1)

            # TODO 2: Reorder x1 according to the optimal transport plan
            # Use the indices from the OT solution to permute x1 to match x0
            x1 = x1[indices]

            # Rest of the training loop is identical to random pairing (provided)
            t = torch.rand(batch_size, 1, device=DEVICE)
            x_t = (1.0 - t) * x0 + t * x1
            velocity_target = x1 - x0

            pred = model(t, x_t)
            loss = ((pred - velocity_target) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model, 0.9)

            losses.append(loss.detach().cpu().item())
            if (step + 1) % 1000 == 0:
                print(f"step {step + 1:5d} | loss = {losses[-1]:.6f}")

        return {'model': model, 'ema': ema_model, 'losses': losses}


    # Test your implementation
    print("Training toy example with OT pairing (your implementation)...")
    toy_ot_manual = train_toy_with_ot(num_steps=5000, batch_size=128)
    print("✓ Training complete!")
    return (toy_ot_manual,)


@app.cell
def _():
    LatentSampler = Callable[[Tuple[int, ...]], torch.Tensor]
    return (LatentSampler,)


@app.cell
def _(BatchSampler, LatentSampler):
    @dataclass
    class FlowConfig:
        """Configuration for flow matching training."""
        target_sampler: BatchSampler
        latent_sampler: LatentSampler
        flow_T: float = 1.0
        seed: int = 3
        lr: float = 5e-4
        ema_decay: float = 0.9
        batch_size: int = 128
        steps: int = 50000
        log_every: int = 200
        grad_clip: float = 1.0
        pairing: Literal["none", "minibatch_ot"] = "none"
        minibatch_ot_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None

    return (FlowConfig,)


@app.cell
def _(FlowConfig):
    @dataclass
    class FlowRun:
        """Container for training artifacts."""
        config: FlowConfig
        model: nn.Module
        ema: nn.Module
        losses: List[float]
        pairing_costs: List[float]

    return (FlowRun,)


@app.cell
def _(DEVICE, FlowConfig, FlowRun):
    def train_flow(cfg: FlowConfig) -> FlowRun:
        """Train a 2D flow-matching model with optional OT pairing."""
        set_seed(cfg.seed)
        model = VelocityMLP(input_dim=2).to(DEVICE)
        ema_model = VelocityMLP(input_dim=2).to(DEVICE)
        ema_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        losses: List[float] = []
        pairing_costs: List[float] = []

        for step in range(cfg.steps):
            x_data = cfg.target_sampler(cfg.batch_size, device=DEVICE, dtype=torch.float32)
            z = cfg.latent_sampler((cfg.batch_size, 2))

            if cfg.pairing == "minibatch_ot":
                if cfg.minibatch_ot_fn is None:
                    raise ValueError("`minibatch_ot_fn` must be provided when pairing='minibatch_ot'.")
                cost_matrix = torch.cdist(x_data, z, p=2).pow(2)
                indices, transport_plan = cfg.minibatch_ot_fn(x_data, z)
                pairing_costs.append(float((transport_plan * cost_matrix).sum().item()))
                z = z[indices]

            t = torch.rand(cfg.batch_size, 1, device=DEVICE)
            x_t = (1.0 - t) * x_data + t * z
            velocity_target = z - x_data

            pred = model(t, x_t)
            loss = ((pred - velocity_target) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            update_ema(ema_model, model, cfg.ema_decay)

            losses.append(loss.detach().cpu().item())
            if (step + 1) % cfg.log_every == 0:
                msg = f"step {step + 1:5d} | loss = {losses[-1]:.6f}"
                if cfg.pairing == "minibatch_ot" and pairing_costs:
                    msg += f" | OT cost ≈ {pairing_costs[-1]:.4f}"
                print(msg)

        return FlowRun(cfg, model, ema_model, losses, pairing_costs)

    return (train_flow,)


@app.cell
def _(FlowConfig, toy_latent_sampler, toy_target_sampler, train_flow):
    # Configuration for toy example without OT
    toy_config_no_ot = {
        'target_sampler': toy_target_sampler,
        'latent_sampler': toy_latent_sampler,
        'steps': 5000,
        'log_every': 1000,
        'batch_size': 128,
    }

    print("Training without OT pairing...")
    toy_no_ot_run = train_flow(FlowConfig(**toy_config_no_ot))

    # Configuration for toy example with OT
    toy_config_ot = {
        'target_sampler': toy_target_sampler,
        'latent_sampler': toy_latent_sampler,
        'pairing': 'minibatch_ot',
        'minibatch_ot_fn': minibatch_ot_pairing,
        'steps': 5000,
        'log_every': 1000,
        'batch_size': 128,
    }
    print("\nTraining with OT pairing...")
    toy_ot_run = train_flow(FlowConfig(**toy_config_ot))
    return toy_no_ot_run, toy_ot_run


@app.function
def compute_ema(values: list[float], alpha: float = 0.05) -> list[float]:
    """Compute exponential moving average for loss plotting.

    Args:
        values: List of scalar values (e.g., training losses)
        alpha: Smoothing factor (0 < alpha <= 1). Lower = more smoothing.
               Typical values: 0.01 (very smooth), 0.05 (moderate)

    Returns:
        List of EMA values with same length as input
    """
    if not values:
        return []
    ema = [values[0]]
    for val in values[1:]:
        ema.append(alpha * val + (1 - alpha) * ema[-1])
    return ema


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### visualizing training loss

    Let's plot the training curves:
    """)
    return


@app.cell
def _(toy_no_ot_run, toy_ot_manual, toy_ot_run):
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4))

    # Left plot: Random vs Your OT implementation
    axes3[0].plot(toy_no_ot_run.losses, color='#D62728', alpha=0.2, linewidth=0.5)
    axes3[0].plot(compute_ema(toy_no_ot_run.losses), label="Random pairing", color='#D62728', linewidth=2)
    axes3[0].plot(toy_ot_manual['losses'], color='#2CA02C', alpha=0.2, linewidth=0.5)
    axes3[0].plot(compute_ema(toy_ot_manual['losses']), label="OT pairing (your implementation)", color='#2CA02C', linewidth=2)
    axes3[0].set_xlabel("Training step")
    axes3[0].set_ylabel("MSE loss")
    axes3[0].set_title("Your Implementation")
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    # Right plot: Random vs train_flow OT
    axes3[1].plot(toy_no_ot_run.losses, color='#D62728', alpha=0.2, linewidth=0.5)
    axes3[1].plot(compute_ema(toy_no_ot_run.losses), label="Random pairing", color='#D62728', linewidth=2)
    axes3[1].plot(toy_ot_run.losses, color='#9467BD', alpha=0.2, linewidth=0.5)
    axes3[1].plot(compute_ema(toy_ot_run.losses), label="OT pairing (train_flow)", color='#9467BD', linewidth=2)
    axes3[1].set_xlabel("Training step")
    axes3[1].set_ylabel("MSE loss")
    axes3[1].set_title("Reference Implementation")
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n✓ If your implementation is correct, both plots should look similar!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing Generated Samples and Trajectories

    Let's examine the generated distributions and flow paths for both models:
    """)
    return


@app.function
def euler_integrate(
    model: nn.Module,
    z0: torch.Tensor,
    flow_T: float,
    *,
    steps: int = 120,
    return_path: bool = False,
) -> torch.Tensor:
    """Integrate the learned velocity field with explicit Euler method."""
    times = torch.linspace(flow_T, 0.0, steps, device=z0.device)
    x = z0
    if return_path:
        states = [x]
    for i in range(len(times) - 1):
        t_curr = times[i].expand(z0.shape[0], 1)
        dt = times[i + 1] - times[i]
        velocity = model(t_curr / flow_T, x)
        x = x + dt * velocity
        if return_path:
            states.append(x)
    if return_path:
        return torch.stack(states, dim=0)
    return x


@app.function
def plot_points(
    ax,
    samples: torch.Tensor | np.ndarray,
    title: str,
    color: str,
    *,
    alpha: float = 0.6,
    limits: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
) -> None:
    """Scatter plot helper with consistent styling."""
    pts = to_numpy(samples)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=alpha, color=color, linewidths=0)
    ax.set_title(title)
    if limits is None:
        limits = compute_limits((pts,))
    (x_min, x_max), (y_min, y_max) = limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', 'box')

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))


@app.function
def plot_source_target_trajectories(
    source: torch.Tensor,
    trajectories: torch.Tensor,
    *,
    ax=None,
    target: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
    max_paths: int = 128,
    latent_color: str = "#000000",
    endpoint_color: str = "#1f77b4",
    path_color: str = "#90EE90",
) -> plt.Axes:
    """Overlay latent samples, trajectories, and endpoints (with optional background).

    Args:
        source: Latent samples at t=1 (N, 2)
        trajectories: Full trajectory (T, N, 2)
        ax: Matplotlib axis to plot on
        target: Optional target samples to show
        background: Optional background samples
        title: Plot title
        max_paths: Maximum number of paths to draw
        latent_color: Color for latent points (default: black)
        endpoint_color: Color for trajectory endpoints (default: blue)
        path_color: Color for trajectory paths (default: light green)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    src = to_numpy(source)
    traj = to_numpy(trajectories)
    if traj.ndim != 3:
        raise ValueError("Trajectories tensor must have shape (T, N, 2).")
    finals = traj[-1]

    to_concat = [src, finals]
    if background is not None:
        bg = _to_numpy(background)
        to_concat.append(bg)
    if target is not None:
        to_concat.append(to_numpy(target))

    limits = compute_limits(tuple(to_concat))

    ax.scatter(src[:, 0], src[:, 1], s=10, alpha=0.45, color=latent_color, label="latents $z$")

    if background is not None:
        ax.scatter(bg[:, 0], bg[:, 1], s=8, alpha=0.15, color="#b0bec5", label="target samples")

    if target is None:
        target = finals
    tgt = to_numpy(target)
    ax.scatter(tgt[:, 0], tgt[:, 1], s=16, alpha=0.7, color=endpoint_color, label="traj endpoint")

    n_paths = min(max_paths, traj.shape[1])
    for idx in range(n_paths):
        path = traj[:, idx, :]
        ax.plot(path[:, 0], path[:, 1], color=path_color, alpha=0.35, linewidth=1.0)

    (x_min, x_max), (y_min, y_max) = limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    return ax


@app.cell
def _(
    DEVICE,
    line_color,
    source_color,
    target_color,
    toy_latent_sampler,
    toy_no_ot_run,
    toy_ot_manual,
    toy_target_sampler,
):
    # Random Pairing
    latents = toy_no_ot_run.config.latent_sampler((256, 2))
    with torch.no_grad():
        trajectories = euler_integrate(toy_no_ot_run.ema, latents, toy_no_ot_run.config.flow_T, steps=120, return_path=True)

    target_samples = toy_no_ot_run.config.target_sampler(4096, device=DEVICE, dtype=torch.float32)
    combined = torch.cat([target_samples, trajectories[-1], latents], dim=0).detach().cpu().numpy()
    x_min4, y_min4 = combined.min(axis=0)
    x_max4, y_max4 = combined.max(axis=0)
    pad_x4 = 0.08 * max(x_max4 - x_min4, 1.0)
    pad_y4 = 0.08 * max(y_max4 - y_min4, 1.0)
    limits = ((x_min4 - pad_x4, x_max4 + pad_x4), (y_min4 - pad_y4, y_max4 + pad_y4))

    fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
    plot_points(axes4[0], target_samples, 'Random Pairing: Target', target_color, limits=limits)
    plot_points(axes4[1], trajectories[-1], 'Random Pairing: Generated', line_color, limits=limits)
    plot_points(axes4[2], latents, 'Random Pairing: Latent', source_color, limits=limits)
    plt.tight_layout()
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(6, 5))
    plot_source_target_trajectories(
        source=latents, trajectories=trajectories, target=trajectories[-1],
        title='Random Pairing: Trajectories', ax=ax4,
        latent_color=source_color, endpoint_color=target_color, path_color=line_color
    )
    plt.tight_layout()
    plt.show()

    # OT Pairing (your implementation)
    latents = toy_latent_sampler((256, 2))
    with torch.no_grad():
        trajectories = euler_integrate(toy_ot_manual['ema'], latents, 1.0, steps=120, return_path=True)

    target_samples = toy_target_sampler(4096, device=DEVICE, dtype=torch.float32)
    combined = torch.cat([target_samples, trajectories[-1], latents], dim=0).detach().cpu().numpy()
    x_min4, y_min4 = combined.min(axis=0)
    x_max4, y_max4 = combined.max(axis=0)
    pad_x4 = 0.08 * max(x_max4 - x_min4, 1.0)
    pad_y4 = 0.08 * max(y_max4 - y_min4, 1.0)
    limits = ((x_min4 - pad_x4, x_max4 + pad_x4), (y_min4 - pad_y4, y_max4 + pad_y4))

    fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
    plot_points(axes4[0], target_samples, 'OT Pairing: Target', target_color, limits=limits)
    plot_points(axes4[1], trajectories[-1], 'OT Pairing: Generated', line_color, limits=limits)
    plot_points(axes4[2], latents, 'OT Pairing: Latent', source_color, limits=limits)
    plt.tight_layout()
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(6, 5))
    plot_source_target_trajectories(
        source=latents, trajectories=trajectories, target=trajectories[-1],
        title='OT Pairing: Trajectories', ax=ax4,
        latent_color=source_color, endpoint_color=target_color, path_color=line_color
    )
    plt.tight_layout()
    plt.show()
    return


@app.function
def compute_distribution_metrics(
    generated_samples: torch.Tensor,
    target_samples: torch.Tensor,
    blur: float = 0.01,
) -> dict:
    """Compute multiple distribution comparison metrics.

    Args:
        generated_samples: Generated samples (N, D)
        target_samples: Target distribution samples (M, D)
        blur: Regularization for Sinkhorn

    Returns:
        Dictionary with metric names and values
    """
    metrics = {}

    # Sinkhorn divergence (W2 approximation)
    if SamplesLoss is not None:
        try:
            metrics['sinkhorn_div'] = compute_w2_distance(generated_samples, target_samples, blur=blur)
        except Exception as e:
            print(f"Warning: Could not compute Sinkhorn divergence: {e}")

    # Simple statistics
    gen_mean = generated_samples.mean(dim=0)
    target_mean = target_samples.mean(dim=0)
    metrics['mean_error'] = float(torch.norm(gen_mean - target_mean).item())

    gen_std = generated_samples.std(dim=0)
    target_std = target_samples.std(dim=0)
    metrics['std_error'] = float(torch.norm(gen_std - target_std).item())

    return metrics


@app.function
def compute_w2_distance(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    blur: float = 0.01,
) -> float:
    """Compute Sinkhorn divergence (approximation of W2) between two sample sets.
    
    Args:
        samples1: First sample set (N, D)
        samples2: Second sample set (M, D)
        blur: Regularization parameter for Sinkhorn
        
    Returns:
        Sinkhorn divergence value
    """
    if SamplesLoss is None:
        raise ImportError("geomloss is required for W2 distance computation. Install with: pip install geomloss")
    
    loss = SamplesLoss("sinkhorn", p=2, blur=blur, backend="tensorized")
    
    # Ensure same device
    device = samples1.device
    samples2 = samples2.to(device)
    
    with torch.no_grad():
        distance = loss(samples1, samples2)
    
    return float(distance.item())


@app.cell
def _(
    DEVICE,
    toy_latent_sampler,
    toy_no_ot_run,
    toy_ot_manual,
    toy_target_sampler,
):
    # Random Pairing metrics
    latents5 = toy_no_ot_run.config.latent_sampler((2048, 2))
    target_samples5 = toy_no_ot_run.config.target_sampler(2048, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        generated = euler_integrate(toy_no_ot_run.ema, latents5, toy_no_ot_run.config.flow_T, steps=150)
    metrics = compute_distribution_metrics(generated, target_samples5, blur=0.01)
    print("\nRandom Pairing:")
    if 'sinkhorn_div' in metrics:
        print(f"  Sinkhorn divergence: {metrics['sinkhorn_div']:.6f}")
    print(f"  Mean error: {metrics['mean_error']:.6f}")
    print(f"  Std error: {metrics['std_error']:.6f}")

    # OT Pairing metrics (your implementation)
    latents5 = toy_latent_sampler((2048, 2))
    target_samples5 = toy_target_sampler(2048, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        generated = euler_integrate(toy_ot_manual['ema'], latents5, 1.0, steps=150)
    metrics = compute_distribution_metrics(generated, target_samples5, blur=0.01)
    print("\nOT Pairing (your implementation):")
    if 'sinkhorn_div' in metrics:
        print(f"  Sinkhorn divergence: {metrics['sinkhorn_div']:.6f}")
    print(f"  Mean error: {metrics['mean_error']:.6f}")
    print(f"  Std error: {metrics['std_error']:.6f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### quantitative evaluation

    Let's compute distribution metrics to quantify the quality of the learned flows:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Another Synthetic Example: Two Moons Distribution

    Let's apply OT pairing to another 2D distribution - the two moons - to explore how it affects training on complex, curved manifolds.
    """)
    return


@app.class_definition
class BaseDistribution2D:
    """Minimal interface shared by the toy 2D distributions."""

    has_log_prob: bool = False

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Analytic log-density not available for this distribution.")


@app.cell
def _(CheckerboardStripes, GridGMM9, NealFunnel2D, TwoMoons, ZScoreWrapper):
    def get_distribution(name: str, **kwargs) -> BaseDistribution2D:
        """Instantiate one of the benchmark 2D distributions."""
        name_l = name.lower()
        if name_l in {"checker", "checkerboard"}:
            return CheckerboardStripes(**kwargs)
        if name_l in {"gridgmm", "gridgmm9"}:
            return GridGMM9(**kwargs)
        if name_l in {"twomoons", "two_moons"}:
            return TwoMoons(**kwargs)
        if name_l in {"funnel", "nealfunnel"}:
            return ZScoreWrapper(NealFunnel2D(**kwargs))
        raise ValueError(f"Unknown distribution '{name}'.")

    return (get_distribution,)


@app.cell
def _(DEVICE):
    class TwoMoons(BaseDistribution2D):
        """2D two moons distribution - sample only (no analytic log-prob)."""

        has_log_prob: bool = False

        def __init__(self, noise: float = 0.05) -> None:
            self.noise = float(noise)

        def sample(
            self,
            n: int,
            *,
            device: Optional[torch.device | str] = None,
            dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            device = torch.device(device) if device is not None else DEVICE
            dtype = dtype or torch.get_default_dtype()
            n1 = n // 2
            n2 = n - n1
            # Moon 1: centered at (0,0), radius 1, theta ∈ [0,π]
            theta1 = math.pi * torch.rand(n1, device=device, dtype=dtype)
            x1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=-1)
            # Moon 2: shift/flip to interleave
            theta2 = math.pi * torch.rand(n2, device=device, dtype=dtype)
            x2 = torch.stack([1.0 - torch.cos(theta2), 1.0 - torch.sin(theta2) - 0.5], dim=-1)
            x = torch.cat([x1, x2], dim=0)
            if self.noise > 0:
                x = x + self.noise * torch.randn_like(x)
            return x

    return (TwoMoons,)


@app.function
def make_latent_sampler(
    name: str | list[tuple[str, dict]],
    device: torch.device,
    dim: int,
    **kwargs
) -> Callable[[Tuple[int, ...]], torch.Tensor]:
    """Factory for latent distributions with optional per-dimension control.

    This function supports both IID sampling (same distribution across all dimensions)
    and non-IID sampling (different distributions per dimension) for componentwise
    noise adaptation to target geometry.

    Args:
        name: Either:
            - String: IID distribution across all dimensions ("gaussian", "uniform", "student_t")
            - List of (dist_name, params) tuples: Different distribution per dimension
        device: Device to create tensors on
        dim: Dimensionality
        **kwargs: Parameters for IID case only (e.g., df=4.0, scale=1.0)

    Examples:
        # IID Gaussian (both dimensions N(0,1))
        >>> sampler = make_latent_sampler("gaussian", DEVICE, 2)

        # IID Student-t with custom df
        >>> sampler = make_latent_sampler("student_t", DEVICE, 2, df=4.0, scale=1.0)

        # Non-IID: different std per dimension
        >>> sampler = make_latent_sampler([
        ...     ("gaussian", {"std": 1.0}),
        ...     ("gaussian", {"std": 2.0})
        ... ], DEVICE, 2)

        # Non-IID: different distributions per dimension (for funnel)
        >>> sampler = make_latent_sampler([
        ...     ("student_t", {"df": 20, "scale": 1}),  # lighter tails for x1
        ...     ("student_t", {"df": 4, "scale": 1})    # heavier tails for x2
        ... ], DEVICE, 2)

    Returns:
        Sampler function that takes shape (batch_size, dim) and returns samples
    """
    # IID case: same distribution for all dimensions
    if isinstance(name, str):
        name_l = name.lower()

        if name_l in {"gaussian", "normal"}:
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            def sample(shape: Tuple[int, ...]) -> torch.Tensor:
                return torch.randn(*shape, device=device) * std + mean
            return sample

        elif name_l in {"uniform", "uni"}:
            low = kwargs.get("low", -2.0)
            high = kwargs.get("high", 2.0)
            def sample(shape: Tuple[int, ...]) -> torch.Tensor:
                return torch.rand(*shape, device=device) * (high - low) + low
            return sample

        elif name_l in {"student_t", "studentt"}:
            df = kwargs.get("df", 4.0)
            scale = kwargs.get("scale", 1.0)
            loc = kwargs.get("loc", 0.0)
            def sample(shape: Tuple[int, ...]) -> torch.Tensor:
                # Create distribution with parameters on the correct device
                dist = torch.distributions.StudentT(
                    df=torch.tensor(df, device=device),
                    loc=torch.tensor(loc, device=device),
                    scale=torch.tensor(scale, device=device)
                )
                return dist.sample(shape)
            return sample

        else:
            raise ValueError(f"Unknown latent distribution '{name}'")

    # Non-IID case: different distribution per dimension
    elif isinstance(name, list):
        if len(name) != dim:
            raise ValueError(
                f"Length of name list ({len(name)}) must match dim ({dim}). "
                f"Provide one (dist_name, params) tuple per dimension."
            )

        # Create individual samplers for each dimension
        samplers = []
        for spec in name:
            if isinstance(spec, tuple) and len(spec) == 2:
                dist_name, params = spec
            else:
                raise ValueError(
                    f"Each element must be a (dist_name, params) tuple, got {spec}"
                )

            # Create 1D sampler with specified parameters
            sampler_1d = _make_1d_sampler(dist_name, device, **params)
            samplers.append(sampler_1d)

        def sample(shape: Tuple[int, ...]) -> torch.Tensor:
            if len(shape) != 2 or shape[1] != dim:
                raise ValueError(
                    f"Expected shape (batch_size, {dim}), got {shape}"
                )

            batch_size = shape[0]
            # Sample each dimension independently
            components = [sampler((batch_size, 1)) for sampler in samplers]
            return torch.cat(components, dim=1)

        return sample

    else:
        raise ValueError(
            f"name must be str (for IID) or list of tuples (for non-IID), got {type(name)}"
        )


@app.cell
def _(DEVICE, FlowConfig, get_distribution, make_batch_sampler, train_flow):
    two_moons_distribution = get_distribution("twomoons")
    two_moons_sampler = make_batch_sampler(two_moons_distribution)
    latent_sampler = make_latent_sampler("gaussian", DEVICE, 2)

    # Configuration for training without OT
    config_no_ot = {
        'target_sampler': two_moons_sampler,
        'latent_sampler': latent_sampler,
        'steps': 15000,
        'log_every': 500,
        'batch_size': 128,
    }

    print("Training without OT pairing...")
    no_ot_run = train_flow(FlowConfig(**config_no_ot))

    # Configuration for training with OT
    config_ot = {
        'target_sampler': two_moons_sampler,
        'latent_sampler': latent_sampler,
        'pairing': 'minibatch_ot',
        'minibatch_ot_fn': minibatch_ot_pairing,
        'steps': 15000,
        'log_every': 500,
        'batch_size': 128,
    }

    print("\nTraining with minibatch OT pairing...")
    ot_run = train_flow(FlowConfig(**config_ot))
    return no_ot_run, ot_run, two_moons_sampler


@app.cell
def _(no_ot_run, ot_run):
    fig6, ax6 = plt.subplots(figsize=(7, 4))

    # Random pairing
    ax6.plot(no_ot_run.losses, color='#D62728', alpha=0.2, linewidth=0.5)
    ax6.plot(compute_ema(no_ot_run.losses), label="Random pairing", color='#D62728', linewidth=2)

    # Minibatch OT
    ax6.plot(ot_run.losses, color='#2CA02C', alpha=0.2, linewidth=0.5)
    ax6.plot(compute_ema(ot_run.losses), label="Minibatch OT", color='#2CA02C', linewidth=2)

    ax6.set_xlabel("Training step")
    ax6.set_ylabel("MSE loss")
    ax6.set_title("Two Moons Training Loss")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### training loss comparison

    Let's compare the training dynamics:
    """)
    return


@app.cell
def _(no_ot_run, ot_run):
    fig7, axes7 = plt.subplots(1, 2, figsize=(12, 5))

    # Random pairing trajectories
    latents7 = no_ot_run.config.latent_sampler((256, 2))
    with torch.no_grad():
        trajectories7 = euler_integrate(no_ot_run.ema, latents7, no_ot_run.config.flow_T, steps=120, return_path=True)
    plot_source_target_trajectories(
        source=latents7, trajectories=trajectories7, target=trajectories7[-1],
        title="Random", ax=axes7[0],
        latent_color="#000000", endpoint_color="#1f77b4", path_color="#90EE90"
    )

    # Minibatch OT trajectories
    latents7 = ot_run.config.latent_sampler((256, 2))
    with torch.no_grad():
        trajectories7 = euler_integrate(ot_run.ema, latents7, ot_run.config.flow_T, steps=120, return_path=True)
    plot_source_target_trajectories(
        source=latents7, trajectories=trajectories7, target=trajectories7[-1],
        title="Minibatch OT", ax=axes7[1],
        latent_color="#000000", endpoint_color="#1f77b4", path_color="#90EE90"
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### flow trejectory

    Let's visualize the learned flow paths:
    """)
    return


@app.cell
def _(DEVICE, no_ot_run, ot_run, two_moons_sampler):
    # Random pairing metrics
    latents8 = no_ot_run.config.latent_sampler((2048, 2))
    target_samples8 = two_moons_sampler(2048, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        generated8 = euler_integrate(no_ot_run.ema, latents8, no_ot_run.config.flow_T, steps=150)
    metrics8 = compute_distribution_metrics(generated8, target_samples8, blur=0.01)
    print("\nRandom:")
    if 'sinkhorn_div' in metrics8:
        print(f"  Sinkhorn divergence: {metrics8['sinkhorn_div']:.6f}")
    print(f"  Mean error: {metrics8['mean_error']:.6f}")
    print(f"  Std error: {metrics8['std_error']:.6f}")

    # Minibatch OT metrics
    latents8 = ot_run.config.latent_sampler((2048, 2))
    target_samples8 = two_moons_sampler(2048, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        generated8 = euler_integrate(ot_run.ema, latents8, ot_run.config.flow_T, steps=150)
    metrics8 = compute_distribution_metrics(generated8, target_samples8, blur=0.01)
    print("\nMinibatch OT:")
    if 'sinkhorn_div' in metrics8:
        print(f"  Sinkhorn divergence: {metrics8['sinkhorn_div']:.6f}")
    print(f"  Mean error: {metrics8['mean_error']:.6f}")
    print(f"  Std error: {metrics8['std_error']:.6f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Quantitative Metrics

    Let's compute distribution quality metrics:
    Questions to Consider

        Compare the training losses and sample quality between random and OT pairing. What differences do you observe?
        Look at the flow trajectories. What characterizes the paths learned with OT vs. random pairing?
        Does solving an optimal transport problem at every training step add significant computational overhead? When might it be worth it?

    Next: Conditional Flow Matching

    In the next notebook, we'll explore conditional flow matching and see how these pairing principles extend to the conditional setting with Y-penalized OT.
    """)
    return


if __name__ == "__main__":
    app.run()
