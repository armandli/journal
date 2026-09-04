import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import math
    import random
    from typing import Callable, List, Optional, Tuple

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_


@app.cell
def _():
    import marimo as mo

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
def _(DEVICE):
    set_seed(42)
    print(f"Using device: {DEVICE}")
    return


@app.cell
def _():
    # data distribution
    return


@app.class_definition
class BaseDistribution2D:
    """Minimal interface shared by the toy 2-D distributions."""
    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


@app.cell
def _(DEVICE):
    class TwoMoons(BaseDistribution2D):
        """2D two moons distribution - sample only (no analytic log-prob)."""
        def __init__(self, noise: float = 0.1) -> None:
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


@app.cell
def _(DEVICE):
    class GridGMM9(BaseDistribution2D):
        """Nine-component Gaussian mixture on a 3x3 grid."""
        def __init__(self, spacing: float = 1.0, var: float = 0.01, weights: Optional[list[float]] = None) -> None:
            coords = (-float(spacing), 0.0, float(spacing))
            self.means = torch.tensor([(x, y) for x in coords for y in coords], dtype=torch.get_default_dtype())
            if weights is None:
                weights = [0.01, 0.1, 0.3, 0.2, 0.02, 0.15, 0.02, 0.15, 0.05]
            w = torch.tensor(weights, dtype=torch.get_default_dtype())
            self.weights = (w / w.sum()).tolist()
            self.var = float(var)
            self._log_weights: Optional[torch.Tensor] = None

        def sample(
            self,
            n: int,
            *,
            device: Optional[torch.device | str] = None,
            dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            device = torch.device(device) if device is not None else DEVICE
            dtype = dtype or torch.get_default_dtype()
            cat = torch.distributions.Categorical(probs=torch.tensor(self.weights, device=device, dtype=dtype))
            idx = cat.sample((n,))
            means = self.means.to(device=device, dtype=dtype)
            noise = torch.randn(n, 2, device=device, dtype=dtype) * math.sqrt(self.var)
            return means[idx] + noise

    return (GridGMM9,)


@app.cell
def _(GridGMM9, TwoMoons):
    def get_distribution(name: str, **kwargs) -> BaseDistribution2D:
        """Instantiate one of the benchmark 2D distributions."""
        name_l = name.lower()
        if name_l in {"twomoons", "two_moons"}:
            return TwoMoons(**kwargs)
        if name_l in {"gridgmm", "gridgmm9"}:
            return GridGMM9(**kwargs)
        raise ValueError(f"Unknown distribution '{name}'.")

    return (get_distribution,)


@app.cell
def _():
    # velocity network
    return


@app.class_definition
class SinusoidalTimeEmbedding(nn.Module):
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
    # visualization
    return


@app.cell
def _(DEVICE):
    def draw_samples(distribution: BaseDistribution2D, n: int = 4096) -> torch.Tensor:
        """Sample from a distribution and return on CPU."""
        samples = distribution.sample(n, device=DEVICE, dtype=torch.float32)
        return samples.detach().cpu()

    return (draw_samples,)


@app.function
def plot_points(ax, samples: torch.Tensor, title: str, color: str, *, alpha: float = 0.6) -> None:
    """Simple scatter plot helper for 2D samples with automatic axis limits."""
    pts = samples.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=alpha, color=color, linewidths=0)
    ax.set_title(title)
    
    # Compute axis limits with some padding
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_aspect('equal', 'box')


@app.cell
def _(draw_samples, get_distribution):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Two Moons
    two_moons = get_distribution("twomoons")
    pts = draw_samples(two_moons, n=2048)
    plot_points(axes[0], pts, "Two Moons", "#E74C3C")

    # Grid GMM
    grid_gmm = get_distribution("gridgmm")
    pts = draw_samples(grid_gmm, n=2048)
    plot_points(axes[1], pts, "Grid GMM", "#E74C3C")

    plt.tight_layout()
    plt.show()
    return


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
def _(DEVICE, get_distribution):
    def train_linear_flow(config):
        """Train a flow matching model with linear interpolation."""
        set_seed(config['seed'])
        sampler = get_distribution(config['target_name'])
        model = VelocityMLP(input_dim=config['dim']).to(DEVICE)
        ema_model = VelocityMLP(input_dim=config['dim']).to(DEVICE)
        ema_model.load_state_dict(model.state_dict())
        latent_sampler = make_latent_sampler(config['latent_name'], DEVICE, config['dim'])

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        losses = []

        for step in range(config['steps']):
            # Sample x0 from data and x1 from latent
            x0 = sampler.sample(config['batch_size'], device=DEVICE, dtype=torch.float32)
            if x0.dim() > 2:
                x0 = x0.view(x0.shape[0], -1)
            x1 = latent_sampler((config['batch_size'], config['dim']))
        
            # 1: Sample random time t ~ Uniform[0,1]
            # What shape do we need? Think about batch_size
            t = torch.rand(config['batch_size'], 1)
        
            # 2: Compute interpolated point x_t
            # Review the "Straight-Line Interpolation" section above for the formula
            x_t = (torch.ones(config['batch_size'], 1) - t) * x0 + t * x1
        
            # 3: Compute target velocity
            # Review the "Straight-Line Interpolation" section: what is dx_t/dt?
            velocity_target = x1 - x0

            # 4: Get model's velocity prediction
            # Check the VelocityMLP.forward() signature - what arguments does it take?
            pred = model(t, x_t)
        
            # 5: Compute MSE loss
            # Structure: compute squared differences, then aggregate appropriately
            loss = nn.functional.mse_loss(pred, velocity_target)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            update_ema(ema_model, model, config.get('ema_decay', 0.9))

            losses.append(loss.detach().cpu().item())
            if (step + 1) % config['log_every'] == 0:
                print(f"step {step + 1:5d} | loss = {losses[-1]:.6f}")

        return {
            'config': config,
            'sampler': sampler,
            'model': model,
            'ema': ema_model,
            'losses': losses,
            'latent_sampler': latent_sampler,
        }

    return (train_linear_flow,)


@app.function
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA update helper for model parameters."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


@app.cell
def _(train_linear_flow):
    config = {
        'target_name': 'twomoons',  # Options: 'gridgmm' or 'twomoons'
        'latent_name': 'gaussian',  # Options: 'gaussian', 'uniform', or 'student_t'
        'dim': 2,
        'flow_T': 1.0,
        'seed': 7,
        'lr': 1e-3,
        'batch_size': 128,
        'steps': 20_000,
        'log_every': 500,
        'grad_clip': 1.0,
    }

    run = train_linear_flow(config)
    return (run,)


@app.cell
def _():
    # training loss
    return


@app.function
def compute_ema(values, alpha=0.05):
    """Compute exponential moving average with given smoothing factor."""
    ema = [values[0]]
    for val in values[1:]:
        ema.append(alpha * val + (1 - alpha) * ema[-1])
    return ema


@app.cell
def _(run):
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    losses = run['losses']
    losses_ema = compute_ema(losses, alpha=0.01)

    ax2.plot(losses, color="#D62728", alpha=0.3, linewidth=0.5, label='Loss')
    ax2.plot(losses_ema, color="#D62728", linewidth=2, label='EMA (α=0.01)')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("MSE loss")
    ax2.set_title("Training loss (linear flow)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    # sampling from learned flow
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
    """
    Integrate the learned velocity field using Euler method.
    
    Starting from latent z0 at time T, integrate backwards to time 0:
        dx/dt = v_t(x)  =>  x_{t+dt} ≈ x_t + dt * v_t(x_t)
    
    Args:
        model: Trained velocity network
        z0: Initial latent samples (batch_size, dim)
        flow_T: Total flow time (usually 1.0)
        steps: Number of integration steps
        return_path: If True, return full trajectory
    
    Returns:
        Final samples at t=0 (or full trajectory if return_path=True)
    """
    times = torch.linspace(flow_T, 0.0, steps, device=z0.device)
    x = z0
    if return_path:
        states = [x]
    
    for i in range(len(times) - 1):
        t_curr = times[i].expand(z0.shape[0], 1)
        dt = times[i + 1] - times[i]
        
        # 1: Predict velocity at current time and position
        # The model takes (time, position) as inputs
        velocity = model(t_curr, x)
        
        # 2: Update position using Euler step
        # Review the ODE integration formula in the docstring above
        x = x + dt * velocity
        
        if return_path:
            states.append(x)
    
    if return_path:
        return torch.stack(states, dim=0)
    return x


@app.cell
def _():
    # comparing generated vs target
    return


@app.cell
def _(draw_samples, run):
    real_samples = draw_samples(run['sampler'], n=4096)
    latents_eval = run['latent_sampler']((4096, run['config']['dim']))
    with torch.no_grad():
        generated = euler_integrate(run['ema'], latents_eval, run['config']['flow_T'], steps=50)

    generated_cpu = generated.detach().cpu()
    latents_cpu = latents_eval.detach().cpu()

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    plot_points(axes3[0], real_samples, 'Target samples', '#E74C3C')
    plot_points(axes3[1], generated_cpu, 'Generated samples', '#1f77b4')
    plot_points(axes3[2], latents_cpu, 'Latent samples', '#000000')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    # visualize flow trejectory
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
        bg = to_numpy(background)
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
def _(draw_samples, run):
    latents_plot = run['latent_sampler']((512, run['config']['dim']))
    target_samples = draw_samples(run['sampler'], n=2048)

    fig4, ax4 = plt.subplots(figsize=(3, 3))
    plot_points(ax4, target_samples, 'Target distribution', '#E74C3C')
    plt.tight_layout()
    plt.show()

    with torch.no_grad():
        trajectory_stack = euler_integrate(
            run['ema'],
            latents_plot,
            run['config']['flow_T'],
            steps=60,
            return_path=True,
        )

    fig4, ax5 = plt.subplots(figsize=(4, 4))
    plot_source_target_trajectories(
        source=latents_plot,
        trajectories=trajectory_stack,
        target=trajectory_stack[-1],
        background=None,
        title='Latents with trajectories',
        ax=ax5,
    
    )
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
