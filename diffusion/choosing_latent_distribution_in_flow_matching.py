import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import math
    import random
    from typing import Callable, List, Optional, Tuple, Literal
    from dataclasses import dataclass

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_

    from geomloss import SamplesLoss



@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing the Latent Distribution in Flow Matching

    A fundamental choice in flow matching is the source distribution from which we sample latent variables. While standard Gaussian noise is the default choice, the geometry and tail behavior of the target distribution may suggest alternative latent distributions.
    Key Question

    Can we improve flow matching by adapting the noise distribution componentwise to match the target geometry?

    In this notebook, we explore this question using Neal's funnel distribution, which exhibits different tail behaviors in each dimension. We'll compare three types of latent distributions:

        Uniform: Bounded support, no tails
        Gaussian: Light tails, standard choice
        Student-t: Heavy tails, with componentwise adaptation
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
    # Define distance metric for comparing distributions
    mmd_loss = SamplesLoss("energy")
    return


@app.cell
def _():
    # Configuration: Number of samples for visualization
    n_samples = 500_000  # Number of samples for visualization plots
    return (n_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding Neal's Funnel
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
def _(DEVICE):
    class NealFunnel2D(BaseDistribution2D):
        """Neal's funnel distribution with analytic log-density."""

        has_log_prob: bool = True

        def __init__(self, sigma1: float = 3.0, alpha: float = 1.0) -> None:
            self.sigma1 = float(sigma1)
            self.alpha = float(alpha)

        def sample(
            self,
            n: int,
            *,
            device: Optional[torch.device | str] = None,
            dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            device = torch.device(device) if device is not None else DEVICE
            dtype = dtype or torch.get_default_dtype()
            x1 = self.sigma1 * torch.randn(n, 1, device=device, dtype=dtype)
            std2 = torch.exp(0.5 * self.alpha * x1)
            x2 = std2 * torch.randn(n, 1, device=device, dtype=dtype)
            return torch.cat([x1, x2], dim=-1)

        def log_prob(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., 0], x[..., 1]
            var2 = torch.exp(self.alpha * x1)
            term1 = -0.5 * ((x1 / self.sigma1) ** 2 + math.log(2 * math.pi * self.sigma1 ** 2))
            term2 = -0.5 * (x2 ** 2 / var2 + math.log(2 * math.pi) + self.alpha * x1)
            return term1 + term2

    return (NealFunnel2D,)


@app.cell
def _(NealFunnel2D):
    class ZScoreWrapper(BaseDistribution2D):
        """Wrap a base sampler to operate in z-scored coordinates for stability."""

        has_log_prob: bool = True

        def __init__(self, base: NealFunnel2D) -> None:
            self.base = base
            self.mean = torch.tensor([0.0, 0.0])
            self.std = torch.tensor(
                [
                    base.sigma1,
                    math.exp(0.25 * (base.alpha ** 2) * (base.sigma1 ** 2)),
                ]
            )

        def sample(self, n: int, *, device=None, dtype=None) -> torch.Tensor:
            raw = self.base.sample(n, device=device, dtype=dtype)
            mean = self.mean.to(device=raw.device, dtype=raw.dtype)
            std = self.std.to(device=raw.device, dtype=raw.dtype)
            return (raw - mean) / std

        def to_raw(self, x: torch.Tensor) -> torch.Tensor:
            """Convert from z-scored coordinates back to raw funnel coordinates."""
            mean = self.mean.to(device=x.device, dtype=x.dtype)
            std = self.std.to(device=x.device, dtype=x.dtype)
            return x * std + mean

        def log_prob(self, x: torch.Tensor) -> torch.Tensor:
            mean = self.mean.to(device=x.device, dtype=x.dtype)
            std = self.std.to(device=x.device, dtype=x.dtype)
            raw = x * std + mean
            base_logp = self.base.log_prob(raw)
            log_det = torch.log(std.abs()).sum()
            return base_logp - log_det

    return (ZScoreWrapper,)


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
def _(DEVICE):
    def draw_samples(distribution: BaseDistribution2D, n: int = 4096) -> torch.Tensor:
        """Sample a batch from the target distribution and detach to CPU."""
        samples = distribution.sample(n, device=DEVICE, dtype=torch.float32)
        return samples.detach().cpu()

    return (draw_samples,)


@app.cell
def _(DEVICE):
    def compute_log_joint_grid_funnel(
        sampler,
        x1_range: tuple[float, float],
        x2_range: tuple[float, float],
        n1: int = 300,
        n2: int = 300,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute log probability on a grid for funnel visualization.

        Args:
            sampler: Distribution sampler with log_prob method (usually the base funnel)
            x1_range: (min, max) for x1 dimension
            x2_range: (min, max) for x2 dimension
            n1: Number of grid points in x1
            n2: Number of grid points in x2

        Returns:
            x1_lin: x1 grid points
            x2_lin: x2 grid points
            logp: Log probabilities on the grid (n1, n2)
        """
        x1_lin = np.linspace(*x1_range, n1)
        x2_lin = np.linspace(*x2_range, n2)
        X2, X1 = np.meshgrid(x2_lin, x1_lin, indexing="xy")
        grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
        grid_tensor = torch.from_numpy(grid).to(dtype=torch.float64, device=DEVICE)
        with torch.no_grad():
            logp = sampler.log_prob(grid_tensor).cpu().numpy().reshape(n1, n2)
        return x1_lin, x2_lin, logp

    return (compute_log_joint_grid_funnel,)


@app.function
def analytic_funnel_x2_pdf(x2_grid: np.ndarray, scale1: float, gh_n: int = 80) -> np.ndarray:
    """Compute the marginal p(x2) via Gauss-Hermite quadrature for Neal's funnel.

    Args:
        x2_grid: Grid of x2 values to evaluate density at
        scale1: Standard deviation parameter (sigma1) of the funnel
        gh_n: Number of Gauss-Hermite quadrature nodes

    Returns:
        Probability densities at x2_grid points
    """
    from numpy.polynomial.hermite import hermgauss

    x2 = x2_grid.astype(np.float64)
    nodes, weights = hermgauss(gh_n)
    x1_vals = (np.sqrt(2.0) * scale1) * nodes
    w_norm = weights / np.sqrt(np.pi)

    var = np.exp(x1_vals)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    var_col = var[:, None]
    coef = inv_sqrt_2pi / np.sqrt(var_col)
    expo = np.exp(-(x2[None, :] ** 2) / (2.0 * var_col))
    pdf_matrix = coef * expo
    pdf = (w_norm[:, None] * pdf_matrix).sum(axis=0)
    return np.maximum(pdf, 1e-300)


@app.cell
def _(compute_log_joint_grid_funnel):
    def plot_funnel_with_marginals(
        samples: torch.Tensor,
        target_sampler,
        *,
        ax=None,
        title: str = "Generated samples on funnel log-density",
        bins_main: int = 200,
        bins_marginal: int = 120,
    ):
        """Create funnel plot with marginal distributions and log-density heatmap.

        This advanced plotting function creates a comprehensive visualization of Neal's funnel:
        - Main plot: Log-density heatmap with generated samples overlaid
        - Top marginal: Histogram of x2 (heavy-tailed dimension) with analytic curve
        - Right marginal: Histogram of x1 (light-tailed dimension)

        Args:
            samples: Generated samples to visualize (N, 2)
            target_sampler: Target distribution (ZScoreWrapper around NealFunnel2D)
            ax: Ignored (function creates its own figure with subplots)
            title: Title for the main plot
            bins_main: Number of bins for density heatmap
            bins_marginal: Number of bins for marginal histograms

        Returns:
            Figure object
        """
        if ax is not None:
            raise ValueError("This helper creates its own figure; pass ax=None.")

        def to_raw(tensor: torch.Tensor) -> torch.Tensor:
            """Convert from z-scored back to raw funnel coordinates."""
            if hasattr(target_sampler, "to_raw"):
                return target_sampler.to_raw(tensor)
            return tensor

        base_sampler = target_sampler.base if hasattr(target_sampler, "base") else target_sampler

        samp_tensor = samples if isinstance(samples, torch.Tensor) else torch.from_numpy(samples)
        gen_raw = to_raw(samp_tensor).to(torch.float64)
        true_raw = base_sampler.sample(gen_raw.shape[0]).to(torch.float64)

        X2_MIN, X2_MAX = -999.0, 999.0
        X1_MIN, X1_MAX = -20.0, 20.0

        x1_d = gen_raw[:, 0].cpu().numpy()
        x2_d = gen_raw[:, 1].cpu().numpy()
        x1_m = true_raw[:, 0].cpu().numpy()
        x2_m = true_raw[:, 1].cpu().numpy()

        bins_x1 = np.linspace(X1_MIN, X1_MAX, bins_marginal)
        bins_x2 = np.linspace(X2_MIN, X2_MAX, bins_marginal)

        fig = plt.figure(figsize=(6, 6), dpi=120)
        gs = plt.GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(gs[1:, :3])
        ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

        teal = "#7fb8c8"
        red = "#e74c3c"

        # Compute and plot log density
        _, _, logp = compute_log_joint_grid_funnel(
            base_sampler, (X1_MIN, X1_MAX), (X2_MIN, X2_MAX), n1=320, n2=360
        )
        ax_main.set_facecolor("black")
        cmap = plt.cm.magma.copy()
        cmap.set_under("black")
        log_floor = -20.0
        vmax = float(np.max(logp))
        ax_main.imshow(
            logp,
            origin="lower",
            extent=[X2_MIN, X2_MAX, X1_MIN, X1_MAX],
            aspect="auto",
            cmap=cmap,
            vmin=log_floor,
            vmax=vmax,
        )

        # Scatter generated samples
        ax_main.scatter(x2_d, x1_d, s=6, alpha=0.5, color=teal, linewidths=0, edgecolors="none")
        ax_main.set_xlabel(r"$x_2$", color="white")
        ax_main.set_ylabel(r"$x_1$", color="white")
        ax_main.set_xlim(X2_MIN, X2_MAX)
        ax_main.set_ylim(X1_MIN, X1_MAX)
        ax_main.set_title(title, color="white")

        # Top marginal (x2) with log scale
        ax_top.set_yscale("log")
        ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0, label="True")
        ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal, label="Generated")
        x2_centers = 0.5 * (bins_x2[:-1] + bins_x2[1:])
        scale1 = float(getattr(base_sampler, "sigma1", torch.tensor(3.0)))
        px2 = analytic_funnel_x2_pdf(x2_centers, scale1=scale1, gh_n=80)
        ax_top.plot(x2_centers, px2, color="#1f77b4", linewidth=2.2, alpha=0.95, label="Analytic")
        ax_top.tick_params(labelbottom=False)
        ax_top.legend(fontsize=8, loc="upper right")
        ax_top.set_ylabel(r"$p(x_2)$", fontsize=9)

        # Right marginal (x1)
        ax_right.hist(
            x1_d,
            bins=bins_x1,
            density=True,
            orientation="horizontal",
            color=teal,
            alpha=0.35,
            edgecolor=teal,
        )
        ax_right.hist(
            x1_m,
            bins=bins_x1,
            density=True,
            orientation="horizontal",
            histtype="step",
            color=red,
            linewidth=2.0,
        )
        ax_right.tick_params(labelleft=False)
        ax_right.set_xlabel(r"$p(x_1)$", fontsize=9)

        return fig

    return (plot_funnel_with_marginals,)


@app.cell
def _(
    draw_samples,
    get_distribution,
    make_batch_sampler,
    n_samples,
    plot_funnel_with_marginals,
):
    set_seed(42)

    # Create Neal's funnel distribution
    funnel = get_distribution("funnel", sigma1=3.0, alpha=1.0)
    funnel_sampler = make_batch_sampler(funnel)
    samples = draw_samples(funnel, n=n_samples)

    # Visualize the funnel with marginals
    fig = plot_funnel_with_marginals(
        samples, 
        funnel,
        title="Neal's Funnel Distribution"
    )
    plt.show()
    plt.close(fig)

    print("Notice: $x_2$ has much heavier tails than $x_1$")
    return funnel, funnel_sampler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### training configuration

    We'll use the same training setup for all experiments to ensure fair comparison.

    By default, we use random pairing between latent and target samples. You can optionally enable `pairing='minibatch_ot'` for structured optimal transport pairings.
    """)
    return


@app.cell
def _(funnel_sampler):
    # Base training configuration
    base_config = {
        'target_sampler': funnel_sampler,
        'latent_sampler': None,  # Will set per experiment
        'pairing': 'none',
        'steps': 20_000,
        'batch_size': 128,
        'log_every': 1000,
        'lr': 2e-4,
        'seed': 42,
        'flow_T': 1.0,
        'ema_decay': 0.99
    }
    return (base_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exploration 1: Uniform Latents

    Let's start with uniform distributions which have bounded support and no tails. We use IID uniform with range [-2, 2] as our baseline.
    """)
    return


@app.function
def make_1d_sampler(name: str, device: torch.device, **kwargs) -> Callable[[Tuple[int, ...]], torch.Tensor]:
    """Helper to create a 1D sampler for a single dimension.

    Args:
        name: Distribution name ("gaussian", "uniform", "student_t")
        device: Device to create tensors on
        **kwargs: Distribution-specific parameters

    Returns:
        Sampler function for 1D samples
    """
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
        raise ValueError(f"Unknown distribution '{name}'")


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
            sampler_1d = make_1d_sampler(dist_name, device, **params)
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
def _(DEVICE, n_samples):
    # Choose your latent sampler here:
    latent_sampler_uniform = make_latent_sampler("uniform", DEVICE, 2)

    # Visualize the latent distribution
    latents = latent_sampler_uniform((n_samples, 2))
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    ax2.scatter(latents.cpu()[:, 0], latents.cpu()[:, 1], s=3, alpha=0.4, color='#3498DB', linewidths=0)
    ax2.set_xlabel('$z_1$')
    ax2.set_ylabel('$z_2$')
    ax2.set_title('Uniform Latent Distribution')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig2, latent_sampler_uniform


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


@app.function
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA update helper for model parameters."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


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
def _(FlowConfig, base_config, latent_sampler_uniform, train_flow):
    print("Training with Uniform latents...")
    print("=" * 60)

    config_uniform = base_config.copy()
    config_uniform['latent_sampler'] = latent_sampler_uniform

    run_uniform = train_flow(FlowConfig(**config_uniform))
    return (run_uniform,)


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
def to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or array to numpy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


@app.function
def plot_latent_colored_by_endpoint_norm(
    latents: torch.Tensor,
    endpoints: torch.Tensor,
    *,
    title: str = "Latent colored by ||x||",
    cmap: str = "viridis",
):
    """Two-panel figure showing latent points colored by endpoint norm.

    This visualization helps understand the transport difficulty:
    - Left panel: Raw latent samples (uncolored)
    - Right panel: Latent samples colored by the norm of their endpoints

    Interpretation:
    - Smooth color gradient → Uniform, easy transport
    - Sharp color changes → Network must stretch/compress heavily

    Args:
        latents: Latent points at t=1 (N, 2)
        endpoints: Corresponding endpoints at t=0 (N, 2)
        title: Title for the right panel
        cmap: Matplotlib colormap name

    Returns:
        Figure object
    """
    L = np.atleast_2d(to_numpy(latents))
    X = np.atleast_2d(to_numpy(endpoints))
    if X.shape[0] == 1:
        norms = np.full(L.shape[0], np.linalg.norm(X[0]))
    else:
        norms = np.linalg.norm(X, axis=1)

    radius = float(np.max(np.abs(L))) * 1.1
    if not np.isfinite(radius) or radius == 0.0:
        radius = 1.0
    limits = (-radius, radius)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120, constrained_layout=True)

    axes[0].scatter(L[:, 0], L[:, 1], s=4, alpha=0.35, color="#888888", linewidths=0)
    axes[0].set_title("Latent samples", fontsize=12)
    axes[0].set_xlabel("$z_1$", fontsize=11)
    axes[0].set_ylabel("$z_2$", fontsize=11)
    axes[0].set_aspect("equal", "box")
    axes[0].set_xlim(*limits)
    axes[0].set_ylim(*limits)
    axes[0].grid(True, alpha=0.2)

    h = axes[1].scatter(L[:, 0], L[:, 1], s=5, c=norms, cmap=cmap, alpha=0.7, linewidths=0)
    axes[1].set_title(title, fontsize=12)
    axes[1].set_xlabel("$z_1$", fontsize=11)
    axes[1].set_ylabel("$z_2$", fontsize=11)
    axes[1].set_aspect("equal", "box")
    axes[1].set_xlim(*limits)
    axes[1].set_ylim(*limits)
    axes[1].grid(True, alpha=0.2)
    cbar = fig.colorbar(h, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("||x||", fontsize=10)

    return fig


@app.cell
def _(
    funnel,
    latent_sampler_uniform,
    n_samples,
    plot_funnel_with_marginals,
    run_uniform,
):
    # Generate samples
    latents3 = latent_sampler_uniform((n_samples, 2))
    with torch.no_grad():
        generated_uniform = euler_integrate(run_uniform.ema, latents3, run_uniform.config.flow_T, steps=150)

    # Advanced funnel plot with marginals
    fig3 = plot_funnel_with_marginals(
        generated_uniform, 
        funnel,
        title="Uniform Latents: Generated vs True Distribution"
    )
    plt.show()
    plt.close(fig3)

    # Latent space analysis
    # Each latent is colored by the norm of its corresponding generated sample (ODE integration endpoint)
    fig3 = plot_latent_colored_by_endpoint_norm(
        latents3,
        generated_uniform,
        title="Uniform: Latent colored by ||x|| (x = generated sample)",
        cmap='plasma'
    )
    plt.show()
    plt.close(fig3)
    return (generated_uniform,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exploration 2: Gaussian Latents

    Now let's try Gaussian distributions which have light tails. We use standard IID Gaussian (mean=0, std=1) - the most common choice in flow matching.
    """)
    return


@app.cell
def _(DEVICE, n_samples):
    # Choose your latent sampler here:
    latent_sampler_gaussian = make_latent_sampler("gaussian", DEVICE, 2)

    # Visualize the latent distribution
    latents4 = latent_sampler_gaussian((n_samples, 2))
    fig4, ax4 = plt.subplots(figsize=(4.5, 4.5))
    ax4.scatter(latents4.cpu()[:, 0], latents4.cpu()[:, 1], s=3, alpha=0.4, color='#3498DB', linewidths=0)
    ax4.set_xlabel('$z_1$')
    ax4.set_ylabel('$z_2$')
    ax4.set_title('Gaussian Latent Distribution')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (latent_sampler_gaussian,)


@app.cell
def _(FlowConfig, base_config, latent_sampler_gaussian, train_flow):
    print("Training with Gaussian latents...")
    print("=" * 60)

    config_gaussian = base_config.copy()
    config_gaussian['latent_sampler'] = latent_sampler_gaussian

    run_gaussian = train_flow(FlowConfig(**config_gaussian))
    return (run_gaussian,)


@app.cell
def _(
    fig2,
    funnel,
    latent_sampler_gaussian,
    n_samples,
    plot_funnel_with_marginals,
    run_gaussian,
):
    # Generate samples
    latents5 = latent_sampler_gaussian((n_samples, 2))
    with torch.no_grad():
        generated_gaussian = euler_integrate(run_gaussian.ema, latents5, run_gaussian.config.flow_T, steps=150)

    # Advanced funnel plot with marginals
    fig5 = plot_funnel_with_marginals(
        generated_gaussian, 
        funnel,
        title="Gaussian Latents: Generated vs True Distribution"
    )
    plt.show()
    plt.close(fig2)

    # Latent space analysis
    # Each latent is colored by the norm of its corresponding generated sample (ODE integration endpoint)
    fig5 = plot_latent_colored_by_endpoint_norm(
        latents5,
        generated_gaussian,
        title="Gaussian: Latent colored by ||x|| (x = generated sample)",
        cmap='plasma'
    )
    plt.show()
    plt.close(fig2)
    return (generated_gaussian,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exploration 3: Student-t Latents (Componentwise Adaptation)
    """)
    return


@app.cell
def _(DEVICE, n_samples):
    # Experiment with Student-t configurations!
    # 
    # Try different degrees of freedom (df) to match the funnel's tail behavior.
    # Lower df = heavier tails, Higher df = lighter tails (approaches Gaussian)
    #
    # Some suggestions to try:
    # - Symmetric: df=10 for both dimensions
    # - Asymmetric: df=20 for x1 (light), df=4 for x2 (heavy)
    # - Extreme: df=100 for x1 (very light), df=2 for x2 (very heavy)

    latent_sampler_studentt = make_latent_sampler([
        ("student_t", {"df": 100., "scale": 1.}),  # Adjust df for x1
        ("student_t", {"df": 2., "scale": 1.})   # Adjust df for x2
    ], DEVICE, 2)

    # Visualize the latent distribution
    latents6 = latent_sampler_studentt((n_samples, 2))
    fig6, ax6 = plt.subplots(figsize=(4.5, 4.5))
    ax6.scatter(latents6.cpu()[:, 0], latents6.cpu()[:, 1], s=3, alpha=0.4, color='#3498DB', linewidths=0)
    ax6.set_xlabel('$z_1$')
    ax6.set_ylabel('$z_2$')
    ax6.set_title('Student-t Latent Distribution')
    ax6.set_aspect('equal','box')
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Experiment with different df values and observe how the tail behavior changes!")
    return (latent_sampler_studentt,)


@app.cell
def _(FlowConfig, base_config, latent_sampler_studentt, train_flow):
    print("Training with Student-t latents (componentwise adaptation)...")
    print("=" * 60)

    config_studentt = base_config.copy()
    config_studentt['latent_sampler'] = latent_sampler_studentt

    run_studentt = train_flow(FlowConfig(**config_studentt))
    return (run_studentt,)


@app.cell
def _(
    funnel,
    latent_sampler_studentt,
    n_samples,
    plot_funnel_with_marginals,
    run_studentt,
):
    # Generate samples
    latents7 = latent_sampler_studentt((n_samples, 2))
    with torch.no_grad():
        generated_studentt = euler_integrate(run_studentt.ema, latents7, run_studentt.config.flow_T, steps=150)

    # Advanced funnel plot with marginals
    fig7 = plot_funnel_with_marginals(
        generated_studentt, 
        funnel,
        title="Student-t (Non-IID): Generated vs True Distribution"
    )
    plt.show()
    plt.close(fig7)

    # Latent space analysis
    # Each latent is colored by the norm of its corresponding generated sample (ODE integration endpoint)
    fig7 = plot_latent_colored_by_endpoint_norm(
        latents7,
        generated_studentt,
        title="Student-t: Latent colored by ||x|| (x = generated sample)",
        cmap='plasma'
    )
    plt.show()
    plt.close(fig7)
    return (generated_studentt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison: Does Componentwise Adaptation Help?

    Let's compare all three approaches to see which one works best for the funnel distribution.
    """)
    return


@app.cell
def _(
    funnel,
    generated_gaussian,
    generated_studentt,
    generated_uniform,
    plot_funnel_with_marginals,
):
    # Compare all three approaches with detailed funnel plots
    for name, generated, color in [
        ("Uniform Latents", generated_uniform, '#E74C3C'),
        ("Gaussian Latents", generated_gaussian, '#3498DB'),
        ("Student-t (Non-IID)", generated_studentt, '#2ECC71')
    ]:
        print(f"\n{name}")
        print("-" * 50)
        fig8 = plot_funnel_with_marginals(
            generated,
            funnel,
            title=f"{name}: Generated vs True Distribution"
        )
        plt.show()
        plt.close(fig8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why Do Tails Matter?

    Key Questions to Consider:

        Why might matching the tail behavior between latent and target distributions matter?
        What does it imply for the velocity field when tails don't match? When they do match?

    Next: Optimal Transport Pairing

    So far we've used random pairing between latent and target samples. In the next notebook, we'll explore minibatch optimal transport to create structured pairings that can further improve learning.
    """)
    return


if __name__ == "__main__":
    app.run()
