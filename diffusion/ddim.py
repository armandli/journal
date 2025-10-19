import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from tqdm import tqdm

    import math
    import random

    import numpy as np

    import torch
    from torch import nn
    from torch import optim
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision import transforms
    from torchvision.transforms import ToTensor

    from datasets import load_dataset
    from torchvision.transforms import (CenterCrop, Compose,InterpolationMode,RandomHorizontalFlip, Resize)

    import matplotlib.pyplot as plt
    return (
        CenterCrop,
        Compose,
        DataLoader,
        InterpolationMode,
        RandomHorizontalFlip,
        Resize,
        ToTensor,
        datasets,
        load_dataset,
        math,
        nn,
        np,
        plt,
        torch,
        torchvision,
        tqdm,
        transforms,
    )


@app.cell
def _(torch):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_built()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    cpu = torch.device("cpu")
    return cpu, device


@app.cell
def _():
    ### unet models
    return


@app.cell
def _():
    model_dir = 'models/'
    return (model_dir,)


@app.cell
def _(torch):
    def check_nan(mtx):
        return torch.sum(torch.isnan(mtx)) > 0
    return


@app.cell
def _(nn, torch):
    class PositionEmbeddingV1(nn.Module):
        def __init__(self, edim, device=torch.device("cpu")):
            super(PositionEmbeddingV1, self).__init__()
            assert edim % 2 == 0, f"position embedding require dimension divisible by 2, edim provided={edim}"
            self.device = device
            self.dim = edim
            # factor = 10000 ^ (i / dim) for i in 0 to dim/2-1
            self.factor = torch.pow(
                10_000.,
                torch.div(torch.arange(start=0, end=self.dim//2, dtype=torch.float32), self.dim // 2)
            ).to(self.device)

        def forward(self, t):
            with torch.no_grad():
                emb = torch.div(t.unsqueeze(-1).repeat(1, self.dim // 2).to(self.device), self.factor)
                emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
                return emb
    return (PositionEmbeddingV1,)


@app.cell
def _(PositionEmbeddingV1, nn, torch):
    class PositionEmbeddingBlockV1(nn.Module):
        def __init__(self, edim, device=torch.device("cpu")):
            super(PositionEmbeddingBlockV1, self).__init__()
            self.layers = nn.Sequential(
                PositionEmbeddingV1(edim, device),
                nn.Linear(edim, edim),
                nn.SiLU(),
                nn.Linear(edim, edim),
            )

        def forward(self, t):
            return self.layers(t)
    return (PositionEmbeddingBlockV1,)


@app.cell
def _(nn):
    class ConvBlockV1(nn.Module):
        def __init__(self, inc, outc, nt, gdivisor=2):
            super(ConvBlockV1, self).__init__()
            self.conv1 = nn.Sequential(
                nn.GroupNorm(inc // gdivisor, inc),
                nn.SiLU(),
                nn.Conv2d(inc, outc, 3, stride=1, padding=1),
            )
            self.tproj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(nt, outc),
            )
            self.conv2 = nn.Sequential(
                nn.GroupNorm(outc // gdivisor, outc),
                nn.SiLU(),
                nn.Conv2d(outc, outc, 3, stride=1, padding=1),
            )
            self.res = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()

        def forward(self, x, t):
            out = x
            out = self.conv1(out)
            out += self.tproj(t).unsqueeze(-1).unsqueeze(-1)
            out = self.conv2(out)
            out += self.res(x)
            return out
    return (ConvBlockV1,)


@app.cell
def _(nn):
    class ConvBlockV2(nn.Module):
        def __init__(self, inc, outc, nt, gdivisor=2):
            super(ConvBlockV2, self).__init__()
            self.is_same_channel = inc == outc
            self.conv1 = nn.Sequential(
                nn.GroupNorm(inc // gdivisor, inc),
                nn.SiLU(),
                nn.Conv2d(inc, outc, 3, stride=1, padding=1),
            )
            self.tproj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(nt, outc),
            )
            self.conv2 = nn.Sequential(
                nn.GroupNorm(outc // gdivisor, outc),
                nn.SiLU(),
                nn.Conv2d(outc, outc, 3, stride=1, padding=1),
            )

        def forward(self, x, t):
            x1 = self.conv1(x)
            out = x1 + self.tproj(t).unsqueeze(-1).unsqueeze(-1)
            out = self.conv2(out)
            if self.is_same_channel:
                out = x + out
            else:
                out = x1 + out
            return out / 1.414
    return (ConvBlockV2,)


@app.cell
def _(nn):
    class AttnBlockV1(nn.Module):
        def __init__(self, nc, nhead, gdivisor=2):
            super(AttnBlockV1, self).__init__()
            self.anorm = nn.GroupNorm(nc // gdivisor, nc)
            self.attn = nn.MultiheadAttention(nc, nhead, batch_first=True)

        def forward(self, x):
            b, c, h, w = x.shape
            out = x.reshape(b, c, h * w)
            out = self.anorm(out)
            out = out.transpose(1, 2)
            out, _ = self.attn(out, out, out)
            out = out.transpose(1, 2).reshape(b, c, h, w)
            out += x
            return out
    return (AttnBlockV1,)


@app.cell
def _(AttnBlockV1, ConvBlockV1, nn):
    class DownBlockV1(nn.Module):
        def __init__(self, inc, outc, nt, nhead, gdivisor=2, downsample=True):
            super(DownBlockV1, self).__init__()
            self.res = ConvBlockV1(inc, outc, nt, gdivisor=gdivisor)
            self.attn = AttnBlockV1(outc, nhead, gdivisor=gdivisor)
            self.downsample = nn.Conv2d(outc, outc, 4, stride=2, padding=1) if downsample else nn.Identity()

        def forward(self, x, t):
            x = self.res(x, t)
            x = self.attn(x)
            x = self.downsample(x)
            return x
    return (DownBlockV1,)


@app.cell
def _(ConvBlockV2, nn):
    class DownBlockV2(nn.Module):
        def __init__(self, inc, outc, nt, gdivisor=2, downsample=True):
            super(DownBlockV2, self).__init__()
            self.res = ConvBlockV2(inc, outc, nt, gdivisor=gdivisor)
            self.downsample = nn.Conv2d(outc, outc, 4, stride=2, padding=1) if downsample else nn.Identity()

        def forward(self, x, t):
            x = self.res(x, t)
            x = self.downsample(x)
            return x
    return (DownBlockV2,)


@app.cell
def _(AttnBlockV1, ConvBlockV1, nn, torch):
    class UpBlockV1(nn.Module):
        def __init__(self, inc, outc, nt, nhead, gdivisor=2, upsample=True):
            super(UpBlockV1, self).__init__()
            self.upsample = nn.ConvTranspose2d(inc, (inc - outc), 4, stride=2, padding=1) if upsample else nn.Identity()
            self.res = ConvBlockV1(inc, outc, nt, gdivisor=gdivisor)
            self.attn = AttnBlockV1(outc, nhead, gdivisor=gdivisor)

        def forward(self, x, d, t):
            x = self.upsample(x)
            x = torch.cat([x, d], dim=1)
            x = self.res(x, t)
            x = self.attn(x)
            return x
    return (UpBlockV1,)


@app.cell
def _(ConvBlockV2, nn, torch):
    class UpBlockV2(nn.Module):
        def __init__(self, inc, outc, nt, gdivisor=2, upsample=True):
            super(UpBlockV2, self).__init__()
            self.upsample = nn.ConvTranspose2d(inc, (inc - outc), 4, stride=2, padding=1) if upsample else nn.Identity()
            self.res = ConvBlockV2(inc, outc, nt, gdivisor=gdivisor)

        def forward(self, x, d, t):
            x = self.upsample(x)
            x = torch.cat([x, d], dim=1)
            x = self.res(x, t)
            return x
    return (UpBlockV2,)


@app.cell
def _(AttnBlockV1, ConvBlockV1, nn):
    class MidBlockV1(nn.Module):
        def __init__(self, nc, nt, nhead, gdivisor=2):
            super(MidBlockV1, self).__init__()
            self.res1 = ConvBlockV1(nc, nc, nt, gdivisor=gdivisor)
            self.res2 = ConvBlockV1(nc, nc, nt, gdivisor=gdivisor)
            self.attn = AttnBlockV1(nc, nhead, gdivisor=gdivisor)

        def forward(self, x, t):
            x = self.res1(x, t)
            x = self.attn(x)
            x = self.res2(x, t)
            return x
    return (MidBlockV1,)


@app.cell
def _(ConvBlockV2, nn):
    class MidBlockV2(nn.Module):
        def __init__(self, nc, nt, gdivisor=2):
            super(MidBlockV2, self).__init__()
            self.res1 = ConvBlockV2(nc, nc, nt, gdivisor=gdivisor)
            self.res2 = ConvBlockV2(nc, nc, nt, gdivisor=gdivisor)

        def forward(self, x, t):
            x = self.res1(x, t)
            x = self.res2(x, t)
            return x
    return (MidBlockV2,)


@app.function
def calculate_group_divisors(ncs):
    return [ncs[i+1] // ncs[i] for i in range(1, len(ncs)-1)]


@app.function
def is_group_divisors_valid(ncs, gdivisors):
    if len(ncs) - 2 != len(gdivisors):
        return False
    for i in range(1, len(ncs)-1):
        if ncs[i] * gdivisors[i-1] != ncs[i+1]:
            return False
    return True


@app.function
def is_image_size_layer_compat(img_sz, ncs):
    if len(img_sz) != 3:
        return False
    if (len(ncs) <= 2):
        return False
    if img_sz[0] != ncs[0]:
        return False
    def factor2(val):
        f = 0
        while val % 2 == 0:
            f += 1
            val = val // 2
        return f
    max_layer = min(factor2(img_sz[1]), factor2(img_sz[2]))
    if len(ncs) > max_layer + 2:
        return False
    return True


@app.cell
def _(DownBlockV1, MidBlockV1, PositionEmbeddingBlockV1, UpBlockV1, nn, torch):
    class UNetV1(nn.Module):
        def __init__(self, ncs, nt, nhead, img_sz, device=torch.device("cpu")):
            super(UNetV1, self).__init__()

            assert len(ncs) > 2, "UNetV1 require at least one image channel size + two intermediate layer channel size"

            gdivisors = calculate_group_divisors(ncs)
            assert is_group_divisors_valid(ncs, gdivisors) == True, "Invalid channel layer size layout"

            assert is_image_size_layer_compat(img_sz, ncs) == True, "Invalid image size and layer setting"

            self.temb = PositionEmbeddingBlockV1(nt, device)
            self.downs = [nn.Conv2d(ncs[0], ncs[1], 3, stride=1, padding=1)]
            for i in range(1, len(ncs)-1):
                self.downs.append(DownBlockV1(ncs[i], ncs[i+1], nt, nhead, gdivisor=gdivisors[i-1]))
            self.downs = nn.ModuleList(self.downs)
            self.ups = []
            for i in range(len(ncs)-1, 1, -1):
                self.ups.append(UpBlockV1(ncs[i], ncs[i-1], nt, nhead, gdivisor=gdivisors[i-2]))
            self.ups.append(nn.Sequential(
                nn.GroupNorm(ncs[1] // 2, ncs[1]),
                nn.SiLU(),
                nn.Conv2d(ncs[1], ncs[0], 3, stride=1, padding=1)
            ))
            self.ups = nn.ModuleList(self.ups)
            self.mid = MidBlockV1(ncs[-1], nt, nhead)

        def forward(self, x, t):
            t = self.temb(t)
            dx = []
            x = self.downs[0](x)
            for i in range(1, len(self.downs)):
                dx.append(x)
                x = self.downs[i](x, t)
            x = self.mid(x, t)
            for i in range(len(self.ups)-1):
                d = dx.pop()
                x = self.ups[i](x, d, t)
            x = self.ups[-1](x)
            return x
    return


@app.cell
def _(DownBlockV2, MidBlockV2, PositionEmbeddingBlockV1, UpBlockV2, nn, torch):
    class UNetV2(nn.Module):
        def __init__(self, ncs, nt, img_sz, device=torch.device("cpu")):
            super(UNetV2, self).__init__()

            assert len(ncs) > 2, "UNetV1 require at least one image channel size + two intermediate layer channel size"

            gdivisors = calculate_group_divisors(ncs)
            assert is_group_divisors_valid(ncs, gdivisors) == True, "Invalid channel layer size layout"

            assert is_image_size_layer_compat(img_sz, ncs) == True, "Invalid image size and layer setting"

            self.temb = PositionEmbeddingBlockV1(nt, device)
            self.downs = [nn.Conv2d(ncs[0], ncs[1], 3, stride=1, padding=1)]
            for i in range(1, len(ncs)-1):
                self.downs.append(DownBlockV2(ncs[i], ncs[i+1], nt, gdivisor=gdivisors[i-1]))
            self.downs = nn.ModuleList(self.downs)
            self.ups = []
            for i in range(len(ncs)-1, 1, -1):
                self.ups.append(UpBlockV2(ncs[i], ncs[i-1], nt, gdivisor=gdivisors[i-2]))
            self.ups.append(nn.Sequential(
                nn.GroupNorm(ncs[1] // 2, ncs[1]),
                nn.SiLU(),
                nn.Conv2d(ncs[1], ncs[0], 3, stride=1, padding=1)
            ))
            self.ups = nn.ModuleList(self.ups)
            self.mid = MidBlockV2(ncs[-1], nt)

        def forward(self, x, t):
            t = self.temb(t)
            dx = []
            x = self.downs[0](x)
            for i in range(1, len(self.downs)):
                dx.append(x)
                x = self.downs[i](x, t)
            x = self.mid(x, t)
            for i in range(len(self.ups)-1):
                d = dx.pop()
                x = self.ups[i](x, d, t)
            x = self.ups[-1](x)
            return x
    return (UNetV2,)


@app.cell
def _():
    ### noise schedule
    return


@app.cell
def _(math, torch):
    def cosine_beta_schedule(timesteps, beta_start, beta_end, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
    return (cosine_beta_schedule,)


@app.cell
def _(cosine_beta_schedule, torch):
    class LinearNoiseScheduler:
        def __init__(self, nts, image_sz, bs=0.0001, be=0.02, beta_schedule='linear', device=torch.device("cpu")):
            self.nts = nts
            self.bs = bs
            self.be = be
            self.image_sz = image_sz
            self.device = device
            self.n_timesteps = nts
            self.beta_schedule = beta_schedule
            self._compute_params()

        def _compute_params(self):
            if self.beta_schedule == 'linear':
                self.beta_t = ((self.be - self.bs) * torch.arange(0, self.n_timesteps + 1, dtype=torch.float32) / self.n_timesteps + self.bs).to(self.device)
            elif self.beta_schedule == 'cosine':
                self.beta_t = cosine_beta_schedule(self.nts, self.bs, self.be).to(self.device)
            self.sqrt_beta_t = torch.sqrt(self.beta_t)
            self.alpha_t = 1. - self.beta_t
            self.log_alpha_t = torch.log(self.alpha_t)
            self.alpha_bar_t = torch.cumsum(self.log_alpha_t, dim=0).exp()
            self.sqrtab = torch.sqrt(self.alpha_bar_t)
            self.oneover_sqrta = 1. / torch.sqrt(self.alpha_t)
            self.sqrtmab = torch.sqrt(1. - self.alpha_bar_t)
            self.mab_over_sqrtmab = ((1. - self.alpha_t) / self.sqrtmab)

        def add_noise(self, x0, eps, ts):
            sqrt_abar = torch.index_select(self.sqrtab, 0, ts)[:, None, None, None]
            sqrt1abar = torch.index_select(self.sqrtmab, 0, ts)[:, None, None, None]
            return sqrt_abar * x0 + sqrt1abar * eps

        def step(self, eps_pred, t, xt):
            z = torch.randn(xt.shape[0], *self.image_sz).to(self.device) if t > 1 else 0.
            x_n = self.oneover_sqrta[t] * (xt - eps_pred * self.mab_over_sqrtmab[t]) + self.sqrt_beta_t[t] * z
            return x_n

        def training_timesteps(self):
            return self.n_timesteps

        def set_timesteps(self, value):
            value = min(self.nts, value)
            self.n_timesteps = value

        @property
        def timesteps(self):
            return [torch.tensor([i]) for i in reversed(range(self.n_timesteps+1))]

        @timesteps.setter
        def timesteps(self, value):
            value = min(self.nts, value)
            self.n_timesteps = value
    return


@app.cell
def _(cosine_beta_schedule, math, torch):
    class DDIMScheduler:
        def __init__(self, nts, image_sz, sampling_n, beta_schedule='linear', sampling_method='uniform', eta=0., bs=0.0001, be=0.02, device=torch.device("cpu")):
            self.nts = nts
            self.bs = bs
            self.be = be
            self.eta = eta
            self.image_sz = image_sz
            self.device = device
            self.n_timesteps = nts
            self.beta_schedule = beta_schedule
            # sampling timesteps does not include T
            if sampling_method == 'uniform':
                self.sampling_timesteps = torch.arange(0, self.n_timesteps, self.n_timesteps // sampling_n) + 1
            elif sampling_method == 'quad':
                self.sampling_timesteps = torch.unique(torch.pow(torch.linspace(0, int(math.sqrt(1000 * 0.8)), sampling_n), 2.).to(dtype=torch.int32), dim=0) + 1
            self.sampling_timesteps_end = len(self.sampling_timesteps)
            self._compute_params()

        def _compute_params(self):
            if self.beta_schedule == 'linear':
                self.beta_t = ((self.be - self.bs) * torch.arange(0, self.n_timesteps + 1, dtype=torch.float32) / self.n_timesteps + self.bs).to(self.device)
            elif self.beta_schedule == 'cosine':
                self.beta_t = cosine_beta_schedule(self.nts, self.bs, self.be).to(self.device)
            self.sqrt_beta_t = torch.sqrt(self.beta_t)
            self.alpha_t = 1. - self.beta_t
            self.log_alpha_t = torch.log(self.alpha_t)
            self.alpha_bar_t = torch.cumsum(self.log_alpha_t, dim=0).exp()
            self.sqrtab = torch.sqrt(self.alpha_bar_t)
            self.sqrtmab = torch.sqrt(1. - self.alpha_bar_t)
            # sampling parameters
            self.alpha_bar_s = self.alpha_bar_t[self.sampling_timesteps]
            self.sqrt_alpha_bar_s = torch.sqrt(self.alpha_bar_s)
            self.alpha_bar_s_prev = torch.cat([self.alpha_bar_t[0:1], self.alpha_bar_t[self.sampling_timesteps[:-1]]])
            self.sqrt_alpha_bar_s_prev = torch.sqrt(self.alpha_bar_s_prev)
            self.sigma = self.eta * torch.sqrt(
                (1. - self.alpha_bar_s_prev) / (1. - self.alpha_bar_s) * 
                (1. - self.alpha_bar_s / self.alpha_bar_s_prev)
            )
            self.sqrtmas = torch.sqrt(1. - self.alpha_bar_s)

        def add_noise(self, x0, eps, ts):
            sqrt_abar = torch.index_select(self.sqrtab, 0, ts)[:, None, None, None]
            sqrt1abar = torch.index_select(self.sqrtmab, 0, ts)[:, None, None, None]
            return sqrt_abar * x0 + sqrt1abar * eps

        def step(self, eps_pred, tau, xt):
            x0 = (xt - self.sqrtmas[tau] * eps_pred) / self.sqrt_alpha_bar_s[tau]
            dxt = torch.sqrt(1. - self.alpha_bar_s_prev[tau] - torch.pow(self.sigma[tau], 2.)) * eps_pred
            z = torch.randn(xt.shape[0], *self.image_sz).to(self.device) if self.eta != 0. and tau >= 1 else 0.
            x_tp = self.sqrt_alpha_bar_s_prev[tau] * x0 + dxt + self.sigma[tau] * z
            return x_tp

        def training_timesteps(self):
            return self.n_timesteps
    
        # sampling timesteps
        def set_timesteps(self, t):
            for idx, ti in enumerate(self.sampling_timesteps):
                if ti == t:
                    self.sampling_timesteps_end = idx + 1
                    return
            self.sampling_timesteps_end = len(self.sampling_timesteps)

        @property
        def timesteps(self):
            return [torch.tensor([t]) for t in reversed(range(self.sampling_timesteps_end))]

        @timesteps.setter
        def timesteps(self, value):
            for idx, ti in enumerate(self.sampling_timesteps):
                if ti == value:
                    self.sampling_timesteps_end = idx + 1
                    return
            self.sampling_timesteps_end = len(self.sampling_timesteps)
    return (DDIMScheduler,)


@app.cell
def _():
    ### training
    return


@app.cell
def _(device, math, torch, tqdm):
    def train(model, scheduler, dataloader, optim, loss, scaler, lrate, n_epoch, T, device=device):
        model.train()
        for ep in range(n_epoch):
            print(f'epoch {ep}')
            optim.param_groups[0]["lr"] = lrate * (1.-ep/n_epoch)
            pbar = tqdm(dataloader)
            loss_ema = None
            for x, _ in pbar:
                x = x.to(device)
                ts = torch.randint(0, T+1, (x.shape[0],)).to(device)
                noise = torch.randn_like(x).to(device)
                xt = scheduler.add_noise(x, noise, ts).to(device)
                with torch.amp.autocast("cuda"):
                    out = model(xt, ts)
                    l = loss(noise, out)
                    if math.isnan(l.item()):
                        print(f"NaN loss, out ({torch.min(out)}, {torch.max(out)}) {torch.sum(torch.isnan(out))}")
                        print(f"xt ({torch.min(xt)}, {torch.max(xt)}) {torch.sum(torch.isnan(xt))}")
                        print(f"noise ({torch.min(noise)}, {torch.max(noise)}) {torch.sum(torch.isnan(noise))}")
                optim.zero_grad()
                scaler.scale(l).backward()
                scaler.step(optim)
                scaler.update()
                if math.isnan(l.item()):
                    break
                optim.step()
                if loss_ema is None:
                    loss_ema = l.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * l.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
    return


@app.cell
def _():
    ### sampling
    return


@app.cell
def _(np, torch):
    @torch.no_grad()
    def ddpm_sample(n_sample, model, scheduler, image_sz, T, device=torch.device("cpu"), seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        scheduler.set_timesteps(T)
        xt = torch.randn(n_sample, *image_sz).to(device)
        xt_store = []
        for t in scheduler.timesteps:
            ts = t.repeat(n_sample).to(device)
            eps_pred = model(xt, ts)
            xt = scheduler.step(eps_pred, t.item(), xt)
            if t % 20 == 0 or t == T or t < 8:
                xt_store.append(xt.detach().cpu().numpy())
        xt_store = np.array(xt_store)
        return xt, xt_store
    return


@app.cell
def _(np, torch):
    @torch.no_grad()
    def ddim_sample(n_sample, model, scheduler, image_sz, T, device=torch.device("cpu"), seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        scheduler.set_timesteps(T)
        xt = torch.randn(n_sample, *image_sz).to(device)
        xt_store = []
        for tau in scheduler.timesteps:
            ts = scheduler.sampling_timesteps[tau.repeat(n_sample)].to(device)
            eps_pred = model(xt, ts)
            xt = scheduler.step(eps_pred, tau.item(), xt)
            if tau % 10 == 0 or tau < 8:
                xt_store.append(xt.detach().cpu().numpy())
        xt_store = np.array(xt_store)
        return xt, xt_store
    return (ddim_sample,)


@app.cell
def _():
    ### datasets
    return


@app.cell
def _():
    data_dir = '/Users/armandli/data/'
    return (data_dir,)


@app.cell
def _(DataLoader, ToTensor, data_dir, datasets, transforms):
    #mnist_transform = transforms.Compose([ToTensor(), transforms.Lambda(lambda x : 2 * (x - 0.5))])
    mnist_transform = transforms.Compose([ToTensor()])
    mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=mnist_transform)
    mnist_loader = DataLoader(mnist, batch_size=32, shuffle=True)
    return


@app.cell
def _(DataLoader, ToTensor, data_dir, datasets, transforms):
    #fmnist_transform = transforms.Compose([ToTensor(), transforms.Lambda(lambda x : 2 * (x - 0.5))])
    fmnist_transform = transforms.Compose([ToTensor()])
    fmnist = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    fmnist_loader = DataLoader(fmnist, batch_size=32, shuffle=True)
    return


@app.cell
def _():
    ### shared training parameters
    return


@app.cell
def _(nn, torch):
    T = 1000
    learning_rate = 0.0001
    n_steps = 20
    scaler = torch.cpu.amp.GradScaler()
    #scaler = torch.cuda.amp.GradScaler()
    loss = nn.MSELoss()
    return T, learning_rate


@app.cell
def _():
    ### DDIM on MNIST
    return


@app.cell
def _(model_dir):
    mnist_modelfile = model_dir + 'mnist_ddpm_v1.pth'
    return (mnist_modelfile,)


@app.cell
def _(UNetV2, device):
    #mnist_model = UNetV1(ncs=[1, 16, 64, 256], nt=16, nhead=16, img_sz=(1, 28, 28), device=device).to(device)
    mnist_model = UNetV2(ncs=[1, 16, 64, 256], nt=16, img_sz=(1, 28, 28), device=device).to(device)
    return (mnist_model,)


@app.cell
def _(cpu, mnist_model, mnist_modelfile, torch):
    # load model if there is one
    mnist_model.load_state_dict(torch.load(mnist_modelfile, map_location=cpu, weights_only=True))
    return


@app.cell
def _(DDIMScheduler, T, device, learning_rate, mnist_model, torch):
    mnist_scheduler = DDIMScheduler(T, (1,28,28), 600, sampling_method='quad', device=device)
    mnist_optimizer = torch.optim.AdamW(mnist_model.parameters(), lr=learning_rate)
    return (mnist_scheduler,)


@app.cell
def _():
    #train(model, scheduler, mnist_loader, optimizer, loss, scaler, learning_rate, nsteps, T, device=device)
    return


@app.cell
def _():
    # save trained model
    #torch.save(mnist_model.state_dict(), mnist_modelfile)
    return


@app.cell
def _(T, cpu, ddim_sample, device, mnist_model, mnist_scheduler):
    mnist_samples, mnist_sample_timesteps = ddim_sample(4, mnist_model, mnist_scheduler, (1, 28, 28), T, device=device)
    mnist_samples = mnist_samples.to(cpu)
    return (mnist_samples,)


@app.cell
def _(mnist_samples, plt, torchvision):
    mnist_grid = torchvision.utils.make_grid(mnist_samples/2+0.5, nrow=1)
    plt.figure(dpi=100)
    plt.imshow(mnist_grid.permute(1,2,0), cmap='binary')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    ### DDIM on Fashion MNIST
    return


@app.cell
def _(model_dir):
    fmnist_modelfile = model_dir + 'fmnist_ddpm_v1.pth'
    return (fmnist_modelfile,)


@app.cell
def _(UNetV2, device):
    # fmnist_model = UNetV1(ncs=[1, 8, 16, 32], nt=8, nhead=8, img_sz=(1, 32, 32), device=device).to(device)
    fmnist_model = UNetV2(ncs=[1, 16, 64, 256], nt=16, img_sz=(1, 28, 28), device=device).to(device)
    return (fmnist_model,)


@app.cell
def _(cpu, fmnist_model, fmnist_modelfile, torch):
    # load model
    fmnist_model.load_state_dict(torch.load(fmnist_modelfile, map_location=cpu, weights_only=True))
    return


@app.cell
def _(DDIMScheduler, T, device, learning_rate, mnist_model, torch):
    fmnist_scheduler = DDIMScheduler(T, (1,28,28), 900, sampling_method='quad', device=device)
    fmnist_optimizer = torch.optim.AdamW(mnist_model.parameters(), lr=learning_rate)
    return (fmnist_scheduler,)


@app.cell
def _():
    #train(fmnist_model, scheduler, fmnist_loader, optimizer, loss, scaler, learning_rate, nsteps, T, device=device)
    return


@app.cell
def _():
    # save trained model
    #torch.save(fmnist_model.state_dict(), fmnist_modelfile)
    return


@app.cell
def _(T, cpu, ddim_sample, device, fmnist_model, fmnist_scheduler):
    fmnist_samples, fmnist_sample_timesteps = ddim_sample(4, fmnist_model, fmnist_scheduler, (1, 28, 28), T, device=device)
    fmnist_samples = fmnist_samples.to(cpu)
    return fmnist_sample_timesteps, fmnist_samples


@app.cell
def _(fmnist_samples, plt, torchvision):
    fmnist_grid = torchvision.utils.make_grid(fmnist_samples/2+0.5, nrow=1)
    plt.figure(dpi=100)
    plt.imshow(fmnist_grid.permute(1,2,0))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(fmnist_sample_timesteps, plt, torch):
    plt.figure(figsize=(10,5), dpi=100)

    for k in range(fmnist_sample_timesteps.shape[0]):
        plt.subplot(8, 10, k+1)
        im = torch.tensor(fmnist_sample_timesteps[k][0]).permute(1,2,0) / 2 + 0.5
        plt.imshow(im, cmap='binary')
        plt.axis('off')
        plt.title(f"{k}")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    ### DDIM on hugging face flower dataset (HF)
    return


@app.cell
def _(
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
    load_dataset,
):
    hf_augmentations = Compose([
        Resize(64, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(64),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    def hf_transforms(examples):
        images = [hf_augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    hf_dataset = load_dataset("huggan/flowers-102-categories", split="train",)
    hf_dataset.set_transform(hf_transforms)
    return (hf_dataset,)


@app.cell
def _(hf_dataset, torch):
    hf_dataloader=torch.utils.data.DataLoader(hf_dataset, batch_size=4, shuffle=True)
    return (hf_dataloader,)


@app.cell
def _(hf_dataloader, plt, torch):
    plt.figure(figsize=(5.9,4),dpi=150)
    for col in range(6):
        imgs=next(iter(hf_dataloader))["input"]
        for row in range(4):
            plt.subplot(4,6,col+1+row*6)
            img=imgs[row].permute(1,2,0) #B
            plt.imshow(torch.clip(img,0,1)) #C
            plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
