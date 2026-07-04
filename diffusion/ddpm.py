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

    import matplotlib.pyplot as plt
    return (
        DataLoader,
        ToTensor,
        datasets,
        math,
        nn,
        np,
        plt,
        random,
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
    model_dir = 'models/'
    return (model_dir,)


@app.cell
def _():
    ### unet models
    return


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
    ### target model
    return


@app.cell
def _(nn):
    class EmbedLayer(nn.Module):
        def __init__(self, input_dim, emb_dim):
            super().__init__()
            self.input_dim = input_dim
            layers = [nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim),]
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            x = x.view(-1, self.input_dim)
            return self.model(x)
    return (EmbedLayer,)


@app.cell
def _(nn):
    class ResidualConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, is_res = False):
            super().__init__()
            self.same_channels = in_channels==out_channels
            self.is_res = is_res
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        def forward(self, x):
            if self.is_res:
                x1 = self.conv1(x)
                x2 = self.conv2(x1)
                if self.same_channels:
                    out = x + x2
                else:
                    out = x1 + x2 
                return out / 1.414
            else:
                x1 = self.conv1(x)
                x2 = self.conv2(x1)
                return x2
    return (ResidualConvBlock,)


@app.cell
def _(ResidualConvBlock, nn):
    class UnetDown(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return self.model(x)
    return (UnetDown,)


@app.cell
def _(ResidualConvBlock, nn, torch):
    class UnetUp(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                ResidualConvBlock(out_channels, out_channels),
                ResidualConvBlock(out_channels, out_channels),
            ]
            self.model = nn.Sequential(*layers)
        def forward(self, x, skip):
            x = torch.cat((x, skip), 1)
            x = self.model(x)
            return x
    return (UnetUp,)


@app.cell
def _():
    ### reverse engineered model
    return


@app.cell
def _(EmbedLayer, ResidualConvBlock, UnetDown, UnetUp, nn, torch):
    class REUNet(nn.Module):
        def __init__(self, in_channels, n_T, n_feat = 256, device=torch.device("cpu")):
            super().__init__()
            self.in_channels = in_channels
            self.n_feat = n_feat
            self.n_T = n_T
            self.device = device
            self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
            self.down1 = UnetDown(n_feat, n_feat)
            self.down2 = UnetDown(n_feat, 2 * n_feat)
            self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
            self.timeembed1 = EmbedLayer(1, 2*n_feat)
            self.timeembed2 = EmbedLayer(1, 1*n_feat)
            self.contextembed1 = EmbedLayer(10, 2*n_feat)
            self.contextembed2 = EmbedLayer(10, 1*n_feat)
            self.up0=nn.Sequential(nn.ConvTranspose2d(2*n_feat,2*n_feat,7,7), nn.GroupNorm(8, 2 * n_feat),nn.ReLU(),)
            self.up1 = UnetUp(4 * n_feat, n_feat)
            self.up2 = UnetUp(2 * n_feat, n_feat)
            self.out = nn.Sequential(
                nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
                nn.GroupNorm(8, n_feat),
                nn.ReLU(),
                nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
            )

        def forward(self, x, t):
            context_mask = torch.bernoulli(torch.zeros(x.shape[0])+0.1).to(self.device)
            t = (t+1.) / self.n_T
            x = self.init_conv(x)
            down1 = self.down1(x)
            down2 = self.down2(down1)
            hiddenvec = self.to_vec(down2)
            context_mask = context_mask[:, None]
            context_mask = context_mask.repeat(1,10)
            context_mask = (-1*(1-context_mask)) 
            cemb1 = self.contextembed1(context_mask).view(-1, self.n_feat * 2, 1, 1)
            temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
            cemb2 = self.contextembed2(context_mask).view(-1, self.n_feat, 1, 1)
            temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
            up1 = self.up0(hiddenvec)
            up2 = self.up1(cemb1*up1+ temb1, down2)  
            up3 = self.up2(cemb2*up2+ temb2, down1)
            out = self.out(torch.cat((up3, x), 1))
            return out
    return


@app.cell
def _():
    ### noise schedule
    return


@app.cell
def _():
    # class LinearNoiseScheduler:
    #     def __init__(self, nts, bs=0.0001, be=0.02, device=torch.device("cpu")):
    #         self.nts = nts
    #         self.bs = bs
    #         self.be = be
    #         self.n_timesteps = self.nts
    #         self.device = device
    #         self._compute_params()

    #     def _compute_params(self):
    #         # 0th index represent step 1, we never take step nts
    #         self.bss = torch.linspace(self.bs, self.be, self.nts).to(self.device)
    #         # at = 1. - bt
    #         self.ass = 1. - self.bss
    #         # a_bar = product(at) for t in 1 to t
    #         self.abar = torch.cumprod(self.ass, dim=0)
    #         self.s_abar = torch.sqrt(self.abar)
    #         self.s1abar = torch.sqrt(1. - self.abar)

    #     def add_noise(self, x0, eps, ts):
    #         sqrt_abar = torch.index_select(self.s_abar, 0, ts)[:, None, None, None]
    #         sqrt1abar = torch.index_select(self.s1abar, 0, ts)[:, None, None, None]
    #         return sqrt_abar * x0 + sqrt1abar * eps

    #     def step(self, eps_pred, t, xt):
    #         # x0 = (xt - sqrt(1-abar) * eps) / sqrt(abar)
    #         x0 = (xt - self.s1abar[t] * eps_pred) / self.s_abar[t]
    #         x0 = torch.clamp(x0, -1., 1.)
    #         # expectation = (1. / sqrt(at)) * (xt - (1 - at) / (sqrt(1 - abar)) * eps)
    #         mean = xt - self.bss[t] * eps_pred / self.s1abar[t]
    #         mean = mean / torch.sqrt(self.ass[t])
    #         if t == 0:
    #             return mean, x0
    #         # variance = (1 - at) * (1 - abar[t-1]) / (1 - abar[t])
    #         var = (1. - self.abar[t-1]) / (1. - self.abar[t])
    #         var = var * self.bss[t]
    #         sig = var ** 0.5
    #         z = torch.randn(xt.shape).to(self.device) #TODO
    #         return mean + sig * z, x0

    #     def training_timesteps(self):
    #         return self.n_timesteps

    #     #TODO: need to return a list of tensor
    #     @property
    #     def timesteps(self):
    #         return [torch.tensor([i]) for i in reversed(range(self.n_timesteps))]

    #     @timesteps.setter
    #     def timesteps(self, value):
    #         value = min(self.nts, value)
    #         self.n_timesteps = value
    return


@app.cell
def _(device, torch):
    class LinearNoiseScheduler:
        def __init__(self, nts, image_sz, bs=0.0001, be=0.02, device=torch.device("cpu")):
            self.nts = nts
            self.bs = bs
            self.be = be
            self.image_sz = image_sz
            self.device = device
            self.n_timesteps = nts
            self._compute_params()

        def _compute_params(self):
            self.beta_t = ((self.be - self.bs) * torch.arange(0, self.n_timesteps + 1, dtype=torch.float32) / self.n_timesteps + self.bs).to(device)
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
    return (LinearNoiseScheduler,)


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
def _():
    # @torch.no_grad()
    # def ddpm_sample(n_sample, model, scheduler, gen_noise, image_sz, device=torch.device("cpu"), seed=None):
    #     model.eval()
    #     if seed is not None:
    #         torch.manual_seed(seed)
    #     scheduler.timesteps = 1000
    #     image = gen_noise(n_sample, image_sz).to(device)
    #     for t in scheduler.timesteps:
    #         ts = torch.tensor([t]).repeat(n_sample).to(device)
    #         eps_pred = model(image, ts)
    #         image, _ = scheduler.step(eps_pred, t, image)
    #     return image
    return


@app.cell
def _(device, np, torch):
    @torch.no_grad()
    def ddpm_sample(n_sample, model, scheduler, image_sz, T, device=device, seed=None):
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
    return (ddpm_sample,)


@app.cell
def _():
    ### datasets
    return


@app.cell
def _():
    data_dir = '../data/'
    return (data_dir,)


@app.cell
def _(DataLoader, ToTensor, data_dir, datasets, transforms):
    #mnist_transform = transforms.Compose([ToTensor(), transforms.Lambda(lambda x : 2 * (x - 0.5))])
    mnist_transform = transforms.Compose([ToTensor()])
    mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=mnist_transform)
    mnist_loader = DataLoader(mnist, batch_size=32, shuffle=True)
    return (mnist,)


@app.cell
def _(DataLoader, ToTensor, data_dir, datasets, transforms):
    #fmnist_transform = transforms.Compose([ToTensor(), transforms.Lambda(lambda x : 2 * (x - 0.5))])
    fmnist_transform = transforms.Compose([ToTensor()])
    fmnist = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    fmnist_loader = DataLoader(fmnist, batch_size=32, shuffle=True)
    return (fmnist,)


@app.cell
def _():
    ### display gradual noise addition
    return


@app.cell
def _(plt, random, torch, torchvision):
    def display_noise_addition(dataset, scheduler, idx=None):
        if idx is None:
            idx = random.randint(0, dataset.data.size()[0]-1)
        tsize = 10
        x0 = dataset[idx][0].unsqueeze(0).repeat(tsize, 1, 1, 1)
        #eps = (torch.rand_like(x0) - 0.5) * 2
        eps = torch.rand_like(x0)
        ts = torch.tensor([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        xt = scheduler.add_noise(x0, eps, ts)
        grid = torchvision.utils.make_grid(xt / 2 + 0.5, nrow=5)
        plt.figure(dpi=100)
        plt.imshow(grid.permute(1,2,0))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return (display_noise_addition,)


@app.cell
def _(LinearNoiseScheduler, display_noise_addition, fmnist):
    display_noise_addition(fmnist, LinearNoiseScheduler(1000, (1,28,28)))
    return


@app.cell
def _(random, torch):
    def noise_min_max(dataset, scheduler, idx=None):
        if idx is None:
            idx = random.randint(0, dataset.data.size()[0]-1)
        tsize = 10
        x0 = dataset[idx][0].unsqueeze(0).repeat(tsize, 1, 1, 1)
        eps = (torch.rand_like(x0) - 0.5) * 2
        #eps = torch.rand_like(x0)
        ts = torch.tensor([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        xt = scheduler.add_noise(x0, eps, ts)
        for i in range(tsize):
            eps_max = torch.max(eps[i])
            eps_min = torch.min(eps[i])
            xt_max = torch.max(xt[i])
            xt_min = torch.min(xt[i])
            x0_max = torch.max(x0[i])
            x0_min = torch.min(x0[i])
            print(f"{i}: eps:({eps_min},{eps_max}) xt:({xt_min},{xt_max}) x0:({x0_min},{x0_max})")
    return (noise_min_max,)


@app.cell
def _(LinearNoiseScheduler, mnist, noise_min_max):
    noise_min_max(mnist, LinearNoiseScheduler(1000, (1, 28, 28)))
    return


@app.cell
def _():
    ### DDPM on MNIST
    return


@app.cell
def _(LinearNoiseScheduler, device, mnist_model, nn, torch):
    T = 1000
    learning_rate = 0.0001
    n_steps = 20
    scaler = torch.cpu.amp.GradScaler()
    #scaler = torch.cuda.amp.GradScaler()
    loss = nn.MSELoss()
    scheduler = LinearNoiseScheduler(T, (1,28,28), device=device)
    optimizer = torch.optim.AdamW(mnist_model.parameters(), lr=learning_rate)
    return T, scheduler


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
def _():
    #train(mnist_model, scheduler, mnist_loader, optimizer, loss, scaler, learning_rate, nsteps, T, device=device)
    return


@app.cell
def _():
    # save trained model
    #torch.save(model.state_dict(), mnist_modelfile)
    return


@app.cell
def _(T, cpu, ddpm_sample, device, mnist_model, scheduler):
    mnist_samples, mnist_sample_timesteps = ddpm_sample(4, mnist_model, scheduler, (1, 28, 28), T, device=device)
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
    ### DDPM on Fashion MNIST
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
def _():
    #train(fmnist_model, scheduler, fmnist_loader, optimizer, loss, scaler, learning_rate, nsteps, T, device=device)
    return


@app.cell
def _(T, cpu, ddpm_sample, device, fmnist_model, scheduler):
    fmnist_samples, fmnist_sample_timesteps = ddpm_sample(4, fmnist_model, scheduler, (1, 28, 28), T, device=device)
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

    for jjj in range(fmnist_sample_timesteps.shape[0]):
        plt.subplot(6, 10, jjj+1)
        im = torch.tensor(fmnist_sample_timesteps[jjj][0]).permute(1,2,0) / 2 + 0.5
        plt.imshow(im, cmap='binary')
        plt.axis('off')
        plt.title(f"{jjj}")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
