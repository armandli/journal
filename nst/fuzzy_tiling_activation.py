import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from tqdm import tqdm

    import torch
    from torch import nn
    from torch import optim
    from torch.utils.data.dataloader import DataLoader
    from torchvision import datasets
    from torchvision import transforms
    return nn, torch


@app.cell
def _(torch):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_built()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device
    cpu = torch.device("cpu")
    return


@app.cell
def _(mo):
    mo.md("""## Implement Fuzzy Tiling Activation Module""")
    return


@app.cell
def _(nn, torch):
    class TilingActivation(nn.Module):
        def __init__(self, l_limit, u_limit, delta):
            super(TilingActivation, self).__init__()
            self.c = nn.Parameter(torch.arange(l_limit, u_limit, delta), requires_grad=False)
            self.expansion_factor = len(self.c)
            self.delta = delta

        def indicator(self, x):
            return (x > 0.) + 0.
    
        def forward(self, x):
            print(self.c.shape)
            x = x.unsqueeze(len(x.shape))
            x = 1. - self.indicator(torch.clip(self.c - x, min=0.) + torch.clip(x - self.delta - self.c, min=0.))
            return x.view(*x.shape[:-2], -1)
    return


@app.cell
def _(nn, torch):
    class FTAV1(nn.Module):
        def __init__(self, l_limit, u_limit, delta, eta):
            super(FTAV1, self).__init__()
            self.c = nn.Parameter(torch.arange(l_limit, u_limit, delta), requires_grad=False)
            self.expansion_factor = len(self.c)
            self.delta = delta
            self.eta = eta

        def indicator(self, x):
            return (x > 0.) + 0.
    
        def fuzzy_indicator(self, x):
            return self.indicator(self.eta - x) * x + self.indicator(x - self.eta)
    
        def forward(self, x):
            x = x.unsqueeze(len(x.shape))
            x = 1. - self.fuzzy_indicator(torch.clip(self.c - x, min=0.) + torch.clip(x - self.delta - self.c, min=0.))
            return x.view(*x.shape[:-2], -1)
    return


@app.cell
def _(mo):
    mo.md("""## Supervised Learning Verification""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
