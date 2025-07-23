import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import torch
    from torch import nn
    return nn, torch


@app.cell
def _():
    ### latent diffusion model
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
def _(EmbedLayer, torch):
    timeembed1=EmbedLayer(1, 512)
    timeembed2=EmbedLayer(1, 256)
    timesteps=torch.tensor([25]).long()
    t=timesteps/1000
    temb1 = timeembed1(t).view(-1, 512, 1, 1)
    temb2 = timeembed2(t).view(-1, 256, 1, 1)
    print("the shape of the first time embedding is", temb1.shape)
    print("the shape of the second time embedding is", temb2.shape)
    return


@app.cell
def _():
    ### generating label embedding
    return


@app.cell
def _(EmbedLayer, torch):
    n_classes=10
    contextembed1=EmbedLayer(n_classes, 512)
    contextembed2=EmbedLayer(n_classes, 256)
    label=torch.tensor([2, 9, 0]).long() # 3 image labels
    context_mask = torch.bernoulli(torch.zeros_like(label)+0.1) # mask to hide label for 10% prob
    onehot = torch.nn.functional.one_hot(label, num_classes=n_classes).type(torch.float) # onehot encode the class label
    context_mask = context_mask[:, None]
    context_mask = context_mask.repeat(1,n_classes)
    context_mask = (-1*(1-context_mask))
    c = onehot * context_mask
    cemb1 = contextembed1(c).view(-1, 512, 1, 1)
    cemb2 = contextembed2(c).view(-1, 256, 1, 1)
    print("the shape of the first label embedding is", cemb1.shape)
    print("the shape of the second label embedding is", cemb2.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
