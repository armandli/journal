import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    #import os
    #import glob
    import random
    #mport cv2
    from tqdm import tqdm
    #import numpy as np

    import torch
    from torch import nn
    from torch import optim
    from torch.utils.data.dataloader import DataLoader
    from torchvision import datasets
    from torchvision import transforms
    return DataLoader, datasets, nn, optim, random, torch, tqdm, transforms


@app.cell
def _(torch):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_built()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device
    cpu = torch.device("cpu")
    return cpu, device, use_cuda, use_mps


@app.cell
def _(mo):
    mo.md(r"""#model components""")
    return


@app.cell
def _(nn):
    def relu_activation():
        return nn.ReLU(inplace=True)
    return (relu_activation,)


@app.cell
def _(nn):
    def leaky_activation():
        return nn.LeakyReLU(inplace=True)
    return (leaky_activation,)


@app.cell
def _(nn):
    def silu_activation():
        return nn.SiLU(inplace=True)
    return (silu_activation,)


@app.cell
def _(nn):
    class VQVAEDecoderV1(nn.Module):
        def __init__(self, csz, ksz, ssz, act):
            super(VQVAEDecoderV1, self).__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(csz[i], csz[i+1], kernel_size=ksz[i], stride=ssz[i], padding=0),
                    nn.BatchNorm2d(csz[i+1]),
                    act,
                )
                for i in range(len(csz)-1)
            ])
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(csz[-2], csz[-1], kernel_size=ksz[-1], stride=ssz[-1], padding=0),
                    nn.Sigmoid(),
                )
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    return (VQVAEDecoderV1,)


@app.cell
def _(nn):
    class VQVAEEncoderV1(nn.Module):
        def __init__(self, csz, ksz, ssz, act):
            super(VQVAEEncoderV1, self).__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(csz[i], csz[i+1], kernel_size=ksz[i], stride=ssz[i], padding=1),
                    nn.BatchNorm2d(csz[i+1]),
                    act,
                )
                for i in range(len(csz)-1)
            ])
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(csz[-2], csz[-1], kernel_size=ksz[-1], stride=ssz[-1], padding=1)
                )
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    return (VQVAEEncoderV1,)


@app.cell
def _(nn, torch):
    class VQVAEQuantizerV1(nn.Module):
        def __init__(self, codebook_sz, latent_dim):
            super(VQVAEQuantizerV1, self).__init__()
            self.embedding = nn.Embedding(codebook_sz, latent_dim)

        def foward(self, x):
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(x.size(0), -1, x.size(-1))

            dist = torch.cdist(x, self.embedding.weight[None,:].repeat((x.size(0), 1, 1)))
            min_idx = torch.argmin(dist, dim=-1)

            quant_out = torch.index_select(self.embedding.weight, 0, min_idx.view(-1))
            x = x.reshape((-1, x.size(-1)))
            commitment_loss = torch.mean((quant_out.detach() - x) ** 2.)
            codebook_loss = torch.mean((quant_out - x.detach()) ** 2.)

            quant_out = x + (quant_out - x).detach()
            quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
            min_idx = min_idx.reshape((-1, quant_out.size(-2), quant_out.size(-1)))

            return quant_out, commitment_loss, codebook_loss, min_idx
    return (VQVAEQuantizerV1,)


@app.cell
def _(VQVAEDecoderV1, VQVAEEncoderV1, VQVAEQuantizerV1, nn):
    class VQVAEV1(nn.Module):
        def __init__(self, e_csz, e_ksz, e_ssz, d_csz, d_ksz, d_ssz, codebook_sz, latent_dim, act):
            super(VQVAEV1, self).__init__()
            self.encoder = VQVAEEncoderV1(e_csz, e_ksz, e_ssz, act)
            self.pre_quant_conv = nn.Conv2d(e_csz[-1], latent_dim, kernel_size=1)
            self.decoder = VQVAEDecoderV1(d_csz, d_ksz, d_ssz, act)
            self.post_quant_conv = nn.Conv2d(latent_dim, d_csz[0], kernel_size=1)
            self.quantizer = VQVAEQuantizerV1(codebook_sz, latent_dim)

        def forward(self, x):
            x = self.encoder(x)
            x = self.pre_quant_conv(x)
            qout, commitment_loss, codebook_loss, qidx = self.quantizer(x)
            out = self.post_quant_conv(qout)
            out = self.decoder(out)
            return out, qout, commitment_loss, codebook_loss, qidx
    return (VQVAEV1,)


@app.cell
def _(mo):
    mo.md(r"""MNIST MODEL""")
    return


@app.cell
def _(VQVAEV1, leaky_activation):
    mnist_model = VQVAEV1(
        [1, 16, 32, 8, 4],
        [3, 3, 3, 2],
        [2, 2, 1, 1],
        [4, 8, 32, 16, 1],
        [3, 4, 4, 4],
        [1, 2, 1, 1],
        5,
        2,
        leaky_activation()
    )
    return (mnist_model,)


@app.cell
def _(mo):
    mo.md("""Simple MNIST Model and Training""")
    return


@app.cell
def _(nn, torch):
    class VQVAE(nn.Module):
        def __init__(self):
            super(VQVAE, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
            )
            self.pre_quant_conv = nn.Conv2d(4, 2, kernel_size=1)
            self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=2)
            self.post_quant_conv = nn.Conv2d(2, 4, kernel_size=1)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )
            self.beta = 0.2

        def forward(self, x):
            # B, C, H, W
            encoded_output = self.encoder(x)
            quant_input = self.pre_quant_conv(encoded_output)

            B, C, H, W = quant_input.shape
            quant_input = quant_input.permute(0, 2, 3, 1)
            quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))

            dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))

            min_encoding_indicies = torch.argmin(dist, dim=-1)

            quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indicies.view(-1))
            quant_input = quant_input.reshape((-1, quant_input.size(-1)))

            commitment_loss = torch.mean((quant_out.detach() - quant_input)**2.)
            codebook_loss = torch.mean((quant_out - quant_input.detach())**2.)
            quantize_loss = codebook_loss + self.beta * commitment_loss

            quant_out = quant_input + (quant_out - quant_input).detach()

            quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
            min_encoding_indicies = min_encoding_indicies.reshape((-1, quant_out.size(-2), quant_out.size(-1)))

            decoder_input = self.post_quant_conv(quant_out)
            output = self.decoder(decoder_input)
            return output, quantize_loss
    return (VQVAE,)


@app.cell
def _():
    data_dir = "/Users/armandli/data"
    mnist_path = data_dir
    return data_dir, mnist_path


@app.cell
def _(transforms):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return (data_transform,)


@app.cell
def _(data_transform, datasets, mnist_path):
    train_data = datasets.MNIST(mnist_path, train=True, transform=data_transform)
    return (train_data,)


@app.cell
def _(data_transform, datasets, mnist_path):
    test_data = datasets.MNIST(mnist_path, train=False, transform=data_transform)
    return (test_data,)


@app.cell
def _(VQVAE):
    model = VQVAE()
    return (model,)


@app.cell
def _(DataLoader, device, nn, optim, tqdm, train_data):
    def train_vqvae(model):
        loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
        model = model.to(device)
        epochs = 20
        optimizer = optim.Adam(model.parameters(), lr=1E-3)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for im, _ in tqdm(loader):
                im = im.to(device)
                optimizer.zero_grad()
                out, quantize_loss = model(im)
                recon_loss = criterion(out, im)
                loss = recon_loss + quantize_loss
                loss.backward()
                optimizer.step()
    return (train_vqvae,)


@app.cell
def _(model, train_vqvae):
    train_vqvae(model)
    return


@app.cell
def _(cpu, random, test_data, transforms):
    def test_model(model):
        model = model.to(cpu)
        model.eval()
        test_idx = random.randint(0, len(test_data))
        input = test_data[test_idx][0]
        C, H, W = input.shape
        input = input.reshape((1, C, H, W))
        out, _ = model(input)
        out = out.reshape((C, H, W))
        img = transforms.functional.to_pil_image(out)
        return img
    return (test_model,)


@app.cell
def _(model, test_model):
    img = test_model(model)
    img
    return (img,)


if __name__ == "__main__":
    app.run()
