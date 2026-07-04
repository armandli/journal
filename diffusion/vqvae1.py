import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import random
    from tqdm import tqdm

    import math
    import numpy as np

    import torch
    from torch import nn
    from torch import optim
    from torch.utils.data.dataloader import DataLoader
    import torchvision
    from torchvision import datasets
    from torchvision import transforms

    import matplotlib.pyplot as plt
    return (
        DataLoader,
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
    device
    cpu = torch.device("cpu")
    return cpu, device


@app.cell
def _(mo):
    mo.md(r"""#model components""")
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
def _(nn):
    def relu_activation():
        return nn.ReLU()
    return


@app.cell
def _(nn):
    def leaky_activation():
        return nn.LeakyReLU()
    return (leaky_activation,)


@app.cell
def _(nn):
    def silu_activation():
        return nn.SiLU()
    return


@app.cell
def _(nn):
    class ConvBlockV1(nn.Module):
        def __init__(self, inc, outc, activation, gdivisor=2):
            super().__init__()
            self.is_same_channel = inc == outc
            self.conv1 = nn.Sequential(
                nn.GroupNorm(inc // gdivisor, inc),
                activation(),
                nn.Conv2d(inc, outc, 3, stride=1, padding=1),
            )
            self.conv2 = nn.Sequential(
                nn.GroupNorm(outc // gdivisor, outc),
                activation(),
                nn.Conv2d(outc, outc, 3, stride=1, padding=1),
            )

        def forward(self, x):
            x1 = self.conv1(x)
            out = self.conv2(x1)
            if self.is_same_channel:
                out = x + out
            else:
                out = x1 + out
            return out / 1.414
    return (ConvBlockV1,)


@app.cell
def _(ConvBlockV1, nn):
    class DownBlockV1(nn.Module):
        def __init__(self, inc, outc, activation, gdivisor=2, downsample=True):
            super().__init__()
            self.res = ConvBlockV1(inc, outc, activation, gdivisor=gdivisor)
            self.downsample = nn.Conv2d(outc, outc, 4, stride=2, padding=1) if downsample else nn.Identity()
        
        def forward(self, x):
            x = self.res(x)
            x = self.downsample(x)
            return x
    return (DownBlockV1,)


@app.cell
def _(ConvBlockV1, nn):
    class UpBlockV1(nn.Module):
        def __init__(self, inc, outc, activation, gdivisor=2, upsample=True):
            super().__init__()
            self.upsample = nn.ConvTranspose2d(inc, outc, 4, stride=2, padding=1) if upsample else nn.Identity()
            self.res = ConvBlockV1(outc, outc, activation, gdivisor=gdivisor)

        def forward(self, x):
            x = self.upsample(x)
            x = self.res(x)
            return x
    return (UpBlockV1,)


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
def _(ConvBlockV1, UpBlockV1, nn):
    class VQVAEDecoderV2(nn.Module):
        def __init__(self, csz, activation, gdivisors):
            super().__init__()
            layers = []
            for i in range(0, len(csz)-2):
                layers.append(UpBlockV1(csz[i], csz[i+1], activation, gdivisor=gdivisors[i]))
            layers.append(ConvBlockV1(csz[-2], csz[-1], activation, 1))
            layers.append(nn.Sigmoid())
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    return (VQVAEDecoderV2,)


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
def _(DownBlockV1, nn):
    class VQVAEEncoderV2(nn.Module):
        def __init__(self, csz, activation, gdivisors):
            super().__init__()
            layers = [nn.Conv2d(csz[0], csz[1], 3, stride=1, padding=1)]
            for i in range(1, len(csz)-1):
                layers.append(DownBlockV1(csz[i], csz[i+1], activation, gdivisor=gdivisors[i-1]))
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    return (VQVAEEncoderV2,)


@app.cell
def _(nn, torch):
    class VQVAEQuantizerV1(nn.Module):
        def __init__(self, codebook_sz, latent_dim):
            super(VQVAEQuantizerV1, self).__init__()
            self.latent_dim = latent_dim
            self.embedding = nn.Embedding(codebook_sz, latent_dim)

        def size(self):
            return self.embedding.weight.shape[0]

        def select(self, qidx, embed_sz):
            batchsize = qidx.shape[0]
            qout = torch.index_select(self.embedding.weight, 0, qidx.view(-1))
            qout = qout.reshape((batchsize, *embed_sz, self.latent_dim)).permute(0, 3, 1, 2)
            return qout
    
        def forward(self, x):
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
    return


@app.cell
def _(VQVAEDecoderV2, VQVAEEncoderV2, VQVAEQuantizerV1, nn):
    class VQVAEV2(nn.Module):
        def __init__(self, csz, codebook_sz, latent_dim, activation, image_sz, embed_sz):
            super().__init__()

            assert len(csz) >= 2, "VQVAEV2 require at least one image channel size + one intermediate layer channel size"
        
            gdivisors = calculate_group_divisors(csz)

            assert is_group_divisors_valid(csz, gdivisors), f"Invalid group divisors {gdivisors} for {csz}"
            assert is_image_size_layer_compat(image_sz, csz), f"Invalid image size and layer setting"

            self.codebook_sz = codebook_sz
            self.embed_sz = embed_sz
        
            self.encoder = VQVAEEncoderV2(csz, activation, gdivisors)
            self.pre_quant_conv = nn.Conv2d(csz[-1], latent_dim, kernel_size=1)
            self.decoder = VQVAEDecoderV2(csz[::-1], activation, gdivisors[::-1])
            self.post_quant_conv = nn.Conv2d(latent_dim, csz[-1], kernel_size=1)
            self.quantizer = VQVAEQuantizerV1(codebook_sz, latent_dim)

        def encode(self, x):
            x = self.encoder(x)
            x = self.pre_quant_conv(x)
            _, _, _, qidx = self.quantizer(x)
            return qidx

        def decode(self, qidx):
            assert qidx.shape[1:] == self.embed_sz, f"Incorrect embedding tensor dimension, expecting {self.embed_sz}"

            out = self.quantizer.select(qidx, self.embed_sz)
            out = self.post_quant_conv(out)
            out = self.decoder(out)
            return out
    
        def forward(self, x):
            x = self.encoder(x)
            x = self.pre_quant_conv(x)

            qout, commitment_loss, codebook_loss, qidx = self.quantizer(x)

            assert qidx.shape[1:] == self.embed_sz, f"Model Embedding matrix size is set incorrectly, decode function is unusable. embedding size should be set to {qidx.shape[1:]}"
        
            out = self.post_quant_conv(qout)
            out = self.decoder(out)
            return out, qout, commitment_loss, codebook_loss, qidx
    return (VQVAEV2,)


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
    return


@app.cell
def _(mo):
    mo.md(r"""### Training and Test""")
    return


@app.class_definition
class EarlyStop:
    def __init__(self, patience=3):
        self.patience = patience
        self.steps = 0
        self.min_loss = float('inf')

    def stop(self, loss):
        to_save = False
        if loss < self.min_loss:
            self.min_loss = loss
            self.steps = 0
            to_save = True
        elif loss >= self.min_loss:
            self.steps += 1
            to_save = False
        if self.steps >= self.patience:
            to_stop = True
        else:
            to_stop = False
        return to_save, to_stop


@app.cell
def _(math, torch, tqdm):
    def train_vqvae(model, train_loader, test_loader, optim, loss, stopper, scaler, n_epoch, modelfile, beta = 0.2, device=torch.device("cpu")):
        model.train()
        for epoch in range(n_epoch):
            pbar = tqdm(train_loader)
            loss_ema = None
            for x, _ in pbar:
                x = x.to(device)
                with torch.amp.autocast("cuda"):
                    out, _, commit_loss, codebook_loss, _ = model(x)
                    l = loss(x, out) + codebook_loss + beta * commit_loss
                    if math.isnan(l.item()):
                        print(f"Loss=NaN. out={torch.sum(torch.isnan(out))} commit_loss={commit_loss} codebook_loss={codebook_loss}")
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
                pbar.set_description(f"train loss: {loss_ema:.4f}")
            model.eval()
            with torch.no_grad():
                test_loss = 0.
                for x, _ in test_loader:
                    batchsize = x.shape[0]
                    x = x.to(device)
                    out, _, commit_loss, codebook_loss, _ = model(x)
                    l = loss(x, out) + codebook_loss + beta * commit_loss
                    test_loss += l.item() * batchsize / len(test_loader.dataset)
                print(f"test loss={test_loss}")
            to_save, to_stop = stopper.stop(test_loss)
            if to_save:
                torch.save(model.state_dict(), modelfile)
            if to_stop:
                break
    return


@app.cell
def _(torch):
    @torch.no_grad()
    def sample(model, n_sample, device=torch.device("cpu"), seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        embed = torch.randint(0, model.codebook_sz, (n_sample, *model.embed_sz), device=device)
        out = model.decode(embed)
        return out
    return (sample,)


@app.cell
def _(np, random, torch):
    @torch.no_grad()
    def sample_comparison(model, dataset, n_sample, device=torch.device("cpu"), seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        idxes = [random.randint(0, len(dataset)-1) for _ in range(n_sample)]
        images = []
        for idx in idxes:
            input = dataset[idx][0]
            input = input.to(device)
            input = input.unsqueeze(0)
            images.append(input.detach().cpu().numpy())
            output, _, _, _, _ = model(input)
            images.append(output.detach().cpu().numpy())
        images = np.array(images)
        return images
    return (sample_comparison,)


@app.cell
def _():
    # def test_model(model):
    #     model = model.to(cpu)
    #     model.eval()
    #     test_idx = random.randint(0, len(test_data))
    #     input = test_data[test_idx][0]
    #     C, H, W = input.shape
    #     input = input.reshape((1, C, H, W))
    #     out, _, _, _, _ = model(input)
    #     out = out.reshape((C, H, W))
    #     img = transforms.functional.to_pil_image(out)
    #     return img
    return


@app.cell
def _():
    ### shared parameters
    return


@app.cell
def _(nn, torch):
    n_epoch = 20
    scaler = torch.cpu.amp.GradScaler()
    #scaler = torch.cuda.amp.GradScaler()
    loss = nn.MSELoss()
    stopper = EarlyStop()
    learning_rate=0.0001
    return (learning_rate,)


@app.cell
def _(mo):
    mo.md(r"""MNIST""")
    return


@app.cell
def _():
    # mnist_model = VQVAEV1(
    #     [1, 16, 32, 8, 4],
    #     [3, 3, 3, 2],
    #     [2, 2, 1, 1],
    #     [4, 8, 32, 16, 1],
    #     [3, 4, 4, 4],
    #     [1, 2, 1, 1],
    #     5,
    #     2,
    #     leaky_activation()
    # )
    return


@app.cell
def _(model_dir):
    mnist_model_file = model_dir + 'mnist_vqvae_v1.pth'
    return (mnist_model_file,)


@app.cell
def _(VQVAEV2, device, leaky_activation):
    mnist_model = VQVAEV2([1, 16, 256, 512], 128, 32, leaky_activation, (1, 28, 28), (7, 7)).to(device)
    return (mnist_model,)


@app.cell
def _(device, mnist_model, mnist_model_file, torch):
    mnist_model.load_state_dict(torch.load(mnist_model_file, map_location=device, weights_only=True))
    return


@app.cell
def _(mo):
    mo.md("""Simple MNIST Model and Training""")
    return


@app.cell
def _():
    data_dir = "../data"
    return (data_dir,)


@app.cell
def _(DataLoader, data_dir, datasets, transforms):
    mnist_path = data_dir
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_dataset = datasets.MNIST(mnist_path, train=True, transform=mnist_transform)
    mnist_test_dataset = datasets.MNIST(mnist_path, train=False, transform=mnist_transform)
    mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=32, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=32, shuffle=True)
    return (mnist_test_dataset,)


@app.cell
def _(learning_rate, mnist_model, torch):
    mnist_optim = torch.optim.AdamW(mnist_model.parameters(), lr=learning_rate)
    return


@app.cell
def _():
    # train_vqvae(mnist_model, mnist_train_loader, mnist_test_loader, mnist_optim, loss, stopper, scaler, n_epoch, mnist_model_file, device=device)
    return


@app.cell
def _():
    #torch.save(mnist_model.state_dict(), mnist_model_file)
    return


@app.cell
def _():
    ### testing sample
    return


@app.cell
def _(cpu, device, mnist_model, sample):
    mnist_images = sample(mnist_model, 8, device=device)
    mnist_images = mnist_images.to(cpu)
    return (mnist_images,)


@app.cell
def _(mnist_images, plt, torchvision):
    mnist_grid = torchvision.utils.make_grid(mnist_images*2.-1.0, nrow=1)
    plt.figure(dpi=100)
    plt.imshow(mnist_grid.permute(1,2,0), cmap='binary')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(device, mnist_model, mnist_test_dataset, sample_comparison):
    mnist_io = sample_comparison(mnist_model, mnist_test_dataset, 8, device=device)
    return (mnist_io,)


@app.cell
def _(mnist_io, plt, torch):
    plt.figure(figsize=(10, 8), dpi=100)

    for jjj in range(mnist_io.shape[0]):
        plt.subplot(6, 10, jjj+1)
        im = torch.tensor(mnist_io[jjj][0]).permute(1,2,0) * 2 - 1.0
        plt.imshow(im, cmap='binary')
        plt.axis('off')
        plt.title(f"{jjj}")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
