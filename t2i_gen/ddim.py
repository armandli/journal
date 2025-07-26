import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    ### loading dataset from hugging face
    return


@app.cell
def _():
    from datasets import load_dataset
    from torchvision.transforms import (CenterCrop, Compose,InterpolationMode,RandomHorizontalFlip, Resize,ToTensor)
    return (
        CenterCrop,
        Compose,
        InterpolationMode,
        RandomHorizontalFlip,
        Resize,
        ToTensor,
        load_dataset,
    )


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
    resolution=64

    augmentations = Compose([
        Resize(resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(resolution),
        RandomHorizontalFlip(),
        ToTensor(),]
    )

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    dataset = load_dataset("huggan/flowers-102-categories", split="train",)
    dataset.set_transform(transforms)
    return dataset, resolution


@app.cell
def _():
    from typing import Union
    import math
    import numpy as np
    from tqdm import tqdm

    from einops import rearrange
    from einops.layers.torch import Rearrange

    import torch
    from torch import nn
    from torch import einsum

    import matplotlib.pyplot as plt
    return Rearrange, Union, einsum, math, nn, np, plt, rearrange, torch, tqdm


@app.cell
def _(torch):
    device="cuda" if torch.cuda.is_available() else "cpu"
    return (device,)


@app.cell
def _(dataset, torch):
    torch.manual_seed(42)
    batch_size=4

    train_dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return batch_size, train_dataloader


@app.cell
def _(batch_size, plt, torch, train_dataloader):
    plt.figure(figsize=(5.9,batch_size),dpi=150)
    for col in range(6):
        imgs=next(iter(train_dataloader))["input"]
        for row in range(batch_size):
            plt.subplot(batch_size,6,col+1+row*6)
            img=imgs[row].permute(1,2,0) #B
            plt.imshow(torch.clip(img,0,1)) #C
            plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    ### generate noisy image for training using DDIM scheduler
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
def _(np, torch):
    def clip(tensor, min_value=None, max_value=None):
        if isinstance(tensor, np.ndarray):
            return np.clip(tensor, min_value, max_value)
        elif isinstance(tensor, torch.Tensor):
            return torch.clamp(tensor, min_value, max_value)

        raise ValueError("Tensor format is not valid is not valid - " f"should be numpy array or torch tensor. Got {type(tensor)}.")
    return (clip,)


@app.function
def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


@app.function
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


@app.cell
def _(Union, clip, cosine_beta_schedule, math, np, torch, tqdm):
    class DDIMScheduler:
        def __init__(
            self,
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="cosine",
            clip_sample=True,
            set_alpha_to_one=True
        ):
            if beta_schedule == "linear":
                self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
            elif beta_schedule == "cosine":
                self.betas = cosine_beta_schedule(num_train_timesteps, beta_start=beta_start, beta_end=beta_end)
            else:
                raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

            self.num_train_timesteps = num_train_timesteps
            self.clip_sample = clip_sample
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

            self.final_alpha_cumprod = np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

            self.num_inference_steps = None
            self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

        def _get_variance(self, timestep, prev_timestep):
            alpha_prod_t = self.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

            return variance

        def set_timesteps(self, num_inference_steps, offset=0):
            self.num_inference_steps = num_inference_steps
            self.timesteps = np.arange(0, 1000, 1000 // num_inference_steps)[::-1].copy()
            self.timesteps += offset

        def step(
            self,
            model_output: Union[torch.FloatTensor, np.ndarray],
            timestep: int,
            sample: Union[torch.FloatTensor, np.ndarray],
            eta: float = 1.0,
            use_clipped_model_output: bool = True,
            generator=None,
        ):
            # 1. get previous step value (=t-1)
            prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

            # 2. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t

            # 3. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)

            # 4. Clip "predicted x_0"
            if self.clip_sample:
                pred_original_sample = clip(pred_original_sample, -1, 1)

            # 5. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            variance = self._get_variance(timestep, prev_timestep)
            std_dev_t = eta * variance**(0.5)

            if use_clipped_model_output:
                # the model_output is always re-derived from the clipped x_0 in Glide
                model_output = (sample - alpha_prod_t**(0.5) * pred_original_sample) / beta_prod_t**(0.5)

            # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2)**(0.5) * model_output

            # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            prev_sample = alpha_prod_t_prev**(0.5) * pred_original_sample + pred_sample_direction

            if eta > 0:
                device = model_output.device if torch.is_tensor(model_output) else "cpu"
                noise = torch.randn(model_output.shape, generator=generator).to(device)
                variance = self._get_variance(timestep, prev_timestep)**(0.5) * eta * noise

                if not torch.is_tensor(model_output):
                    variance = variance.numpy()

                prev_sample = prev_sample + variance

            return prev_sample

        def add_noise(self, original_samples, noise, timesteps):
            timesteps = timesteps.cpu()
            sqrt_alpha_prod = self.alphas_cumprod[timesteps]**0.5
            sqrt_alpha_prod = match_shape(sqrt_alpha_prod, original_samples)
            sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps])**0.5
            sqrt_one_minus_alpha_prod = match_shape(sqrt_one_minus_alpha_prod, original_samples)

            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
            return noisy_samples

        @torch.no_grad()
        def generate(self,model,device,batch_size=1,generator=None,eta=1.0,use_clipped_model_output=True,num_inference_steps=50):
            # save intermediate steps
            imgs=[]
            # the initial random noise
            image=torch.randn((batch_size,model.in_channels,model.sample_size, model.sample_size),generator=generator).to(device)
            self.set_timesteps(num_inference_steps)
            # start denoising
            for t in tqdm(self.timesteps):
                model_output = model(image, t)["sample"]
                image = self.step(model_output,t,image,eta, use_clipped_model_output=use_clipped_model_output)
                img = unnormalize_to_zero_to_one(image)
                img = img.cpu().permute(0, 2, 3, 1).numpy()
                imgs.append(img)
            # output the final clean image and the intermediate steps    
            image = unnormalize_to_zero_to_one(image)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            return image, imgs

        @torch.no_grad()
        def interpolate(
            self,
            model,
            a_idx,
            b_idx,
            batch_size=1,
            generator=None,
            eta=1.0,
            use_clipped_model_output=True,
            num_inference_steps=50,
            device=None
        ):
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            image0 = torch.randn((batch_size, model.in_channels, model.sample_size, model.sample_size),generator=generator,).to(device)
            image = torch.zeros((batch_size, model.in_channels, model.sample_size, model.sample_size)).to(device)
            # pick two initial noise tensors
            a,b=image0[a_idx],image0[b_idx]
            # interpolate with different weights
            for i in range(10):
                ab=torch.sin(torch.tensor(0.5*math.pi*(0.05+i/10)))*a+torch.cos(torch.tensor(0.5*math.pi*(0.05+i/10)))*b
                image[i]=ab
            self.set_timesteps(num_inference_steps)
            # generate 
            for t in tqdm(self.timesteps):
                model_output = model(image, t)["sample"]
                image = self.step(model_output,t,image,eta,use_clipped_model_output=use_clipped_model_output,generator=generator)
            image = unnormalize_to_zero_to_one(image)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            return image

        def __len__(self):
            return self.num_train_timesteps
    return (DDIMScheduler,)


@app.cell
def _(DDIMScheduler, plt, torch, train_dataloader):
    noise_scheduler=DDIMScheduler(num_train_timesteps=1000)

    imgs2=next(iter(train_dataloader))["input"]
    imgs2=imgs2.permute(0,2,3,1).reshape(-1,64,3)

    allimgs=[imgs2]
    for step in [200,400,600,800,1000]:
        timesteps=torch.tensor([step-1]).long()
        noisy_image=noise_scheduler.add_noise(imgs2, torch.randn(imgs2.shape), timesteps)
        allimgs.append(noisy_image)
    plt.figure(figsize=(10,8),dpi=80)
    for i in range(6):
        plt.subplot(1,6,i+1)
        plt.imshow(torch.clip(allimgs[i],0,1))
        plt.axis('off')
        plt.title(f"t={200*i}",fontsize=16)
    plt.tight_layout(w_pad=-0.1)
    plt.show()
    return (noise_scheduler,)


@app.cell
def _():
    ### DDIM Unet
    return


@app.cell
def _(math, torch):
    # fixed positional embedding, applying sine to even position and cos to odd position
    def sinusoidal_embedding(timesteps, dim):
        half_dim = dim // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        exponent = exponent / (half_dim - 1.0)
        emb = torch.exp(exponent).to(device=timesteps.device)
        emb = timesteps[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
    return (sinusoidal_embedding,)


@app.cell
def _(einsum, nn, rearrange):
    class Attention(nn.Module):
        def __init__(self, dim, heads=4, dim_head=32):
            super().__init__()
            self.scale = dim_head**-0.5
            self.heads = heads
            hidden_dim = dim_head * heads

            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            qkv = self.to_qkv(x).chunk(3, dim=1)
            q, k, v = map(
                lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
                qkv)

            q = q * self.scale

            sim = einsum('b h d i, b h d j -> b h i j', q, k)
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h d j -> b h i d', attn, v)

            out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
            return self.to_out(out)  
    return (Attention,)


@app.cell
def _(nn, torch):
    class ResidualBlock(nn.Module):
        def __init__(self,
                     in_channels,
                     out_channels,
                     temb_channels,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     groups=8):
            super(ResidualBlock, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            self.time_emb_proj = nn.Sequential(nn.SiLU(), torch.nn.Linear(temb_channels, out_channels))

            self.residual_conv = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

            self.conv1 = nn.Conv2d(in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
            self.conv2 = nn.Conv2d(out_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)

            self.norm1 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
            self.nonlinearity = nn.SiLU()

        def forward(self, x, temb):
            residual = self.residual_conv(x)

            x = self.conv1(x)
            x = self.norm1(x)
            x = self.nonlinearity(x)

            temb = self.time_emb_proj(self.nonlinearity(temb))
            x += temb[:, :, None, None]

            x = self.conv2(x)
            x = self.norm2(x)
            x = self.nonlinearity(x)

            return x + residual
    return (ResidualBlock,)


@app.cell
def _(nn):
    class Residual(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, *args, **kwargs):
            return self.fn(x, *args, **kwargs) + x
    return (Residual,)


@app.cell
def _(nn, torch):
    class LayerNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

        def forward(self, x):
            eps = 1e-5 if x.dtype == torch.float32 else 1e-3
            var = torch.var(x, dim=1, unbiased=False, keepdim=True)
            mean = torch.mean(x, dim=1, keepdim=True)
            return (x - mean) * (var + eps).rsqrt() * self.g
    return (LayerNorm,)


@app.cell
def _(LayerNorm, nn):
    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super().__init__()
            self.fn = fn
            self.norm = LayerNorm(dim)

        def forward(self, x):
            x = self.norm(x)
            return self.fn(x)
    return (PreNorm,)


@app.cell
def _(Attention, PreNorm, Residual, nn):
    def get_attn_layer(in_dim, is_last):
        if is_last:
            return Residual(PreNorm(in_dim, Attention(in_dim)))
        else:
            return nn.Identity()
    return (get_attn_layer,)


@app.cell
def _(nn):
    def get_upsample_layer(in_dim, hidden_dim, is_last):
        if not is_last:
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_dim, hidden_dim, 3, padding=1))
        else:
            return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
    return (get_upsample_layer,)


@app.cell
def _(Rearrange, nn):
    def get_downsample_layer(in_dim, hidden_dim, is_last):
        if not is_last:
            return nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
                nn.Conv2d(in_dim * 4, hidden_dim, 1)
            )
        else:
            return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
    return (get_downsample_layer,)


@app.cell
def _(
    Attention,
    PreNorm,
    Residual,
    ResidualBlock,
    get_attn_layer,
    get_downsample_layer,
    get_upsample_layer,
    nn,
    sinusoidal_embedding,
    torch,
):
    class UNet(nn.Module):
        def __init__(self, in_channels, hidden_dims=[128, 256, 512, 1024], image_size=64):
            super(UNet, self).__init__()

            self.sample_size = image_size
            self.in_channels = in_channels
            self.hidden_dims = hidden_dims

            timestep_input_dim = hidden_dims[0]
            time_embed_dim = timestep_input_dim * 4

            self.time_embedding = nn.Sequential(
                nn.Linear(timestep_input_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )

            self.init_conv = nn.Conv2d(in_channels,
                                       out_channels=hidden_dims[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

            down_blocks = []

            in_dim = hidden_dims[0]
            for idx, hidden_dim in enumerate(hidden_dims[1:]):
                is_last = idx >= (len(hidden_dims) - 2)
                down_blocks.append(
                    nn.ModuleList([
                        ResidualBlock(in_dim, in_dim, time_embed_dim),
                        ResidualBlock(in_dim, in_dim, time_embed_dim),
                        get_attn_layer(in_dim, is_last),
                        get_downsample_layer(in_dim, hidden_dim, is_last)
                    ]))
                in_dim = hidden_dim

            self.down_blocks = nn.ModuleList(down_blocks)

            mid_dim = hidden_dims[-1]
            self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)

            up_blocks = []
            in_dim = mid_dim
            for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
                is_last = idx >= (len(hidden_dims) - 2)
                up_blocks.append(
                    nn.ModuleList([
                        ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                        ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                        get_attn_layer(in_dim, is_last),
                        get_upsample_layer(in_dim, hidden_dim, is_last)
                    ]))
                in_dim = hidden_dim

            self.up_blocks = nn.ModuleList(up_blocks)

            self.out_block = ResidualBlock(hidden_dims[0] * 2, hidden_dims[0],
                                           time_embed_dim)
            self.conv_out = nn.Conv2d(hidden_dims[0], out_channels=3, kernel_size=1)

        def forward(self, sample, timesteps):
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps],
                                         dtype=torch.long,
                                         device=sample.device)

            timesteps = torch.flatten(timesteps)
            timesteps = timesteps.broadcast_to(sample.shape[0])

            t_emb = sinusoidal_embedding(timesteps, self.hidden_dims[0])
            t_emb = self.time_embedding(t_emb)

            x = self.init_conv(sample)
            r = x.clone()

            skips = []
            for block1, block2, attn, downsample in self.down_blocks:
                x = block1(x, t_emb)
                skips.append(x)

                x = block2(x, t_emb)
                x = attn(x)
                skips.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t_emb)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t_emb)

            for block1, block2, attn, upsample in self.up_blocks:
                x = torch.cat((x, skips.pop()), dim=1)
                x = block1(x, t_emb)

                x = torch.cat((x, skips.pop()), dim=1)
                x = block2(x, t_emb)
                x = attn(x)

                x = upsample(x)

            x = self.out_block(torch.cat((x, r), dim=1), t_emb)
            out = self.conv_out(x)
            return {"sample": out}
    return (UNet,)


@app.cell
def _(UNet, device, resolution):
    model = UNet(3,hidden_dims=[128,256,512,1024], image_size=resolution).to(device)
    num = sum(p.numel() for p in model.parameters())
    print("number of parameters: %.2fM" % (num/1e6,)) 
    return (model,)


@app.cell
def _():
    ### training
    return


@app.cell
def _(model, torch, train_dataloader):
    from diffusers.optimization import get_scheduler

    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001, betas=(0.95,0.999),weight_decay=0.00001,eps=1e-8)
    lr_scheduler=get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=300, num_training_steps=(len(train_dataloader) * 100))
    return


@app.cell
def _():
    class EarlyStop:
        def __init__(self, patience=3):
            self.patience = patience
            self.steps = 0
            self.min_loss = float('inf')

        def stop(self, loss):
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

    stopper = EarlyStop()
    return


@app.cell
def _():
    model_dir = 'models/'
    return (model_dir,)


@app.cell
def _(model_dir):
    model_file = model_dir + 'ddim.pth'
    model_file_ref = model_dir + 'ddim_ref.pth'
    return (model_file_ref,)


@app.cell
def _():
    # for epoch in range(100):
    #     model.train()
    #     tloss = 0
    #     print(f"start epoch {epoch}")
    #     for step2, batch in enumerate(train_dataloader):
    #         clean_images = batch["input"].to(device)*2-1
    #         nums = clean_images.shape[0]
    #         noise = torch.randn(clean_images.shape).to(device)
    #         timesteps2 = torch.randint(0, noise_scheduler.num_train_timesteps, (nums, ), device=device).long()
    #         noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps2)
    #         noise_pred = model(noisy_images, timesteps2)["sample"]
    #         loss = torch.nn.functional.l1_loss(noise_pred, noise)
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         tloss += loss.detach().item()
    #         if step2%100==0:
    #             print(f"step {step2}, average loss {tloss/(step2+1)}")
    # torch.save(model.state_dict(), model_file)
    return


@app.cell
def _():
    ### image generation and interpolation
    return


@app.cell
def _(device, model, model_file_ref, torch):
    model.load_state_dict(torch.load(model_file_ref, map_location=device, weights_only=True))
    return


@app.cell
def _():
    ### use trained denoising diffusion model to generate image
    return


@app.cell
def _(device, model, noise_scheduler, plt, torch):
    with torch.no_grad():
        generator = torch.manual_seed(100)
        generated_images, _ = noise_scheduler.generate(
            model,
            device=device,
            num_inference_steps=50,
            generator=generator,
            eta=0,
            use_clipped_model_output=True,
            batch_size=10
        )

    imgnp=generated_images

    plt.figure(figsize=(10,4),dpi=100)
    for ii in range(10):
        ax = plt.subplot(2, 5, ii + 1)
        plt.imshow(imgnp[ii])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    ### transition from one image to another
    return


@app.cell
def _(model, noise_scheduler, plt, torch):
    with torch.no_grad():
        generator2 = torch.manual_seed(100)
        generated_images3 = noise_scheduler.interpolate(
            model,1,7,
            num_inference_steps=50,
            generator=generator2,
            eta=0,
            use_clipped_model_output=True,
            batch_size=10
        )

    imgnp3=generated_images3

    plt.figure(figsize=(10,4),dpi=300)
    for iii in range(10):
        _ = plt.subplot(1,10, iii + 1) 
        plt.imshow(imgnp3[iii])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    ### generate 3 sets of composite images
    return


@app.cell
def _(model, noise_scheduler, np, plt, torch):
    with torch.no_grad():
        generator3 = torch.manual_seed(100)
        generated_images4 = noise_scheduler.interpolate(
            model,2,5,
            num_inference_steps=50,
            generator=generator3,
            eta=0,
            use_clipped_model_output=True,
            batch_size=10
        )

    imgnp4 = generated_images4

    with torch.no_grad():
        generator4 = torch.manual_seed(100)
        generated_images5 = noise_scheduler.interpolate(
            model,7,2,
            num_inference_steps=50,
            generator=generator4,
            eta=0,
            use_clipped_model_output=True,
            batch_size=10
        )

    imgnp5 = generated_images5

    with torch.no_grad():
        generator5 = torch.manual_seed(100)
        generated_images6 = noise_scheduler.interpolate(
            model,3,1,
            num_inference_steps=50,
            generator=generator5,
            eta=0,
            use_clipped_model_output=True,
            batch_size=10
        )

    imgnp6 = generated_images6

    img_array=np.concatenate([imgnp4,imgnp5,imgnp6])
    fig, axs = plt.subplots(nrows=3,ncols=10,sharex=True,sharey=True,figsize=(10,3),dpi=100)
    for row2 in range(3):
        for col2 in range(10):
            axs[row2, col2].clear()
            axs[row2, col2].set_xticks([])
            axs[row2, col2].set_yticks([])
            axs[row2, col2].imshow(img_array[col2+row2*10])

    plt.subplots_adjust(bottom=0.001,right=0.999,top=0.999,left=0.001, hspace=-0.1,wspace=-0.1)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
