import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    ### visualize forward diffusion process
    return


@app.cell
def _():
    ### linear forward diffusion schedule
    return


@app.cell
def _():
    data_dir = '../data/'
    model_dir = 'models/'
    return data_dir, model_dir


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt
    import PIL
    import numpy as np
    import torchvision
    from torch.utils.data import DataLoader
    return DataLoader, PIL, np, plt, torch, torchvision


@app.cell
def _(torch):
    T=1000
    t=torch.arange(0, T + 1,dtype=torch.float32)/T
    def linear_scheduler():
        beta1, beta2 = 0.0001, 0.02
        # increasing linearly over time
        beta_t = (beta2 - beta1) * t + beta1
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
        # weight placed on clean image x0
        sqrtab = torch.sqrt(alphabar_t)
        # weight placed on noise epsilon0
        sqrtmab = torch.sqrt(1 - alphabar_t) 
        return {"sqrtab": sqrtab,"sqrtmab": sqrtmab}
    return linear_scheduler, t


@app.cell
def _(PIL, data_dir, linear_scheduler, np, plt, torch):
    def linear_noisy_image(image, timestep):
        alphabar_t = linear_scheduler()["sqrtab"][timestep]
        noisy = image * torch.sqrt(alphabar_t) + torch.randn_like(image) * torch.sqrt(1 - alphabar_t)
        return torch.clip(noisy, min=-1, max=1)

    imgs=[]
    for name in ["bird","dog","mountains","horse"]:
        img=np.array(PIL.Image.open(data_dir + f"images/{name}.png"))
        img=torch.tensor(2*(img/255)-1)
        for timestep in [0,200,400,600,800,1000]:
            imgs.append(linear_noisy_image(img,timestep)/2+0.5)

    plt.figure(figsize=(12,8),dpi=100)

    for i in range(24):
        plt.subplot(4,6,i+1)
        plt.imshow(imgs[i])
        if i < 6:
            plt.title(f't={200*(i%6)}',fontsize=12,c="r")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    return (linear_noisy_image,)


@app.cell
def _():
    ### cosine forward diffusion schedule
    return


@app.cell
def _(linear_scheduler, plt, t, torch):
    import math
    def cosine_scheduler():
        sqrtab = torch.cos(0.5*math.pi*t)
        sqrtmab = torch.sin(0.5*math.pi*t)
        return {"sqrtab": sqrtab,"sqrtmab": sqrtmab}

    plt.figure(figsize=(10,6),dpi=100)
    plt.plot(t, linear_scheduler()["sqrtmab"], linewidth=3, linestyle="solid",c="r", label="linear schedule")
    plt.plot(t, cosine_scheduler()["sqrtmab"], linewidth=3, linestyle="dotted",c="g", label="cosine schedule")
    plt.xlabel(r"relative time, t/T", fontsize=12)
    plt.ylabel(r"weight on $\epsilon$: $\sqrt{1-\bar{\alpha_t}}$", fontsize=12)
    plt.legend()
    plt.show()
    return (cosine_scheduler,)


@app.cell
def _():
    ### weight on original image over step t
    return


@app.cell
def _(cosine_scheduler, linear_scheduler, plt, t):
    plt.figure(figsize=(10,6),dpi=100)
    plt.plot(t, linear_scheduler()["sqrtab"], linewidth=3, linestyle="solid",c="r", label="linear schedule")
    plt.plot(t, cosine_scheduler()["sqrtab"], linewidth=3, linestyle="dotted",c="g", label="cosine schedule")
    plt.xlabel(r"relative time, t/T", fontsize=12)
    plt.ylabel(r"weight on $x_0$: $\sqrt{\bar{\alpha_t}}$", fontsize=12)
    plt.legend()
    plt.show()
    return


@app.cell
def _(PIL, cosine_scheduler, data_dir, linear_noisy_image, np, plt, torch):
    def cosine_noisy_image(image,timestep): #A
        alphabar_t=cosine_scheduler()["sqrtab"][timestep]
        noisy=image * torch.sqrt(alphabar_t) + torch.randn_like(image) * torch.sqrt(1 - alphabar_t)
        return torch.clip(noisy,min=-1,max=1)

    img2=np.array(PIL.Image.open(data_dir + "images/bunny.png"))
    img2=torch.tensor(2*(img2/255)-1)
    imgs2=[]

    for timestep2 in [0,166,332,498,664,830,996]:
        imgs2.append(linear_noisy_image(img2,timestep2)/2+0.5)
    for timestep2 in [0,166,332,498,664,830,996]:
        imgs2.append(cosine_noisy_image(img2,timestep2)/2+0.5)
    plt.figure(figsize=(13,4),dpi=100)
    for ii in range(14):
        plt.subplot(2,7,ii+1)
        plt.imshow(imgs2[ii])
        plt.xticks([])
        plt.yticks([])
        if ii==0:
            plt.ylabel("linear schedule",fontsize=15,c="r")
        if ii==7:
            plt.ylabel("cosine schedule",fontsize=15,c="r")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    ### use diffusers library to create a UNet diffusion model on fashion MNIST
    return


@app.cell
def _(torch):
    import diffusers

    device="cuda" if torch.cuda.is_available() else "cpu"

    model=diffusers.UNet2DModel(
        sample_size=32, in_channels=1, out_channels=1,layers_per_block=2, block_out_channels=(128,128,256,512),
        down_block_types=("DownBlock2D","DownBlock2D", "AttnDownBlock2D","DownBlock2D",),
        up_block_types=("UpBlock2D","AttnUpBlock2D", "UpBlock2D","UpBlock2D",),
    ).to(device)
    return device, diffusers, model


@app.cell
def _(torch):
    from torchvision import transforms

    # manual seed so result is reproducible
    torch.manual_seed(42)

    # need to transform 1,28,28 -> 1,32,32 due to diffusers model requirement, ToTensor convert from 0-255 -> 0. -> 1., then to range 
    # -1 to 1
    tf = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(), transforms.Lambda(lambda x: 2*(x-0.5)),])
    return (tf,)


@app.cell
def _(data_dir, tf, torchvision):
    dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf,)
    return (dataset,)


@app.cell
def _():
    text_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return (text_labels,)


@app.cell
def _():
    ### visualize the dataset
    return


@app.cell
def _(dataset, plt, text_labels):
    plt.figure(dpi=200,figsize=(8,4))
    for iii in range(24):
        ax=plt.subplot(3, 8, iii + 1)
        img3=dataset[iii+888][0]
        img3=img3/2+0.5
        img3=img3.reshape(32,32)
        plt.imshow(img3, cmap="binary")
        plt.axis('off')
        plt.title(text_labels[dataset[iii+888][1]],fontsize=8)
    plt.show()
    return


@app.cell
def _(DataLoader, dataset):
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    return


@app.cell
def _(diffusers):
    noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    return (noise_scheduler,)


@app.cell
def _(device, torch):
    @torch.no_grad()
    def sample(n_sample, model, noise_scheduler, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        noise_scheduler.set_timesteps(1000)
        image=torch.randn((n_sample,1,32,32)).to(device)
        for t in noise_scheduler.timesteps:
            model_output=model(image,t)['sample']
            image=noise_scheduler.step(model_output,int(t), image,generator=None)['prev_sample']
        return image
    return (sample,)


@app.cell
def _(model, torch):
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    return


@app.cell
def _(model_dir):
    model_file = model_dir + 'diffusion_fashion.pth'
    model_file_ref = model_dir + 'diffusion_fashion_ref.pth'
    return (model_file,)


@app.cell
def _():
    # from tqdm import tqdm
    # from torchvision.utils import save_image, make_grid

    # scaler = torch.cpu.amp.GradScaler()
    # for iiii in range(10):
    #     pbar = tqdm(dataloader)
    #     loss_ema = None
    #     for x, _ in pbar:
    #         x = x.to(device)
    #         with torch.amp.autocast("cuda"):
    #             noise=torch.randn_like(x).to(device)
    #             timesteps=torch.randint(0, noise_scheduler.config.num_train_timesteps, (x.shape[0],),device=device)
    #             noisy=noise_scheduler.add_noise(x, noise, timesteps)
    #             noise_pred=model(noisy,timesteps)["sample"]
    #             loss = torch.nn.functional.l1_loss(noise_pred,noise)
    #         optim.zero_grad()
    #         scaler.scale(loss).backward()
    #         scaler.step(optim)
    #         scaler.update()
    #         if loss_ema is None:
    #             loss_ema = loss.item()
    #         else:
    #             loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
    #         pbar.set_description(f"loss: {loss_ema:.4f}")
    #     xh = sample(32, model, noise_scheduler)
    #     grid = make_grid(0.5-xh/2, nrow=8)
    #     save_image(grid, data_dir + f"images/diffusion_fashion{iiii}.png")
    # torch.save(model, model_dir + "diffusion_fashion.pth")
    return


@app.cell
def _():
    ### visualize output
    return


@app.cell
def _(device, model, model_file, torch):
    #model.load_state_dict(torch.load(model_file_ref, map_location=device, weights_only=True))
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    return


@app.cell
def _(data_dir, model, noise_scheduler, plt, sample, torch, torchvision):
    torch.manual_seed(42)

    generated_images = sample(32, model, noise_scheduler)
    grid = torchvision.utils.make_grid(0.5-generated_images/2, nrow=8)
    torchvision.utils.save_image(grid, data_dir + f"images/diffusion_fashion.png")
    plt.figure(dpi=100)
    plt.imshow(grid.cpu().permute(1,2,0))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
