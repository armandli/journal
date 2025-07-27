import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    ### image generation with diffusers library
    return


@app.cell
def _():
    import numpy as np

    import torch
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL

    import matplotlib.pyplot as plt
    return (
        AutoencoderKL,
        CLIPTextModel,
        CLIPTokenizer,
        StableDiffusionPipeline,
        plt,
        torch,
    )


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (device,)


@app.cell
def _():
    data_dir = '/Users/armandli/data/'
    model_dir = 'models/'
    return (data_dir,)


@app.cell
def _(torch):
    torch.manual_seed(42)
    return


@app.cell
def _(StableDiffusionPipeline, device, torch):
    prompt="an astronaut in a spacesuit riding a unicorn"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        variant="fp16",
        torch_dtype=torch.float16
    ).to(device)
    return pipe, prompt


@app.cell
def _(pipe, prompt):
    image = pipe(prompt).images[0]
    image.save(f"files/{prompt}.png")
    image
    return


@app.cell
def _(data_dir, pipe, torch):
    from PIL import Image

    torch.manual_seed(0)

    prompts = [
        "a panda eating a bowl of noodles",
        "a dog with sunglasses and a straw hat",
        "a cat stretching on the floor"
    ] 

    images = pipe(prompts).images

    grid = Image.new('RGB', size=(3*images[0].size[0], images[0].size[1]))
    for i, img in enumerate(images):
        grid.paste(img, box=(i*images[0].size[0], 0))
    grid.save(data_dir + "three_prompts.png")
    grid
    return (images,)


@app.cell
def _(data_dir, pipe, torch):
    torch.manual_seed(42)

    prompt = "dogs running on a sandy beach under blue sky"

    images = pipe(prompt,width=768,height=512, guidance_scale=12).images

    images[0].save(data_dir + f"/{prompt}.png")
    images[0]
    return images, prompt


@app.cell
def _():
    class Config:
        height = 512
        width = 512
        guidance_scale = 7.5
        num_inference_steps=50

    config=Config()

    prompts1 = ["a lion"]
    prompts2 = ["a tiger"]
    return config, prompts1, prompts2


@app.cell
def _(CLIPTextModel, CLIPTokenizer, device):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    return text_encoder, tokenizer


@app.cell
def _(device, prompts1, prompts2, text_encoder, tokenizer, torch):
    tokens1=tokenizer(prompts1,padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    tokens2=tokenizer(prompts2,padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        text_embeddings1=text_encoder(tokens1.input_ids.to(device))[0]
        text_embeddings2=text_encoder(tokens2.input_ids.to(device))[0]
    print("text embedding 1 shape:", text_embeddings1.shape)
    print("text embedding 2 shape:", text_embeddings1.shape)
    return text_embeddings1, text_embeddings2


@app.cell
def _(device, text_encoder, tokenizer, torch):
    unconditional_prompt = [""]
    uncond_input = tokenizer(unconditional_prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")

    with torch.no_grad():
        uncond_embeds=text_encoder(uncond_input.input_ids.to(device))[0]
    return (uncond_embeds,)


@app.cell
def _(device):
    from diffusers import LMSDiscreteScheduler
    from diffusers import UNet2DConditionModel

    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    unet = unet.to(device)
    return scheduler, unet


@app.cell
def _(config, device, scheduler, torch, uncond_embeds, unet):
    from tqdm import tqdm

    def gen_latents(text_embeddings, seed):
        torch.manual_seed(seed)
        scheduler.set_timesteps(config.num_inference_steps)
        latents = torch.randn((1, unet.config.in_channels, config.height // 8, config.width // 8)).to(device)
        latents = latents * scheduler.init_noise_sigma
        text_embeddings=torch.cat([uncond_embeds, text_embeddings])
        with torch.autocast(device):
            for i, t in tqdm(enumerate(scheduler.timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred=noise_pred_uncond+config.guidance_scale*(noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        # reverse a scaling of 0.18215 during training in stable diffusion
        # 0.18215 is the latent space standard deviation
        return 1 / 0.18215 * latents
    return (gen_latents,)


@app.cell
def _(gen_latents, text_embeddings1, text_embeddings2):
    latents1=gen_latents(text_embeddings1,seed=5)
    latents2=gen_latents(text_embeddings2,seed=5)
    print(f"latent image size is {latents1[0].shape}")
    return latents1, latents2


@app.cell
def _(AutoencoderKL, device):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    return (vae,)


@app.cell
def _(latents1, latents2, torch, vae):
    with torch.no_grad():
        images1 = vae.decode(latents1).sample
        images2 = vae.decode(latents2).sample
    print(f"final image size is {images2[0].shape}")
    return images1, images2


@app.cell
def _(torch):
    def latent_to_display_image(latent, upsample_factor=8):
        image = latent[0].permute(1, 2, 0).detach().cpu()
        image = torch.clamp(image, -1, 1)
        image = image.repeat_interleave(upsample_factor, dim=0).repeat_interleave(upsample_factor, dim=1)
        image = (image + 1) / 2
        return image.numpy()
    return (latent_to_display_image,)


@app.cell
def _(torch):
    def output_to_display_image(output):
        image = output[0].permute(1, 2, 0).detach().cpu()
        image = (image + 1) / 2
        image = torch.clamp(image, 0, 1)
        return image.numpy()
    return (output_to_display_image,)


@app.cell
def _(
    images1,
    images2,
    latent_to_display_image,
    latents1,
    latents2,
    output_to_display_image,
    plt,
):
    display_imgs = [
        latent_to_display_image(latents1),
        output_to_display_image(images1),
        latent_to_display_image(latents2),
        output_to_display_image(images2)
    ]
    captions = [
        "latent image\n a lion",
        "final output\n a lion",
        "latent image\n a tiger",
        "final output\n a tiger"
    ]

    plt.figure(figsize=(8, 5), dpi=100)
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(display_imgs[i])
        plt.title(captions[i], fontsize=15)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(gen_latents, text_embeddings1, text_embeddings2, torch, vae):
    images=[]
    for weight in [0.9, 0.7, 0.5, 0.3, 0.1]:
        text_embeddings = weight * text_embeddings1 + (1-weight) * text_embeddings2
        latents=gen_latents(text_embeddings,seed=5)
        with torch.no_grad():
            images.append(vae.decode(latents).sample)
    return (images,)


@app.cell
def _(images, plt, torch):
    plt.figure(figsize=(8,5),dpi=100)
    for i in range(5):
        plt.subplot(1,5,i+1)
        img=torch.clip(images[i][0].detach().cpu().permute(1,2,0)/2+0.5,0,1)
        plt.imshow(img)
        plt.title(f"{90-20*i}% lion\n {10+20*i}% tiger", fontsize=15)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
