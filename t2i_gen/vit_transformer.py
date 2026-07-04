import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    ### download cifar10 dataset
    return


@app.cell
def _():
    import torchvision

    data_dir = '../data/'
    model_dir = 'models/'

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    testset=torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    return data_dir, model_dir, testset, torchvision, trainset


@app.cell
def _():
    ### label names
    return


@app.cell
def _():
    names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return (names,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    # plt.figure(figsize=(12,6),dpi=100)
    # for i in range(3):
    #     for j in range(6):
    #         plt.subplot(3, 6, 6*i+j+1)
    #         plt.imshow(trainset[6*i+j][0])
    #         plt.axis('off')
    #         plt.title(names[trainset[6*i+j][1]], fontsize=12)
    # plt.subplots_adjust(hspace=0.20)
    # plt.show()
    return math, np, plt


@app.cell
def _():
    ### prepare dataset for training and testing
    return


@app.cell
def _(trainset):
    import torchvision.transforms as transforms
    trainset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32),antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2,antialias=True), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return (transforms,)


@app.cell
def _(trainset):
    import torch
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    return torch, trainloader


@app.cell
def _(testset, torch, transforms):
    testset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32),antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    return (testloader,)


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (device,)


@app.cell
def _():
    class Config:
        patch_size=4
        hidden_size=48
        num_hidden_layers=4
        num_attention_heads=4
        intermediate_size=4 * 48
        image_size=32
        num_classes=10
        num_channels=3
    config = Config()
    return Config, config


@app.cell
def _():
    ### divide image into 64 patches
    return


@app.cell
def _():
    from torch import nn
    from torch import optim
    return nn, optim


@app.cell
def _(nn):
    class PatchEmbeddings(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.projection = nn.Conv2d(config.num_channels, config.hidden_size,  kernel_size=config.patch_size,  stride=config.patch_size)
        def forward(self, x):
            x = self.projection(x)
            # output dim 64 * 48
            x = x.flatten(2).transpose(1, 2)
            return x
    return (PatchEmbeddings,)


@app.cell
def _(PatchEmbeddings, config, torch):
    patchembed=PatchEmbeddings(config)
    img=torch.randn((1,3,32,32))
    out=patchembed(img)
    print(out.shape)
    return img, out, patchembed


@app.cell
def _():
    ### adding position to each patch
    return


@app.cell
def _(PatchEmbeddings, nn, torch):
    class Embeddings(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.patch_embeddings = PatchEmbeddings(config)
            # class token used for aggregating information across the entire image for image analysis task
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))  
            num_patches = (config.image_size // config.patch_size) ** 2
            # learning the positional embedding
            self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))  
        def forward(self, x):
            x = self.patch_embeddings(x)
            batch_size, _, _ = x.size()
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.position_embeddings
            return x
    return (Embeddings,)


@app.cell
def _(Embeddings, config, torch):
    embed=Embeddings(config)
    img2=torch.randn((1,3,32,32))
    out2=embed(img2)
    print(out2.shape)
    print(embed.position_embeddings.shape)
    return embed, img2, out2


@app.cell
def _():
    ### split the 48 dim key, value, query into 4 heads of 12 dim, each pays attention to to different part of the input to form a more broad and contextual understanding of the image
    return


@app.cell
def _(math, nn, torch):
    class AttentionHead(nn.Module):
        def __init__(self, hidden_size, attention_head_size, bias=True):
            super().__init__()
            self.hidden_size = hidden_size
            self.attention_head_size = attention_head_size
            self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
            self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
            self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)   
        def forward(self, x):
            query = self.query(x)
            key = self.key(x)
            value = self.value(x)
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_probs, value)
            return (attention_output, attention_probs)
    return (AttentionHead,)


@app.cell
def _(AttentionHead, nn, torch):
    class MultiHeadAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = self.hidden_size // self.num_attention_heads
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            self.heads = nn.ModuleList([])
            for _ in range(self.num_attention_heads):
                head = AttentionHead(self.hidden_size, self.attention_head_size)
                self.heads.append(head)
            self.output_projection = nn.Linear(self.all_head_size, self.hidden_size) 
        def forward(self, x, output_attentions=False):
            attention_outputs = [head(x) for head in self.heads]
            attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
            attention_output = self.output_projection(attention_output)
            if not output_attentions:
                return (attention_output, None)
            else:
                attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
                return (attention_output, attention_probs)
    return (MultiHeadAttention,)


@app.cell
def _(math, nn, torch):
    class GELU(nn.Module):
        def forward(self, input):
            return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    return (GELU,)


@app.cell
def _(GELU, nn):
    class MLP(nn.Module):
        """
        A multi-layer perceptron module.
        """

        def __init__(self, config):
            super().__init__()
            # intermediate_size > hidden_size
            self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
            self.activation = GELU()
            self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)

        def forward(self, x):
            x = self.dense_1(x)
            x = self.activation(x)
            x = self.dense_2(x)
            return x
    return (MLP,)


@app.cell
def _(MLP, MultiHeadAttention, nn):
    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attention = MultiHeadAttention(config)
            self.layernorm_1 = nn.LayerNorm(config.hidden_size)
            self.mlp = MLP(config)
            self.layernorm_2 = nn.LayerNorm(config.hidden_size)
        def forward(self, x, output_attentions=False):
            attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
            x = x + attention_output
            mlp_output = self.mlp(self.layernorm_2(x))
            x = x + mlp_output  
            if not output_attentions:
                return (x, None)
            else:
                return (x, attention_probs)
    return (Block,)


@app.cell
def _(Block, nn):
    class Encoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.blocks = nn.ModuleList([])
            for _ in range(config.num_hidden_layers):
                block = Block(config)
                self.blocks.append(block)
        def forward(self, x, output_attentions=False):
            all_attentions = []
            for block in self.blocks:
                x, attention_probs = block(x, output_attentions=output_attentions)
                if output_attentions:
                    all_attentions.append(attention_probs)
            if not output_attentions:
                return (x, None)
            else:
                return (x, all_attentions)
    return (Encoder,)


@app.cell
def _(Embeddings, Encoder, nn, torch):
    class ViTForClassfication(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.image_size = config.image_size
            self.hidden_size = config.hidden_size
            self.num_classes = config.num_classes
            self.embedding = Embeddings(config)
            self.encoder = Encoder(config)
            self.classifier = nn.Linear(self.hidden_size, self.num_classes)
            self.apply(self._init_weights)

        def forward(self, x, output_attentions=False):
            embedding_output = self.embedding(x)
            encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
            logits = self.classifier(encoder_output[:, 0, :])
            if not output_attentions:
                return (logits, None)
            else:
                return (logits, all_attentions)

        def _init_weights(self, module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, Embeddings):
                module.position_embeddings.data = nn.init.trunc_normal_(module.position_embeddings.data.to(torch.float32), mean=0.0,std=0.02,).to(module.position_embeddings.dtype)
                module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data.to(torch.float32), mean=0.0,std=0.02,).to(module.cls_token.dtype)
    return (ViTForClassfication,)


@app.cell
def _(ViTForClassfication, config, device):
    model = ViTForClassfication(config).to(device)
    return (model,)


@app.cell
def _():
    ### training parameters
    return


@app.cell
def _(model, nn, optim, torch):
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    # mixed precision package for speeding up training
    scaler = torch.cpu.amp.GradScaler() # amp moved to specific arch package
    return loss_fn, optimizer, scaler


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
    return EarlyStop, stopper


@app.cell
def _(device, loss_fn, model, optimizer, scaler, torch, trainloader):
    def train_batch(batch):
        batch = [t.to(device) for t in batch]
        images, labels = batch
        with torch.amp.autocast(device):
            loss = loss_fn(model(images)[0], labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item() * len(images) / len(trainloader.dataset)
    return (train_batch,)


@app.cell
def _():
    ### training ViT Classifier
    return


@app.cell
def _(model_dir):
    model_file = model_dir + "ViT.pth"
    return (model_file,)


@app.cell
def _():
    # from tqdm import tqdm
    # import os
    # os.makedirs(data_dir, exist_ok=True)
    # for i in range(100):
    #     print(f'Epoch {i+1}')
    #     model.train()
    #     trainL, testL = 0, 0
    #     for batch in tqdm(trainloader):
    #         loss = train_batch(batch)
    #         trainL += loss
    #     model.eval()
    #     with torch.no_grad():
    #         for batch in testloader:
    #             batch = [t.to(device) for t in batch]
    #             images, labels = batch
    #             logits, _ = model(images)
    #             loss = loss_fn(logits, labels)
    #             testL += loss.item() * len(images) / len(testloader.dataset)
    #     print(f'Train and test losses: {trainL:.4f}, {testL:.4f}')
    #     to_save, to_stop = stopper.stop(testL)
    #     if to_save == True:
    #         torch.save(model, model_file)
    #     if to_stop == True:
    #         break
    return


@app.cell
def _():
    ### using trained model to classify images
    return


@app.cell
def _(device, model_file, names, testloader, torch):
    import torch.nn.functional as F

    model2 = torch.load(model_file, map_location=device, weights_only=False)
    model2.eval()
    with torch.no_grad():
        batch=next(iter(testloader))
        batch = [t.to(device) for t in batch]
        images, labels = batch
        logits, attention_maps = model2(images, output_attentions=True)
        predictions = torch.argmax(logits, dim=1)

    print(predictions)
    print([names[i] for i in predictions.tolist()])
    return (
        F,
        attention_maps,
        batch,
        images,
        labels,
        logits,
        model2,
        predictions,
    )


@app.cell
def _():
    ### exploring attention maps
    return


@app.cell
def _(attention_maps, torch):
    for attn in attention_maps:
        print(attn.shape)
    block0_image0_head0=attention_maps[0][0,0,:,:] 
    print(block0_image0_head0.shape)
    probs_sum=torch.sum(block0_image0_head0,dim=1)
    print(probs_sum)
    return attn, block0_image0_head0, probs_sum


@app.cell
def _(F, attention_maps, math, plt, torch):
    with torch.no_grad():
        attention_maps2 = torch.cat(attention_maps, dim=1)
        print(f"attention map shape: {attention_maps2.shape}")
        attention_maps2 = attention_maps2[:, :, 0, 1:] 
        print(f"attention map shape: {attention_maps2.shape}")
        attention_maps2 = attention_maps2.mean(dim=1)
        print(f"attention map shape: {attention_maps2.shape}")
        num_patches = attention_maps2.size(-1)
        size = int(math.sqrt(num_patches))
        attention_maps2 = attention_maps2.view(-1, size, size)
        print(f"attention map shape: {attention_maps2.shape}")
        attention_maps2 = attention_maps2.unsqueeze(1)
        attention_maps2 = F.interpolate(attention_maps2, size=(32, 32), mode='bilinear', align_corners=False)
        attention_maps2 = attention_maps2.squeeze(1)
        print(f"attention map shape: {attention_maps2.shape}")

    fig = plt.figure(figsize=(8, 8),dpi=100)
    for i in range(16):
        ax = fig.add_subplot(4,4, i+1, xticks=[], yticks=[])
        ax.imshow(attention_maps2[i].cpu(), alpha=0.5, cmap='jet')
    plt.tight_layout()
    plt.show()
    return attention_maps2, ax, fig, i, num_patches, size


@app.cell
def _():
    ### comparing attention map with original image
    return


@app.cell
def _(attention_maps2, ax, images, labels, names, np, plt, predictions):
    fig2 = plt.figure(figsize=(8, 5),dpi=200)
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)

    for ii in range(16):
        ax2 = fig2.add_subplot(4,4, ii+1, xticks=[], yticks=[])
        img3 = np.concatenate((images[ii].cpu(), images[ii].cpu()), axis=-1)
        ax2.imshow(img3.transpose(1,2,0)/2+0.5)
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps2[ii].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax2.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        gt = names[labels[ii]]
        pred = names[predictions[ii]]
        ax.set_title(f"Actual: {gt} / Pred: {pred}", color=("green" if gt==pred else "red"), fontsize=10)
    plt.tight_layout()
    plt.show()
    return ax2, extended_attention_map, fig2, gt, ii, img3, mask, pred


@app.cell
def _():
    ### test accuracy
    return


@app.cell
def _(device, model2, testloader, torch):
    model2.eval()
    acc = 0
    with torch.no_grad():
        for bbatch in testloader:
            bbatch = [t.to(device) for t in bbatch]
            images2, labels2 = bbatch
            logits2, _ = model2(images2)
            predictions2 = torch.argmax(logits2, dim=1)
            acc += torch.sum(predictions2 == labels2).item()/ len(testloader.dataset)

    print(f'the prediction accuracy is {acc:.4f}')
    return acc, bbatch, images2, labels2, logits2, predictions2


if __name__ == "__main__":
    app.run()
