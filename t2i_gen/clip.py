import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    ### load data
    return


@app.cell
def _():
    import pandas as pd

    data_dir = '/Users/armandli/data/'
    model_dir = 'models/'

    df=pd.read_csv(data_dir + r'flickr8k/captions.txt', delimiter=",")
    return data_dir, df, model_dir


@app.cell
def _(df):
    # each image have 5 descriptions
    print(df.head(n=12))
    return


@app.cell
def _():
    ### visualize image and caption pairs
    return


@app.cell
def _(PIL, data_dir, df, os, plt):
    imgfolder=data_dir + r"flickr8k/Images"
    with os.scandir(imgfolder) as fb:
        files=[f.name for f in fb]
    start=100
    imgs=files[start:start+10]
    dfi=df[df["image"].isin(imgs)].copy()
    dfi["length"]=dfi["caption"].str.len()
    dfi=dfi.sort_values(['image',"length"])
    dfi=dfi.groupby("image").first()


    plt.figure(dpi=200,figsize=(15,10))

    for i in range(10):
        plt.subplot(5,2, i+1)
        img=f"{imgfolder}/{dfi.index[i]}"
        nparray=PIL.Image.open(img)
        plt.imshow(nparray)
        plt.title(f"{dfi.iloc[i]['caption']}") #D
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import os
    import cv2
    from tqdm import tqdm
    import pickle

    import torch
    from torch import nn
    import torch.nn.functional as F
    import numpy as np

    import timm # for resnet50
    import albumentations as A
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

    import PIL
    from matplotlib import pyplot as plt
    from PIL import Image
    return (
        A,
        DistilBertConfig,
        DistilBertModel,
        DistilBertTokenizer,
        F,
        Image,
        PIL,
        cv2,
        nn,
        np,
        os,
        pickle,
        plt,
        timm,
        torch,
        tqdm,
    )


@app.cell
def _():
    ### tokenize caption with DistilBERT tokenizer
    return


@app.cell
def _(DistilBertTokenizer):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    encoded=tokenizer(["two dogs run","an eagle flies in the sky"],padding=True, truncation=True,max_length=200)
    print(encoded)
    for indexes in encoded['input_ids']:
        tokens = tokenizer.convert_ids_to_tokens(indexes)
        print(tokens)
    return (tokenizer,)


@app.cell
def _():
    ### preprocessing images and text for training
    return


@app.cell
def _(df, np):
    image_ids = np.arange(0, len(df))
    np.random.seed(42)
    valid_ids = np.random.choice(image_ids, size=int(0.2 * len(df)), replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train = df[df.index.isin(train_ids)].reset_index(drop=True)
    valid = df[df.index.isin(valid_ids)].reset_index(drop=True)
    return train, valid


@app.cell
def _(data_dir, torch):
    class CFG:
        image_path = data_dir + r"flickr8k/Images"
        captions_path = r"files"
        batch_size = 32
        head_lr = 1e-3
        weight_decay = 1e-3
        patience = 1
        factor = 0.8
        epochs = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = 'resnet50'
        image_embedding = 2048
        text_encoder_model = "distilbert-base-uncased"
        text_embedding = 768
        text_tokenizer = "distilbert-base-uncased"
        max_length = 200
        pretrained = True # for both image encoder and text encoder
        trainable = False # for both image encoder and text encoder
        temperature = 1.0
        # image size
        size = 224
        # for projection head; used for both image and text encoders
        num_projection_layers = 1
        projection_dim = 256 
        dropout = 0.1
    return (CFG,)


@app.cell
def _(CFG, cv2, torch):
    class CLIPDataset(torch.utils.data.Dataset):
        def __init__(self,image_filenames,captions,tokenizer, transforms):
            self.image_filenames = image_filenames
            self.captions = list(captions)
            self.encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=CFG.max_length)
            self.transforms = transforms
        def __getitem__(self, idx):
            item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}
            image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
            item['image'] = torch.tensor(image).permute(2, 0, 1).float()
            item['caption'] = self.captions[idx]
            return item
        def __len__(self):
            return len(self.captions)
    return (CLIPDataset,)


@app.cell
def _(A, CFG):
    def get_transforms():
        return A.Compose([
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])  
    return (get_transforms,)


@app.cell
def _():
    ### creating data loaders
    return


@app.cell
def _(CFG, CLIPDataset, get_transforms, tokenizer, torch, train, valid):
    transforms = get_transforms()
    trainset = CLIPDataset(train["image"].values, train["caption"].values, tokenizer=tokenizer, transforms=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CFG.batch_size,shuffle=True)
    valset = CLIPDataset(valid["image"].values, valid["caption"].values,tokenizer=tokenizer, transforms=transforms)
    valloader = torch.utils.data.DataLoader(valset, batch_size=CFG.batch_size,shuffle=False)
    return trainloader, valloader


@app.cell
def _(trainloader):
    batch0=next(iter(trainloader))
    print(batch0.keys())
    return (batch0,)


@app.cell
def _(batch0):
    print(batch0['input_ids'][0]) # token IDs
    print(batch0['attention_mask'][0]) # masking out the paddings in token IDs
    print(batch0['image'][0].shape)
    print(batch0['caption'][0])
    return


@app.cell
def _(CFG, DistilBertConfig, DistilBertModel, nn):
    class TextEncoder(nn.Module):
        def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
            super().__init__()
            if pretrained:
                self.model = DistilBertModel.from_pretrained(model_name)
            else:
                self.model = DistilBertModel(config=DistilBertConfig())            
            for p in self.model.parameters():
                p.requires_grad = trainable
            self.target_token_idx = 0
        def forward(self, input_ids, attention_mask):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state
            # text embedding is the output associated with CLS token in the last layer
            return last_hidden_state[:,self.target_token_idx,:]
    return (TextEncoder,)


@app.cell
def _(TextEncoder):
    textencoder = TextEncoder()
    num_trainable = sum([p.numel() for p in textencoder.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_trainable}")
    non_trainable = sum([p.numel() for p in textencoder.parameters() if not p.requires_grad])
    print(f"Number of untrainable parameters: {non_trainable}")
    return (textencoder,)


@app.cell
def _(batch0, textencoder):
    # no matter how long the caption, the embedding for CLS token is always the same
    encoded_text=textencoder(batch0['input_ids'], batch0['attention_mask'])
    # we need to reshape the 768 dim embedding into 256 dim later with additional linear mapping layer
    print(encoded_text.shape)
    return


@app.cell
def _(CFG, nn, timm):
    class ImageEncoder(nn.Module):
        def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable):
            super().__init__()
            self.model = timm.create_model(model_name, pretrained, num_classes=0,global_pool="avg")
            for p in self.model.parameters():
                p.requires_grad = trainable
        def forward(self, x):
            return self.model(x)
    return (ImageEncoder,)


@app.cell
def _(ImageEncoder):
    imageencoder=ImageEncoder()
    num_trainable2 = sum([p.numel() for p in imageencoder.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_trainable2}")
    non_trainable2 = sum([p.numel() for p in imageencoder.parameters() if not p.requires_grad])
    print(f"Number of untrainable parameters: {non_trainable2}")
    return (imageencoder,)


@app.cell
def _(batch0, imageencoder):
    encoded_image = imageencoder(batch0['image'])
    # need to project the 2048 dim embedding from resnet50 to 256 dim
    print(encoded_image.shape)
    return


@app.cell
def _(CFG, nn):
    class ProjectionHead(nn.Module):
        def __init__(self,embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout):
            super().__init__()
            self.projection = nn.Linear(embedding_dim, projection_dim)
            self.gelu = nn.GELU()
            self.fc = nn.Linear(projection_dim, projection_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(projection_dim)
        def forward(self, x):
            projected = self.projection(x)
            x = self.gelu(projected)
            x = self.fc(x)
            x = self.dropout(x)
            x = x + projected
            x = self.layer_norm(x)
            return x   
    return (ProjectionHead,)


@app.cell
def _(CFG, F, ImageEncoder, ProjectionHead, TextEncoder, cross_entropy, nn):
    class CLIPModel(nn.Module):
        def __init__(self,temperature=CFG.temperature, image_embedding=CFG.image_embedding, text_embedding=CFG.text_embedding):
            super().__init__()
            self.image_encoder = ImageEncoder()
            self.text_encoder = TextEncoder()
            self.image_projection = ProjectionHead(embedding_dim=image_embedding)
            self.text_projection = ProjectionHead(embedding_dim=text_embedding)
            self.temperature = temperature
        
        def forward(self, batch):
            # Getting Image and Text Features
            image_features = self.image_encoder(batch["image"])
            text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            # Getting Image and Text Embeddings (with same dimension)
            image_embeddings = self.image_projection(image_features)
            text_embeddings = self.text_projection(text_features)

            # Calculating the Loss
            logits = (text_embeddings @ image_embeddings.T) / self.temperature
            images_similarity = image_embeddings @ image_embeddings.T
            texts_similarity = text_embeddings @ text_embeddings.T
            targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)    
            texts_loss = cross_entropy(logits, targets, reduction='none')
            images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            # makes sure both encoders are equally trained
            loss =  (images_loss + texts_loss) / 2.0
            return loss.mean()
    return (CLIPModel,)


@app.cell
def _():
    ### training
    return


@app.cell
def _(CFG, CLIPModel):
    model = CLIPModel().to(CFG.device)
    num_trainable3 = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_trainable3}")
    non_trainable3 = sum([p.numel() for p in model.parameters() if not p.requires_grad])
    print(f"Number of untrainable parameters: {non_trainable3}") 
    return (model,)


@app.cell
def _(model_dir):
    model_file = model_dir + 'clip.pth'
    model_file_ref = model_dir + 'clip_ref.pth'
    return model_file, model_file_ref


@app.cell
def _(CFG, model, torch):
    import itertools

    params = [
        {"params": itertools.chain(model.image_projection.parameters(), model.text_projection.parameters()),
         "lr": CFG.head_lr,
         "weight_decay": CFG.weight_decay
        }
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=CFG.patience,
        factor=CFG.factor
    )
    return


@app.cell
def _():
    best_loss = float('inf')
    return (best_loss,)


@app.cell
def _(nn):
    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    return (cross_entropy,)


@app.cell
def _(CFG, best_loss, model, model_file, torch, tqdm, valloader):
    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            tqdm_object = tqdm(valloader, total=len(valloader))
            for batch in tqdm_object:
                batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
                loss = model(batch)
                losses.append(loss.item())
                avgloss=sum(losses)/len(losses)
                tqdm_object.set_description(f"valid_loss={avgloss:.2f}")
        if avgloss < best_loss:
            best_loss = avgloss
            torch.save(model.state_dict(), model_file)
            print("Saved Best Model!")
    return


@app.cell
def _():
    # scaler = torch.cpu.amp.GradScaler()
    # for epoch in range(10):
    #     print(f"Epoch: {epoch + 1}")
    #     model.train()
    #     losses = []
    #     tqdm_object = tqdm(trainloader, total=len(trainloader))
    #     for batch in tqdm_object:
    #         batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
    #         with torch.amp.autocast("cuda"):
    #             loss = model(batch)
    #         optimizer.zero_grad()
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         losses.append(loss.item())
    #         avgloss=sum(losses)/len(losses)
    #         tqdm_object.set_description(f"loss is {avgloss:.5f}")
    #     evaluate()
    #     lr_scheduler.step(avgloss)
    return


@app.cell
def _(CFG, model, model_file_ref, torch):
    model.load_state_dict(torch.load(model_file_ref, map_location=CFG.device, weights_only=True))
    return


@app.cell
def _(CFG, model, torch, tqdm, valloader):
    image_embeds = []
    with torch.no_grad():
        for batch in tqdm(valloader):
            image_features3 = model.image_encoder(batch["image"].to(CFG.device))
            image_embeds.append(model.image_projection(image_features3))
    image_embeddings = torch.cat(image_embeds)
    return (image_embeddings,)


@app.cell
def _(data_dir, image_embeddings, pickle):
    with open(data_dir + "image_embeds.p","wb") as f:
        pickle.dump(image_embeddings, f)
    return


@app.cell
def _():
    ### selecting image based on prompt
    return


@app.cell
def _(CFG, image_embeddings, model, tokenizer, torch, valid):
    def match(prompt):
        encoded = tokenizer([prompt])
        batch = {key: torch.tensor(values).to(CFG.device) for key, values in encoded.items()}
        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            text_embeddings = model.text_projection(text_features)
        dot_similarity = text_embeddings @ image_embeddings.T
        values, idx = torch.topk(dot_similarity.squeeze(0), 1) 
        img=valid['image'].values[idx.item()]
        caption=valid['caption'].values[idx.item()]
        return img, caption
    return (match,)


@app.cell
def _(PIL, data_dir, match, plt):
    prompt="students having a class in the classroom"
    file,cap=match(prompt)
    plt.imshow(PIL.Image.open(data_dir + rf"flickr8k/Images/{file}"))
    plt.title(f"Prompt: {prompt}\nOriginal caption: {cap}")
    plt.axis("off")
    plt.show()
    return


@app.cell
def _():
    ### use pretrained clip model 
    return


@app.cell
def _(CFG):
    import clip

    clip_model, preprocess = clip.load("ViT-B/32", device=CFG.device)
    return clip, clip_model, preprocess


@app.cell
def _(CFG, Image, clip_model, data_dir, os, pickle, preprocess, torch):
    with os.scandir(data_dir + 'flickr8k/Images/') as files2:
        names=[file.name for file in files2]

    images=[]
    for ii in names:
        images.append(preprocess(Image.open(data_dir + f"flickr8k/Images/{ii}")).unsqueeze(0).to(CFG.device))

    image=torch.cat(images)
    print(image.shape)
    with torch.no_grad():
        image_features = clip_model.encode_image(image) #B
    with open(data_dir + "imgfeas.p","wb") as fb2:
        pickle.dump(image_features,fb2)
    return image_features, names


@app.cell
def _(CFG, PIL, clip, clip_model, data_dir, image_features, names, plt, torch):
    def find_match2(prompt):
        print(f"prompt is {prompt}")
        text = clip.tokenize([prompt]).to(CFG.device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        simu=text_features@image_features.T
        values, indices = torch.topk(simu[0], 5) 
        plots=[]
        for i in indices:
            plots.append(names[i])
        plt.figure(dpi=200,figsize=(10,2))
        for i in range(5):
            plt.subplot(1,5, i+1)
            img=data_dir + f"flickr8k/Images/{plots[i]}"
            nparray=PIL.Image.open(img)
            plt.imshow(nparray)
            plt.axis("off")
        plt.tight_layout()
        plt.show()
    return (find_match2,)


@app.cell
def _(data_dir, find_match2, pickle):
    with open(data_dir + "imgfeas.p","rb") as fb3:
        image_features2=pickle.load(fb3)
    find_match2("people eating at the restaurant")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
