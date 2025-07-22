import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json

    data_dir = '/Users/armandli/data/'
    model_dir = 'models/'

    with open(data_dir + 'flickr8k_caption_datasets/dataset_flickr8k.json', 'r') as fb:
        data = json.load(fb)
    return data, data_dir, fb, json, model_dir


@app.cell
def _():
    ### split into training and test set
    return


@app.cell
def _(data, data_dir):
    from collections import Counter

    train_image_paths = []
    train_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    max_len=50
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
        if len(captions) == 0:
            continue
        path = data_dir + "flickr8k/Images/" + img['filename']
        if img['split'] in {'train', 'val', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
    return (
        Counter,
        c,
        captions,
        img,
        max_len,
        path,
        test_image_captions,
        test_image_paths,
        train_image_captions,
        train_image_paths,
        word_freq,
    )


@app.cell
def _(
    test_image_captions,
    test_image_paths,
    train_image_captions,
    train_image_paths,
):
    assert len(train_image_paths)==len(train_image_captions)
    assert len(test_image_paths)==len(test_image_captions)
    print(f"there are {len(train_image_paths)} training images")
    print(f"there are {len(test_image_paths)} test images")
    return


@app.cell
def _():
    ### build a vocabulary of tokens
    return


@app.cell
def _(word_freq):
    min_word_freq=0
    words = [w for w in word_freq.keys() if word_freq[w]>min_word_freq]
    word2idx = {k:v + 4 for v,k in enumerate(words)}
    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<end>'] = 2
    word2idx['<unk>'] = 3
    return min_word_freq, word2idx, words


@app.cell
def _(test_image_captions, word2idx):
    indexes=[word2idx.get(token,3) for token in test_image_captions[0][0]]
    print(indexes)
    return (indexes,)


@app.cell
def _(indexes, word2idx):
    idx2word={v:k for k, v in word2idx.items()}
    tokens=[idx2word.get(idx,"<unk>") for idx in indexes]
    print(tokens)
    print(f"there are {len(idx2word)} unique tokens")
    return idx2word, tokens


@app.cell
def _():
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    import math
    from torch import nn
    from torch.distributions import Categorical

    import torchvision

    import matplotlib.pyplot as plt
    return (
        Categorical,
        Dataset,
        Image,
        math,
        nn,
        plt,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def center_crop():
    def center_crop(img):   
        width, height = img.size   
        new=min(width,height)
        left = (width - new)/2
        top = (height - new)/2
        right = (width + new)/2
        bottom = (height + new)/2
        im = img.crop((left, top, right, bottom))
        return im
    return (center_crop,)


@app.cell
def _(Dataset, Image, center_crop, torch, transforms):
    class FlickrD(Dataset):
        def __init__(self,images,captions,word2idx):
            self.images=images
            self.captions=captions
            self.word2idx=word2idx
            self._max_len = 50
            self.image_size=128
            self._image_transform = self._construct_image_transform(self.image_size)
            self._data = self._create_input_label_mappings()
            self._dataset_size = len(self._data)
            self._start_idx = 1
            self._end_idx = 2
            self._pad_idx = 0
            self._UNK_idx = 3
            self._START_token = "<start>"
            self._END_token = "<end>"
            self._PAD_token = "<pad>"
            self._UNK_token = "<unk>"       

        def _construct_image_transform(self, image_size):
            # ImageNet normalization statistics
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            preprocessing = transforms.Compose([transforms.ToTensor(), normalize,])
            return preprocessing
        def _load_and_process_images(self):
            # Load images
            images_raw = [center_crop(Image.open(path)).resize((self.image_size, self.image_size)) for path in self.images]
            # Adapt the images to CNN trained on ImageNet { PIL -> Tensor }
            image_tensors = [self._image_transform(img) for img in images_raw]
            images_processed = {img_name: img_tensor for img_name, img_tensor in zip(self.images, image_tensors)}
            return images_processed

        def _group_captions(self):
            grouped_captions = {self.images[i]:self.captions[i] for i in range(len(self.images))}
            return grouped_captions
        def _create_input_label_mappings(self):
            processed_data = []
            for img, caps in self._group_captions().items():
                for cap in caps:
                    pair = (img, cap)
                    processed_data.append(pair)
            return processed_data    

        def _load_and_prepare_image(self, image_name):
            img_pil = center_crop(Image.open(image_name)).resize((self.image_size, self.image_size))
            image_tensor = self._image_transform(img_pil)
            return image_tensor

        def __len__(self):
            return self._dataset_size

        def __getitem__(self, index):
            # Extract the caption data
            image_id, tokens = self._data[index]
            # Load and preprocess image
            image_tensor = self._load_and_prepare_image(image_id)
            # Pad the token and label sequences
            tokens = tokens[:self._max_len]
            tokens = [token.strip().lower() for token in tokens]
            tokens = [self._START_token] + tokens + [self._END_token]
            # Extract input and target output
            input_tokens = tokens[:-1].copy()
            tgt_tokens = tokens[1:].copy()

            # Number of words in the input token
            sample_size = len(input_tokens)
            padding_size = self._max_len - sample_size

            if padding_size > 0:
                padding_vec = [self._PAD_token for _ in range(padding_size)]
                input_tokens += padding_vec.copy()
                tgt_tokens += padding_vec.copy()

            # Apply the vocabulary mapping to the input tokens
            input_tokens = [self.word2idx.get(token, self._UNK_idx) for token in input_tokens]
            tgt_tokens = [self.word2idx.get(token, self._UNK_idx) for token in tgt_tokens]

            input_tokens = torch.Tensor(input_tokens).long()
            tgt_tokens = torch.Tensor(tgt_tokens).long()

            # Index from which to extract the model prediction
            # Define the padding masks
            attn_mask = torch.zeros([self._max_len, ])
            attn_mask[:sample_size] = 1.0
            attn_mask = attn_mask.bool()

            return image_tensor, input_tokens, tgt_tokens, attn_mask
    return (FlickrD,)


@app.cell
def _(
    FlickrD,
    test_image_captions,
    test_image_paths,
    train_image_captions,
    train_image_paths,
    word2idx,
):
    trainset=FlickrD(train_image_paths, train_image_captions,word2idx)
    testset=FlickrD(test_image_paths, test_image_captions,word2idx)
    return testset, trainset


@app.cell
def _(testset, trainset):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=True)
    return DataLoader, test_loader, train_loader


@app.cell
def _(data_dir, test_loader, torch):
    test_images, test_tokens, test_targets,test_mask=next(iter(test_loader))
    torch.save((test_images,test_tokens), data_dir + "flickr8k/tests.pt")
    return test_images, test_mask, test_targets, test_tokens


@app.cell
def _():
    ### create model
    return


@app.cell
def _(torch):
    def extract_patches(image_tensor, patch_size=16):
        # Get the dimensions of the image tensor
        bs, c, h, w = image_tensor.size()
        # Define the Unfold layer with appropriate parameters
        unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        # Apply Unfold to the image tensor
        unfolded = unfold(image_tensor)
        # Reshape the unfolded tensor to match the desired output shape
        # Output shape: BSxLxH, where L is the number of patches in each dimension
        unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

        return unfolded
    return (extract_patches,)


@app.cell
def _(extract_patches, test_images):
    image=test_images[0].unsqueeze(0)
    patches=extract_patches(image,patch_size=8)
    print(patches.shape)
    return image, patches


@app.cell
def _(math, nn, torch):
    # fixed snusoidal positional embedding
    class SinusoidalPosEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            device = x.device
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = x[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb
    return (SinusoidalPosEmb,)


@app.cell
def _(nn, torch):
    class AttentionBlock(nn.Module):
        def __init__(self, hidden_size=128, num_heads=4, masking=True):
            super(AttentionBlock, self).__init__()
            self.masking = masking

            # Multi-head attention mechanism
            self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=0.0)

        def forward(self, x_in, kv_in, key_mask=None):
            # Apply causal masking if enabled
            if self.masking:
                bs, l, h = x_in.shape
                mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
            else:
                mask = None

            # Perform multi-head attention operation
            return self.multihead_attn(x_in, kv_in, kv_in, attn_mask=mask, key_padding_mask=key_mask)[0]
    return (AttentionBlock,)


@app.cell
def _(AttentionBlock, nn):
    class TransformerBlock(nn.Module):
        def __init__(self, hidden_size=128, num_heads=4, decoder=False, masking=True):
            super(TransformerBlock, self).__init__()
            self.decoder = decoder

            # Layer normalization for the input
            self.norm1 = nn.LayerNorm(hidden_size)
            # Self-attention mechanism
            self.attn1 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, masking=masking)

            # Layer normalization for the output of the first attention layer
            if self.decoder:
                self.norm2 = nn.LayerNorm(hidden_size)
                # Self-attention mechanism for the decoder with no masking
                self.attn2 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, masking=False)

            # Layer normalization for the output before the MLP
            self.norm_mlp = nn.LayerNorm(hidden_size)
            # Multi-layer perceptron (MLP)
            self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.ELU(), nn.Linear(hidden_size * 4, hidden_size))

        def forward(self, x, input_key_mask=None, cross_key_mask=None, kv_cross=None):
            # Perform self-attention operation
            x = self.attn1(x, x, key_mask=input_key_mask) + x
            x = self.norm1(x)

            # If decoder, perform additional cross-attention layer
            if self.decoder:
                x = self.attn2(x, kv_cross, key_mask=cross_key_mask) + x
                x = self.norm2(x)

            # Apply MLP and layer normalization
            x = self.mlp(x) + x
            return self.norm_mlp(x)
    return (TransformerBlock,)


@app.cell
def _(TransformerBlock, extract_patches, nn, torch):
    class VisionEncoder(nn.Module):
        def __init__(self, image_size, channels_in, patch_size=16, hidden_size=128, num_layers=3, num_heads=4):
            super(VisionEncoder, self).__init__()

            self.patch_size = patch_size
            self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)

            seq_length = (image_size // patch_size) ** 2
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))

            # Create multiple transformer blocks as layers
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_size, num_heads, decoder=False, masking=False) for _ in range(num_layers)
            ])

        def forward(self, image):
            patch_seq = extract_patches(image, patch_size=self.patch_size)
            patch_emb = self.fc_in(patch_seq)

            # Add a unique embedding to each token embedding
            embs = patch_emb + self.pos_embedding

            # Pass the embeddings through each transformer block
            for block in self.blocks:
                embs = block(embs)

            return embs
    return (VisionEncoder,)


@app.cell
def _(SinusoidalPosEmb, TransformerBlock, nn, torch):
    class Decoder(nn.Module):
        def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
            super(Decoder, self).__init__()

            # Create an embedding layer for tokens
            self.embedding = nn.Embedding(num_emb, hidden_size)
            # Initialize the embedding weights
            self.embedding.weight.data = 0.001 * self.embedding.weight.data

            # Initialize sinusoidal positional embeddings
            self.pos_emb = SinusoidalPosEmb(hidden_size)

            # Create multiple transformer blocks as layers
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_size, num_heads, decoder=True) for _ in range(num_layers)
            ])

            # Define a linear layer for output prediction
            self.fc_out = nn.Linear(hidden_size, num_emb)

        def forward(self, input_seq, encoder_output, input_padding_mask=None, 
                    encoder_padding_mask=None):        
            # Embed the input sequence
            input_embs = self.embedding(input_seq)
            bs, l, h = input_embs.shape

            # Add positional embeddings to the input embeddings
            seq_indx = torch.arange(l, device=input_seq.device)
            pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
            embs = input_embs + pos_emb

            # Pass the embeddings through each transformer block
            for block in self.blocks:
                embs = block(embs, input_key_mask=input_padding_mask, cross_key_mask=encoder_padding_mask, kv_cross=encoder_output)

            return self.fc_out(embs)
    return (Decoder,)


@app.cell
def _(Decoder, VisionEncoder, nn):
    class VisionEncoderDecoder(nn.Module):
        def __init__(self, image_size, channels_in,
                     num_emb, patch_size=16, 
                     hidden_size=128, num_layers=(3, 3),
                     num_heads=4):
            super(VisionEncoderDecoder, self).__init__()    
            # Create an encoder and decoder with specified parameters
            self.encoder = VisionEncoder(image_size=image_size, channels_in=channels_in, patch_size=patch_size, hidden_size=hidden_size, num_layers=num_layers[0], num_heads=num_heads)
            self.decoder = Decoder(num_emb=num_emb, hidden_size=hidden_size, num_layers=num_layers[1], num_heads=num_heads)
        def forward(self, input_image, target_seq, padding_mask):
            # Generate padding masks for the target sequence
            bool_padding_mask = padding_mask == 0
            # Encode the input sequence
            encoded_seq = self.encoder(image=input_image)
            # Decode the target sequence using the encoded sequence
            decoded_seq = self.decoder(input_seq=target_seq, encoder_output=encoded_seq, input_padding_mask=bool_padding_mask)
            return decoded_seq
    return (VisionEncoderDecoder,)


@app.cell
def _():
    ### training
    return


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = 192
    num_layers = (6, 6)
    num_heads = 8
    patch_size = 8
    return device, hidden_size, num_heads, num_layers, patch_size


@app.cell
def _(
    VisionEncoderDecoder,
    device,
    hidden_size,
    num_heads,
    num_layers,
    patch_size,
    word2idx,
):
    caption_model = VisionEncoderDecoder(image_size=128, channels_in=3, num_emb=len(word2idx), patch_size=patch_size, num_layers=num_layers,hidden_size=hidden_size, num_heads=num_heads).to(device)
    return (caption_model,)


@app.cell
def _(model_dir):
    model_file = model_dir + 'img_caption.pth'
    model_file_ref = model_dir + 'img_caption_ref.pth'
    return model_file, model_file_ref


@app.cell
def _(caption_model, nn, torch):
    optimizer = torch.optim.Adam(caption_model.parameters(), lr=0.0001)
    scaler = torch.cpu.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    return loss_fn, optimizer, scaler


@app.cell
def _(caption_model):
    num_model_params = 0
    for param in caption_model.parameters():
        num_model_params += param.flatten().shape[0]
    print(f"This model has {num_model_params} parameters")
    return num_model_params, param


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
def _(caption_model, device, loss_fn, optimizer, scaler, torch, train_loader):
    def train_batch(batch):
        images, inputs, targets, masks = batch
        images, inputs, targets, masks = images.to(device), inputs.to(device), targets.to(device), masks.to(device)
        with torch.amp.autocast("cuda"):
            pred = caption_model(images, inputs, padding_mask=masks)
            loss = (loss_fn(pred.transpose(1,2), targets) * masks).mean()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item() * len(images) / len(train_loader.dataset)
    return (train_batch,)


@app.cell
def _(caption_model, device, loss_fn, test_loader):
    def test_batch(batch):
        images, inputs, targets, masks = batch
        images, inputs, targets, masks = images.to(device), inputs.to(device), targets.to(device), masks.to(device)
        pred = caption_model(images, inputs, padding_mask=masks)
        loss = (loss_fn(pred.transpose(1,2), targets) * masks).mean()
        return loss.item() * len(images) / len(test_loader.dataset)
    return (test_batch,)


@app.cell
def _():
    # from tqdm import tqdm

    # for i in range(2):
    #     print(f'Epoch {i+1}')
    #     caption_model.train()
    #     trainL, testL = 0, 0
    #     for batch in tqdm(train_loader):
    #         loss = train_batch(batch)
    #         trainL += loss
    #     caption_model.eval()
    #     with torch.no_grad():
    #         for batch in test_loader:
    #             loss = test_batch(batch)
    #             testL += loss
    #     print(f'Train and test losses: {trainL:.4f}, {testL:.4f}')
    #     to_save, to_stop = stopper.stop(testL)
    #     if to_save == True:
    #         torch.save(caption_model, model_file)
    #     if to_stop == True:
    #         break
    return


@app.cell
def _(caption_model, device, model_file, model_file_ref, torch):
    caption_model2 = torch.load(model_file, map_location=device, weights_only=False)
    caption_model.load_state_dict(torch.load(model_file_ref, map_location=device, weights_only=False))
    return (caption_model2,)


@app.cell
def _(Categorical, device, idx2word, torch):
    def caption(image, model, temp=1.0):
        sos_token = 1 * torch.ones(1, 1).long()
        log_tokens = [sos_token]
        model.eval()
        with torch.no_grad():
            image_embedding = model.encoder(image.to(device))
            for i in range(50):
                input_tokens = torch.cat(log_tokens, 1)
                data_pred = model.decoder(input_tokens.to(device),image_embedding)
                dist = Categorical(logits=data_pred[:, -1] / temp)
                next_tokens = dist.sample().reshape(1, 1)
                log_tokens.append(next_tokens.cpu())
                if next_tokens.item() == 2:
                    break
        pred_text = torch.cat(log_tokens, 1)
        pred_text_strings = [idx2word.get(i,"<unk>") for i in pred_text[0].tolist() if i>3]
        pred_text = " ".join(pred_text_strings)
        return pred_text
    return (caption,)


@app.cell
def _(caption, idx2word, plt, torchvision):
    def compare(images, captions, index, model, temp=1.0):
        image = images[index].unsqueeze(0)
        capi=captions[index]
        capt=[idx2word.get(i,"UNK") for i in capi.tolist() if i>3]
        cap=" ".join(capt)
        pred=caption(image, model, temp=temp)
        out=torchvision.utils.make_grid(image, 1, normalize=True)
        plt.figure(figsize=(5,10),dpi=100)
        out = torchvision.utils.make_grid(image, 1, normalize=True)
        plt.imshow(out.numpy().transpose((1, 2, 0)))
        plt.title(f"**Original caption:\n"+cap+"\n**Generated caption:\n"+pred, wrap=True, loc="left", fontsize=18)
        plt.axis("off")
        plt.show()
    return (compare,)


@app.cell
def _(caption_model, compare, test_images, test_tokens):
    compare(test_images, test_tokens, 10, caption_model, temp=0.95)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
