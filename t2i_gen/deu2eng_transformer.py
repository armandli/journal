import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""### download dataset""")
    return


@app.cell
def _():
    import requests
    import os
    import tarfile

    url = ("https://raw.githubusercontent.com/neychev/" "small_DL_repo/master/datasets/Multi30k/training.tar.gz")
    dest_dir = "/Users/armandli/data/deu2eng/"
    model_dir = 'model/'
    dest_file = dest_dir + "training.tar.gz"
    os.makedirs(dest_dir, exist_ok=True)
    if not os.path.exists(dest_file):
        fb1 = requests.get(url)
        with open(dest_file, "wb") as f:
            f.write(fb1.content)

    train = tarfile.open(dest_file)
    train.extractall(dest_dir)
    train.close()
    return (
        dest_dir,
        dest_file,
        f,
        fb1,
        model_dir,
        os,
        requests,
        tarfile,
        train,
        url,
    )


@app.cell
def _():
    ### open dataset
    return


@app.cell
def _(dest_dir):
    with open(dest_dir + "train.de") as df:
        trainde = df.readlines()
    with open(dest_dir + "train.en") as df:
        trainen = df.readlines()
    trainde = [i.strip() for i in trainde]
    trainen = [i.strip() for i in trainen]
    return df, trainde, trainen


@app.cell
def _(trainde, trainen):
    from pprint import pprint
    print(f"the length of the list trainde is {len(trainde)}")
    print(f"the length of the list trainen is {len(trainen)}")
    print(f"the first five elements of the list trainde are")
    pprint(trainde[:5])
    print(f"the first five elements of the list trainen are")
    pprint(trainen[:5])
    print(f"the last five elements of the list trainde are")
    pprint(trainde[-5:])
    print(f"the last five elements of the list trainen are")
    pprint(trainen[-5:])
    return (pprint,)


@app.cell
def _():
    ### convert training data phrases into tokens
    return


@app.cell
def _(os):
    import spacy

    try:
        de_tokenizer = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        de_tokenizer = spacy.load("de_core_news_sm")

    try:
        en_tokenizer = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        en_tokenizer = spacy.load("en_core_web_sm")
    return de_tokenizer, en_tokenizer, spacy


@app.cell
def _(de_tokenizer, en_tokenizer, trainde, trainen):
    tokenized_de = [tok.text for tok in de_tokenizer.tokenizer(trainde[0])]
    tokenized_en = [tok.text for tok in en_tokenizer.tokenizer(trainen[0])]
    print(tokenized_de)
    print(tokenized_en)
    return tokenized_de, tokenized_en


@app.cell
def _():
    ### create dictionary to map English tokens to indexes
    return


@app.cell
def _(en_tokenizer, trainen):
    from collections import Counter

    en_tokens = [["BOS"] + [tok.text for tok in en_tokenizer.tokenizer(x)] + ["EOS"] for x in trainen]
    PAD=0
    UNK=1
    word_counter = Counter()
    for sen in en_tokens:
        for word in sen:
            word_counter[word] += 1
    freq = word_counter.most_common(50_000)
    total_en_words = len(freq) + 2

    en_word_dict = {w[0] : idx+2 for idx, w in enumerate(freq)}
    en_word_dict["PAD"] = PAD
    en_word_dict["UNK"] = UNK

    en_idx_dict = {v:k for k, v in en_word_dict.items()}
    return (
        Counter,
        PAD,
        UNK,
        en_idx_dict,
        en_tokens,
        en_word_dict,
        freq,
        sen,
        total_en_words,
        word,
        word_counter,
    )


@app.cell
def _(UNK, en_word_dict, tokenized_en):
    enidx=[en_word_dict.get(i,UNK) for i in tokenized_en]
    print(enidx)
    return (enidx,)


@app.cell
def _(en_idx_dict, enidx):
    entokens=[en_idx_dict.get(i,"UNK") for i in enidx]
    print(entokens)
    en_phrase=" ".join(entokens)
    for x in '''?:;.,'("-!&)%''':
        en_phrase=en_phrase.replace(f" {x}",f"{x}")
    print(en_phrase)
    return en_phrase, entokens, x


@app.cell
def _():
    ### create dictionary to map German tokens to indexes
    return


@app.cell
def _(Counter, PAD, UNK, de_tokenizer, trainde):
    de_tokens = [["BOS"] + [tok.text for tok in de_tokenizer.tokenizer(x)] + ["EOS"] for x in trainde]
    de_word_count = Counter()
    for desen in de_tokens:
        for deword in desen:
            de_word_count[deword] += 1
    defreq = de_word_count.most_common(50_000)
    total_de_words = len(defreq) + 2
    de_word_dict = {w[0]: idx+2 for idx, w in enumerate(defreq)}
    de_word_dict["PAD"] = PAD
    de_word_dict["UNK"] = UNK

    de_idx_dict = {v:k for k,v in de_word_dict.items()}
    return (
        de_idx_dict,
        de_tokens,
        de_word_count,
        de_word_dict,
        defreq,
        desen,
        deword,
        total_de_words,
    )


@app.cell
def _(UNK, de_word_dict, tokenized_de):
    deidx=[de_word_dict.get(i,UNK) for i in tokenized_de]
    print(deidx)
    return (deidx,)


@app.cell
def _(de_idx_dict, deidx):
    detokens=[de_idx_dict.get(i,"UNK") for i in deidx]
    print(detokens)
    de_phrase=" ".join(detokens)
    for dex in '''?:;.,'("-!&)%''':
        de_phrase=de_phrase.replace(f" {dex}",f"{dex}")
    print(de_phrase)
    return de_phrase, detokens, dex


@app.cell
def _():
    ### pad the sequences so batches has the same size, if initially different in size, PAD is added to the short one
    return


@app.cell
def _(UNK, de_tokens, de_word_dict, en_tokens, en_word_dict):
    out_en_ids=[[en_word_dict.get(w,UNK) for w in s] for s in en_tokens]
    out_de_ids=[[de_word_dict.get(w,UNK) for w in s] for s in de_tokens]
    sorted_ids=sorted(range(len(out_de_ids)),
    key=lambda x:len(out_de_ids[x]))
    out_de_ids=[out_de_ids[x] for x in sorted_ids]
    out_en_ids=[out_en_ids[x] for x in sorted_ids]
    return out_de_ids, out_en_ids, sorted_ids


@app.cell
def _(de_tokens):
    import numpy as np
    batch_size=128
    idx_list=np.arange(0,len(de_tokens),batch_size)
    np.random.shuffle(idx_list)
    batch_indexs=[]
    for idx in idx_list:
        batch_indexs.append(np.arange(idx,min(len(de_tokens), idx+batch_size)))
    return batch_indexs, batch_size, idx, idx_list, np


@app.cell
def _(PAD, np):
    def seq_padding(X, padding=PAD):
        L = [len(x) for x in X]
        ML = max(L)
        padded_seq = np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])
        return padded_seq
    return (seq_padding,)


@app.cell
def _():
    import math
    from copy import deepcopy
    import torch
    from torch import nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return deepcopy, device, math, nn, torch


@app.cell
def _(np, torch):
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape),
                                  k=1).astype('uint8')
        output = torch.from_numpy(subsequent_mask) == 0
        return output
    return (subsequent_mask,)


@app.cell
def _(subsequent_mask):
    def make_std_mask(tgt, pad):
        tgt_mask=(tgt != pad).unsqueeze(-2)
        output=tgt_mask & subsequent_mask(\
            tgt.size(-1)).type_as(tgt_mask.data)
        return output
    return (make_std_mask,)


@app.cell
def _(device, make_std_mask, torch):
    class Batch:
        def __init__(self, src, trg=None, pad=0):
            src = torch.from_numpy(src).to(device).long()
            self.src = src
            # source mask to hide padding at the end
            self.src_mask = (src != pad).unsqueeze(-2)
            if trg is not None:
                trg = torch.from_numpy(trg).to(device).long()
                # input to decoder
                self.trg = trg[:, :-1]
                # target to decoder
                self.trg_y = trg[:, 1:]
                # target mask
                self.trg_mask = make_std_mask(self.trg, pad)
                self.ntokens = (self.trg_y != pad).data.sum()
    return (Batch,)


@app.cell
def _(de_word_dict, en_word_dict):
    src_vocab = len(de_word_dict)
    tgt_vocab = len(en_word_dict)
    print(f"there are {src_vocab} distinct German tokens")
    print(f"there are {tgt_vocab} distinct English tokens")
    return src_vocab, tgt_vocab


@app.cell
def _(Batch, batch_indexs, out_de_ids, out_en_ids, seq_padding):
    batches=[]
    for b in batch_indexs:
        batch_en=[out_en_ids[x] for x in b]
        batch_de=[out_de_ids[x] for x in b]
        batch_en=seq_padding(batch_en)
        batch_de=seq_padding(batch_de)
        batches.append(Batch(batch_de,batch_en))
    return b, batch_de, batch_en, batches


@app.cell
def _():
    ### create the model
    return


@app.cell
def _(math, nn):
    class Embeddings(nn.Module):
        def __init__(self, d_model, vocab):
            super().__init__()
            self.lut = nn.Embedding(vocab, d_model)
            self.d_model = d_model

        def forward(self, x):
            out = self.lut(x) * math.sqrt(self.d_model)
            return out
    return (Embeddings,)


@app.cell
def _(device, math, nn, torch):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout, max_len=5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model, device=device)
            position = torch.arange(0., max_len, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0., d_model, 2, device=device) * -(math.log(10000.0) / d_model))
            pe_pos = torch.mul(position, div_term)
            pe[:, 0::2] = torch.sin(pe_pos)
            pe[:, 1::2] = torch.cos(pe_pos)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)  

        def forward(self, x):
            x = x + self.pe[:, :x.size(1)].requires_grad_(False)
            out = self.dropout(x)
            return out
    return (PositionalEncoding,)


@app.cell
def _(math, nn, torch):
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    return (attention,)


@app.cell
def _(attention, deepcopy, nn):
    class MultiHeadedAttention(nn.Module):
        def __init__(self, h, d_model, dropout=0.1):
            super().__init__()
            assert d_model % h == 0
            self.d_k = d_model // h
            self.h = h
            self.linears = nn.ModuleList([deepcopy(nn.Linear(d_model, d_model)) for i in range(4)])
            self.attn = None
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, query, key, value, mask=None):
            if mask is not None:
                mask = mask.unsqueeze(1)
            nbatches = query.size(0)  
            query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
            x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
            x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            output = self.linears[-1](x)
            return output
    return (MultiHeadedAttention,)


@app.cell
def _(nn, torch):
    class LayerNorm(nn.Module):
        def __init__(self, features, eps=1e-6):
            super().__init__()
            self.a_2 = nn.Parameter(torch.ones(features))
            self.b_2 = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True) 
            std = x.std(-1, keepdim=True)
            x_zscore = (x - mean) / torch.sqrt(std ** 2 + self.eps)
            output = self.a_2*x_zscore+self.b_2
            return output
    return (LayerNorm,)


@app.cell
def _(LayerNorm, deepcopy, nn):
    class Encoder(nn.Module):
        def __init__(self, layer, N):
            super().__init__()
            self.layers = nn.ModuleList([deepcopy(layer) for i in range(N)])
            self.norm = LayerNorm(layer.size)

        def forward(self, x, mask):
            for layer in self.layers:
                x = layer(x, mask)
                output = self.norm(x)
            return output
    return (Encoder,)


@app.cell
def _(LayerNorm, nn):
    class SublayerConnection(nn.Module):
        def __init__(self, size, dropout):
            super().__init__()
            self.norm = LayerNorm(size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, sublayer):
            output = x + self.dropout(sublayer(self.norm(x)))
            return output
    return (SublayerConnection,)


@app.cell
def _(SublayerConnection, deepcopy, nn):
    class EncoderLayer(nn.Module):
        def __init__(self, size, self_attn, feed_forward, dropout):
            super().__init__()
            self.self_attn = self_attn
            self.feed_forward = feed_forward
            self.sublayer = nn.ModuleList([deepcopy(SublayerConnection(size, dropout)) for i in range(2)])
            self.size = size  

        def forward(self, x, mask):
            x = self.sublayer[0](
                x, lambda x: self.self_attn(x, x, x, mask))
            output = self.sublayer[1](x, self.feed_forward)
            return output
    return (EncoderLayer,)


@app.cell
def _(LayerNorm, deepcopy, nn):
    class Decoder(nn.Module):
        def __init__(self, layer, N):
            super().__init__()
            self.layers = nn.ModuleList(
                [deepcopy(layer) for i in range(N)])
            self.norm = LayerNorm(layer.size)

        def forward(self, x, memory, src_mask, tgt_mask):
            for layer in self.layers:
                x = layer(x, memory, src_mask, tgt_mask)
            output = self.norm(x)
            return output
    return (Decoder,)


@app.cell
def _(SublayerConnection, deepcopy, nn):
    class DecoderLayer(nn.Module):
        def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
            super().__init__()
            self.size = size
            self.self_attn = self_attn
            self.src_attn = src_attn
            self.feed_forward = feed_forward
            self.sublayer = nn.ModuleList([deepcopy(
            SublayerConnection(size, dropout)) for i in range(3)])

        def forward(self, x, memory, src_mask, tgt_mask):
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
            output = self.sublayer[2](x, self.feed_forward)
            return output
    return (DecoderLayer,)


@app.cell
def _(nn):
    class Transformer(nn.Module):
        def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.src_embed = src_embed
            self.tgt_embed = tgt_embed
            self.generator = generator

        def encode(self, src, src_mask):
            return self.encoder(self.src_embed(src), src_mask)

        def decode(self, memory, src_mask, tgt, tgt_mask):
            return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

        def forward(self, src, tgt, src_mask, tgt_mask):
            memory = self.encode(src, src_mask)
            output = self.decode(memory, src_mask, tgt, tgt_mask)
            return output
    return (Transformer,)


@app.cell
def _(nn):
    class Generator(nn.Module):
        def __init__(self, d_model, vocab):
            super().__init__()
            self.proj = nn.Linear(d_model, vocab)

        def forward(self, x):
            out = self.proj(x)
            probs = nn.functional.log_softmax(out, dim=-1)
            return probs
    return (Generator,)


@app.cell
def _(nn):
    class PositionwiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            h1 = self.w_1(x)
            h2 = self.dropout(h1)
            return self.w_2(h2)
    return (PositionwiseFeedForward,)


@app.cell
def _(
    Decoder,
    DecoderLayer,
    Embeddings,
    Encoder,
    EncoderLayer,
    Generator,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    Transformer,
    deepcopy,
    device,
    nn,
):
    def create_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout=0.1):
        attn=MultiHeadedAttention(h, d_model).to(device)
        ff=PositionwiseFeedForward(d_model, d_ff, dropout).to(device)
        pos=PositionalEncoding(d_model, dropout).to(device)
        model = Transformer(
            Encoder(EncoderLayer(d_model,deepcopy(attn),deepcopy(ff), dropout).to(device),N).to(device),
            Decoder(DecoderLayer(d_model,deepcopy(attn),deepcopy(attn),deepcopy(ff), dropout).to(device),N).to(device),
            nn.Sequential(Embeddings(d_model, src_vocab).to(device), deepcopy(pos)),
            nn.Sequential(Embeddings(d_model, tgt_vocab).to(device), deepcopy(pos)),
            Generator(d_model, tgt_vocab)).to(device)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model.to(device)
    return (create_model,)


@app.cell
def _(
    Decoder,
    DecoderLayer,
    Embeddings,
    Encoder,
    EncoderLayer,
    Generator,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    Transformer,
    deepcopy,
    nn,
):
    def create_model2(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout=0.1):
        attn=MultiHeadedAttention(h, d_model)
        ff=PositionwiseFeedForward(d_model, d_ff, dropout)
        pos=PositionalEncoding(d_model, dropout)
        model = Transformer(
            Encoder(EncoderLayer(d_model,deepcopy(attn),deepcopy(ff), dropout),N),
            Decoder(DecoderLayer(d_model,deepcopy(attn),deepcopy(attn),deepcopy(ff), dropout),N),
            nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(pos)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(pos)),
            Generator(d_model, tgt_vocab))
        return model
    return (create_model2,)


@app.cell
def _(create_model, src_vocab, tgt_vocab):
    model = create_model(src_vocab, tgt_vocab, N=6, d_model=256, d_ff=1024, h=8, dropout=0.1)
    return (model,)


@app.cell
def _():
    ### training parameters
    return


@app.cell
def NoamOpt():
    class NoamOpt:
        def __init__(self, model_size, factor, warmup, optimizer):
            self.optimizer = optimizer
            self._step = 0
            self.warmup = warmup
            self.factor = factor
            self.model_size = model_size
            self._rate = 0

        def step(self):
            self._step += 1
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()

        def rate(self, step=None):
            if step is None:
                step = self._step
            output = self.factor * (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
            return output
    return (NoamOpt,)


@app.cell
def SimpleLossCompute():
    class SimpleLossCompute:
        def __init__(self, generator, criterion, opt=None):
            self.generator = generator
            self.criterion = criterion
            self.opt = opt

        def __call__(self, x, y, norm):
            x = self.generator(x)
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                                  y.contiguous().view(-1)) / norm
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
            return loss.data.item() * norm.float()
    return (SimpleLossCompute,)


@app.cell
def _(nn, torch):
    class LabelSmoothing(nn.Module):
        def __init__(self, size, padding_idx, smoothing=0.1):
            super().__init__()
            self.criterion = nn.KLDivLoss(reduction='sum')  
            self.padding_idx = padding_idx
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.size = size
            self.true_dist = None

        def forward(self, x, target):
            assert x.size(1) == self.size
            true_dist = x.data.clone()
            true_dist.fill_(self.smoothing / (self.size - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            self.true_dist = true_dist
            output = self.criterion(x, true_dist.clone().detach())
            return output
    return (LabelSmoothing,)


@app.cell
def _(LabelSmoothing, NoamOpt, SimpleLossCompute, model, tgt_vocab, torch):
    optimizer = NoamOpt(256, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.0)
    loss_func = SimpleLossCompute(model.generator, criterion, optimizer)
    return criterion, loss_func, optimizer


@app.cell
def _(model_dir):
    model_file = model_dir + "de2en.pth"
    return (model_file,)


@app.cell
def _(batches, loss_func, model, model_file, torch):
    for epoch in range(1):
        model.train()
        tloss = 0
        tokens = 0
        for batch in batches:
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_func(out, batch.trg_y, batch.ntokens)
            tloss += loss
            tokens += batch.ntokens
        print(f"Epoch {epoch}, average loss: {tloss/tokens}")
    torch.save(model,model_file)
    return batch, epoch, loss, out, tloss, tokens


@app.cell
def _(device, model_file, torch):
    model2 = torch.load(model_file, map_location=device, weights_only=False)
    return (model2,)


@app.cell
def _():
    ### inference
    return


@app.cell
def _(
    UNK,
    de_tokenizer,
    de_word_dict,
    device,
    en_idx_dict,
    en_word_dict,
    subsequent_mask,
    torch,
):
    def de2en(ger, mod):
        tokenized_ger = [tok.text for tok in de_tokenizer.tokenizer(ger)]
        tokenized_ger = ["BOS"] + tokenized_ger + ["EOS"]
        geridx = [de_word_dict.get(i, UNK) for i in tokenized_ger]
        src = torch.tensor(geridx).long().to(device).unsqueeze(0)
        src_mask = (src!=0).unsqueeze(-2)
        memory = mod.encode(src, src_mask)
        start_symbol = en_word_dict["BOS"]
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        translation = []
        for i in range(100):
            out = mod.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
            prob = mod.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            sym = en_idx_dict[ys[0,-1].item()]
            if sym != 'EOS':
                translation.append(sym)
            else:
                break
        trans = " ".join(translation)
        for x in '''?:;.,'("-!&)%''':
            trans = trans.replace(f" {x}", f"{x}")
        return trans
    return (de2en,)


@app.cell
def _(de2en, model2, trainde, trainen):
    for i in range(5):
        print("original Ger:", trainde[100+i])
        print("original Eng:", trainen[100+i])
        print("translated Eng:", de2en(trainde[100+i], model2))
        print("")
    return (i,)


if __name__ == "__main__":
    app.run()
