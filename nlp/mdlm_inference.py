import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")

with app.setup:
    import os
    from tqdm import tqdm
    import s3fs

    import datasets
    from datasets import load_from_disk, load_dataset

    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForMaskedLM, get_scheduler
    from tokenizers.processors import TemplateProcessing
    from accelerate import Accelerator

    from safetensors.torch import load_file

    from rich.live import Live
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.text import Text


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    path = '../data/nlp/experiment/'
    sft_model_path = path + 'sft_model/'
    pretrained_model_path = path + 'pretrained_model/'
    return (sft_model_path,)


@app.cell
def _():
    model_name = 'jhu-clsp/mmBERT-base'
    return (model_name,)


@app.function
def create_sft_tokenizer(
    model_name='jhu-clsp/mmBERT-base',
    bos_token='<BOS>',
    eos_token='<EOS>',
    start_token='<START_ID>',
    end_token='<END_ID>',
    eot_token='<EOT_ID>'
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {
        'bos_token' : bos_token,
        'eos_token' : eos_token,
        'additional_special_tokens' : [
            start_token, end_token, eot_token
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = eos_token
    tokenizer.cls_token = bos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f'{bos_token} $A {eos_token}',
        special_tokens=[
            (bos_token, tokenizer.bos_token_id),
            (eos_token, tokenizer.eos_token_id),
        ]
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token if loop.first else '' }}"
        f"{{{{ '{start_token}' + message['language'] + '{end_token}' }}}}"
        "{{ message['content'] }}"
        f"{{{{ '{eot_token}' if message['role'] == 'user' else '{eos_token}' }}}}"
        "{% endfor %}"
    )
    return tokenizer


@app.function
def create_mt_tokenizer(
    model_name='jhu-clsp/mmBERT-base',
    bos_token='<BOS>',
    eos_token='<EOS>',
    start_token='<START_ID>',
    end_token='<END_ID>',
    eot_token='<EOT_ID>'
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {
        'bos_token' : bos_token,
        'eos_token' : eos_token,
        'additional_special_tokens' : [
            start_token, end_token, eot_token
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = eos_token
    tokenizer.cls_token = bos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f'{bos_token} $A {eos_token}',
        special_tokens=[
            (bos_token, tokenizer.bos_token_id),
            (eos_token, tokenizer.eos_token_id),
        ]
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token }}"
        f"{{{{ '{start_token}' + 'source' + '{end_token}' }}}}"
        "{{ message['content'] }}"
        "{{ eot_token }}"
        f"{{{{ '{start_token}' + message['language'] + '{end_token}' }}}}"
        "{% endfor %}"
    )
    return tokenizer


@app.function
def load_model_and_tokenizer(model_path, model_name, device='cpu'):
    tokenizer = create_mt_tokenizer(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, device_map=device)
    model.resize_token_embeddings(len(tokenizer))

    accelerator = Accelerator(log_with=None)

    model = accelerator.prepare_model(model)
    accelerator.load_state(model_path)
    model.tie_weights()
    model.eval()
    return model, tokenizer


@app.cell
def _(model_name, sft_model_path):
    model, tokenizer = load_model_and_tokenizer(sft_model_path, model_name)
    return model, tokenizer


@app.cell
def _(tokenizer):
    template = [
        {'content' : "hello world", 'language': "Japanese"}
    ]
    tokenized = tokenizer.apply_chat_template(
        template,
        tokenize=True,
        add_special_tokens=True,
        add_generation_prmpt=False,
    )
    print(tokenized)
    return


@app.function
def prepare_unconditional_tokens_for_inference(seq_len, mask_token_id, device='cpu'):
    input_tokens = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device) 
    return input_tokens, mask, attention_mask


@app.function
def prepare_conditional_tokens_for_inference(seq_len, tokenizer, language, text, device='cpu'):
    template = [
        {'content' : text, 'language': language}
    ]
    tokenized = tokenizer.apply_chat_template(
        template,
        tokenize=True,
        add_special_tokens=True,
        add_generation_prmpt=False,
    )
    text_tokens = torch.tensor(tokenized).to(device)
    input_tokens, mask, attention_mask = prepare_unconditional_tokens_for_inference(
        seq_len, tokenizer.mask_token_id, device
    )
    input_tokens[0, :len(text_tokens)] = text_tokens
    mask[0, :len(text_tokens)] = False
    return input_tokens, mask, attention_mask


@app.function
def format_display_for_qa(user_text, translated_text):
    output = Text()
    output.append("SOURCE: ", style="bold green")
    output.append(user_text + "\n\n")
    output.append("TRANSLATION: ", style="bold cyan")
    output.append(translated_text, style="white")
    return output


@app.cell
def _(model, tokenizer):
    def diffusion_translate_v1(input_tokens, mask, attention_mask, source_text, language, num_steps, device="cpu", show_mask=True):
        ### Nice Printing Stuff ##
        console = Console(highlight=False)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            ### What Controls our Progress Bar ###
            task = progress.add_task("Generating...", total=num_steps)
            ### Get Timesteps for Inference ###
            times = torch.linspace(1, 0, num_steps + 1, device=device)
            with Live("", refresh_per_second=5, console=console) as live:
                for t, s in zip(times[:-1], times[1:]):
                    ### Compute Logits ###
                    logits = model(input_tokens, attention_mask=attention_mask).logits
                    ### Sample Gen Token from Masked Tokens ###
                    probs = torch.softmax(logits[mask], dim=-1)
                    input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    ### All Tokens are Randomly Remasked ###
                    ### For Every Position, sample a value betweewn 0 and 1 ###
                    remask_probs = torch.rand_like(mask, dtype=torch.float, device=device)
                    ### If less than proportion token is selected to be remasked ###
                    remask_probs = (remask_probs < s/t)
                    ### Only replace if our mask token was previous True and is again True ###
                    ### once a token is false (no more masking) it is here to stay! ###
                    mask = mask & remask_probs
                    ### Set those tokens back to mask ###
                    input_tokens[mask] = tokenizer.mask_token_id
                    if show_mask:
                        ### Get all of the Tokens ###
                        decoded_tokens = tokenizer.convert_ids_to_tokens(input_tokens[0])

                        ### Keep [MASK] tokens, drop all other special tokens ###
                        cleaned_tokens = []
                        for tok in decoded_tokens:
                            if tok == tokenizer.mask_token:  # keep mask tokens
                                cleaned_tokens.append(tok)
                            elif tok in tokenizer.all_special_tokens:  # drop all other specials
                                continue
                            else:
                                cleaned_tokens.append(tok)
                        ### Put all the tokens back together into a string ###
                        decoded_after = tokenizer.convert_tokens_to_string(cleaned_tokens)
                    else:
                        decoded_after = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]
                    #TODO: buggy, what if source_text contains language names or the word source ?
                    translated_text = decoded_after.replace(source_text, "").replace(language, "").replace("source", "").strip()
                    format_text = format_display_for_qa(source_text, translated_text)
                    live.update(format_text)
                    progress.update(task, advance=1)
    return (diffusion_translate_v1,)


@app.cell
def _(model, tokenizer):
    def diffusion_translate_v2(input_tokens, mask, attention_mask, source_text, language, num_steps, device="cpu", show_mask=True):
        ### Nice Printing Stuff ##
        console = Console(highlight=False)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            ### What Controls our Progress Bar ###
            task = progress.add_task("Generating...", total=num_steps)
            ### Get Timesteps for Inference ###
            times = torch.linspace(1, 0, num_steps + 1, device=device)
            with Live("", refresh_per_second=5, console=console) as live:
                for t, s in zip(times[:-1], times[1:]):
                    ### Compute Logits ###
                    logits = model(input_tokens, attention_mask=attention_mask).logits
                    ### Sample Gen Token from Masked Tokens ###
                    probs = torch.softmax(logits[mask], dim=-1)
                    input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    ### Low confidence Tokens are Randomly Remasked ###
                    ### Compute Probs for all Tokens ###
                    probs_all = torch.nn.functional.softmax(logits, dim=-1)
                    ### Get the probability of the actually selected token ###
                    ### probs_all: 1 x seq_len x vocab_size
                    ### input_tokens: 1 x seq_len
                    chosen_token_probs = torch.gather(probs_all, dim=-1, index=input_tokens.unsqueeze(-1)).squeeze(-1)
                    ### Make sure to set all tokens already selected to not be remasked to again ###
                    ### not be selected to be remasked. We can just set them to 1 because we want ###
                    ### low confidence (prob) tokens to be replaced! (set False to 1) ###
                    chosen_token_probs[~mask] = 1.0
                    ### Compute Proportion of Tokens to Remask out of the tokens that are currently masked ###
                    num_to_remask = int((s/t) * mask.sum().item())
                    if num_to_remask > 0:
                        ### Find the lowest prob tokens ###
                        lowest_confidence_idx = torch.topk(chosen_token_probs, num_to_remask, largest=False).indices
                        ### Create a New Mask (where everything is set to False) ###
                        new_mask = torch.zeros_like(mask)
                        ### Set the lowest confidence tokens to be remasked ###
                        new_mask[0, lowest_confidence_idx] = True
                        mask = new_mask
                        ### Update our Input Tokens with Mask Tokens ###
                        input_tokens[mask] = tokenizer.mask_token_id
                    if show_mask:
                        ### Get all of the Tokens ###
                        decoded_tokens = tokenizer.convert_ids_to_tokens(input_tokens[0])

                        ### Keep [MASK] tokens, drop all other special tokens ###
                        cleaned_tokens = []
                        for tok in decoded_tokens:
                            if tok == tokenizer.mask_token:  # keep mask tokens
                                cleaned_tokens.append(tok)
                            elif tok in tokenizer.all_special_tokens:  # drop all other specials
                                continue
                            else:
                                cleaned_tokens.append(tok)
                        ### Put all the tokens back together into a string ###
                        decoded_after = tokenizer.convert_tokens_to_string(cleaned_tokens)
                    else:
                        decoded_after = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]
                    #TODO: buggy, what if source_text contains language names or the word source ?
                    translated_text = decoded_after.replace(source_text, "").replace(language, "").replace("source", "").strip()
                    format_text = format_display_for_qa(source_text, translated_text)
                    live.update(format_text)
                    progress.update(task, advance=1)
    return


@app.cell
def _(diffusion_translate_v1, tokenizer):
    def machine_translate(source_text, target_language, seq_len=2048, num_steps=10, device="cpu", show_mask=True):
        input_tokens, mask, attention_mask = prepare_conditional_tokens_for_inference(
            seq_len,
            tokenizer,
            target_language,
            source_text,
            device
        )
        diffusion_translate_v1(
            input_tokens,
            mask,
            attention_mask,
            source_text,
            target_language,
            num_steps,
            device=device,
            show_mask=show_mask
        )
    return


@app.cell
def _():
    #machine_translate("good morning sunshine", "Japanese", device='cuda')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
