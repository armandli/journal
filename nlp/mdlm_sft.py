import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import os
    import json
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


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    config_file = "s3config.json"
    return (config_file,)


@app.cell
def _(config_file):
    with open(config_file) as file:
        config_data = json.load(file)
    return (config_data,)


@app.cell
def _(config_data):
    s3_path = config_data['S3_DATA_PATH'] + 'sft_dataset_v1/'
    return (s3_path,)


@app.cell
def _(config_data):
    storage_options = {
        'aws_access_key_id': config_data['S3_KEY'],
        'aws_secret_access_key': config_data['S3_SECRET'],
        'endpoint_url': config_data['S3_ENDPOINT'],
        # Other options like region, endpoint_url, etc. can also be included here
    }
    return (storage_options,)


@app.cell
def _(config_data):
    storage_options2 = {
        's3': {
            'client_kwargs': {
                'aws_access_key_id': config_data['S3_KEY'],
                'aws_secret_access_key': config_data['S3_SECRET'],
                'endpoint_url': config_data['S3_ENDPOINT'],
                # Other options like region, endpoint_url, etc. can also be included here
            }
        }
    }
    return (storage_options2,)


@app.cell
def _(storage_options):
    s3 = s3fs.S3FileSystem(client_kwargs=storage_options)
    return


@app.cell
def _():
    num_procs = 8
    n_training_steps = 100000
    return n_training_steps, num_procs


@app.cell
def _(s3_path, storage_options2):
    dataset = load_from_disk(s3_path, storage_options=storage_options2)
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _():
    model_name='jhu-clsp/mmBERT-base'
    return (model_name,)


@app.cell
def _():
    path = '../data/nlp/experiment/'
    return (path,)


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
def SFTCollator(model_name="jhu-clsp/mmBERT-base"):

    tokenizer = create_sft_tokenizer(model_name)
    eos_token = tokenizer.eos_token_id

    def _collate_fn(batch):

        inputs = [torch.tensor(b["input_ids"]) for b in batch]
        query_masks = [torch.tensor(b["pt_mask"]) for b in batch]

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value=eos_token, batch_first=True)
        query_masks = torch.nn.utils.rnn.pad_sequence(query_masks, padding_value=1, batch_first=True)

        return {"input_ids": inputs, "pt_mask": query_masks}

    return _collate_fn


@app.cell
def _():
    # gpu batch size // gradient accumulation steps
    mini_batchsize = 4 // 1
    return (mini_batchsize,)


@app.cell
def _():
    tokenizer = create_sft_tokenizer()
    return (tokenizer,)


@app.cell
def _(path):
    accelerator = Accelerator(project_dir=path, log_with=None)
    return (accelerator,)


@app.cell
def _(
    accelerator,
    dataset,
    mini_batchsize,
    model_name,
    n_training_steps,
    num_procs,
    path,
    tokenizer,
):
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_dataloader = DataLoader(dataset["train"], batch_size=mini_batchsize, collate_fn=SFTCollator(model_name), shuffle=True)
    eval_dataloader = DataLoader(dataset["test"], batch_size=mini_batchsize, collate_fn=SFTCollator(model_name), shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)

    scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=1000 * num_procs, num_training_steps=n_training_steps * num_procs)

    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    # could change later
    accelerator.load_state(path + 'pretrained_model')
    return (model,)


@app.cell
def _(accelerator, model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    accelerator.print("Number of Parameters:", params)
    return


@app.cell
def _():
    loss_func = nn.CrossEntropyLoss(reduction="none")
    return


@app.function
def evaluate(
    model,
    tokenizer,
    eval_dataloader,
    accelerator,
    loss_func,
    completed_steps: int,
    num_training_steps: int,
    progress_bar,
    total_eval_steps=10000
):
    """Run evaluation loop and return validation loss."""
    if accelerator.is_main_process:
        progress_bar.write("Evaluating Model!!")
    model.eval()

    total_loss = 0.0
    num_losses = 0
    eval_steps = 0

    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(accelerator.device)
        query_mask = batch["pt_mask"].to(accelerator.device)
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)

        t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len)
        mask = torch.bernoulli(t).bool()
        mask = mask * query_mask
        mask = mask.bool()

        masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)

        with torch.inference_mode():
            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"]

        num_classes = logits.shape[-1]
        loss = loss_func(logits.reshape(batch_size * seq_len, num_classes), labels.flatten())
        loss = loss.reshape(batch_size, seq_len) / t
        answer_lengths = query_mask.sum(dim=1, keepdim=True)
        answer_lengths = answer_lengths.clamp_min(1)
        loss = loss / answer_lengths
        loss = loss.sum(dim=1).mean()

        loss = loss.detach()
        if accelerator.num_processes > 1:
            loss = torch.mean(accelerator.gather_for_metrics(loss))

        total_loss += loss
        num_losses += 1
        eval_steps += 1
        if eval_steps >= total_eval_steps:
            break

    val_loss = total_loss / num_losses
    logging_string = f"[{completed_steps}/{num_training_steps}] Validation Loss: {val_loss}"
    if accelerator.is_main_process:
        progress_bar.write(logging_string)

    model.train()
    return val_loss


@app.function
def sft_train(
    model,
    tokenizer,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    accelerator,
    loss_func,
    path_to_experiment: str,
    num_training_steps: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    evaluation_interval: int = 1000,
    completed_steps: int = 1,
):
    train = True
    progress_bar = tqdm(range(completed_steps, num_training_steps))

    while train:
        ### Keep Track of Accumulated Mini-Steps ###
        accumulate_steps = 0
        ### Accumulated Loss ###
        accumulate_loss = 0

        for batch in train_dataloader:
            ### Grab Input IDs ###
            input_ids = batch["input_ids"].to(accelerator.device)
            query_mask = batch["pt_mask"].to(accelerator.device)
            ### Attend to All Tokens (EVEN EOS) ###
            batch_size, seq_len = input_ids.shape
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)
            ### Random sample t to mask each token with that probability ###
            t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len).clamp_min(1e-5)
            mask = torch.bernoulli(t)
            ### Mask only valid where it is not query ###
            mask = mask * query_mask
            mask = mask.bool()
            ### Mask Data and Dont Compute Loss for Unmasked Data ###
            masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
            labels = input_ids.masked_fill(~mask, -100)
            ### Compute Logits ###
            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"]
            ### Compute Loss (per token) ###
            num_classes = logits.shape[-1]
            loss = loss_func(logits.reshape(batch_size * seq_len, num_classes), labels.flatten())
            ### loss per sample by the t ###
            loss = loss.reshape(batch_size, seq_len) / t
            ### Different answers are of different lengths, so lets scale by that too ###
            answer_lengths = query_mask.sum(dim=1, keepdim=True)
            answer_lengths = answer_lengths.clamp_min(1)
            loss = loss / answer_lengths
            ### Add up all the per-token losses and average across batch ###
            loss = loss.sum(dim=1).mean()
            ### Scale Loss by Gradient Accumulation Steps ###
            loss = loss / gradient_accumulation_steps
            accumulate_loss += loss
            ### Compute Gradients ###
            accelerator.backward(loss)
            accumulate_steps += 1

            if accumulate_steps % gradient_accumulation_steps == 0:
                ### Update Model ###
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ### Update Scheduler ###
                scheduler.step()

                ### Log Results ###
                if completed_steps % evaluation_interval == 0:
                    accumulate_loss = accumulate_loss.detach()
                    if accelerator.state.num_processes > 1:
                        accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))
                    log = {"train_loss": accumulate_loss, "learning_rate": scheduler.get_last_lr()[0]}
                    logging_string = f"[{completed_steps}/{num_training_steps}] Training Loss: {accumulate_loss}"
                    if accelerator.is_main_process:
                        progress_bar.write(logging_string)

                ### Evaluation Loop ###
                if completed_steps % evaluation_interval == 0:
                    evaluate(
                        model=model,
                        tokenizer=tokenizer,
                        eval_dataloader=eval_dataloader,
                        accelerator=accelerator,
                        loss_func=loss_func,
                        completed_steps=completed_steps,
                        num_training_steps=num_training_steps,
                        progress_bar=progress_bar,
                    )

                ### Checkpoint Model ###
                if completed_steps % evaluation_interval == 0:
                    path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")
                    if accelerator.is_main_process:
                        progress_bar.write(f"Saving Checkpoint to {path_to_checkpoint}")
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        accelerator.save_state(output_dir=path_to_checkpoint)

                if completed_steps >= num_training_steps:
                    train = False
                    if accelerator.is_main_process:
                        progress_bar.write("Completed Training!!")
                    break

                ### Iterate Progress Bar and Completed Steps ###
                completed_steps += 1
                progress_bar.update(1)
                ### Reset Loss Accumulate For Next Accumulation ###
                accumulate_loss = 0

    return completed_steps


@app.cell
def _():
    # sft_train(
    #     model,
    #     tokenizer,
    #     train_dataloader,
    #     eval_dataloader,
    #     optimizer,
    #     scheduler,
    #     accelerator,
    #     loss_func,
    #     path,
    #     n_training_steps,
    #     evaluation_interval=50000,
    # )
    return


@app.cell
def _():
    # path_to_checkpoint = os.path.join(path, f"sft_model")
    # accelerator.save_state(output_dir=path_to_checkpoint)
    # accelerator.end_training()
    return


if __name__ == "__main__":
    app.run()
