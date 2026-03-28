import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer
    from datasets import load_dataset, load_from_disk, concatenate_datasets
    return AutoTokenizer, TemplateProcessing, load_dataset


@app.cell
def _(AutoTokenizer, TemplateProcessing):
    # modify existing tokenizer to add missing required token types
    def get_tokenizer(model_name='answerdotai/ModernBERT-base',
                     bos_token='<BOS>',
                     eos_token='<EOS>',
                     start_token='<START_ID>',
                     end_token='<END_ID>',
                     eot_token='<EOT_ID>'):
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
            single=f"{bos_token} $A {eos_token}",
            special_tokens=[
                (bos_token, tokenizer.bos_token_id),
                (eos_token, tokenizer.eos_token_id),
            ]
        )
        # chat template for SFT
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ bos_token if loop.first else '' }}"
            f"{{{{ '{start_token}' + message['role'] + '{end_token}' }}}}\n"
            "{{ message['content'] }}"
            f"{{{{ '{eot_token}' if message['role'] == 'user' else eos_token }}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{start_token}' + assistant + '{end_token}' }}}}"
            "{% endif %}"
        )
        return tokenizer
    return (get_tokenizer,)


@app.cell
def _(get_tokenizer):
    def test_tokenizer():
        tokenizer = get_tokenizer()
        text = "Hello World"
        ids = tokenizer(text, padding=True, return_tensors='pt')['input_ids'][0]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        # chatbot message format example
        messages = [
            {"role" : "user", "content" : "what is the capital of France ?"},
            {"role" : "assistant", "content" : "Paris"},
        ]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        return decoded
    return (test_tokenizer,)


@app.cell
def _(test_tokenizer):
    print(test_tokenizer())
    return


@app.cell
def _():
    ### prepare dataset
    return


@app.cell
def _():
    data_dir = '../data/nlp/'
    cache_dir = '../data/nlp/cache/'
    return (cache_dir,)


@app.cell
def _(cache_dir, load_dataset):
    gutenberg = load_dataset("manu/project_gutenberg", split="en", cache_dir=cache_dir, num_proc=8)
    gutenberg = gutenberg.remove_columns([col for col in gutenberg.column_names if col != "text"])
    return (gutenberg,)


@app.cell
def _(gutenberg):
    gutenberg
    return


@app.cell
def _(gutenberg):
    dataset = gutenberg.train_test_split(test_size=0.005, seed=42)
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset):
    dataset['train']['text'][0]
    return


@app.cell
def _(get_tokenizer):
    tokenizer = get_tokenizer()
    return (tokenizer,)


@app.cell
def _():
    context_length = 1024
    return (context_length,)


@app.cell
def _(context_length, tokenizer):
    ### Tokenize Dataset ###
    def compute_tokens(examples):
        tokenized = tokenizer(examples["text"], 
                              return_attention_mask=False, 
                              add_special_tokens=True,
                              max_length=None,
                              truncation=False)

        ### Chunk Text ###
        input_ids_list = []
        for ids in tokenized["input_ids"]:
            for i in range(0, len(ids), context_length):
                chunk = ids[i:i+context_length]
                if len(chunk) < context_length:
                    chunk = chunk + [tokenizer.pad_token_id] * (context_length - len(chunk))
                input_ids_list.append(chunk)
        
        return {"input_ids": input_ids_list}
    return (compute_tokens,)


@app.cell
def _(compute_tokens, dataset):
    tokenized_data = dataset.map(
        compute_tokens, 
        batched=True, 
        batch_size=8,
        num_proc=8, 
        remove_columns="text"
    )
    return (tokenized_data,)


@app.cell
def _(path_to_save, tokenized_data):
    tokenized_data.save_to_disk(path_to_save)
    return


if __name__ == "__main__":
    app.run()
