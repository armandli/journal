import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import csv
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer, pipeline
    from datasets import load_dataset, load_from_disk, concatenate_datasets

    return (
        AutoTokenizer,
        TemplateProcessing,
        concatenate_datasets,
        csv,
        load_dataset,
    )


@app.cell
def _():
    data_dir = '../data/machine_translation/'
    cache_dir = '../data/nlp/cache/'
    return cache_dir, data_dir


@app.cell
def _():
    n_proc = 16
    return (n_proc,)


@app.cell
def _():
    context_length = 8192 # max sequence length for mmBERT
    return (context_length,)


@app.cell
def _():
    files = [
        # ("cmn.txt", "Mandarin Chinese"),
        # ("dan.txt", "Danish"),
        # ("deu.txt", "German"),
        # ("fin.txt", "Finnish"),
        # ("fra.txt", "French"),
        # ("heb.txt", "Hebrew"),
        # ("hin.txt", "Hindi"),
        # ("isl.txt", "Icelandic"),
        # ("ita.txt", "Italian"),
        ("jpn.txt", "Japanese"),
        # ("kor.txt", "Korean"),
        # ("nld.txt", "Dutch"),
        # ("nno.txt", "Norwegian Nynorsk"),
        # ("nob.txt", "Norwegian Bokmål"),
        # ("pes.txt", "Persian"),
        # ("rus.txt", "Russian"),
        # ("spa.txt", "Spanish"),
        # ("swe.txt", "Swedish"),
        # ("tha.txt", "Thai"),
        # ("ukr.txt", "Ukranian"),
        # ("vie.txt", "Vietanese"),
        # ("yue.txt", "Cantonese Chinese"),
    ]
    return (files,)


@app.cell
def _():
    ### test if tokenizer can parse every character in dataset
    return


@app.cell
def _(data_dir):
    def create_raw_dataset(files):
        dataset = []
        for file, lang in files:
            with open(data_dir + file, 'r', encoding='UTF-8') as f:
                while line := f.readline():
                    dataset.append((*line.rstrip().split('\t')[:2], lang))
        return dataset

    return (create_raw_dataset,)


@app.cell
def _(create_raw_dataset, files):
    test_raw_data = create_raw_dataset(files)
    return (test_raw_data,)


@app.cell
def _(test_raw_data):
    len(test_raw_data)
    return


@app.cell
def _(AutoTokenizer):
    orig_tokenizer = AutoTokenizer.from_pretrained('jhu-clsp/mmBERT-base')
    return (orig_tokenizer,)


@app.cell
def _(orig_tokenizer):
    def test_original_tokenizer(dataset):
        for eng, tlang, lang in dataset:
            ids = orig_tokenizer(tlang, padding=True, return_tensors='pt')['input_ids'][0]
            for id in ids:
                if id == orig_tokenizer.unk_token_id:
                    print(tlang)
                    return False
        return True

    return (test_original_tokenizer,)


@app.cell
def _(test_original_tokenizer, test_raw_data):
    test_original_tokenizer(test_raw_data)
    return


@app.cell
def _(AutoTokenizer, TemplateProcessing):
    def create_pretraining_tokenizer(
        model_name='jhu-clsp/mmBERT-base',
        bos_token='<BOS>',
        eos_token='<EOS>',
        start_token='<START_ID>',
        end_token='<END_ID>'
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens = {
            'bos_token' : bos_token,
            'eos_token' : eos_token,
            'additional_special_tokens' : [
                start_token, end_token
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
        # assume only one message
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ bos_token }}"
            f"{{{{ '{start_token}' + message['language'] + '{end_token}' }}}}"
            "{{ message['content'] }}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
        return tokenizer

    return (create_pretraining_tokenizer,)


@app.cell
def _(AutoTokenizer, TemplateProcessing):
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

    return (create_sft_tokenizer,)


@app.cell
def _():
    ### create pre-training dataset
    return


@app.cell
def _(create_pretraining_tokenizer):
    pretrain_tokenizer = create_pretraining_tokenizer()
    return (pretrain_tokenizer,)


@app.cell
def _(data_dir):
    pt_file = data_dir + 'en-ja.bicleaner05.txt'
    return (pt_file,)


@app.cell
def _(cache_dir, csv, load_dataset, pt_file):
    pt_raw_dataset = load_dataset('csv', data_files=pt_file, delimiter='\t', quoting=csv.QUOTE_NONE, column_names=['source1', 'source2', 'score', 'SourceText', 'TargetText'], num_proc=8, cache_dir=cache_dir, split='train')
    pt_raw_dataset = pt_raw_dataset.remove_columns([col for col in pt_raw_dataset.column_names if col != "SourceText" and col != 'TargetText'])
    pt_raw_dataset = pt_raw_dataset.add_column('SourceLanguage', ['English'] * len(pt_raw_dataset))
    pt_raw_dataset = pt_raw_dataset.add_column('TargetLanguage', ['Japanese'] * len(pt_raw_dataset))

    # need to remove half in order to fit in limited space
    pt_raw_dataset = pt_raw_dataset.select(range(0, len(pt_raw_dataset), 2))
    return (pt_raw_dataset,)


@app.cell
def _(pretrain_tokenizer):
    def apply_pretrain_chat_template(language, sentence):
        return pretrain_tokenizer.apply_chat_template([
            {'language': language, 'content': sentence}
        ], tokenize=True, add_special_tokens=True)

    return (apply_pretrain_chat_template,)


@app.cell
def _(apply_pretrain_chat_template):
    def pt_preprocess_source(example):
        sentence = example['SourceText']
        language = example['SourceLanguage']
        tokenized = apply_pretrain_chat_template(language, sentence)
        return {'input_ids': tokenized, 'length': len(tokenized)}

    return (pt_preprocess_source,)


@app.cell
def _(apply_pretrain_chat_template):
    def pt_preprocess_target(example):
        sentence = example['TargetText']
        language = example['TargetLanguage']
        tokenized = apply_pretrain_chat_template(language, sentence)
        return {'input_ids': tokenized, 'length': len(tokenized)}

    return (pt_preprocess_target,)


@app.cell
def _(context_length):
    def keep_within_context(example):
        if example['length'] > context_length:
            print(f"context length > {context_length} detected")
        return example['length'] <= context_length

    return (keep_within_context,)


@app.cell
def _(pretrain_tokenizer):
    def get_pt_mask(example):
        tokenized = example['input_ids']
        end_id = pretrain_tokenizer.convert_tokens_to_ids("<END_ID>")
        pt_mask = []
        is_text = False
        for t in tokenized:
            check = t == end_id
            if not is_text:
                pt_mask.append(0)
            else:
                pt_mask.append(1)
            if check:
                is_text = True
        example['pt_mask'] = pt_mask
        return example

    return (get_pt_mask,)


@app.cell
def _(
    concatenate_datasets,
    get_pt_mask,
    keep_within_context,
    n_proc,
    pt_preprocess_source,
    pt_preprocess_target,
    pt_raw_dataset,
):
    pt_english_dataset = pt_raw_dataset.map(pt_preprocess_source, num_proc=n_proc, remove_columns=['SourceLanguage', 'TargetLanguage', 'SourceText', 'TargetText'])
    pt_japanese_dataset = pt_raw_dataset.map(pt_preprocess_target, num_proc=n_proc, remove_columns=['SourceLanguage', 'TargetLanguage', 'SourceText', 'TargetText'])
    pt_dataset = concatenate_datasets([pt_english_dataset, pt_japanese_dataset])
    pt_dataset = pt_dataset.train_test_split(test_size=0.05)
    pt_dataset = pt_dataset.filter(keep_within_context, num_proc=n_proc)
    pt_dataset = pt_dataset.remove_columns(['length'])
    pt_dataset = pt_dataset.map(get_pt_mask, num_proc=n_proc)
    return (pt_dataset,)


@app.cell
def _(data_dir):
    pt_dataset_file = data_dir + 'pretrain_dataset_v2'
    return (pt_dataset_file,)


@app.cell
def _(pt_dataset, pt_dataset_file):
    pt_dataset.save_to_disk(pt_dataset_file)
    return


@app.cell
def _():
    ### create fine tuning dataset
    return


@app.cell
def _(create_sft_tokenizer):
    sft_tokenizer = create_sft_tokenizer()
    return (sft_tokenizer,)


@app.cell
def _(cache_dir, concatenate_datasets, csv, data_dir, load_dataset):
    def create_sft_raw_dataset(files):
        datasets = []
        for file, lang in files:
            dataset = load_dataset('csv', data_files=data_dir + file, delimiter='\t', quoting=csv.QUOTE_NONE, column_names=['SourceText', 'TargetText', 'Source'], cache_dir=cache_dir, split='train')
            dataset = dataset.remove_columns(['Source'])
            dataset = dataset.add_column('SourceLanguage', ['English'] * len(dataset))
            dataset = dataset.add_column('TargetLanguage', [lang] * len(dataset))
            datasets.append(dataset)
        return concatenate_datasets(datasets)

    return (create_sft_raw_dataset,)


@app.cell
def _(create_sft_raw_dataset, files):
    sft_raw_dataset = create_sft_raw_dataset(files)
    return (sft_raw_dataset,)


@app.cell
def _(sft_tokenizer):
    def apply_sft_chat_template(source, target, language):
        return sft_tokenizer.apply_chat_template([
            {'role': 'user', 'language': 'source', 'content': source},
            {'role': 'assistant', 'language': language, 'content': target},
        ])

    return (apply_sft_chat_template,)


@app.cell
def _(apply_sft_chat_template):
    def sft_std_preprocess(example):
        source = example['SourceText']
        target = example['TargetText']
        language = example['TargetLanguage']
        tokenized = apply_sft_chat_template(source, target, language)
        return {'input_ids': tokenized, 'length': len(tokenized)}

    return (sft_std_preprocess,)


@app.cell
def _(apply_sft_chat_template):
    def sft_dts_preprocess(example):
        source = example['TargetText']
        target = example['SourceText']
        language = example['SourceLanguage']
        tokenized = apply_sft_chat_template(source, target, language)
        return {'input_ids': tokenized, 'length': len(tokenized)}

    return (sft_dts_preprocess,)


@app.cell
def _(sft_tokenizer):
    def get_sft_mask(example):
        tokenized = example['input_ids']
        end_id = sft_tokenizer.convert_tokens_to_ids("<END_ID>")
        count = 0
        pt_mask = []
        is_text = False
        for t in tokenized:
            check = t == end_id
            if not is_text:
                pt_mask.append(0)
            else:
                pt_mask.append(1)
            if check:
                count += 1
                if count >= 2:
                    is_text = True
        example['pt_mask'] = pt_mask
        return example

    return (get_sft_mask,)


@app.cell
def _(
    concatenate_datasets,
    get_sft_mask,
    keep_within_context,
    n_proc,
    sft_dts_preprocess,
    sft_raw_dataset,
    sft_std_preprocess,
):
    sft_std_dataset = sft_raw_dataset.map(sft_std_preprocess, num_proc=n_proc, remove_columns=['SourceLanguage', 'TargetLanguage', 'SourceText', 'TargetText'])
    sft_dts_dataset = sft_raw_dataset.map(sft_dts_preprocess, num_proc=n_proc, remove_columns=['SourceLanguage', 'TargetLanguage', 'SourceText', 'TargetText'])
    sft_dataset = concatenate_datasets([sft_std_dataset, sft_dts_dataset])
    sft_dataset = sft_dataset.train_test_split(test_size=0.1)
    sft_dataset = sft_dataset.filter(keep_within_context, num_proc=n_proc)
    sft_dataset = sft_dataset.remove_columns(['length'])
    sft_dataset = sft_dataset.map(get_sft_mask, num_proc=n_proc)
    return (sft_dataset,)


@app.cell
def _(data_dir):
    sft_dataset_file = data_dir + 'sft_dataset_v1'
    return (sft_dataset_file,)


@app.cell
def _(sft_dataset, sft_dataset_file):
    sft_dataset.save_to_disk(sft_dataset_file)
    return


if __name__ == "__main__":
    app.run()
