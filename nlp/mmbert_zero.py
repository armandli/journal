import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import torch
    return AutoModelForMaskedLM, AutoTokenizer, torch


@app.cell
def _(AutoModelForMaskedLM, AutoTokenizer):
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    model = AutoModelForMaskedLM.from_pretrained("jhu-clsp/mmBERT-base")
    return model, tokenizer


@app.cell
def _(model, tokenizer, torch):
    def predict_masked_token(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        mask_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
        predictions = outputs.logits[mask_indices]
        top_tokens, top_indices = torch.topk(predictions, 5, dim=-1)
        return [tokenizer.decode(token) for token in top_indices[0]]
    return (predict_masked_token,)


@app.cell
def _():
    texts = [
        "The capital of France is <mask>.",
        "La capital de España es <mask>.",
        "Die Hauptstadt von Deutschland ist <mask>.",
    ]
    return (texts,)


@app.cell
def _(predict_masked_token, texts):
    for text in texts:
        predictions = predict_masked_token(text)
        print(f"Text: {text}")
        print(f"Predictions: {predictions}\n")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
