import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from transformers import pipeline
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity, pipeline


@app.cell
def _():
    ### modernbert zero-shot fill mask task
    return


@app.cell
def _(pipeline):
    fill_task = pipeline(
        task='fill-mask',
        model='answerdotai/ModernBERT-base',
        tokenizer='answerdotai/ModernBERT-base'
    )
    return (fill_task,)


@app.cell
def _():
    example_text1 = "The capital of France is [MASK]."
    return (example_text1,)


@app.cell
def _(example_text1, fill_task):
    predictions1 = fill_task(example_text1)
    return (predictions1,)


@app.cell
def _(example_text1, predictions1):
    print("Masked Text:", example_text1)
    print("Predictions:")

    for pred in predictions1:
        print(f"  - {pred['sequence']} (score: {pred['score']:.4f})")
    return


@app.cell
def _():
    ### ModernBERT zero-shot feature extraction task
    return


@app.cell
def _(pipeline):
    feature_extractor = pipeline(
        task="feature-extraction",
        model="answerdotai/ModernBERT-base",
        tokenizer="answerdotai/ModernBERT-base",
    )
    return (feature_extractor,)


@app.cell
def _():
    example_text2 = "ModernBERT is a robust model for natural language understanding."
    return (example_text2,)


@app.cell
def _(example_text2, feature_extractor):
    features = feature_extractor(example_text2)
    return (features,)


@app.cell
def _(features):
    print(f"Extracted feature shape: {len(features)} x {len(features[0])}")
    return


@app.cell
def _():
    ### ModernBERT zero-shot sentence similarity
    return


@app.cell
def _():
    sentence_1 = "ModernBERT is a great language model."
    sentence_2 = "ModernBERT excels in understanding language."
    return sentence_1, sentence_2


@app.cell
def _(feature_extractor, sentence_1, sentence_2):
    embedding_1 = feature_extractor(sentence_1)[0][0]
    embedding_2 = feature_extractor(sentence_2)[0][0]
    return embedding_1, embedding_2


@app.cell
def _(cosine_similarity, embedding_1, embedding_2):
    similarity = cosine_similarity([embedding_1], [embedding_2])
    print(f"Similarity between sentences: {similarity[0][0]:.4f}")
    return


@app.cell
def _():
    ### ModernBERT zero-shot next word prediction
    return


@app.cell
def _():
    example_text3 = "ModernBERT is designed for [MASK]."
    return (example_text3,)


@app.cell
def _(example_text3, fill_task):
    predictions2 = fill_task(example_text3)
    return (predictions2,)


@app.cell
def _(example_text3, predictions2):
    print("Masked Text:", example_text3)
    print("Next Word Predictions:")
    for pred2 in predictions2:
        print(f"  - {pred2['sequence']} (score: {pred2['score']:.4f})")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
