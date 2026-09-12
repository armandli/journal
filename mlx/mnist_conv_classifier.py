import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import mlx
    import mlx.nn as nn
    import mlx.core as mx
    import mlx.optimizers as optim
    import mlx.utils
    from mlx.data.datasets import load_mnist

    import os
    import time
    from functools import partial

    import matplotlib
    import matplotlib.pyplot as plt


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    data_dir = '../data/mnist'
    return


@app.cell
def _():
    train_ds = load_mnist(root="../data/mnist", train=True)
    test_ds = load_mnist(root="../data/mnist", train=False)
    return test_ds, train_ds


@app.function
def make_datasets(train_ds, test_ds, batch_size):
    def normalize(x):
        return x.astype("float32") / 255.0

    shuffled = train_ds.shuffle()
    train_iter = (
        shuffled
        .to_stream()
        .key_transform("image", normalize)
        .batch(batch_size)
    )
    test_iter = (
        test_ds
        .to_stream()
        .key_transform("image", normalize)
        .batch(batch_size)
    )
    return train_iter, test_iter


@app.cell
def _():
    ### model
    return


@app.class_definition
class ClassifierV1(nn.Module):
    def __init__(self, img_sz, inc, nclass):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inc    , inc * 2, 3, 1, 1),
            nn.Conv2d(inc * 2, inc * 4, 3, 1, 1),
        )
        self.ffn = nn.Sequential(
            nn.Linear((img_sz) * (img_sz) * inc * 4, nclass),
            nn.Softmax(),
        )

    def __call__(self, x):
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.ffn(x)
        return x


@app.cell
def _():
    lr = 0.01
    decay = 0.0001
    epochs = 10
    batch_size = 32
    return batch_size, decay, epochs, lr


@app.cell
def _(batch_size, decay, lr, test_ds, train_ds):
    train_iter, test_iter = make_datasets(train_ds, test_ds, batch_size)

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=decay)
    loss = nn.losses.cross_entropy
    return optimizer, test_iter, train_iter


@app.cell
def _():
    model = ClassifierV1(28, 1, 10)
    return (model,)


@app.cell
def _():
    ### basic training code
    return


@app.function
def compute_loss(model, x, y):
    pred = model(x)
    return mx.mean(nn.losses.cross_entropy(pred, y))


@app.function
def train_epoch(model, train_iter, optimizer, epoch):
    model.train(True)
    train_iter.reset()
    losses = 0.
    count = 0
    lng = nn.value_and_grad(model, compute_loss)
    for batch in train_iter:
        x = mx.array(batch['image'])
        y = mx.array(batch['label'])
        loss, grad = lng(model, x, y)
        optimizer.update(model, grad)
        mx.eval(loss, model.parameters())
        losses += loss.item()
        count += 1
    return losses / count


@app.function
def validation_epoch(model, data_iter, epoch):
    model.train(False)
    data_iter.reset()
    losses = 0.
    count = 0
    for batch in data_iter:
        x = mx.array(batch['image'])
        y = mx.array(batch['label'])
        loss = compute_loss(model, x, y)
        mx.eval(loss)
        losses += loss.item()
        count += 1
    return losses / count


@app.function
def train(model, train_iter, test_iter, optimizer, epochs):
    total_time = 0.
    count = 0
    for epoch in range(epochs):
        tic = time.perf_counter()
        avg_loss = train_epoch(model, train_iter, optimizer, epoch)
        eval_loss = validation_epoch(model, test_iter, epoch)
        toc = time.perf_counter()
        total_time += (toc - tic)
        count += 1
        print(f"{epoch}: train loss {avg_loss} validation loss {eval_loss} time {toc - tic} sec")
    print(f"total time {total_time} sec")


@app.cell
def _(epochs, model, optimizer, test_iter, train_iter):
    train(model, train_iter, test_iter, optimizer, epochs)
    return


@app.cell
def _(mo, model, test_iter, train_iter):
    _train_acc = compute_accuracy(model, train_iter)
    _test_acc = compute_accuracy(model, test_iter)
    mo.md(f"**Basic training — train accuracy:** {_train_acc:.4f} | **test accuracy:** {_test_acc:.4f}")
    return


@app.cell
def _():
    ### use mx.compile for training
    return


@app.function
@mx.compile
def train_lng(model, x, y, optimizer):
    @mx.compile
    def train_loss(model, x, y):
        pred = model(x)
        return mx.mean(nn.losses.cross_entropy(pred, y))
    lng = nn.value_and_grad(model, train_loss)
    loss, grad = lng(model, x, y)
    optimizer.update(model, grad)
    return loss


@app.function
def opt_train_epoch(model, train_iter, optimizer, epoch):
    model.train(True)
    train_iter.reset()
    losses = 0.
    count = 0
    for batch in train_iter:
        x = mx.array(batch['image'])
        y = mx.array(batch['label'])
        loss = train_lng(model, x, y, optimizer)
        mx.eval(loss, model.parameters())
        losses += loss.item()
        count += 1
    return losses / count


@app.function
@mx.compile
def validate_lng(model, x, y):
    pred = model(x)
    loss = mx.mean(nn.losses.cross_entropy(pred, y))
    return loss


@app.function
def opt_validation_epoch(model, data_iter, epoch):
    model.train(False)
    data_iter.reset()
    losses = 0.
    count = 0
    for batch in data_iter:
        x = mx.array(batch['image'])
        y = mx.array(batch['label'])
        loss = validate_lng(model, x, y)
        mx.eval(loss)
        losses += loss.item()
        count += 1
    return losses / count


@app.function
def compute_accuracy(model, data_iter):
    model.train(False)
    data_iter.reset()
    correct = 0
    total = 0
    for batch in data_iter:
        x = mx.array(batch['image'])
        y = mx.array(batch['label'])
        pred = model(x)
        predicted = mx.argmax(pred, axis=1)
        correct += mx.sum(predicted == y).item()
        total += y.shape[0]
    return correct / total


@app.function
def opt_train(model, train_iter, test_iter, optimizer, epochs):
    total_time = 0.
    count = 0
    for epoch in range(epochs):
        tic = time.perf_counter()
        avg_loss = opt_train_epoch(model, train_iter, optimizer, epoch)
        eval_loss = opt_validation_epoch(model, test_iter, epoch)
        toc = time.perf_counter()
        total_time += (toc - tic)
        count += 1
        print(f"{epoch}: train loss {avg_loss} validation loss {eval_loss} time {toc - tic}")
    print(f"total time {total_time}")


@app.cell
def _(epochs, model, optimizer, test_iter, train_iter):
    opt_train(model, train_iter, test_iter, optimizer, epochs)
    return


@app.cell
def _(mo, model, test_iter, train_iter):
    _train_acc = compute_accuracy(model, train_iter)
    _test_acc = compute_accuracy(model, test_iter)
    mo.md(f"**Optimized training — train accuracy:** {_train_acc:.4f} | **test accuracy:** {_test_acc:.4f}")
    return


@app.cell
def _(model):
    os.makedirs("models", exist_ok=True)
    model.save_weights("models/mnist_conv_classifier.npz")
    print("Model saved to models/mnist_conv_classifier.npz")
    return


@app.cell
def _():
    #### conclusion: it does not appear mx.compile helps to run training loop faster, this is probably because compile is enabled by default
    return


if __name__ == "__main__":
    app.run()
