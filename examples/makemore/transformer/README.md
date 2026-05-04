# A Simple GPT (Decoder-Only Transformer)

This example tries to train the same character-based GPT language model as in the
[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=8)
video of the "Neural Networks: Zero to Hero" series taught by Andrej Karpathy.

## How to Train?

```
go run .
```

Add `--interactive` for interactive training so that you can change learning rate
and training epochs dynamically:

```
go run . --interactive
```

## Model Architecture

The Transformer model is built in a hierarchical manner.

### Masked Self-Attention

Take embedding dimension = 2 and key/value/query vector dimension = 2 as an example:
![Masked Self-Attention Illustration](/assets/masked-self-attention-illustration.png)

### Multi-Head Attention

Take embedding dimension = 32 and head number = 2 as an example:
![Multi-Head Attention Illustration](/assets/multi-head-attention-illustration.png)

### Attention Block

Take embedding dimension = 96 and head number = 3 as an example:
![Attention Block Architecture](/assets/attention-block-architecture.png)

### Transformer

Take embedding dimension = 96 and attention block layers = 3 as an example:
![Transformer Architecture](/assets/transformer-architecture.png)

## Performance

The following shows the model performance of generating 200 characters out of nothing:

```
Wan hactr warst
fine,
Der mond mull:
Go ars frre gour pll keasesplan heclis, mar hare aither, thy mafar yolle walaves bute see!

Se
Leald thun moth acibly:
Wess gou brparestollenspraneen hitre iarmod
```

This model is trained upon the first 100K characters in the "shakespeare.txt" dataset
with following settings:

| Context length | Attention layers | Attention heads | Embedding dimension |
| -------------- | ---------------- | --------------- | ------------------- |
| 8 | 3 | 4 | 64 |
