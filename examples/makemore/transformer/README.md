# A Simple GPT (Decoder-Only Transformer)

This example tries to train the same character-based GPT language model as in the
[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=8)
video of the "Neural Networks: Zero to Hero" series taught by Andrej Karpathy.

## Model structure

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
| 8 | 3 | 3 | 64 |
