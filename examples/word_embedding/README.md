# A Simple Word Embedding Model

This example tries to train the same neural-net based word embedding as in the
[Word Embedding in PyTorch + Lightning](https://www.youtube.com/watch?v=Qf06XDYXCXI)
video in Josh Starmer's StatQuest series.

## How to train?

```
go run .
```

## Model structure

![Word Embedding Model](/assets/word-embedding-model.png)

## Training result

Training output:

```
Vocabulary: map[<EOS>:0 Godzilla:1 Ironman:2 great:3 is:4]
Inputs: [[0] [1] [4] [3] [0] [2] [4] [3]]
Labels: [1 4 3 0 2 4 3 0]
Sample: {X:[4] Y:[3]}
Sample: {X:[1] Y:[4]}
Sample: {X:[3] Y:[0]}
Sample: {X:[4] Y:[3]}
Sample: {X:[0] Y:[2]}
Sample: {X:[0] Y:[1]}
Sample: {X:[3] Y:[0]}
Sample: {X:[2] Y:[4]}
Embeddings: -------------------
     <EOS> | [0.0491798, -2.07063e-06] | [-543.288, 1.49801]
  Godzilla | [-0.0217535, -9.23983] | [1.02529, -1.06333]
   Ironman | [-0.0217615, -9.23983] | [1.0302, -1.06333]
     great | [-0.00083626, 12.2884] | [-545.064, -0.580115]
        is | [-16.2202, 1.17246] | [-543.69, -3.32659]
(Godzilla - Ironman) = [0.0000 0.0000] | [-0.0049 -0.0000]
```

One can see that after training, word "Godzilla" and word "Ironman" reach almost
the same embedding vector (with their difference approaching zero).
