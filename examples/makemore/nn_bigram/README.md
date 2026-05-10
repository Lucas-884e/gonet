# A Character-level Neural Net Bigram Model

This example tries to train the same neural-net based bigram model as in the
[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
video of the "Neural Networks: Zero to Hero" series taught by Andrej Karpathy.

## How to train?

- Non-interactive mode

  ```
  go run .
  ```

- Interactive mode (dynamic learning rate and training epochs setting)

  ```
  go run . --interactive
  ```

- Use Linear layer (with one-hot encoding as input) instead of Embedding layer
  (with token index as input) for comparison.

  ```
  go run . --linear --interactive
  ```

## Model structure

Two types of implementations:

- Approach I: a model of one linear layer with one-hot encoding as input
- Approach II: a model of one embedding layer with token index as input

### Approach I

![Linear Model](/assets/bigram-linear-model.png)

### Approach II

![Embedding Model](/assets/bigram-embedding-model.png)

## Name Generation

After training, we reached a Cross Entropy Loss around 2.454, which is basically
at the same level as word-counting based bigram model. And the model can generate
names like the following ones:

```plain/text
qanion
alyne
paxorinilele
lahliaddr
hynsliyn
ayahleniriceleshakem
cosuerooma
sinn
lesee
kheann
meederela
kafomah
b
ken
remadipristta
xiolis
cava
mesan
nnnnnimy
se
```

Not very good? Try the [nn_quadgram example](/examples/makemore/nn_quadgram/).

## Performance test

On my 2022 MacBook Air with M2 chip and 16GB memory (GPU is not used), the approach
using an embedding layer and accepting token index as neural net input shows a significant
performance gain (memory consumption, convergence speed in terms of training epochs,
etc.) comparing with the approach using a linear layer and accepting one-hot encoding
as network input.

| Approach | Memory | Time cost per epoch | Mini-batch size |
| -------- | -------- | -------- | -------- |
| Embedding | 50MB | 0.2 sec | 1000 |
| Linear | 110MB | 0.8 sec | 1000 |
