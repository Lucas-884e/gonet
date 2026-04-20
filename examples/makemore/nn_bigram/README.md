# A Character-level Neural Net Bigram Model

This example tries to train the same neural-net based bigram model as in the
[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
video of the "Neural Networks: Zero to Hero" series taught by Andrej Karpathy.

## Model structure

Two types of implementations:

- Approach I: a model of one linear layer with one-hot encoding as input
- Approach II: a model of one embedding layer with token index as input

### Approach I

![Linear Model](/assets/bigram-linear-model.png)

### Approach II

![Embedding Model](/assets/bigram-embedding-model.png)

## How to run

- Non-interactive mode

  ```
  go run .
  ```

- Interactive mode

  ```
  go run . --interactiv
  ```

- Use Linear layer (with one-hot encoding as input) instead of Embedding layer
  (with token index as input) for comparison.

  ```
  go run . --linear --interactiv
  ```

## Performance test

On my 2022 MacBook Air with M2 chip and 16GB memory, the approach using an embedding
layer and accepting token index as neural net input shows a significant performance
gain comparing with the approach using a linear layer and accepting one-hot encoding
as network input. The convergence speed of the embedding-layer approach in terms
of training epochs is also much faster.

| Approach | Dataset size for loss evaluation | Memory | Time cost per epoch | Mini-batch size |
| -------- | -------- | -------- | -------- | -------- |
| Embedding | 10000 | 200MB | 0.2 sec | 1000 |
| Embedding | 228146 (all samples) | 3.8GB | 0.2 sec | 1000 |
| Linear | 10000 | 5.4GB | 12s | 1000 |
| Linear | 228146 (all samples) | OOM | N/A | 1000 |
