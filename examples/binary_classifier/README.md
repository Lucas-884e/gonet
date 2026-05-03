# Binary classification example

This example tries to train the same neural-net based binary classifier as in the
[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
video of the "Neural Networks: Zero to Hero" series taught by Andrej Karpathy.

## How to train?

- Use the computational graph approach:

  ```
  go run .
  ```

- Use precomputed gradient formula (array-based MLP) approach:

  ```
  go run . --arr
  ```

## Network Illustration

![Binary Classification Model](/assets/binary-classification-model.png)

![Binary Classification Data Illustration](/assets/binary-classification-data-illustration.png)
