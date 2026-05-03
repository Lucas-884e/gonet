# A Language Model with a WaveNet-like structure

This example tries to train the same WaveNet-like language model as in the
[Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
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

## Model structure

![WaveNet Language Model](/assets/wavenet-language-model.png)
