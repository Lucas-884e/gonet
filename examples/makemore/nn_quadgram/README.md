# Neural Net Language Model from Bengio et al. 2003 Paper

This example tries to train the same neural-net based language model as in the
[Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
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

## Model Structure

![Neural Net Language Model](/assets/bengio-nnlm.png)

## Name Generation

After training on the [names.txt](../dataset/names.txt) dataset, we reached a
Cross Entropy Loss around 2.17 on the training set (10-dim embedding, 36-dim
hidden layer size, 2385 parameters), which is far better than the bigram model.
And the model can generate better-looking names:

```plain/text
toyt
tai
maka
amellaw
kamien
abel
amison
pelian
reig
kias
adee
neadmi
elm
dari
ailgion
quidalygandra
navianee
jelts
khani
diz
```

Still feeling missing some quintessence of the ingredients in English names?
Try the [WaveNet example](/examples/makemore/wavenet/).

## Character embedding (2D) distribution

By setting embedding dimension to 2, we can obtain the following 2-dimensional
character embedding distribution:
![Character Embedding Distribution](/assets/character-embedding-distribution.png)

It is quite obvious that the vocabulary (the set of all characters) can be divided
into 5 groups (clusters):

- `.`: the SoS/EoS symbol
- `a/e/i/o/u`: mainly generate vowel sounds
- `y`: can generate both vowel and consonant sounds
- `q`: a special consonant appearing rarely (only 272 times in the dataset, significantly
  less than other consonants) and often followed by the letter `u`
- `b/c/d/f/g/...`: other common consonants
