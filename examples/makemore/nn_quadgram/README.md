# Neural Net Language Model from Bengio et al. 2003 Paper

This example tries to train the same neural-net based language model as in the
[Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
video of the "Neural Networks: Zero to Hero" series taught by Andrej Karpathy.

## Model structure

![Neural Net Language Model](/assets/bengio-nnlm.png)

## Character embedding (2D) distribution

![Character Embedding Distribution](/assets/character-embedding-distribution.png)

It is quite obvious that vocabulary (the character set) can be divided into 5 groups:

- ".": the SoS/EoS symbol
- a/e/i/o/u: mainly generates vowel sounds
- y: can generate both vowel and consonant sounds
- q: a special consonant appearing rarely (only 272 times in the dataset, significantly
  less than other consonants) and often followed by the letter "u"
- b/c/d/f/g/...: other common consonants
