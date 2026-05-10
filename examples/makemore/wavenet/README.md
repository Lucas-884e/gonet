# A Language Model with a WaveNet-like structure

This example tries to train the same WaveNet-like language model as in the
[Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
video of the "Neural Networks: Zero to Hero" series taught by Andrej Karpathy.
But the difference is that we did not include the normalization layer.

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

## Name Generation

After training, we reached a Cross Entropy Loss around 2.125, which is slightly
lower than the [nn_quadgram example](/examples/makemore/nn_quadgram/). And this
model can achieve (maybe?) a slightly better performance of English name generation.
Take a few for example:

```plain/text
agen
ireiel
jiemarean
toel
siea
praadiya
lahce
meden
neya
talloby
afraylah
elioron
azdienesi
viku
carlei
takdiontera
debataleanni
shankila
pitre
lynnen
```
