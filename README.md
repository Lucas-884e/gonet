# gonet

## Motivation

`gonet` is a neural network library written in Go as a project for learning,
experimentation, and fun. It is not intended for production use. The goal is to
make neural-network internals easier to inspect by implementing the core pieces
directly: **fully connected layers**, **embeddings**, **attention**, **normalization**,
**losses**, **training loops**, and **automatic differentiation**.

The repository contains two independent implementations:

- The root `gonet` package builds dynamic computation graphs and performs
  reverse-mode automatic differentiation.
- The `arrimpl` package contains an array-based fully connected network with
  explicit forward and backward propagation (which is meant for double check).

## Examples

The `examples/` directory contains small demonstrational mini-projects that
exercise different parts of the framework:

- [Binary classifier](examples/binary_classifier/) - trains a small neural-net
  binary classifier inspired by Karpathy's micrograd introduction. It can run
  with either the computation-graph implementation or the array-based MLP.
- [Digit OCR](examples/digit_ocr/) - trains a digit classifier on sklearn digits
  or MNIST-style data, again with both graph and array-based training modes.
- [Word embedding](examples/word_embedding/) - trains a tiny word embedding model
  and shows how similar words can converge toward similar learned vectors.
- [Makemore neural bigram](examples/makemore/nn_bigram/) - implements a
  character-level neural bigram language model, with both one-hot linear and
  embedding-based variants.
- [Makemore neural quadgram](examples/makemore/nn_quadgram/) - implements an MLP
  character language model in the style of Bengio et al. 2003, using multiple
  previous characters as context.
- [Makemore WaveNet](examples/makemore/wavenet/) - builds a character language
  model with a WaveNet-like hierarchical structure.
- [Makemore decoder-only transformer](examples/makemore/transformer/) - trains a
  small character-level GPT-style model with masked self-attention, multi-head
  attention, attention blocks, and token generation.

## Decoder-Only Transformer Highlight

The [decoder-only transformer example](examples/makemore/transformer/) is the
most sophisticated example in this repository for now. It demonstrates that this
small Go deep-learning framework can express and train a relatively complex
model: token embeddings, positional/context handling, masked self-attention,
multi-head attention, stacked transformer blocks, and autoregressive character
generation.

The example is still primarily educational. Performance is not competitive with
production deep-learning frameworks (eg, PyTorch), but that is also the point:
the model is implemented in a way that keeps the mechanics visible and hackable
instead of hiding them behind highly optimized kernels (tensor operations).
