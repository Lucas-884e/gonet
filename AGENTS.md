# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

gonet is a neural network library written in Go. It provides two independent implementations of fully-connected neural networks (MLPs): an array-based approach and a computational-graph-based approach with automatic differentiation.

## Build and Test Commands

Run all tests:

```
go test ./...
```

Run a single test:

```
go test -run TestMLP ./
go test -run TestFCNNet ./arrimpl/
```

Format code:

```
gofmt -w .
```

Run examples (from repo root):

```
go run ./examples/binary_classifier -i data/data.csv
go run ./examples/binary_classifier -i data/data.csv -g   # computational graph mode
go run ./examples/digit_ocr -ds sklearn
go run ./examples/digit_ocr -ds sklearn -g                 # computational graph mode
```

## Architecture

### Two Parallel NN Implementations

The library has two completely separate neural network implementations that share only the `util` package:

1. **Root package (`gonet`)** — Computational graph with autograd. Each `Node` carries closure-based `forward` and `backward` functions. Calling `node.Backward()` performs topological sort then reverse-order gradient propagation through the graph. The `MLP` type builds a graph of nodes when `Feed()` is called, and `Learn()` updates all weight nodes.

2. **`arrimpl` package** — Array-based FCNNet. Neurons store weights as `[]Weight` structs with explicit indices. Forward/backward propagation is done via manual matrix-style loops across layers. The `Trainer` type handles the training loop with epoch/batch iteration.

### Key Design Details

- **Neuron indexing**: In the root package, index 0 in each layer is a bias node (fixed output=1). Actual neurons are indices 1..size. Weight `W_{n,p}` connects neuron `n` in current layer to neuron `p` in previous layer.

- **Activator interface**: Activation functions implement `A([]float64) []float64` and `D(ys []float64) func(row, column int) float64`. The derivative `D` returns a closure representing a matrix (for softmax, it's a full Jacobian; for element-wise activations, only diagonal entries are non-zero).

- **Loss functions**: In the root package, `LossFunction` is a string enum with a `Grads()` method. In `graph`, loss functions are `func(actual, predicted []*Node) *Node` that construct graph nodes with backward closures.

- **Graph node operators**: `Plus`, `Multiply`, `Relu`, `Sigmoid`, `Tanh`, `Softmax` are constructor functions that return new `Node`(s) with wired `forward`/`backward` closures referencing their input nodes.

- **Input nodes**: In the graph package, `isInput` nodes have their gradients skipped during backprop (they represent data, not learnable parameters).

- **Batch training in graph mode**: A `SampleBatch` is pre-allocated and reused across iterations. The loss graph is built once from the batch, then `loss.Backward()` is called after updating sample values in-place.

### Package Dependencies

```
examples/ → gonet (root), arrimpl, util
gonet (root) → util
arrimpl → util
util → (no internal deps)
```

### Testing

Tests use `github.com/stretchr/testify`. The test in `arrimpl/fcnn_test.go` manually computes expected forward/backward values with hand-set weights. Graph tests (`./*_test.go`) verify node values and gradients through the autograd system.
