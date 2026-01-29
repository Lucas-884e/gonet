package gonet

import (
	"fmt"
	"strings"
)

// Weight implements the weight element `W_{n,p}` in certain layer of a neural
// network. `W_{n,p}` connects two neurons located at two successive layers,
// where `n` is the index the neuron at next layer and `p` is the index the
// neuron at previous layer.
type Weight struct {
	// The index of the neuron at next layer that this weight connects.
	n int
	// The index of the neuron at previous layer that this weight connects.
	p int
	// Value of the weight.
	v float64
	// Value of the weight gradient
	grad float64
}

// Neuron defines a neuron at certain layer of a neural network.
type Neuron struct {
	// Index of the neuron within the layer.
	n int
	// Weights that connects to this neuron from neurons (including bias) in
	// previous layer, `W_{n,p}` where p = 0, 1, 2, ...
	weights []Weight
	// Value of the neuron output:
	//   y_j = φ (Σ_k [W_{j,k} * y_k(previous_layer)])
	output float64
	// Value of local gradient of the neuron output:
	//          / φ '(v) * δ_j(loss) , for output layer
	//   δ_j = {
	//          \ φ '(v) * Σ_k [δ_k(next_layer) * W_{k,j}] , for hidden layer
	// where `d_j` is the desired response whose estimate is `y_j`, `Σ_k` means
	// summation over index `k` and `δ_k(next_layer)` is the local gradient for
	// k-th neuron in next layer.
	grad float64
}

func (n *Neuron) zeroWeightGrads() {
	for i := range n.weights {
		n.weights[i].grad = 0
	}
}

func (n *Neuron) loadWeights(weights []float64) {
	for i := range n.weights {
		n.weights[i].v = weights[i]
	}
}

func (n *Neuron) updateWeights(eta float64) float64 {
	var delta float64
	for k, w := range n.weights {
		n.weights[k].v -= eta * w.grad
		delta += w.grad * w.grad
	}
	return delta
}

func (n *Neuron) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("#%d: ", n.n))
	for _, w := range n.weights {
		sb.WriteString(fmt.Sprintf(" W(%d,%d)=%g", w.n, w.p, w.v))
	}
	sb.WriteString(fmt.Sprintf("  Activation=%g  Gradient=%g", n.output, n.grad))
	return sb.String()
}
