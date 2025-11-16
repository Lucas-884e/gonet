package nnet

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/Lucas-884e/gonet/training"
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
	// Value of local gradient:
	//          / φ '(v) * (d_j - y_j) , for output layer
	//   δ_j = {
	//          \ φ '(v) * Σ_k [δ_k(next_layer) * W_{k,j}] , for hidden layer
	// where `d_j` is the desired response whose estimate is `y_j`, `Σ_k` means
	// summation over index `k` and `δ_k(next_layer)` is the local gradient for
	// k-th neuron in next layer.
	localGrad float64
	// Activation function and its derivative.
	activator Activator
}

func (n *Neuron) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("#%d: ", n.n))
	for _, w := range n.weights {
		sb.WriteString(fmt.Sprintf(" W(%d,%d)=%g", w.n, w.p, w.v))
	}
	sb.WriteString(fmt.Sprintf("  Activation(%s)=%g  Gradient=%g", n.activator, n.output, n.localGrad))
	return sb.String()
}

// Layer implements the layer structure in a neural network.
type Layer struct {
	// Number of neurons in this layer.
	size    int
	neurons []*Neuron
}

// FCNNet Implements the Fully-Connect Neural Network.
type FCNNet struct {
	layers           []*Layer
	desiredOutputs   []float64
	defaultActivator Activator
}

// NewFCNNet returns a new FCNNet with its input dimension equal to `inputSize`.
func NewFCNNet(inputSize int, activator Activator) *FCNNet {
	if activator == nil {
		activator = IDActivator()
	}

	net := &FCNNet{defaultActivator: activator}
	// Add input layer.
	net.AddLayer(inputSize)
	return net
}

// AddLayer adds a new layer with the default activation function and its derivative.
func (net *FCNNet) AddLayer(size int) {
	net.AddLayerWithActivator(size, nil)
}

// AddLayerWithActivator adds a new layer with the given size (number of neurons)
// to the FCNNet. This new layer will be considered the output layer until there
// is another new layer being added to the network.
func (net *FCNNet) AddLayerWithActivator(size int, activator Activator) {
	if size == 0 {
		return
	}

	switch {
	case activator != nil:
	case net.defaultActivator != nil:
		activator = net.defaultActivator
	default:
		activator = IDActivator()
	}

	layer := &Layer{
		size: size,
		// Place a fake neuron as bias node (which has index 0 and fixed output 1)
		// in advance.
		neurons: []*Neuron{{n: 0, output: 1}},
	}
	for n := 1; n <= size; n++ {
		neuron := &Neuron{
			n:         n,
			activator: activator,
		}
		if depth := len(net.layers); depth > 0 {
			for p := 0; p <= net.layers[depth-1].size; p++ {
				neuron.weights = append(neuron.weights, Weight{
					n: n, // Range: [1, current_layer_size]
					p: p, // Range: [0, previous_layer_size]
				})
			}
		}
		layer.neurons = append(layer.neurons, neuron)
	}
	net.layers = append(net.layers, layer)
}

// RandomizeInitialWeights initialize the network weights to random values.
func (net *FCNNet) RandomizeInitialWeights() {
	rand.Seed(time.Now().Unix())
	for i, layer := range net.layers {
		if i == 0 {
			continue
		}
		for j, n := range layer.neurons {
			if j == 0 {
				continue
			}
			max := math.Sqrt(3 / float64(len(n.weights)))
			for k := range n.weights {
				// Uniform random distribution in the range: [-sqrt(3/m), sqrt(3/m)],
				// where `m` is the number of synaptic connections of neuron `n`.
				n.weights[k].v = training.RandomUniformSample(-max, max)
			}
		}
	}
}

// feedInputSample feeds a training sample to the network input and output.
func (net *FCNNet) feedInputSample(xs, ys []float64) {
	if len(xs) != net.layers[0].size {
		panic("Input sample size must match network input dimension")
	}
	if len(ys) != 0 && len(ys) != net.layers[len(net.layers)-1].size {
		panic("Output sample size must either be 0 or match network output dimension")
	}
	for i, x := range xs {
		net.layers[0].neurons[i+1].output = x
	}
	if len(net.desiredOutputs) != len(ys) {
		net.desiredOutputs = make([]float64, len(ys))
	}
	for i, y := range ys {
		net.desiredOutputs[i] = y
	}
}

// forwardPropagate does the forward pass computation:
//
//	y_j = φ (Σ_k [W_{j,k} * y_k(previous_layer)])
//
// where `Σ_k` is the sum over neuron index `k` (starting from 0 which corresponds
// to the bias term) in previous layer and `φ (v)` is the activation function.
func (net *FCNNet) forwardPropagate() {
	for l := 1; l < len(net.layers); l++ {
		prev, curr := net.layers[l-1], net.layers[l]
		for i := 1; i <= curr.size; i++ {
			var v float64
			n := curr.neurons[i]
			for _, w := range n.weights {
				v += w.v * prev.neurons[w.p].output
			}
			n.output = n.activator.A(v)
		}
	}
}

// backwardPropagate does the backward propagation computation:
//
//	       / φ '(v) * (d_j - y_j) , for output layer
//	δ_j = {
//	       \ φ '(v) * Σ_k [δ_k(next_layer) * W_{k,j}] , for hidden layer
//
// where `d_j` is the desired response whose estimate is `y_j`, `Σ_k` means
// summation over index `k` and `δ_k(next_layer)` is the local gradient for
// k-th neuron in next layer.
func (net *FCNNet) backwardPropagate() {
	lc := len(net.layers)
	outputLayer := net.layers[lc-1]
	for i, d := range net.desiredOutputs {
		n := outputLayer.neurons[i+1]
		n.localGrad = n.activator.D(n.output) * (d - n.output)
	}
	for l := lc - 2; l >= 1; l-- {
		curr, next := net.layers[l], net.layers[l+1]
		tmp := make(map[int]float64, curr.size)
		for k := 1; k <= next.size; k++ {
			n := next.neurons[k]
			for _, w := range n.weights {
				if w.p > 0 {
					tmp[w.p] += n.localGrad * w.v
				}
			}
		}
		for j, v := range tmp {
			n := curr.neurons[j]
			n.localGrad = n.activator.D(n.output) * v
		}
	}
}

// PropagateSample propagates with one training sample to tune the weights.
func (net *FCNNet) PropagateSample(xs, ys []float64) {
	net.feedInputSample(xs, ys)
	net.forwardPropagate()
	net.backwardPropagate()
}

// Predict predicts the result with the trained model given the input `xs`.
func (net *FCNNet) Predict(xs []float64) (prediction []float64) {
	net.feedInputSample(xs, nil)
	net.forwardPropagate()
	for j, n := range net.layers[len(net.layers)-1].neurons {
		if j == 0 {
			continue
		}
		prediction = append(prediction, n.output)
	}
	return prediction
}

// UpdateWeights update weights with the given learning rate `η` for one-round
// propagation and returns how much the weights has changed (in terms of the
// norm of all weights, eps) in this update.
func (net *FCNNet) UpdateWeights(eta float64) (eps float64) {
	//var eps float64 // change in the norm of weights
	for l, layer := range net.layers {
		if l == 0 {
			continue
		}
		prev := net.layers[l-1]
		for j, n := range layer.neurons {
			if j == 0 {
				continue
			}
			for i := range n.weights {
				dw := n.localGrad * prev.neurons[i].output
				eps += dw * dw
				n.weights[i].v += eta * dw
			}
		}
	}
	return math.Sqrt(eps)
}

// Print prints the network to the console.
func (net *FCNNet) Print() {
	fmt.Println("\n## Neural network")
	for i, l := range net.layers {
		fmt.Printf("--------------- Layer %d: %d neurons ---------------\n", i, l.size)
		for j, n := range l.neurons {
			if j > 0 {
				fmt.Printf("Neuron %s\n", n.String())
			}
		}
	}
	fmt.Println("Desired outputs:", net.desiredOutputs)
}
