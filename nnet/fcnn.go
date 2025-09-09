package nnet

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/Lucas-884e/gonet/training"
)

// ActivationFunc defines the functional signature of activation functions and
// their derivative and local gradient functions.
type ActivationFunc func(float64) float64

// IDActivator defines the identity activation function.
func IDActivator(v float64) float64 { return v }

// DIDActivator defines the derivative of identity activation function.
func DIDActivator(float64) float64 { return 1 }

// LogisticActivator returns the Logistic activation function:
//
//	                     1
//	y = φ (v) = --------------------
//	              1 + exp(- a * v)
//
// with the given parameter `a`.
func LogisticActivator(a float64) ActivationFunc {
	return func(v float64) float64 {
		return 1 / (1 + math.Exp(-a*v))
	}
}

// DLogisticActivator returns the derivative function of logistic activation
// function, but in terms of `y = φ (v)` instead of `v`:
//
//	φ '(v) = a * φ (v) [1 - φ (v)] = a * y * (1 - y)
func DLogisticActivator(a float64) ActivationFunc {
	return func(y float64) float64 {
		return a * y * (1 - y)
	}
}

// TanhActivator returns a tanh activation function:
//
//	y = φ (v) = a * tanh(b * v)
//
// with the given parameter `a` and `b`.
func TanhActivator(a, b float64) ActivationFunc {
	return func(v float64) float64 {
		return a * math.Tanh(b*v)
	}
}

// DTanhActivator returns the derivative function of tanh activation function,
// but in terms of `y = φ (v)` instead of `v`:
//
//	φ '(v) = (b / a) * [a - φ (v)] * [a + φ (v)] = (b / a) * (a - y) * (a + y)
func DTanhActivator(a, b float64) ActivationFunc {
	return func(y float64) float64 {
		return (b / a) * (a - y) * (a + y)
	}
}

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
	// Activation function
	activator ActivationFunc
	// Derivative of activation function
	dActivator ActivationFunc
}

// Layer implements the layer structure in a neural network.
type Layer struct {
	// Number of neurons in this layer.
	size    int
	neurons []*Neuron
}

// FCNNet Implements the Fully-Connect Neural Network.
type FCNNet struct {
	layers            []*Layer
	desiredOutputs    []float64
	defaultActivator  ActivationFunc
	defaultDActivator ActivationFunc
}

// NewFCNNet returns a new FCNNet with its input dimension equal to `inputSize`.
func NewFCNNet(inputSize int, activator, dActivator ActivationFunc) *FCNNet {
	if activator == nil {
		activator = IDActivator
		dActivator = DIDActivator
	} else if dActivator == nil {
		panic("Must provide a derivative function for non-nil activation function")
	}

	net := &FCNNet{
		defaultActivator:  activator,
		defaultDActivator: dActivator,
	}
	// Add input layer.
	net.AddLayer(inputSize)
	return net
}

// AddLayer adds a new layer with the default activation function and its derivative.
func (net *FCNNet) AddLayer(size int) {
	net.AddLayerWithActivator(size, nil, nil)
}

// AddLayerWithActivator adds a new layer with the given size (number of neurons)
// to the FCNNet. This new layer will be considered the output layer until there
// is another new layer being added to the network.
func (net *FCNNet) AddLayerWithActivator(size int, activator, dActivator ActivationFunc) {
	switch {
	case activator != nil:
		if dActivator == nil {
			panic("Must provide a derivative function for non-nil activation function")
		}
	case net.defaultActivator != nil:
		activator = net.defaultActivator
		dActivator = net.defaultDActivator
	default:
		activator = IDActivator
		dActivator = DIDActivator
	}

	layer := &Layer{
		size: size,
		// Place a fake neuron as bias node (which has index 0 and fixed output 1)
		// in advance.
		neurons: []*Neuron{{n: 0, output: 1}},
	}
	for n := 1; n <= size; n++ {
		neuron := &Neuron{
			n:          n,
			activator:  activator,
			dActivator: dActivator,
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
			n.output = n.activator(v)
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
		n.localGrad = n.dActivator(n.output) * (d - n.output)
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
			n.localGrad = n.dActivator(n.output) * v
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
// propagation and returns whether the learning process should stop according to
// some stop criterion.
func (net *FCNNet) UpdateWeights(eta float64) bool {
	var eps float64 // change in the norm of weights
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
	return false
}

// Print prints the network to the console.
func (net *FCNNet) Print() {
	fmt.Println("\n## Neural network")
	for i, l := range net.layers {
		fmt.Printf("--------------- Layer %d: %d neurons ---------------\n", i, l.size)
		for _, n := range l.neurons {
			fmt.Printf("Neuron%+v\n", *n)
		}
	}
	fmt.Println("Desired outputs:", net.desiredOutputs)
}
