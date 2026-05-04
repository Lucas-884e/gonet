package arrimpl

import (
	"fmt"
	"math"

	"github.com/LucasInOz/gonet/util"
)

type MLP struct {
	layers           []*Layer
	loss             LossFunction
	defaultActivator Activator
	desiredOutputs   []float64
}

// NewMLP returns a new Multi-Layer Perceptron with its input dimension equal to `inputSize`.
func NewMLP(inputSize int, loss LossFunction, activator Activator) *MLP {
	if activator == nil {
		activator = LinearActivator()
	}

	net := &MLP{
		loss:             loss,
		defaultActivator: activator,
	}
	// Add input layer.
	net.AddLayerWithActivator(inputSize, LinearActivator())
	return net
}

// AddLayer adds a new layer with the default activation function and its derivative.
func (net *MLP) AddLayer(size int) {
	net.AddLayerWithActivator(size, nil)
}

// AddLayerWithActivator adds a new layer with the given size (number of neurons)
// to the MLP. This new layer will be considered the output layer until there
// is another new layer being added to the network.
func (net *MLP) AddLayerWithActivator(size int, activator Activator) {
	if size == 0 {
		return
	}

	switch {
	case activator != nil:
	case net.defaultActivator != nil:
		activator = net.defaultActivator
	default:
		activator = LinearActivator()
	}

	layer := &Layer{
		size: size,
		// Place a fake neuron as bias node (which has index 0 and fixed output 1)
		// in advance.
		neurons:   []*Neuron{{n: 0, output: 1}},
		activator: activator,
	}
	for n := 1; n <= size; n++ {
		neuron := &Neuron{n: n}
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
func (net *MLP) RandomizeInitialWeights() {
	for i, layer := range net.layers {
		if i == 0 {
			continue
		}
		weights := GenerateRandomLayerWeights(layer.size, net.layers[i-1].size+1)
		layer.loadWeights(weights)
	}
}

func (net *MLP) ZeroGrads() {
	for i, l := range net.layers {
		if i == 0 {
			continue
		}
		l.zeroWeightGrads()
	}
}

// feedInputSample feeds a training sample to the network input and output.
func (net *MLP) feedInputSample(xs, ys []float64) {
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
	copy(net.desiredOutputs, ys)
}

// forwardPropagate does the forward pass computation:
//
//	y_j = φ (Σ_k [W_{j,k} * y_k(previous_layer)])
//
// where `Σ_k` is the sum over neuron index `k` (starting from 0 which corresponds
// to the bias term) in previous layer and `φ (v)` is the activation function.
func (net *MLP) forwardPropagate() {
	for l := 1; l < len(net.layers); l++ {
		prev, curr := net.layers[l-1], net.layers[l]
		// prod is the matrix product of (curr.neurons.weights · prev.neurons.output)
		prod := make([]float64, curr.size)
		for i := 1; i <= curr.size; i++ {
			for _, w := range curr.neurons[i].weights {
				prod[i-1] += w.v * prev.neurons[w.p].output
			}
		}
		ys := curr.activator.A(prod)
		for i := 1; i <= curr.size; i++ {
			curr.neurons[i].output = ys[i-1]
		}
	}
}

func (net *MLP) lossGrads() []float64 {
	outputLayer := net.layers[len(net.layers)-1]
	return net.loss.Grads(outputLayer.output(), net.desiredOutputs)
}

func (net *MLP) backwardPropagate() {
	l := len(net.layers) - 1
	// Output layer back propagation.
	net.layers[l].backward(net.layers[l-1], nil, net.lossGrads())
	// Hidden layer back propagation.
	for l--; l >= 1; l-- {
		net.layers[l].backward(net.layers[l-1], net.layers[l+1], nil)
	}
}

// PropagateSamples propagates with one training sample to tune the weights.
func (net *MLP) PropagateSamples(samples []util.Sample) {
	net.ZeroGrads()
	for _, sample := range samples {
		net.feedInputSample(sample.X, sample.Y)
		net.forwardPropagate()
		net.backwardPropagate()
	}

	count := len(samples)
	for i, l := range net.layers {
		if i > 0 {
			l.normalizeGrads(count)
		}
	}
}

// Predict predicts the result with the trained model given the input `xs`.
func (net *MLP) Predict(xs []float64) (prediction []float64) {
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
// norm of all weights, delta) in this update.
func (net *MLP) UpdateWeights(eta float64) float64 {
	// delta is the change in the norm of weights
	var delta float64
	for i, l := range net.layers {
		if i == 0 {
			continue
		}
		delta += l.updateWeights(eta)
	}
	return eta * math.Sqrt(delta)
}

// Print prints the network to the console.
func (net *MLP) Print() {
	fmt.Println("\n## Neural network")
	for i, l := range net.layers {
		fmt.Printf("--------------- Layer %d: %d neurons (activator: %s) ---------------\n", i, l.size, l.activator)
		if i == 0 {
			fmt.Println("Input layer neurons not printed.")
			continue
		}
		for j, n := range l.neurons {
			if j > 0 {
				fmt.Printf("Neuron %s\n", n.String())
			}
		}
	}
	fmt.Println()
}
