package graph

import (
	"fmt"
	"math"
	"strings"

	"github.com/Lucas-884e/gonet/util"
)

type Neuron struct {
	lidx    int     // layer index
	nidx    int     // neuron index in the layer
	weights []*Node // first node is bias
}

func NewNeuron(inputSize, lidx, nidx int) *Neuron {
	var (
		max = math.Sqrt(3 / float64(inputSize))
		b   = util.RandomUniformSample(-max, max) // bias
		bn  = NewNode(b, fmt.Sprintf("B_%d%d", lidx, nidx))
		n   = &Neuron{
			lidx:    lidx,
			nidx:    nidx,
			weights: []*Node{bn},
		}
	)

	for widx := 1; widx <= inputSize; widx++ {
		w := util.RandomUniformSample(-max, max)
		wn := NewNode(w, fmt.Sprintf("W_%d%d%d", lidx, nidx, widx))
		n.weights = append(n.weights, wn)
	}
	return n
}

func (n *Neuron) Feed(input []*Node, activator Operator) *Node {
	wx := []*Node{n.weights[0]}
	for i, xn := range input {
		wx = append(wx, Multiply(n.weights[i+1], xn))
	}
	sum := Plus(wx...)

	switch activator {
	case OpNone, OpSoftmax: // Linear activation (Softmax computation will be done in Layer.Feed)
		return sum
	case OpRelu:
		return Relu(sum)
	case OpSigmoid:
		return Sigmoid(sum)
	case OpTanh:
		return Tanh(sum)
	default:
		panic("Unexpected activator: " + activator.String())
	}
}

func (n *Neuron) Learn(rate float64) (delta float64) {
	for _, w := range n.weights {
		delta += w.Learn(rate)
	}
	return
}

func (n *Neuron) LoadWeights(ws []float64) {
	for i, w := range n.weights {
		w.v = ws[i]
	}
}

func (n *Neuron) W() []*Node {
	return n.weights
}

func (n *Neuron) String() string {
	ws := make([]string, len(n.weights))
	for i, w := range n.weights {
		if i == 0 {
			ws[0] = fmt.Sprintf("Bias=%g", w.V())
		} else {
			ws[i] = fmt.Sprintf("W[%d]=%g", i, w.V())
		}
	}
	return strings.Join(ws, " | ")
}

type Layer struct {
	lidx      int // layer index
	neurons   []*Neuron
	activator Operator
}

func NewLayer(inputSize, outputSize, lidx int, activator Operator) *Layer {
	l := &Layer{
		lidx:      lidx,
		activator: activator,
	}
	for nidx := 1; nidx <= outputSize; nidx++ {
		n := NewNeuron(inputSize, lidx, nidx)
		l.neurons = append(l.neurons, n)
	}
	return l
}

func (l *Layer) Feed(input []*Node) []*Node {
	var out []*Node
	for _, n := range l.neurons {
		out = append(out, n.Feed(input, l.activator))
	}
	if l.activator == OpSoftmax {
		return Softmax(1, out...)
	}
	return out
}

func (l *Layer) Learn(rate float64) (delta float64) {
	for _, n := range l.neurons {
		delta += n.Learn(rate)
	}
	return
}

func (l *Layer) LoadWeights(ws [][]float64) {
	for i, n := range l.neurons {
		n.LoadWeights(ws[i])
	}
}

func (l *Layer) N() []*Neuron {
	return l.neurons
}

// MLP defines a Multi-Layer Perceptron
type MLP struct {
	inputSize  int
	outputSize int
	layers     []*Layer
}

func NewMLP(inputSize int) *MLP {
	return &MLP{
		inputSize: inputSize,
		// Use input layer size as output layer size when there is no output layer
		outputSize: inputSize,
	}
}

func (mlp *MLP) AddLayer(outputSize int, activator Operator) {
	lidx := len(mlp.layers) + 1
	// Use the outputSize of the old output layer as the inputSize of the new output layer.
	layer := NewLayer(mlp.outputSize, outputSize, lidx, activator)
	mlp.layers = append(mlp.layers, layer)
	// Update new output layer size.
	mlp.outputSize = outputSize
}

func (mlp *MLP) Feed(input []*Node) []*Node {
	output := input
	for _, l := range mlp.layers {
		output = l.Feed(output)
	}
	return output
}

func (mlp *MLP) Learn(rate float64) (delta float64) {
	for _, l := range mlp.layers {
		delta += l.Learn(rate)
	}
	return
}

func (mlp *MLP) LoadWeights(ws [][][]float64) {
	for i, l := range mlp.layers {
		l.LoadWeights(ws[i])
	}
}

func (mlp *MLP) RandomizeInitialWeights() {
	for _, l := range mlp.layers {
		weights := util.GenerateRandomLayerWeights(len(l.neurons), len(l.neurons[0].weights))
		l.LoadWeights(weights)
	}
}

func (mlp *MLP) L() []*Layer {
	return mlp.layers
}

// Print prints the network to the console.
func (mlp *MLP) String() string {
	w := new(strings.Builder)
	fmt.Fprintf(w, "\n## Neural network (#input=%d | #output=%d | #layers=%d)\n", mlp.inputSize, mlp.outputSize, len(mlp.layers))
	for _, l := range mlp.layers {
		fmt.Fprintf(w, "--------------- Layer %d: %d neurons (activator: %s) ---------------\n", l.lidx, len(l.neurons), l.activator)
		for _, n := range l.neurons {
			fmt.Fprintf(w, "Neuron #%d:  %s\n", n.nidx, n.String())
		}
	}
	return w.String()
}
