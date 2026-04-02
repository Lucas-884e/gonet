package gonet

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
	hasBias bool
}

func NewNeuron(inputSize, lidx, nidx int, withBias bool) *Neuron {
	var (
		max = math.Sqrt(3 / float64(inputSize))
		n   = &Neuron{
			lidx:    lidx,
			nidx:    nidx,
			hasBias: withBias,
		}
	)

	for widx := 1; widx <= inputSize; widx++ {
		w := util.RandomUniformSample(-max, max)
		wn := NewNode(w, fmt.Sprintf("W_%d_%d_%d", lidx, nidx, widx))
		n.weights = append(n.weights, wn)
	}

	if withBias {
		b := util.RandomUniformSample(-max, max) // bias
		bn := NewNode(b, fmt.Sprintf("B_%d_%d", lidx, nidx))
		n.weights = append(n.weights, bn)
	}

	return n
}

func (n *Neuron) Feed(input []*Node, activator Operator) *Node {
	var wx []*Node
	for i, xn := range input {
		wx = append(wx, Multiply(n.weights[i], xn))
	}
	if n.hasBias {
		wx = append(wx, n.weights[len(input)])
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

func (n *Neuron) Parameters() []util.Parameter {
	return util.ListConvert[*Node, util.Parameter](n.weights)
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
	const maxPrint = 10 // Print at most 10 weights
	ws := make([]string, min(len(n.weights), maxPrint+1))
	for i, w := range n.weights {
		if i == maxPrint {
			ws[i] = "..."
			break
		}

		if i == len(n.weights)-1 && n.hasBias {
			ws[i] = fmt.Sprintf("Bias=%.6g", w.V())
		} else {
			ws[i] = fmt.Sprintf("W[%d]=%.6g", i, w.V())
		}
	}
	return strings.Join(ws, " | ")
}

type Layer struct {
	lidx      int // layer index
	neurons   []*Neuron
	activator Operator
}

func NewLayer(inputSize, outputSize, lidx int, activator Operator, withBias bool) *Layer {
	l := &Layer{
		lidx:      lidx,
		activator: activator,
	}
	for nidx := 1; nidx <= outputSize; nidx++ {
		n := NewNeuron(inputSize, lidx, nidx, withBias)
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

func (l *Layer) Parameters() (p []util.Parameter) {
	for _, n := range l.neurons {
		p = append(p, n.Parameters()...)
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

func (mlp *MLP) AddLayer(outputSize int, activator Operator, withBias bool) {
	lidx := len(mlp.layers) + 1
	// Use the outputSize of the old output layer as the inputSize of the new output layer.
	layer := NewLayer(mlp.outputSize, outputSize, lidx, activator, withBias)
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

func (mlp *MLP) Parameters() (p []util.Parameter) {
	for _, l := range mlp.layers {
		p = append(p, l.Parameters()...)
	}
	return
}

func (mlp *MLP) Output(input []float64) []float64 {
	xs := NewInputNodeBatch(len(input), "X_%d")
	for i, n := range xs {
		n.SetV(input[i])
	}
	ys := mlp.Feed(xs)
	for _, y := range ys {
		y.Forward()
	}
	return NodeValues(ys)
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

// String returns the string representation of the MLP.
func (mlp *MLP) String() string {
	w := new(strings.Builder)
	fmt.Fprintf(w, "\n## Neural network (#input=%d | #output=%d | #layers=%d)\n", mlp.inputSize, mlp.outputSize, len(mlp.layers))
	for _, l := range mlp.layers {
		fmt.Fprintf(w, "--------------- Layer %d: %d neurons (activator: %s) ---------------\n", l.lidx, len(l.neurons), l.activator)
		for i, n := range l.neurons {
			// Print at most 21 lines of neurons (or 20 lines of neurons plus one line of ellipsis).
			if len(l.neurons) <= 21 || i < 20 {
				fmt.Fprintf(w, "Neuron #%d:  %s\n", n.nidx, n.String())
			} else {
				fmt.Fprintf(w, "Neuron #%d→%d: ...\n", i, len(l.neurons)-1)
				break
			}
		}
	}
	return w.String()
}
