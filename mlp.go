package gonet

import (
	"fmt"
	"strings"

	"github.com/Lucas-884e/gonet/util"
)

// MLP defines a Multi-Layer Perceptron
type MLP struct {
	inputSize  int
	outputSize int
	layers     []Layer

	inputBatch  [][]*Node
	outputBatch [][]*Node
}

func NewMLP(inputSize int) *MLP {
	return &MLP{
		inputSize: inputSize,
		// Use input layer size as output layer size when there is no output layer
		outputSize: inputSize,
	}
}

func (mlp *MLP) AddLayer(outputSize int, activator Operator, withBias bool) {
	// Clear input and output batch in case Feed is called before AddLayer.
	mlp.inputBatch = nil
	mlp.outputBatch = nil

	// Use the outputSize of the old output layer as the inputSize of the new output layer.
	mlp.layers = append(mlp.layers, LinearLayer(mlp.outputSize, outputSize, withBias))
	// Update new output layer size.
	mlp.outputSize = outputSize

	switch activator {
	case OpNone:
	case OpRelu:
		mlp.layers = append(mlp.layers, ReluLayer())
	case OpSigmoid:
		mlp.layers = append(mlp.layers, SigmoidLayer())
	case OpTanh:
		mlp.layers = append(mlp.layers, TanhLayer())
	case OpSoftmax: // Linear activation (Softmax computation will be done in Layer.Feed)
		mlp.layers = append(mlp.layers, SoftmaxLayer(1))
	default:
		panic("Unexpected activator: " + activator.String())
	}
}

func (mlp *MLP) FeedOne(xs []float64) (out []*Node) {
	return mlp.BatchFeed(xs)[0]
}

func (mlp *MLP) BatchFeed(inBatch ...[]float64) (outBatch [][]*Node) {
	l := len(mlp.inputBatch)
	for i, xs := range inBatch {
		if i < l {
			ins := mlp.inputBatch[i]
			for j, in := range ins {
				in.SetV(xs[j])
			}
			outBatch = append(outBatch, mlp.outputBatch[i])
		} else {
			ins := make([]*Node, len(xs))
			for j, x := range xs {
				ins[j] = NewInputNode(x, fmt.Sprintf("X%d_%d", i, j))
			}
			mlp.inputBatch = append(mlp.inputBatch, ins)

			outs := mlp.Feed(ins)
			outBatch = append(outBatch, outs)
			mlp.outputBatch = append(mlp.outputBatch, outs)
		}
	}
	return outBatch
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

func (mlp *MLP) Predict(xs []float64) []float64 {
	out := mlp.FeedOne(xs)
	for _, o := range out {
		o.Forward()
	}
	return NodeValues(out)
}

func (mlp *MLP) LazyPredict(xs []*Node) func() []float64 {
	ys := mlp.Feed(xs)
	return func() []float64 {
		for _, y := range ys {
			y.Forward()
		}
		return NodeValues(ys)
	}
}

func (mlp *MLP) LoadWeights(ws [][][]float64) {
	var i int
	for _, l := range mlp.layers {
		if ll, ok := l.(*linearLayer); ok {
			ll.loadWeights(ws[i])
			i++
		}
	}
}

func (mlp *MLP) L() []Layer {
	return mlp.layers
}

// String returns the string representation of the MLP.
func (mlp *MLP) String() string {
	w := new(strings.Builder)
	fmt.Fprintf(w, "\n## Neural network (#input=%d | #output=%d | #layers=%d)\n", mlp.inputSize, mlp.outputSize, len(mlp.layers))
	for i, l := range mlp.layers {
		fmt.Fprintf(w, "--------------- Layer %d: %s ---------------\n", i, l.Name())
		if ll, ok := l.(*linearLayer); ok {
			fmt.Fprintf(w, "--------------- %d neurons ---------------\n", len(ll.neurons))
			for j, n := range ll.neurons {
				// Print at most 21 lines of neurons (or 20 lines of neurons plus one line of ellipsis).
				if len(ll.neurons) <= 21 || j < 20 {
					fmt.Fprintf(w, "Neuron #%d:  %s\n", j, n.String())
				} else {
					fmt.Fprintf(w, "Neuron #%d→%d: ...\n", j, len(ll.neurons)-1)
					break
				}
			}
		}
	}
	return w.String()
}
