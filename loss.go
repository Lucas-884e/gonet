package gonet

import (
	"fmt"
	"math"

	"github.com/Lucas-884e/gonet/util"
)

type LossFunction func(actual, predicted []*Node) *Node

type FeedForwarder interface {
	Feed([]*Node) []*Node
}

type Sample struct {
	X []*Node
	Y []*Node
}

func NewSample(inputSize, outputSize int) *Sample {
	return &Sample{
		X: NewInputNodeBatch(inputSize, "X_%d"),
		Y: NewInputNodeBatch(outputSize, "Y_%d"),
	}
}

func FromSample(us util.Sample) *Sample {
	s := NewSample(len(us.X), len(us.Y))
	s.Update(us)
	return s
}

func (s *Sample) Update(us util.Sample) {
	for i, n := range s.X {
		n.v = us.X[i]
	}
	for i, n := range s.Y {
		n.v = us.Y[i]
	}
}

type SampleBatch []*Sample

func NewSampleBatch(inputSize, outputSize, batchSize int) SampleBatch {
	sb := make(SampleBatch, batchSize)
	for i := range sb {
		sb[i] = NewSample(inputSize, outputSize)
	}
	return sb
}

func (sb SampleBatch) Update(samples []util.Sample) {
	for i, s := range samples {
		sb[i].Update(s)
	}
}

func ModelLossFunc(model FeedForwarder, lf LossFunction) func([]*Sample) *Node {
	return func(samples []*Sample) *Node {
		var losses []*Node
		for _, s := range samples {
			losses = append(losses, lf(s.Y, model.Feed(s.X)))
		}
		return BatchLoss(losses...)
	}
}

func BatchLoss(losses ...*Node) *Node {
	if len(losses) == 1 {
		return losses[0]
	}
	return Multiply(Plus(losses...), NewNode(1/float64(len(losses)), "mean"))
}

// ResidualSumSquaredLoss is the Residual Sum of Squared (RSS) or Sum of Squared Errors (SSE).
func ResidualSumSquaredLoss(actual, predicted []*Node) *Node {
	if len(predicted) != len(actual) {
		panic("Residual-Sum-of-Squared loss function must receive the same number of predicted values and actual values")
	}

	out := &Node{
		name: fmt.Sprintf("RSS[count=%d]", len(actual)),
		prev: predicted,
	}
	out.forward = func() {
		out.v = 0
		for i, n := range predicted {
			diff := n.v - actual[i].v
			out.v += diff * diff / 2
		}
	}
	out.backward = func() {
		for i, n := range predicted {
			n.g += (n.v - actual[i].v) * out.g
		}
	}
	return out
}

// CrossEntropyLoss defines the cross-entropy loss function. `actual` represents
// the actual probability of each predefined class which should contain only one
// non-vanishing entry with value 1, meaning this class is observed in the data
// set sample (hence probability equals 1).
// Note, calculating the concrete value of cross-entropy is meaningless, so the
// returned node does not contain a valid `Node.v`, it only has `Node.backward`
// and `Node.prev` assigned for backward propagation.
func CrossEntropyLoss(actual, predicted []*Node) *Node {
	if len(predicted) != len(actual) {
		panic(fmt.Sprintf("Cross-Entropy loss function must receive the same number of predicted values and actual values, got actual %v", actual))
	}

	out := &Node{
		name: fmt.Sprintf("cross_entropy[count=%d]", len(actual)),
		prev: predicted,
	}
	out.forward = func() {
		var observed int
		for idx, a := range actual {
			if a.v > 0 {
				observed = idx
				break
			}
		}

		n := predicted[observed]
		out.v = -math.Log(n.v)
	}
	out.backward = func() {
		var observed int
		for idx, a := range actual {
			if a.v > 0 {
				observed = idx
				break
			}
		}

		n := predicted[observed]
		n.g -= out.g / n.v
	}
	return out
}

func MaxMarginLoss(actual, predicted []*Node) *Node {
	if len(predicted) != 1 || len(actual) != 1 {
		panic("Max-Margin loss function must receive scalar values for prediction or label")
	}

	out := &Node{
		name: fmt.Sprintf("MaxMargin[count=%d]", len(actual)),
		prev: predicted,
	}
	out.forward = func() {
		out.v = max(0, 1-predicted[0].v*actual[0].v)
	}
	out.backward = func() {
		n := predicted[0]
		if d := actual[0]; d.v*n.v < 1 {
			n.g -= d.v * out.g
		}
	}
	return out
}
