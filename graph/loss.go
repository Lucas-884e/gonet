package graph

import (
	"fmt"

	"github.com/Lucas-884e/gonet/util"
)

type LossFunction func(actual []float64, predicted []*Node) *Node

type FeedForwarder interface {
	Feed([]float64) []*Node
}

func ModelLossFunc(model FeedForwarder, lf LossFunction) func([]util.Sample) *Node {
	return func(samples []util.Sample) *Node {
		var losses []*Node
		for _, s := range samples {
			losses = append(losses, lf(s.Y, model.Feed(s.X)))
		}
		return BatchLoss(losses)
	}
}

func BatchLoss(losses []*Node) *Node {
	if len(losses) == 1 {
		return losses[0]
	}
	return Multiply(Plus(losses...), NewNode(1/float64(len(losses)), "mean"))
}

// CrossEntropyLoss defines the cross-entropy loss function. `actual` represents
// the actual probability of each predefined class which should contain only one
// non-vanishing entry with value 1, meaning this class is observed in the data
// set sample (hence probability equals 1).
// Note, calculating the concrete value of cross-entropy is meaningless, so the
// returned node does not contain a valid `Node.v`, it only has `Node.backward`
// and `Node.prev` assigned for backward propagation.
func CrossEntropyLoss(actual []float64, predicted []*Node) *Node {
	if len(predicted) < 2 {
		panic("Cross-Entropy loss function must have at least two predicted nodes")
	}

	var observed int
	for idx, a := range actual {
		if a > 0 {
			observed = idx
			break
		}
	}

	out := &Node{
		name: fmt.Sprintf("cross_entropy[observed=%d]", observed),
		prev: predicted,
	}
	out.backward = func() {
		n := predicted[observed]
		n.g -= out.g / n.v
	}
	return out
}

// Residual Sum of Squared (RSS) or Sum of Squared Errors (SSE).
func ResidualSumSquaredLoss(actual []float64, predicted []*Node) *Node {
	if len(predicted) != len(actual) {
		panic("Residual-Sum-of-Squared loss function must receive the same number of predicted values and actual values")
	}

	out := &Node{
		name: fmt.Sprintf("RSS[count=%d]", len(actual)),
		prev: predicted,
	}
	out.backward = func() {
		for i, n := range predicted {
			n.g += n.v - actual[i]
		}
	}
	return out
}

func MaxMarginLoss(actual []float64, predicted []*Node) *Node {
	if len(predicted) != len(actual) {
		panic("Max-Margin loss function must receive the same number of predicted values and actual values")
	}

	out := &Node{
		name: fmt.Sprintf("MaxMargin[count=%d]", len(actual)),
		prev: predicted,
	}
	out.backward = func() {
		for i, n := range predicted {
			if d := actual[i]; d*n.v < 1 {
				n.g -= d
			}
		}
	}
	return out
}
