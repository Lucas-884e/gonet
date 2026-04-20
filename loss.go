package gonet

import (
	"cmp"
	"fmt"
	"math"
	"slices"

	"github.com/Lucas-884e/gonet/util"
)

type (
	LossFunction func(actual, predicted []*Node) *Node
	E2ELoss      func([]util.Sample) *Node
)

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

func ModelLossFunc(model FeedForwarder, lf LossFunction) E2ELoss {
	var (
		batch SampleBatch
		loss  *Node
	)

	return func(samples []util.Sample) *Node {
		if len(batch) == 0 || loss == nil {
			batch = NewSampleBatch(len(samples[0].X), len(samples[0].Y), len(samples))

			losses := make([]*Node, len(samples))
			for i, s := range batch {
				out := model.Feed(s.X)
				losses[i] = lf(s.Y, out)
			}
			loss = BatchLoss(losses...)
		}

		batch.Update(samples)
		loss.Forward()
		return loss
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
func CrossEntropyLoss(actual, predicted []*Node) *Node {
	out := &Node{
		name: fmt.Sprintf("cross_entropy[count=%d]", len(predicted)),
		prev: predicted,
	}

	if len(actual) == 1 {
		// In this case, predicted are actually logits, and the softmax layer is
		// fused with this loss function.
		var (
			qs       = make([]float64, len(predicted))
			observed int
		)

		out.forward = func() {
			var (
				vmax = slices.MaxFunc(predicted, func(a, b *Node) int { return cmp.Compare(a.v, b.v) }).v
				sum  float64
			)
			for i, n := range predicted {
				qs[i] = math.Exp(n.v - vmax)
				sum += qs[i]
			}
			for i, q := range qs {
				qs[i] = q / sum
			}

			observed = int(actual[0].v)
			out.v = -math.Log(qs[observed])
		}

		out.backward = func() {
			for i, n := range predicted {
				if i == observed {
					n.g += (qs[i] - 1) * out.g
				} else {
					n.g += qs[i] * out.g
				}
			}
		}
		return out
	}

	if len(predicted) != len(actual) {
		panic(fmt.Sprintf("Cross-Entropy loss function must receive the same number of predicted values and actual values, got actual %d", len(actual)))
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
