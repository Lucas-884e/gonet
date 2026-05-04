package gonet

import (
	"cmp"
	"fmt"
	"math"
	"slices"

	"github.com/LucasInOz/gonet/util"
)

type (
	LossFunction   func(actual, predicted []*Node) *Node
	E2ELoss        func([]util.Sample) *Node
	E2EPredictLoss func([]util.Sample) float64
)

type FeedForwarder interface {
	Feed([]*Node) []*Node
}

type Sample struct {
	X []*Node
	Y []*Node
}

func NewSample(inputSize, outputSize int, noGrad bool) *Sample {
	return &Sample{
		X: NewInputNodeBatch(inputSize, "X_%d", noGrad),
		Y: NewInputNodeBatch(outputSize, "Y_%d", noGrad),
	}
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
		sb[i] = NewSample(inputSize, outputSize, false)
	}
	return sb
}

func (sb SampleBatch) Update(samples []util.Sample) {
	for i, s := range samples {
		sb[i].Update(s)
	}
}

func PredictLossFunc(model FeedForwarder, lf LossFunction) E2EPredictLoss {
	var (
		sample *Sample
		loss   *Node
	)

	return func(samples []util.Sample) float64 {
		if s0 := samples[0]; sample == nil || loss == nil {
			sample = NewSample(len(s0.X), len(s0.Y), true)
			loss = lf(sample.Y, model.Feed(sample.X))
		}

		var sum float64
		for _, s := range samples {
			sample.Update(s)
			loss.Forward()
			sum += loss.V()
		}
		return sum / float64(len(samples))
	}
}

func TrainLossFunc(model FeedForwarder, lf LossFunction) E2ELoss {
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
			loss = Mean(losses...)
		}

		batch.Update(samples)
		loss.Forward()
		return loss
	}
}

// ResidualSumSquaredLoss is the Residual Sum of Squared (RSS) or Sum of Squared Errors (SSE).
func ResidualSumSquaredLoss(actual, predicted []*Node) *Node {
	if len(predicted) != len(actual) {
		panic("Residual-Sum-of-Squared loss function must receive the same number of predicted values and actual values")
	}

	var (
		noGrad = predicted[0].noGrad
		out    = &Node{
			name:   fmt.Sprintf("RSS[count=%d]", len(actual)),
			prev:   predicted,
			noGrad: noGrad,
		}
	)
	out.forward = func() {
		out.v = 0
		for i, n := range predicted {
			diff := n.v - actual[i].v
			out.v += diff * diff / 2
		}
	}
	if !noGrad {
		out.backward = func() {
			for i, n := range predicted {
				n.g += (n.v - actual[i].v) * out.g
			}
		}
	}
	return out
}

// RawCrossEntropyLoss defines the cross-entropy loss function. `actual` represents
// the actual probability of each predefined class which should contain only one
// non-vanishing entry with value 1, meaning this class is observed in the data
// set sample (hence probability equals 1).
func RawCrossEntropyLoss(actual, predicted []*Node) *Node {
	if len(predicted) != len(actual) {
		panic(fmt.Sprintf("Cross-Entropy loss function must receive the same number of predicted values and actual values, got actual %d", len(actual)))
	}

	var (
		noGrad = predicted[0].noGrad
		out    = &Node{
			name:   fmt.Sprintf("cross_entropy[count=%d]", len(predicted)),
			prev:   predicted,
			noGrad: noGrad,
		}
	)

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

	if !noGrad {
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
	}
	return out
}

// CrossEntropyLoss takes softmax activation into account already, so the values
// in `predicted` nodes are logits rather than probabilities actaully. If you
// would like to use the cross entropy function without softmax fused into it,
// use RawCrossEntoryLoss instead. Also note that, different from RawCrossEntoryLoss,
// all values in `actual` nodes are indexes of ground-truth. So do expect
// `actual` (indexes) and `predicted` (logits) to be of different lengths.
func CrossEntropyLoss(actual, predicted []*Node) *Node {
	if len(predicted)%len(actual) != 0 {
		panic("number of predicted logits is not a multiple of actual indexes")
	}

	var (
		noGrad  = predicted[0].noGrad
		outs    = make([]*Node, len(actual))
		nLogits = len(predicted) / len(actual)
	)
	for i := range outs {
		outs[i] = &Node{
			name:   fmt.Sprintf("cross_entropy[%d]", i),
			prev:   predicted[i*nLogits : (i+1)*nLogits],
			noGrad: noGrad,
		}
	}

	for i, out := range outs {
		var (
			qs       = make([]float64, nLogits) // softmax
			observed int
		)
		out.forward = func() {
			var (
				vmax = slices.MaxFunc(out.prev, func(a, b *Node) int { return cmp.Compare(a.v, b.v) }).v
				sum  float64
			)
			for j, n := range out.prev {
				qs[j] = math.Exp(n.v - vmax)
				sum += qs[j]
			}
			for j, q := range qs {
				qs[j] = q / sum
			}

			observed = int(actual[i].v)
			out.v = -math.Log(qs[observed])
		}

		if !noGrad {
			out.backward = func() {
				for j, n := range out.prev {
					if j == observed {
						n.g += (qs[j] - 1) * out.g
					} else {
						n.g += qs[j] * out.g
					}
				}
			}
		}
	}

	return Mean(outs...)
}

func MaxMarginLoss(actual, predicted []*Node) *Node {
	if len(predicted) != 1 || len(actual) != 1 {
		panic("Max-Margin loss function must receive scalar values for prediction or label")
	}

	var (
		noGrad = predicted[0].noGrad
		out    = &Node{
			name:   fmt.Sprintf("MaxMargin[count=%d]", len(actual)),
			prev:   predicted,
			noGrad: noGrad,
		}
	)
	out.forward = func() {
		out.v = max(0, 1-predicted[0].v*actual[0].v)
	}
	if !noGrad {
		out.backward = func() {
			n := predicted[0]
			if d := actual[0]; d.v*n.v < 1 {
				n.g -= d.v * out.g
			}
		}
	}
	return out
}
