package gonet

import (
	"testing"

	"github.com/Lucas-884e/gonet/util"
	"github.com/stretchr/testify/assert"
)

func TestMLP(t *testing.T) {
	var (
		eps   = 1e-10 // error should be less than this
		input = []float64{-5, 2}
		mlp   = NewMLP(2)
	)

	mlp.AddLayer(2, OpSigmoid, true)
	{
		mlp.LoadWeights([][][]float64{
			{{0.5, 0, 1}, {-1, 0, -0.5}},
		})
		out := mlp.BatchFeed(input)

		y1 := out[0][0]
		y1.Forward()
		assert.InDelta(t, 0.18242552380635635, y1.V(), eps)
		assert.Equal(t, "σ(W_0×X0_0+W_1×X0_1+B)", y1.Name())

		y2 := out[0][1]
		y2.Forward()
		assert.InDelta(t, 0.9890130573694068, y2.V(), eps)
		assert.Equal(t, "σ(W_0×X0_0+W_1×X0_1+B)", y2.Name())
	}

	mlp.AddLayer(2, OpSoftmax, true)
	{
		mlp.LoadWeights([][][]float64{
			{{0.5, 0, 1}, {-1, 0, -0.5}},
			{{-0.5, 1, 0}, {1, 2, -1}},
		})
		out := mlp.BatchFeed(input)

		z1 := out[0][0]
		z1.Forward()
		assert.InDelta(t, 0.43471206139788876, z1.V(), eps)

		z2 := out[0][1]
		z2.Forward()
		assert.InDelta(t, 0.5652879386021112, z2.V(), eps)
	}

	var (
		sample = util.Sample{
			X: []float64{-5, 2},
			Y: []float64{1, 0},
		}
		lossFn = TrainLossFunc(mlp, RawCrossEntropyLoss)
		loss   = lossFn([]util.Sample{sample})
	)
	loss.Backward()

	layers := mlp.L()

	{ // layer-2
		n := layers[2].(*linearLayer).neurons

		w1 := n[0].weights                                         // weights of first neuron
		assert.InDelta(t, -0.10312294830090556, w1[0].G(), eps)    // w11 gradient
		assert.InDelta(t, -0.5590771524509236, w1[1].G(), eps)     // w12 gradient
		assert.InDelta(t, -0.5652879386021112, n[0].bias.G(), eps) // bias gradient

		w2 := n[1].weights
		assert.InDelta(t, 0.10312294830090556, w2[0].G(), eps)    // w21 gradient
		assert.InDelta(t, 0.5590771524509236, w2[1].G(), eps)     // w22 gradient
		assert.InDelta(t, 0.5652879386021112, n[1].bias.G(), eps) // bias gradient
	}

	{ // layer-1
		n := layers[0].(*linearLayer).neurons

		w1 := n[0].weights
		assert.InDelta(t, -0.6323301783049279, w1[0].G(), eps)     // w11 gradient
		assert.InDelta(t, 0.25293207132197115, w1[1].G(), eps)     // w12 gradient
		assert.InDelta(t, 0.12646603566098558, n[0].bias.G(), eps) // bias gradient

		w2 := n[1].weights
		assert.InDelta(t, -0.030712743000268498, w2[0].G(), eps)  // w21 gradient
		assert.InDelta(t, 0.0122850972001074, w2[1].G(), eps)     // w22 gradient
		assert.InDelta(t, 0.0061425486000537, n[1].bias.G(), eps) // bias gradient
	}
}
