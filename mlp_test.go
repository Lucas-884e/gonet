package gonet

import (
	"testing"

	"github.com/Lucas-884e/gonet/util"
	"github.com/stretchr/testify/assert"
)

func TestMLP(t *testing.T) {
	var (
		eps   = 1e-10 // error should be less than this
		input = []*Node{
			NewInputNode(-5, "X_1"),
			NewInputNode(2, "X_2"),
		}
		mlp = NewMLP(2)
	)

	mlp.AddLayer(2, OpSigmoid, true)
	{
		mlp.LoadWeights([][][]float64{
			{{0.5, 0, 1}, {-1, 0, -0.5}},
		})
		out := mlp.Feed(input)

		y1 := out[0]
		y1.Forward()
		assert.InDelta(t, 0.18242552380635635, y1.V(), eps)
		assert.Equal(t, "σ(W_1_1_1×X_1+W_1_1_2×X_2+B_1_1)", y1.Name())

		y2 := out[1]
		y2.Forward()
		assert.InDelta(t, 0.9890130573694068, y2.V(), eps)
		assert.Equal(t, "σ(W_1_2_1×X_1+W_1_2_2×X_2+B_1_2)", y2.Name())
	}

	mlp.AddLayer(2, OpSoftmax, true)
	{
		mlp.LoadWeights([][][]float64{
			{{0.5, 0, 1}, {-1, 0, -0.5}},
			{{-0.5, 1, 0}, {1, 2, -1}},
		})
		out := mlp.Feed(input)

		z1 := out[0]
		z1.Forward()
		assert.InDelta(t, 0.43471206139788876, z1.V(), eps)

		z2 := out[1]
		z2.Forward()
		assert.InDelta(t, 0.5652879386021112, z2.V(), eps)
	}

	var (
		sample = FromSample(util.Sample{
			X: []float64{-5, 2},
			Y: []float64{1, 0},
		})
		lossFn = ModelLossFunc(mlp, CrossEntropyLoss)
		loss   = lossFn([]*Sample{sample})
	)
	loss.Backward()

	layers := mlp.L()

	{ // layer-2
		n := layers[1].N()

		w1 := n[0].W()                                          // weights of first neuron
		assert.InDelta(t, -0.10312294830090556, w1[0].G(), eps) // w11 gradient
		assert.InDelta(t, -0.5590771524509236, w1[1].G(), eps)  // w12 gradient
		assert.InDelta(t, -0.5652879386021112, w1[2].G(), eps)  // bias gradient

		w2 := n[1].W()                                         // weights of second neuron
		assert.InDelta(t, 0.10312294830090556, w2[0].G(), eps) // w21 gradient
		assert.InDelta(t, 0.5590771524509236, w2[1].G(), eps)  // w22 gradient
		assert.InDelta(t, 0.5652879386021112, w2[2].G(), eps)  // bias gradient
	}

	{ // layer-1
		n := layers[0].N()

		w1 := n[0].W()                                         // weights of first neuron
		assert.InDelta(t, -0.6323301783049279, w1[0].G(), eps) // w11 gradient
		assert.InDelta(t, 0.25293207132197115, w1[1].G(), eps) // w12 gradient
		assert.InDelta(t, 0.12646603566098558, w1[2].G(), eps) // bias gradient

		w2 := n[1].W()                                           // weights of second neuron
		assert.InDelta(t, -0.030712743000268498, w2[0].G(), eps) // w21 gradient
		assert.InDelta(t, 0.0122850972001074, w2[1].G(), eps)    // w22 gradient
		assert.InDelta(t, 0.0061425486000537, w2[2].G(), eps)    // bias gradient
	}
}
