package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMLP(t *testing.T) {
	var (
		eps = 1e-10 // error should be less than this
		mlp = NewMLP(2)
	)

	mlp.AddLayer(2, OpSigmoid)
	mlp.LoadWeights([][][]float64{
		{{1, 0.5, 0}, {-0.5, -1, 0}},
	})

	out := mlp.Feed([]float64{-5, 2})
	y1 := out[0]
	y2 := out[1]
	assert.InDelta(t, 0.18242552380635635, y1.V(), eps)
	assert.Equal(t, "σ(B_11+W_111×X_1+W_112×X_2)", y1.Name())
	assert.InDelta(t, 0.9890130573694068, y2.V(), eps)
	assert.Equal(t, "σ(B_12+W_121×X_1+W_122×X_2)", y2.Name())

	mlp.AddLayer(2, OpSoftmax)
	mlp.LoadWeights([][][]float64{
		{{1, 0.5, 0}, {-0.5, -1, 0}},
		{{0, -0.5, 1}, {-1, 1, 2}},
	})

	out = mlp.Feed([]float64{-5, 2})
	z1 := out[0]
	z2 := out[1]
	assert.InDelta(t, 0.43471206139788876, z1.V(), eps)
	assert.InDelta(t, 0.5652879386021112, z2.V(), eps)
}
