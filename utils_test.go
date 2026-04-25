package gonet

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLayerNorm(t *testing.T) {
	var (
		xs = []*Node{
			NewNode(0.6, "x1"),
			NewNode(1.1, "x2"),
			NewNode(0.7, "x3"),
		}
		gamma = []*Node{
			NewNode(0.9, "gamma"),
			NewNode(1.0, "gamma"),
			NewNode(1.1, "gamma"),
		}
		beta = []*Node{
			NewNode(0.5, "beta"),
			NewNode(0.6, "beta"),
			NewNode(0.4, "beta"),
		}
		ln  = LayerNorm(xs, gamma, beta, 1e-5)
		sum = Plus(ln...)
	)
	sum.ForwardBackward()

	assert.InDelta(t, -0.18028742, ln[0].v, 1e-6)
	assert.InDelta(t, 1.73381257, ln[1].v, 1e-6)
	assert.InDelta(t, -0.01573133, ln[2].v, 1e-6)

	assert.EqualValues(t, 1.0, beta[0].g)
	assert.EqualValues(t, 1.0, beta[1].g)
	assert.EqualValues(t, 1.0, beta[2].g)

	assert.InDelta(t, -0.75587493, gamma[0].g, 1e-6)
	assert.InDelta(t, 1.13381254, gamma[1].g, 1e-6)
	assert.InDelta(t, -0.37793758, gamma[2].g, 1e-6)

	assert.InDelta(t, -0.32395410, xs[0].g, 1e-6)
	assert.InDelta(t, -0.08097529, xs[1].g, 1e-6)
	assert.InDelta(t, 0.40492939, xs[2].g, 1e-6)
}
