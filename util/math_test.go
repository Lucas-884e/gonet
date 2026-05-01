package util

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTranspose(t *testing.T) {
	mat := [][]int{
		{1, 2, 3},
		{4, 5, 6},
	}
	matTr := [][]int{
		{1, 4},
		{2, 5},
		{3, 6},
	}
	assert.Equal(t, matTr, Transpose(mat))
}

func TestMaskedAtention(t *testing.T) {
	var (
		ks = [][]float64{{1, 2}, {3, 4}, {5, 6}}
		qs = [][]float64{{-0.1, 0.2}, {-0.3, 0.4}, {-0.5, 0.6}}
		vs = [][]float64{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}

		multiply = func(a, b []float64) float64 {
			var sum float64
			for i, v := range b {
				sum += a[i] * v
			}
			return sum
		}

		softmax = func(x []float64) []float64 {
			return Softmax(math.Sqrt(2), x)
		}
		attention = MaskedAttention(ks, qs, vs, multiply, softmax)
	)

	var (
		// a = 0.4647034688926673, 0.5352965311073327
		a = Softmax(math.Sqrt(2), []float64{0.5, 0.7})
		// b = 0.2874549195939612, 0.33112217060709076, 0.3814229097989481
		b = Softmax(math.Sqrt(2), []float64{0.7, 0.9, 1.1})
	)
	assert.InDelta(t, vs[0][0], attention[0], 1e-10)
	assert.InDelta(t, vs[0][1], attention[1], 1e-10)
	assert.InDelta(t, vs[0][2], attention[2], 1e-10)
	assert.InDelta(t, a[0]*vs[0][0]+a[1]*vs[1][0], attention[3], 1e-10)
	assert.InDelta(t, a[0]*vs[0][1]+a[1]*vs[1][1], attention[4], 1e-10)
	assert.InDelta(t, a[0]*vs[0][2]+a[1]*vs[1][2], attention[5], 1e-10)
	assert.InDelta(t, b[0]*vs[0][0]+b[1]*vs[1][0]+b[2]*vs[2][0], attention[6], 1e-10)
	assert.InDelta(t, b[0]*vs[0][1]+b[1]*vs[1][1]+b[2]*vs[2][1], attention[7], 1e-10)
	assert.InDelta(t, b[0]*vs[0][2]+b[1]*vs[1][2]+b[2]*vs[2][2], attention[8], 1e-10)
}
