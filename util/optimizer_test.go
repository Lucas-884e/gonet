package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDefaultAdamLearningRateFunc(t *testing.T) {
	var (
		v   = 10.0
		g   = func(v float64) float64 { return 2 * v }
		opt = NewDefaultAdamOptimizer(1.05)

		expect = []float64{8.950, 7.904, 6.867, 5.843, 4.838}
	)

	for _, exp := range expect {
		opt.Step()
		v -= opt.LearningRate(g(v))
		assert.InDelta(t, exp, v, 0.005)
	}
}
