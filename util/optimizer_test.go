package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type Param struct {
	v float64
}

func (p *Param) Learn(d float64) {
	p.v -= d
}

func (p *Param) G() float64 {
	return 2 * p.v
}

func NewParam(v float64) *Param {
	return &Param{v: v}
}

func TestDefaultAdamLearningRateFunc(t *testing.T) {
	var (
		p = NewParam(10.0)
		// g   = func(v float64) float64 { return 2 * v }
		opt = NewDefaultAdamOptimizer([]Parameter{p}, 1.05)

		expect = []float64{8.950, 7.904, 6.867, 5.843, 4.838}
	)

	for _, exp := range expect {
		opt.Learn()
		assert.InDelta(t, exp, p.v, 0.005)
	}
}
