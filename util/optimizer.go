package util

import "math"

type Parameter interface {
	Learn(delta float64)
	G() float64
}

type LearningRateFunc func(grad float64) (rate float64)

// AnnealingLearningRate returns an annealed learning rate by the following formula:
//
//	η = η_0 / (1 + decay * epoch)
func AnnealingLearningRate(eta0, decay float64, epoch int) float64 {
	return eta0 / (1 + decay*float64(epoch))
}

func AnnealingLearningRateFunc(eta, decay float64, epoch int) LearningRateFunc {
	eta = eta / (1 + decay*float64(epoch))
	return func(g float64) float64 {
		return eta * g
	}
}

type AdamOptimizer struct {
	params  []Parameter
	eta     float64
	beta1   float64
	beta2   float64
	epsilon float64

	beta1t float64
	beta2t float64
	m      []float64 // momentum
	v      []float64 // velocity
}

func NewAdamOptimizer(params []Parameter, eta, beta1, beta2, epsilon float64) *AdamOptimizer {
	return &AdamOptimizer{
		params:  params,
		eta:     eta,
		beta1:   beta1,
		beta2:   beta2,
		epsilon: epsilon,
		beta1t:  1.0,
		beta2t:  1.0,
		m:       make([]float64, len(params)),
		v:       make([]float64, len(params)),
	}
}

func NewDefaultAdamOptimizer(params []Parameter, eta float64) *AdamOptimizer {
	return NewAdamOptimizer(params, eta, 0.9, 0.999, 1e-8)
}

func (ao *AdamOptimizer) Learn() float64 {
	ao.beta1t *= ao.beta1
	ao.beta2t *= ao.beta2

	var diff float64
	for i, p := range ao.params {
		delta := ao.learningRate(i, p.G())
		p.Learn(delta)
		diff += delta * delta
	}
	return diff
}

func (ao *AdamOptimizer) learningRate(i int, g float64) float64 {
	ao.m[i] = ao.beta1*ao.m[i] + (1-ao.beta1)*g
	ao.v[i] = ao.beta2*ao.v[i] + (1-ao.beta2)*g*g

	mHat := ao.m[i] / (1 - ao.beta1t)
	vHat := ao.v[i] / (1 - ao.beta2t)
	return ao.eta * mHat / (math.Sqrt(vHat) + ao.epsilon)
}
