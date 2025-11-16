package training

import "math/rand/v2"

// AnnealingLearningRate returns an annealed learning rate by the following formula:
//
//	η = η_0 / (1 + n / tau)
func AnnealingLearningRate(eta0 float64, tau, n int64) float64 {
	return eta0 / (1 + float64(n/tau))
}

func RandomUniformSample(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
