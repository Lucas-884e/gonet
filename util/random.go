package util

import "math/rand/v2"

func RandomUniformSample(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

type Sample struct {
	X []float64 // input
	Y []float64 // output
}
