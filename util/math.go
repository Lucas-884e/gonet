package util

import (
	"math"
	"slices"
)

func Softmax(xs []float64) []float64 {
	var (
		ys   = make([]float64, len(xs))
		xmax = slices.Max(xs)
		sum  float64
	)
	for i, x := range xs {
		ys[i] = math.Exp(x - xmax)
		sum += ys[i]
	}
	for i := range ys {
		ys[i] /= sum
	}
	return ys
}
