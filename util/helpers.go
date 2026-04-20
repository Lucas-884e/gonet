package util

import (
	"math/rand/v2"

	"golang.org/x/exp/constraints"
)

func SliceConvert[From, To any](s []From) []To {
	t := make([]To, len(s))
	for i, e := range s {
		t[i] = any(e).(To)
	}
	return t
}

func NumberSliceConvert[From, To constraints.Integer | constraints.Float](vs []From) []To {
	result := make([]To, len(vs))
	for i, v := range vs {
		result[i] = To(v)
	}
	return result
}

func Must1[T any](x T, err error) T {
	if err != nil {
		panic(err)
	}
	return x
}

func RandMultinomial(probDist []float64) int {
	var (
		accum float64
		r     = rand.Float64()
	)
	for idx, p := range probDist {
		if accum += p; r < accum {
			return idx
		}
	}
	return len(probDist) - 1
}
