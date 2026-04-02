package util

import "math/rand/v2"

func ListConvert[S, T any](s []S) []T {
	t := make([]T, len(s))
	for i, e := range s {
		t[i] = any(e).(T)
	}
	return t
}

func Must1[T any](x T, err error) T {
	if err != nil {
		panic(err)
	}
	return x
}

func RandMultinomial(probDist []float64) int {
	var accum float64
	for idx, p := range probDist {
		r := rand.Float64()
		if accum += p; r < accum {
			return idx
		}
	}
	return len(probDist) - 1
}
