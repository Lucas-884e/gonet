package util

import (
	"math"
	"slices"
)

func Softmax(t float64, xs []float64) []float64 {
	var (
		ys   = make([]float64, len(xs))
		xmax = slices.Max(xs)
		sum  float64
	)
	for i, x := range xs {
		ys[i] = math.Exp((x - xmax) / t)
		sum += ys[i]
	}
	for i := range ys {
		ys[i] /= sum
	}
	return ys
}

func Transpose[T any](mat [][]T) [][]T {
	out := make([][]T, len(mat[0]))
	for j := range out {
		out[j] = make([]T, len(mat))
	}
	for i, row := range mat {
		for j, elem := range row {
			out[j][i] = elem
		}
	}
	return out
}

func MaskedAttention[T any](ks, qs, vs [][]T, mul func([]T, []T) T, softmax func([]T) []T) (out []T) {
	// ws is the masked attention weights (a lower-triangular matrix):
	// [1, 0, 0, ...]
	// [x, x, 0, ...]  (sum_x = 1)
	// [y, y, y, ...]  (sum_y = 1)
	// ...
	ws := [][]T{{}}
	for i, q := range qs {
		if i == 0 {
			continue // Skip the first row (no need to compute)
		}
		ps := make([]T, i+1)
		for j := range i + 1 {
			ps[j] = mul(q, ks[j])
		}
		ws = append(ws, softmax(ps))
	}

	// First row is easy, just simple element moving due to the fact that the
	// softmatx of [x, -∞, -∞, ...] is always [1, 0, 0, ...].
	out = append(out, vs[0]...)
	vs = Transpose(vs)

	for i, w := range ws {
		if i == 0 {
			continue // Skip first row as vs[0] is already appended to `out`.
		}
		for _, v := range vs {
			// Since ws is lower-triangular matrix, so we only take the first `i+1`
			// elements of transposed vs for vector multiplication (dot product).
			out = append(out, mul(w, v[:i+1]))
		}
	}
	return out
}
