package gonet

import "math"

func Mean(xs ...*Node) *Node {
	if len(xs) == 1 {
		return xs[0]
	}

	divisor := float64(len(xs))
	mean := &Node{
		name: "mean",
		prev: xs,
	}
	mean.forward = func() {
		var sum float64
		for _, x := range xs {
			sum += x.v
		}
		mean.v = sum / divisor
	}
	mean.backward = func() {
		for _, x := range xs {
			x.g += mean.g / divisor
		}
	}
	return mean
}

func MeanVariance(xs ...*Node) (mean, variance *Node) {
	// Using Bessel's correction (count-1).
	correctedCount := float64(len(xs) - 1)
	mean = Mean(xs...)

	variance = &Node{
		name: "variance",
		prev: append(xs, mean),
	}
	variance.forward = func() {
		variance.v = 0
		for _, x := range xs {
			diff := x.v - mean.v
			variance.v += diff * diff / correctedCount
		}
	}
	variance.backward = func() {
		for _, x := range xs {
			x.g += 2 / correctedCount * (x.v - mean.v) * variance.g
			mean.g += 2 / correctedCount * (mean.v - x.v) * variance.g
		}
	}
	return mean, variance
}

func Normalize(eps float64, xs ...*Node) (ys []*Node) {
	mean, variance := MeanVariance(xs...)

	for _, x := range xs {
		y := &Node{
			name: "normalized",
			prev: []*Node{x, mean, variance},
		}
		y.forward = func() {
			y.v = (x.v - mean.v) / math.Sqrt(variance.v+eps)
		}
		y.backward = func() {
			var (
				v      = variance.v + eps
				sigma  = math.Sqrt(v)
				sigma3 = sigma * v
			)
			x.g += y.g / sigma
			mean.g -= y.g / sigma
			variance.g += (mean.v - x.v) * y.g / sigma3 / 2
		}
		ys = append(ys, y)
	}
	return ys
}

func LayerNorm(xs, gamma, beta []*Node, eps float64) (ys []*Node) {
	for i, n := range Normalize(eps, xs...) {
		y := Plus(Multiply(gamma[i], n), beta[i])
		ys = append(ys, y)
	}
	return ys
}
