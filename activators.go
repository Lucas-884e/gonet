package gonet

import (
	"math"
)

const (
	LinearActivation  = "linear"
	ReluActivation    = "relu"
	SigmoidActivation = "sigmoid"
	SoftmaxActivation = "softmax"
	TanhActivation    = "tanh"
)

type Activator interface {
	A([]float64) []float64 // Activate
	// Returns a function that determines the element of the derivative matrix
	// at (row, column), which is used for vector and matrix multiplication. Eg,
	//              ┌ d_11, d_12 d_13 ┐
	// [v1 v2 v3] · │ d_21, d_22 d_23 │
	//              └ d_31, d_32 d_33 ┘
	// where d_ij is determined by func(i, j)
	D(ys []float64) func(row, column int) float64 // Derivative
	String() string
}

func LinearActivator() Activator {
	return linearActivator{}
}

func SigmoidActivator(k float64) Activator {
	return sigmoidActivator{k: k}
}

func TanhActivator(k float64) Activator {
	return tanhActivator{k: k}
}

func SoftmaxActivator(t float64) Activator {
	return softmaxActivator{t: t}
}

func ReluActivator() Activator {
	return reluActivator{}
}

type linearActivator struct{}

func (linearActivator) A(xs []float64) []float64 {
	ys := make([]float64, len(xs))
	copy(ys, xs)
	return ys
}

func (linearActivator) D([]float64) func(int, int) float64 {
	return func(i, j int) float64 {
		if i != j {
			return 0
		}
		return 1
	}
}

func (linearActivator) String() string { return LinearActivation }

// sigmoidActivator returns the Logistic Sigmoid function:
//
//	                    1
//	y = φ (x) = ──────────────────
//	             1 + exp(- k * x)
//
// with the given parameter `k`.
type sigmoidActivator struct {
	k float64
}

func (a sigmoidActivator) A(xs []float64) []float64 {
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = 1 / (1 + math.Exp(-a.k*x))
	}
	return ys
}

// D returns the derivative function of logistic activation function, but in
// terms of `y = φ (x)` instead of `x`:
//
//	φ '(x) = k * φ (x) [1 - φ (x)] = k * y * (1 - y)
func (a sigmoidActivator) D(ys []float64) func(int, int) float64 {
	return func(i, j int) float64 {
		if i != j {
			return 0
		}
		return a.k * ys[i] * (1 - ys[i])
	}
}

func (sigmoidActivator) String() string { return SigmoidActivation }

// tanhActivator returns a tanh activation function:
//
//	y = φ (x) = tanh(k * x)
//
// with the given parameter `k`.
type tanhActivator struct {
	k float64
}

func (a tanhActivator) A(xs []float64) []float64 {
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = math.Tanh(a.k * x)
	}
	return ys
}

// D returns the derivative function of tanh activation function, but in terms
// of `y = φ (x)` instead of `x`:
//
//	φ '(x) = k * [1 - φ (x)] * [1 + φ (x)] = k * (1 - y) * (1 + y)
func (a tanhActivator) D(ys []float64) func(int, int) float64 {
	return func(i, j int) float64 {
		if i != j {
			return 0
		}
		return a.k * (1 - ys[i]) * (1 + ys[i])
	}
}

func (tanhActivator) String() string { return TanhActivation }

type softmaxActivator struct {
	t float64 // temperature
}

func (a softmaxActivator) A(xs []float64) []float64 {
	var sum float64
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = math.Exp(x / a.t)
		sum += ys[i]
	}
	for i, y := range ys {
		ys[i] = y / sum
	}
	return ys
}

// D returns the derivative function of softmax activation function, but in terms
// of `y_j = φ_j (x)` instead of `x`:
//
//		∂φ_j (x)    ┌  y_j * (1 - y_i) / t  (for j == i)
//	  ──────── = <│
//	    ∂x_i      └    - y_j * y_i / t    (for j != i)
func (a softmaxActivator) D(ys []float64) func(int, int) float64 {
	return func(i, j int) float64 {
		if i == j {
			return ys[i] * (1 - ys[j]) / a.t
		}
		return -ys[i] * ys[j] / a.t
	}
}

func (softmaxActivator) String() string { return SoftmaxActivation }

type reluActivator struct{}

func (a reluActivator) A(xs []float64) []float64 {
	ys := make([]float64, len(xs))
	for i, x := range xs {
		if x > 0 {
			ys[i] = x
		}
	}
	return ys
}

func (a reluActivator) D(ys []float64) func(int, int) float64 {
	return func(i, j int) float64 {
		if i != j || ys[i] <= 0 {
			return 0
		}
		return 1
	}
}

func (reluActivator) String() string { return ReluActivation }
