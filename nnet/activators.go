package nnet

import "math"

type Activator interface {
	A(float64) float64 // Activate
	D(float64) float64 // Derivative
	String() string
}

func IDActivator() Activator {
	return idActivator{}
}

func SigmoidActivator(k float64) Activator {
	return sigmoidActivator{k: k}
}

func TanhActivator(c, k float64) Activator {
	return tanhActivator{c: c, k: k}
}

type idActivator struct{}

func (idActivator) A(x float64) float64 { return x }
func (idActivator) D(float64) float64   { return 1 }
func (idActivator) String() string      { return "identity" }

// sigmoidActivator returns the Logistic Sigmoid function:
//
//	                     1
//	y = φ (x) = --------------------
//	              1 + exp(- k * x)
//
// with the given parameter `k`.
type sigmoidActivator struct {
	k float64
}

func (a sigmoidActivator) A(x float64) float64 {
	return 1 / (1 + math.Exp(-a.k*x))
}

// DLogisticActivator returns the derivative function of logistic activation
// function, but in terms of `y = φ (x)` instead of `x`:
//
//	φ '(x) = k * φ (x) [1 - φ (x)] = k * y * (1 - y)
func (a sigmoidActivator) D(y float64) float64 {
	return a.k * y * (1 - y)
}

func (sigmoidActivator) String() string { return "sigmoid" }

// TanhActivator returns a tanh activation function:
//
//	y = φ (x) = c * tanh(k * x)
//
// with the given parameter `c` and `k`.
type tanhActivator struct {
	c, k float64
}

func (a tanhActivator) A(x float64) float64 {
	return a.c * math.Tanh(a.k*x)
}

// DTanhActivator returns the derivative function of tanh activation function,
// but in terms of `y = φ (x)` instead of `x`:
//
//	φ '(x) = (k / c) * [c - φ (x)] * [c + φ (x)] = (k / c) * (c - y) * (c + y)
func (a tanhActivator) D(y float64) float64 {
	return (a.k / a.c) * (a.c - y) * (a.c + y)
}

func (tanhActivator) String() string { return "tanh" }
