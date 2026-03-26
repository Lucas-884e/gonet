package arrimpl

// Layer implements the layer structure in a neural network.
type Layer struct {
	// Number of neurons in this layer.
	size    int
	neurons []*Neuron
	// Activation function and its derivative.
	activator Activator
}

func (l *Layer) zeroNeuronGrads() {
	for i := 1; i <= l.size; i++ {
		l.neurons[i].grad = 0
	}
}

func (l *Layer) zeroWeightGrads() {
	for i := 1; i <= l.size; i++ {
		l.neurons[i].zeroWeightGrads()
	}
}

func (l *Layer) loadWeights(weights [][]float64) {
	for i, n := range l.neurons {
		if i == 0 {
			continue
		}
		n.loadWeights(weights[i-1])
	}
}

func (l *Layer) normalizeGrads(sampleCount int) {
	for i := 1; i <= l.size; i++ {
		n := l.neurons[i]
		for j := range n.weights {
			n.weights[j].grad /= float64(sampleCount)
		}
	}
}

func (l *Layer) output() []float64 {
	ys := make([]float64, l.size)
	for i := 0; i < l.size; i++ {
		ys[i] = l.neurons[i+1].output
	}
	return ys
}

// backward does the backward propagation computation for a single layer:
//
//	               ┌ φ '(x) * δ_j(loss) , for output layer
//	δ_j(neuron) = <│
//	               └ φ '(x) * Σ_k [δ_k(next_layer) * W_{k,j}] , for hidden layer
//
//	                ┌ δ_j(neuron of hidden layer) * x_i(input layer) , for first hidden layer
//	δ_ji(weight) = <│
//	                └ δ_j(neuron of current layer) * y_i(output of previous layer), for other hidden layers and output layer
//
// where `d_j` is the desired response whose estimate is `y_j`, `Σ_k` means
// summation over index `k` and `δ_k(neuron)` is the gradient for k-th neuron
// in some layer, `δ_ji(weight)` is the gradient for the i-th weight of the
// j-th neuron in some layer.
func (l *Layer) backward(prev, next *Layer, lossGrads []float64) {
	// Set all neuron gradients to zero for every training sample.
	l.zeroNeuronGrads()
	// derivative (matrix) function
	df := l.activator.D(l.output())

	// If there is no next layer, we use gradients of loss function to compute gradients of output layer.
	if next == nil {
		for i := 1; i <= l.size; i++ {
			for j, g := range lossGrads {
				l.neurons[i].grad += g * df(j, i-1)
			}
		}
	} else {
		for j := 1; j <= next.size; j++ {
			n := next.neurons[j]
			for _, w := range n.weights {
				if i := w.p - 1; i >= 0 {
					l.neurons[i+1].grad += df(i, i) * n.grad * w.v
				}
			}
		}
	}

	for i := 1; i <= l.size; i++ {
		n := l.neurons[i]
		for j, w := range n.weights {
			n.weights[j].grad += n.grad * prev.neurons[w.p].output
		}
	}
}

func (l *Layer) updateWeights(eta float64) float64 {
	var delta float64
	for j, n := range l.neurons {
		if j == 0 {
			continue
		}
		delta += n.updateWeights(eta)
	}
	return delta
}
