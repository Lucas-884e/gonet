package nnet

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

// setWeights sets layer weight values in an orderly fashion only for test.
func (layer *Layer) setWeights(weights [][]float64) {
	for j, neuronWeights := range weights {
		neuron := layer.neurons[j+1]
		for i, w := range neuronWeights {
			neuron.weights[i].v = w
		}
	}
}

func TestFCNNet(t *testing.T) {
	var (
		equal bool
		msg   string

		logisticActivator  = LogisticActivator(1)
		dLogisticActivator = DLogisticActivator(1)

		net = NewFCNNet(2, logisticActivator, dLogisticActivator)

		expectLayer0 = &Layer{
			size: 2,
			neurons: []*Neuron{
				{n: 0, output: 1},
				{n: 1},
				{n: 2},
			},
		}
		expectLayer1 = &Layer{
			size: 2,
			neurons: []*Neuron{
				{n: 0, output: 1},
				{n: 1, activator: IDActivator, dActivator: DIDActivator, weights: []Weight{
					{n: 1, p: 0},
					{n: 1, p: 1},
					{n: 1, p: 2},
				}},
				{n: 2, activator: IDActivator, dActivator: DIDActivator, weights: []Weight{
					{n: 2, p: 0},
					{n: 2, p: 1},
					{n: 2, p: 2},
				}},
			},
		}
		expectLayer2 = &Layer{
			size: 3,
			neurons: []*Neuron{
				{n: 0, output: 1},
				{n: 1, activator: logisticActivator, dActivator: dLogisticActivator, weights: []Weight{
					{n: 1, p: 0},
					{n: 1, p: 1},
					{n: 1, p: 2},
				}},
				{n: 2, activator: logisticActivator, dActivator: dLogisticActivator, weights: []Weight{
					{n: 2, p: 0},
					{n: 2, p: 1},
					{n: 2, p: 2},
				}},
				{n: 3, activator: logisticActivator, dActivator: dLogisticActivator, weights: []Weight{
					{n: 3, p: 0},
					{n: 3, p: 1},
					{n: 3, p: 2},
				}},
			},
		}
		expectLayer3 = &Layer{
			size: 2,
			neurons: []*Neuron{
				{n: 0, output: 1},
				{n: 1, activator: logisticActivator, dActivator: dLogisticActivator, weights: []Weight{
					{n: 1, p: 0},
					{n: 1, p: 1},
					{n: 1, p: 2},
					{n: 1, p: 3},
				}},
				{n: 2, activator: logisticActivator, dActivator: dLogisticActivator, weights: []Weight{
					{n: 2, p: 0},
					{n: 2, p: 1},
					{n: 2, p: 2},
					{n: 2, p: 3},
				}},
			},
		}
	)

	// test NewFCNNet
	assert.Equal(t, 1, len(net.layers))
	assert.True(t, equalFuncs(logisticActivator, net.defaultActivator))
	assert.True(t, equalFuncs(dLogisticActivator, net.defaultDActivator))
	equal, msg = equalLayers(net.layers[0], expectLayer0)
	assert.Truef(t, equal, msg)

	// test AddLayer
	// add layer 1
	net.AddLayerWithActivator(2, IDActivator, DIDActivator)
	assert.Equal(t, 2, len(net.layers))
	equal, msg = equalLayers(net.layers[1], expectLayer1)
	assert.Truef(t, equal, msg)
	// add layer 2
	net.AddLayer(3)
	assert.Equal(t, 3, len(net.layers))
	equal, msg = equalLayers(net.layers[2], expectLayer2)
	assert.Truef(t, equal, msg)
	// add layer 3
	net.AddLayer(2)
	assert.Equal(t, 4, len(net.layers))
	equal, msg = equalLayers(net.layers[3], expectLayer3)
	assert.Truef(t, equal, msg)

	// test feedInputSample
	var ( // network sample inputs and desired outputs
		y01 = 0.3
		y02 = 0.7
		d1  = 0.9
		d2  = 0.1
	)
	net.feedInputSample([]float64{y01, y02}, []float64{d1, d2})
	expectLayer0.neurons[1].output = 0.3
	expectLayer0.neurons[2].output = 0.7
	equal, msg = equalLayers(net.layers[0], expectLayer0)
	assert.Truef(t, equal, msg)
	assert.Equal(t, []float64{0.9, 0.1}, net.desiredOutputs)

	// Set initial weights
	// 1st layer
	layer1Weights := [][]float64{
		{0.5, 1, 0.3}, // 1st neuron
		{0.7, 0.8, 1}, // 2nd neuron
	}
	net.layers[1].setWeights(layer1Weights)
	// 2nd layer
	layer2Weights := [][]float64{
		{1, 0.2, 0.6}, // 1st neuron
		{0.4, 1, 1},   // 2nd neuron
		{1, 1, 0.9},   // 3rd neuron
	}
	net.layers[2].setWeights(layer2Weights)
	// 3rd layer
	layer3Weights := [][]float64{
		{0.6, 1, 0.5, 1},   // 1st neuron
		{1, 0.25, 1, 0.75}, // 2nd neuron
	}
	net.layers[3].setWeights(layer3Weights)

	// Set expected weights and compute neuron outputs and local gradients.
	expectLayer1.setWeights(layer1Weights)
	expectLayer2.setWeights(layer2Weights)
	expectLayer3.setWeights(layer3Weights)
	var (
		// y^1_j = W^1_{j0} + W^1_{j1} * y^0_1 + W^1_{j2} * y^0_2
		y11 = 0.5 + 1*y01 + 0.3*y02
		y12 = 0.7 + 0.8*y01 + 1*y02
		// y^2_j = φ (W^2_{j0} + W^2_{j1} * y^1_1 + W^2_{j2} * y^1_2)
		// where φ is the logistic activation function.
		y21 = logisticActivator(1 + 0.2*y11 + 0.6*y12)
		y22 = logisticActivator(0.4 + 1*y11 + 1*y12)
		y23 = logisticActivator(1 + 1*y11 + 0.9*y12)
		// y^3_j = φ (W^3_{j0} + W^3_{j1} * y^2_1 + W^3_{j2} * y^2_2 + W^3_{j3} * y^2_3)
		y31 = logisticActivator(0.6 + 1*y21 + 0.5*y22 + 1*y23)
		y32 = logisticActivator(1 + 0.25*y21 + 1*y22 + 0.75*y23)
		// Compute local gradients
		// δ^3_j = φ '(y^3_j) * (d_j - y^3_j)
		g31 = dLogisticActivator(y31) * (d1 - y31)
		g32 = dLogisticActivator(y32) * (d2 - y32)
		// δ^2_j = φ '(y^2_j) * (g^3_1 * W^3_{1j} + g^3_2 * W^3_{2j})
		g21 = dLogisticActivator(y21) * (g31*1 + g32*0.25)
		g22 = dLogisticActivator(y22) * (g31*0.5 + g32*1)
		g23 = dLogisticActivator(y23) * (g31*1 + g32*0.75)
		// δ^1_j = g^2_1 * W^2_{1j} + g^2_2 * W^2_{2j} + g^2_3 * W^2_{3j}
		g11 = g21*0.2 + g22*1 + g23*1
		g12 = g21*0.6 + g22*1 + g23*0.9
	)

	// test forwardPropagate
	net.forwardPropagate()
	{ // Set expected outputs
		expectLayer1.neurons[1].output = y11
		expectLayer1.neurons[2].output = y12
		equal, msg = equalLayers(net.layers[1], expectLayer1)
		assert.Truef(t, equal, msg)
		expectLayer2.neurons[1].output = y21
		expectLayer2.neurons[2].output = y22
		expectLayer2.neurons[3].output = y23
		equal, msg = equalLayers(net.layers[2], expectLayer2)
		assert.Truef(t, equal, msg)
		expectLayer3.neurons[1].output = y31
		expectLayer3.neurons[2].output = y32
		equal, msg = equalLayers(net.layers[3], expectLayer3)
		assert.Truef(t, equal, msg)
	}

	// test backwardPropagate
	net.backwardPropagate()
	{ // Set expected local gradients
		expectLayer3.neurons[1].localGrad = g31
		expectLayer3.neurons[2].localGrad = g32
		equal, msg = equalLayers(net.layers[3], expectLayer3)
		assert.Truef(t, equal, msg)
		expectLayer2.neurons[1].localGrad = g21
		expectLayer2.neurons[2].localGrad = g22
		expectLayer2.neurons[3].localGrad = g23
		equal, msg = equalLayers(net.layers[2], expectLayer2)
		assert.Truef(t, equal, msg)
		expectLayer1.neurons[1].localGrad = g11
		expectLayer1.neurons[2].localGrad = g12
		equal, msg = equalLayers(net.layers[1], expectLayer1)
		assert.Truef(t, equal, msg)
	}

	// test UpdateWeights
	eta := 0.1
	assert.False(t, net.UpdateWeights(eta))
	// Set expected weights. Weight updates:
	//   ΔW^l_{ji} = η * δ^l_j * y^{l-1}_i
	// where eta is the learning rate and
	//   l = 1, 2, 3
	//   j = 1, 2, ...
	//   i = 0, 1, ...
	expectLayer1.neurons[1].weights[0].v += eta * g11       // W^1_{10}
	expectLayer1.neurons[1].weights[1].v += eta * g11 * y01 // W^1_{11}
	expectLayer1.neurons[1].weights[2].v += eta * g11 * y02 // W^1_{12}
	expectLayer1.neurons[2].weights[0].v += eta * g12       // W^1_{20}
	expectLayer1.neurons[2].weights[1].v += eta * g12 * y01 // W^1_{21}
	expectLayer1.neurons[2].weights[2].v += eta * g12 * y02 // W^1_{22}
	equal, msg = equalLayers(net.layers[1], expectLayer1)
	assert.Truef(t, equal, msg)
	expectLayer2.neurons[1].weights[0].v += eta * g21       // W^2_{10}
	expectLayer2.neurons[1].weights[1].v += eta * g21 * y11 // W^2_{11}
	expectLayer2.neurons[1].weights[2].v += eta * g21 * y12 // W^2_{12}
	expectLayer2.neurons[2].weights[0].v += eta * g22       // W^2_{20}
	expectLayer2.neurons[2].weights[1].v += eta * g22 * y11 // W^2_{21}
	expectLayer2.neurons[2].weights[2].v += eta * g22 * y12 // W^2_{22}
	expectLayer2.neurons[3].weights[0].v += eta * g23       // W^2_{30}
	expectLayer2.neurons[3].weights[1].v += eta * g23 * y11 // W^2_{31}
	expectLayer2.neurons[3].weights[2].v += eta * g23 * y12 // W^2_{32}
	equal, msg = equalLayers(net.layers[2], expectLayer2)
	assert.Truef(t, equal, msg)
	expectLayer3.neurons[1].weights[0].v += eta * g31       // W^3_{10}
	expectLayer3.neurons[1].weights[1].v += eta * g31 * y21 // W^3_{11}
	expectLayer3.neurons[1].weights[2].v += eta * g31 * y22 // W^3_{12}
	expectLayer3.neurons[1].weights[3].v += eta * g31 * y23 // W^3_{13}
	expectLayer3.neurons[2].weights[0].v += eta * g32       // W^3_{20}
	expectLayer3.neurons[2].weights[1].v += eta * g32 * y21 // W^3_{21}
	expectLayer3.neurons[2].weights[2].v += eta * g32 * y22 // W^3_{22}
	expectLayer3.neurons[2].weights[3].v += eta * g32 * y23 // W^3_{22}
	equal, msg = equalLayers(net.layers[3], expectLayer3)
	assert.Truef(t, equal, msg)
}

func equalLayers(l1, l2 *Layer) (bool, string) {
	if l1.size != l2.size {
		return false, fmt.Sprintf("Layer size: %d != %d", l1.size, l2.size)
	}
	if len(l1.neurons) != len(l2.neurons) {
		return false, fmt.Sprintf("Neuron count: %d != %d", len(l1.neurons), len(l2.neurons))
	}
	for i, n := range l1.neurons {
		if equal, msg := equalNeurons(n, l2.neurons[i]); !equal {
			return false, fmt.Sprintf("%d-th %s", i, msg)
		}
	}
	return true, ""
}

func equalNeurons(n1, n2 *Neuron) (bool, string) {
	if n1.n != n2.n {
		return false, fmt.Sprintf("neuron index: %d != %d", n1.n, n2.n)
	}
	if n1.output != n2.output {
		return false, fmt.Sprintf("neuron output: %g != %g", n1.output, n2.output)
	}
	if n1.localGrad != n2.localGrad {
		return false, fmt.Sprintf("neuron local gradient: %g != %g", n1.localGrad, n2.localGrad)
	}
	if !equalFuncs(n1.activator, n2.activator) {
		return false, fmt.Sprintf("neuron activator: %v != %v", n1.activator, n2.activator)
	}
	if !equalFuncs(n1.dActivator, n2.dActivator) {
		return false, fmt.Sprintf("neuron activator derivative: %v != %v", n1.dActivator, n2.dActivator)
	}
	if len(n1.weights) != len(n2.weights) {
		return false, fmt.Sprintf("neuron weight count: %d != %d", len(n1.weights), len(n2.weights))
	}
	for i, w1 := range n1.weights {
		if w2 := n2.weights[i]; w1 != w2 {
			return false, fmt.Sprintf("neuron %d-th weight: %+v != %+v", i, w1, w2)
		}
	}
	return true, ""
}

func equalFuncs(f1, f2 ActivationFunc) bool {
	return fmt.Sprintf("%v", f1) == fmt.Sprintf("%v", f2)
}
