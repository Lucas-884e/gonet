// Package arrimpl provides an array based implementation of Fully Connected
// Neural Network, which is much faster for large and deep networks.
package arrimpl

import (
	"math"

	"github.com/Lucas-884e/gonet/util"
)

func GenerateRandomLayerWeights(numNeurons, weightsPerNeuron int) [][]float64 {
	var (
		// Uniform random distribution in the range: [-sqrt(3/m), sqrt(3/m)],
		// where `m` is the number of synaptic connections of neuron `n`.
		max     = math.Sqrt(3 / float64(weightsPerNeuron))
		weights = make([][]float64, numNeurons)
	)
	for i := range weights {
		ws := make([]float64, weightsPerNeuron)
		for j := range ws {
			ws[j] = util.RandomUniformSample(-max, max)
		}
		weights[i] = ws
	}
	return weights
}
