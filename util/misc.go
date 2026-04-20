// Package util ...
package util

import (
	"encoding/csv"
	"math"
	"math/rand/v2"
	"os"
)

type Sample struct {
	X []float64 // input
	Y []float64 // output
}

type TrainConfig struct {
	BatchSize    int
	Epochs       int
	StopEps      float64
	LearningRate float64
}

type IsCorrectFunc func(pred, label []float64) bool

type Predictor interface {
	Predict([]float64) []float64
}

func PredictionPrecision(p Predictor, testSet []Sample, isCorrect IsCorrectFunc) float32 {
	var correctCount int
	for _, sample := range testSet {
		if isCorrect(p.Predict(sample.X), sample.Y) {
			correctCount++
		}
	}
	return float32(correctCount) / float32(len(testSet))
}

func RandomUniformSample(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

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
			ws[j] = RandomUniformSample(-max, max)
		}
		weights[i] = ws
	}
	return weights
}

func ReadCSVDataSet(file string) ([][]string, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	r := csv.NewReader(f)
	return r.ReadAll()
}

func SplitDataSet[T any](samples []T) (training, validation, testing []T) {
	var (
		total           = len(samples)
		trainingSplit   = 7 * total / 10
		validationSplit = 9 * total / 10
	)
	return samples[:trainingSplit], samples[trainingSplit:validationSplit], samples[validationSplit:]
}

func ShuffleSamples[T any](samples []T) {
	rand.Shuffle(len(samples), func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})
}
