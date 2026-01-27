package util

import (
	"encoding/csv"
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

type IsCorrectFunc func(pred, actual []float64) bool

func RandomUniformSample(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func ReadCSVDataSet(file string) ([][]string, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
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

// AnnealingLearningRate returns an annealed learning rate by the following formula:
//
//	η = η_0 / (1 + decay * epoch)
func AnnealingLearningRate(eta0, decay float64, epoch int) float64 {
	return eta0 / (1 + decay*float64(epoch))
}
