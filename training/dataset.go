package training

import "math/rand/v2"

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
