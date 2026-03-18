package util

type LearningRateFunc func(grad float64) (rate float64)

// AnnealingLearningRate returns an annealed learning rate by the following formula:
//
//	η = η_0 / (1 + decay * epoch)
func AnnealingLearningRate(eta0, decay float64, epoch int) float64 {
	return eta0 / (1 + decay*float64(epoch))
}

func ConstantLearningRateFunc(rate float64) LearningRateFunc {
	return func(g float64) float64 {
		return rate * g
	}
}

func AnnealingLearningRateFunc(eta0, decay float64, epoch int) LearningRateFunc {
	eta := eta0 / (1 + decay*float64(epoch))
	return func(g float64) float64 {
		return eta * g
	}
}

func AdamLearningRateFunc(beta1, beta2, epsilon float64) LearningRateFunc {
	return func(g float64) float64 {
		return 0
	}
}
