package gonet

import (
	"log"
)

// AnnealingLearningRate returns an annealed learning rate by the following formula:
//
//	η = η_0 / (1 + decay * epoch)
func AnnealingLearningRate(eta0, decay float64, epoch int) float64 {
	return eta0 / (1 + decay*float64(epoch))
}

type IsCorrectFunc func(pred, actual []float64) bool

type TrainConfig struct {
	BatchSize    int
	Epochs       int
	StopEps      float64
	LearningRate float64
}

func NewTrainer(model *FCNNet, isCorrect IsCorrectFunc) *Trainer {
	return &Trainer{model: model, isCorrect: isCorrect}
}

type Trainer struct {
	model     *FCNNet
	isCorrect IsCorrectFunc
}

func (t *Trainer) Train(cfg TrainConfig, trainingSet, validationSet []Sample) {
	tsSize := len(trainingSet)
	var delta float64
train:
	for ep := 0; ep < cfg.Epochs; ep++ {
		// Shuffle before each epoch.
		ShuffleSamples(trainingSet)
		for start := 0; start < tsSize; start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > tsSize {
				end = tsSize
			}
			t.model.PropagateSamples(trainingSet[start:end])

			learningRate := AnnealingLearningRate(cfg.LearningRate, 1, ep)
			if delta = t.model.UpdateWeights(learningRate); delta < cfg.StopEps {
				log.Printf("* Reached stopping criterion (delta = %g < %g).", delta, cfg.StopEps)
				break train
			}
		}

		if ep%5 == 0 || ep+1 == cfg.Epochs {
			precision := t.PredictionPrecision(validationSet)
			log.Printf("[Epoch %d] Validation set prediction precision (delta=%g): %g", ep+1, delta, precision)
		}
	}
}

func (t *Trainer) PredictionPrecision(dataset []Sample) float32 {
	var correctCount int
	for _, sample := range dataset {
		if t.isCorrect(t.model.Predict(sample.X), sample.Y) {
			correctCount++
		}
	}
	return float32(correctCount) / float32(len(dataset))
}
