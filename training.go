package gonet

import (
	"log"

	"github.com/Lucas-884e/gonet/util"
)

func NewTrainer(model *FCNNet, isCorrect util.IsCorrectFunc) *Trainer {
	return &Trainer{model: model, isCorrect: isCorrect}
}

type Trainer struct {
	model     *FCNNet
	isCorrect util.IsCorrectFunc
}

func (t *Trainer) Train(cfg util.TrainConfig, trainingSet, validationSet []util.Sample) {
	tsSize := len(trainingSet)
	var delta float64
train:
	for ep := 0; ep < cfg.Epochs; ep++ {
		// Shuffle before each epoch.
		util.ShuffleSamples(trainingSet)
		for start := 0; start < tsSize; start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, tsSize)
			t.model.PropagateSamples(trainingSet[start:end])

			learningRate := util.AnnealingLearningRate(cfg.LearningRate, 1, ep)
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

func (t *Trainer) PredictionPrecision(dataset []util.Sample) float32 {
	var correctCount int
	for _, sample := range dataset {
		if t.isCorrect(t.model.Predict(sample.X), sample.Y) {
			correctCount++
		}
	}
	return float32(correctCount) / float32(len(dataset))
}
