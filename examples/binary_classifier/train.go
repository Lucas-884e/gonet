package main

import (
	"log"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/arrimpl"
	"github.com/Lucas-884e/gonet/util"
)

func constructNonGraphNetwork(hiddenLayerSizes ...int) *arrimpl.MLP {
	// Must use Tanh activator because the training data has target values within range: [-1, 1]
	nn := arrimpl.NewMLP(2, arrimpl.LossMaxMargin, arrimpl.ReluActivator())
	for _, size := range hiddenLayerSizes {
		nn.AddLayer(size)
	}
	nn.AddLayerWithActivator(1, arrimpl.LinearActivator())
	nn.RandomizeInitialWeights()
	return nn
}

func nonGraphTrain(trainingSet, validationSet, testSet []util.Sample) {
	nn := constructNonGraphNetwork(8)
	tr := arrimpl.NewTrainer(nn, func(pred, actual []float64) bool {
		return pred[0]*actual[0] > 0
	})
	log.Printf("(Before training) Prediction precision: validation set = %g | test set = %g",
		tr.PredictionPrecision(validationSet), tr.PredictionPrecision(testSet))

	tr.Train(util.TrainConfig{
		BatchSize:    20,
		Epochs:       20,
		StopEps:      0,
		LearningRate: 0.3,
	}, trainingSet, validationSet)

	nn.Print()
	log.Println("(After training) Test set prediction precision:", tr.PredictionPrecision(testSet))
}

func constructGraphNetwork(hiddenLayerSizes ...int) *gonet.MLP {
	mlp := gonet.NewMLP(2)
	for _, size := range hiddenLayerSizes {
		mlp.AddLayer(size, gonet.OpRelu, true)
	}
	mlp.AddLayer(1, gonet.OpNone, true)
	return mlp
}

func graphTrain(trainingSet, validationSet, testSet []util.Sample) {
	var (
		mlp       = constructGraphNetwork(8)
		lossFn    = gonet.TrainLossFunc(mlp, gonet.MaxMarginLoss)
		isCorrect = func(pred, actual []float64) bool { return pred[0]*actual[0] > 0 }
		tsSize    = len(trainingSet)
		delta     float64
	)

	precision := util.PredictionPrecision(mlp, validationSet, isCorrect)
	log.Printf("[Before training] Validation set prediction precision: %g", precision)

	var (
		cfg = util.TrainConfig{
			BatchSize:    20,
			Epochs:       20,
			StopEps:      0,
			LearningRate: 0.03,
		}
		optimizer = util.DefaultAdamOptimizer(mlp.Parameters(), cfg.LearningRate)
		loss      *gonet.Node
	)
train:
	for ep := 0; ep < cfg.Epochs; ep++ {
		// Shuffle before each epoch.
		util.ShuffleSamples(trainingSet)
		for start := 0; start < tsSize; start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, tsSize)
			loss = lossFn(trainingSet[start:end])
			loss.Backward()

			if delta = optimizer.Learn(); delta < cfg.StopEps {
				log.Printf("* Reached stopping criterion (delta = %g < %g).", delta, cfg.StopEps)
				break train
			}
		}

		if ep%5 == 0 || ep+1 == cfg.Epochs {
			precision := util.PredictionPrecision(mlp, validationSet, isCorrect)
			log.Printf("[Epoch %d] Validation set prediction precision (delta=%g): %g", ep+1, delta, precision)
		}
	}

	precision = util.PredictionPrecision(mlp, testSet, isCorrect)
	log.Printf("[After training] Test set prediction precision: %g", precision)
}
