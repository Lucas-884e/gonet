package main

import (
	"log"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/arrimpl"
	"github.com/Lucas-884e/gonet/util"
)

func constructNonGraphNetwork(inputLayerSize int, hiddenLayerSizes ...int) *arrimpl.FCNNet {
	nn := arrimpl.NewFCNNet(inputLayerSize, arrimpl.LossCrossEntropy, arrimpl.ReluActivator())
	for _, size := range hiddenLayerSizes {
		nn.AddLayer(size)
	}
	nn.AddLayerWithActivator(10, arrimpl.SoftmaxActivator(1))
	nn.RandomizeInitialWeights()
	return nn
}

func nonGraphTrain(trainingSet, validationSet, testSet []util.Sample) {
	nn := constructNonGraphNetwork(len(trainingSet[0].X), 32, 16)
	tr := arrimpl.NewTrainer(nn, isCorrect)
	log.Printf("(Before training) Prediction precision: validation set = %g | testing set = %g",
		tr.PredictionPrecision(validationSet), tr.PredictionPrecision(testSet))

	tr.Train(util.TrainConfig{
		BatchSize:    10,
		Epochs:       20,
		StopEps:      0,
		LearningRate: 0.05,
	}, trainingSet, validationSet)

	log.Println("(After training) Testing set prediction precision:", tr.PredictionPrecision(testSet))
}

func constructGraphNetwork(inputLayerSize int, hiddenLayerSizes ...int) *gonet.MLP {
	mlp := gonet.NewMLP(inputLayerSize)
	for _, size := range hiddenLayerSizes {
		mlp.AddLayer(size, gonet.OpRelu, true)
	}
	mlp.AddLayer(10, gonet.OpSoftmax, true)
	return mlp
}

func graphTrain(trainingSet, validationSet, testSet []util.Sample) {
	var (
		inputLayerSize = len(trainingSet[0].X)
		mlp            = constructGraphNetwork(inputLayerSize, 32, 16)
		lossFn         = gonet.TrainLossFunc(mlp, gonet.CrossEntropyLoss)
		tsSize         = len(trainingSet)
		delta          float64
	)

	precision := util.PredictionPrecision(mlp, validationSet, isCorrect)
	log.Printf("[Before training] Validation set prediction precision: %g", precision)

	var (
		cfg = util.TrainConfig{
			BatchSize:    10,
			Epochs:       20,
			StopEps:      0,
			LearningRate: 0.005,
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

func isCorrect(pred, actual []float64) bool {
	findPrediction := func(ys []float64) int {
		maxIdx := -1
		var maxProb float64
		for i, y := range ys {
			if y > maxProb {
				maxProb = y
				maxIdx = i
			}
		}
		return maxIdx
	}
	return findPrediction(pred) == findPrediction(actual)
}
