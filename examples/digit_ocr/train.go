package main

import (
	"log"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/arrimpl"
	"github.com/Lucas-884e/gonet/util"
)

func constructNonGraphNetwork(inputLayerSize int, hiddenLayerSizes ...int) *arrimpl.MLP {
	nn := arrimpl.NewMLP(inputLayerSize, arrimpl.LossCrossEntropy, arrimpl.ReluActivator())
	for _, size := range hiddenLayerSizes {
		nn.AddLayer(size)
	}
	nn.AddLayerWithActivator(10, arrimpl.SoftmaxActivator(1))
	nn.RandomizeInitialWeights()
	return nn
}

func nonGraphTrain(trainingSet, validationSet []util.Sample) {
	nn := constructNonGraphNetwork(len(trainingSet[0].X), 32, 16)
	tr := arrimpl.NewTrainer(nn, isCorrect)
	log.Printf("[Before training] Validation set prediction precision: %g", tr.PredictionPrecision(validationSet))

	tr.Train(util.TrainConfig{
		BatchSize:    10,
		Epochs:       20,
		StopEps:      0,
		LearningRate: 0.05,
	}, trainingSet, validationSet)

	log.Printf("[After training] Validation set prediction precision: %g", tr.PredictionPrecision(validationSet))
}

func constructGraphNetwork(inputLayerSize int, hiddenLayerSizes ...int) gonet.Model {
	var (
		fanIn  = inputLayerSize
		layers []gonet.Layer
	)
	for _, fanOut := range hiddenLayerSizes {
		layers = append(layers,
			gonet.LinearLayer(fanIn, fanOut, true),
			gonet.ReluLayer())
		fanIn = fanOut
	}
	layers = append(layers,
		gonet.LinearLayer(fanIn, 10, true),
		gonet.SoftmaxLayer(1))
	return gonet.SequentialModel(layers...)
}

func graphTrain(trainingSet, validationSet []util.Sample) {
	var (
		inputLayerSize = len(trainingSet[0].X)
		mlp            = constructGraphNetwork(inputLayerSize, 32, 16)
	)
	precision := util.PredictionPrecision(mlp, validationSet, isCorrect)
	log.Printf("[Before training] Validation set prediction precision: %g", precision)

	var (
		cfg = util.TrainConfig{
			BatchSize:        10,
			Epochs:           20,
			StopEps:          0,
			LearningRate:     0.005,
			LogEpochInterval: 5,
		}
		timeCost = gonet.Train(mlp, trainingSet, &cfg, gonet.CrossEntropyLoss)
	)
	log.Printf("Training time cost: %s", timeCost)

	precision = util.PredictionPrecision(mlp, validationSet, isCorrect)
	log.Printf("[After training] Validation set prediction precision: %g", precision)
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
